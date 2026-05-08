"""Text deduplication: pluggable similarity strategies + near-dup orchestrators.

Pure helpers:

- :func:`sha256_text` — canonical SHA-256 of (optionally normalized) text
- :func:`normalize_text_for_dedup` — Unicode/whitespace normalization

Similarity strategies (Protocol + reference implementations):

- :class:`SimilarityStrategy` — Protocol; ``pairs_within`` + ``pairs_across``
- :class:`TfidfCosineStrategy` — default lexical near-dedup (TF-IDF + cosine).
  Recommended default. Scales to 100K+ texts (sklearn sparse + ANN k-NN).
- :class:`ExactNormalizedHashStrategy` — SHA-256-bucket exact-paraphrase dedup.
  O(n) hashing; trivially scales to millions of texts.
- :class:`EmbeddingCosineStrategy` — cosine on caller-supplied embeddings.
  Scale follows the embedder + sklearn k-NN; recommended for semantic
  near-dedup when lexical signal is insufficient.
- :class:`JaccardNgramStrategy` — set-based n-gram Jaccard. **Diagnostic /
  small-corpus only**: brute-force pairwise (O(n²) memory + compute). Stalls
  on corpora above ~1K texts. Prefer :class:`TfidfCosineStrategy` or
  :class:`EmbeddingCosineStrategy` (with a MinHash/LSH-backed embedder) at
  any production scale.

Orchestrators (strategy-agnostic):

- :func:`near_dedup` — forward-scan greedy near-deduplication
- :func:`cross_dedup` — drop eval rows near-duplicate to any train row
- :class:`DedupReport` — frozen audit-trail of which rows were dropped and why

Notes
-----

**NFC normalization asymmetry across strategies.** Only
:class:`ExactNormalizedHashStrategy` applies Unicode NFC normalization
before similarity (via :func:`normalize_text_for_dedup`). The TF-IDF,
embedding, and Jaccard strategies treat composed and decomposed accents
as different inputs by default. If your corpus mixes NFC and NFD forms
(common when concatenating data from different OSes), normalize once at
load time with :func:`normalize_text_for_dedup` and then call any
strategy.

References
----------
.. [1] Lee, K., et al. "Deduplicating training data makes language models
       better." ACL 2022. (NearDup pipeline; modern authority on
       dedup-and-LM-quality.)
.. [2] Penedo, G., et al. "The RefinedWeb dataset for Falcon LLM."
       NeurIPS Datasets & Benchmarks, 2023.
.. [3] Unicode Standard Annex #15 (Unicode Normalization Forms).
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DedupReport",
    "EmbeddingCosineStrategy",
    "ExactNormalizedHashStrategy",
    "JaccardNgramStrategy",
    "SimilarityStrategy",
    "TfidfCosineStrategy",
    "cross_dedup",
    "near_dedup",
    "normalize_text_for_dedup",
    "sha256_text",
]

DEFAULT_DEDUP_THRESHOLD = 0.9


@dataclass(frozen=True, slots=True)
class DedupReport:
    """Outcome of a near-dedup pass.

    Parameters
    ----------
    kept_indices : list[int]
        Sorted positions in the input ``texts`` retained after dedup.
    dropped_pairs : list[tuple[int, int, float]]
        Triples ``(dropped_idx, kept_idx, similarity)`` for audit.
    threshold : float
        Cosine-similarity threshold the dedup pass used.
    n_input : int
        Length of the input list (so kept + dropped invariant is checkable).

    Examples
    --------
    >>> report = DedupReport(
    ...     kept_indices=[0, 2],
    ...     dropped_pairs=[(1, 0, 0.95)],
    ...     threshold=0.9,
    ...     n_input=3,
    ... )
    >>> report.n_kept + report.n_dropped == report.n_input
    True

    Notes
    -----
    The ``kept_indices`` and the first element of each ``dropped_pairs``
    triple form a disjoint partition of ``range(n_input)``. Callers can use
    this property to reconstruct the dropped-set without re-running dedup.
    """

    kept_indices: list[int]
    dropped_pairs: list[tuple[int, int, float]]
    threshold: float
    n_input: int

    @property
    def n_kept(self) -> int:
        """Number of input rows retained after deduplication."""
        return len(self.kept_indices)

    @property
    def n_dropped(self) -> int:
        """Number of input rows dropped as near-duplicates."""
        return len(self.dropped_pairs)


def normalize_text_for_dedup(text: str) -> str:
    """Canonical text normalization for hashing and deduplication.

    Applies:

    1. Unicode NFC normalization (composes accent characters)
    2. Lowercase
    3. Collapse runs of whitespace to single spaces
    4. Strip leading/trailing whitespace

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Normalized text.

    Examples
    --------
    >>> normalize_text_for_dedup("Hello   World")
    'hello world'
    >>> normalize_text_for_dedup("  Foo\\tBar\\n")
    'foo bar'
    >>> normalize_text_for_dedup(normalize_text_for_dedup("X")) == normalize_text_for_dedup("X")
    True
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    nfc = unicodedata.normalize("NFC", text)
    lowered = nfc.lower()
    collapsed = " ".join(lowered.split())
    return collapsed


def sha256_text(text: str, *, normalize: bool = True) -> str:
    """SHA-256 hex digest of (optionally normalized) text.

    Parameters
    ----------
    text : str
    normalize : bool, optional
        If True (default), apply :func:`normalize_text_for_dedup` first.
        Set False to hash the raw bytes.

    Returns
    -------
    str
        64-character hex digest.

    Examples
    --------
    >>> sha256_text("Hello   World") == sha256_text("hello world")
    True
    >>> len(sha256_text("foo"))
    64
    >>> sha256_text("foo", normalize=False) != sha256_text("FOO", normalize=False)
    True
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    canonical = normalize_text_for_dedup(text) if normalize else text
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SimilarityStrategy Protocol + reference implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class SimilarityStrategy(Protocol):
    """Pluggable similarity backend for :func:`near_dedup` / :func:`cross_dedup`.

    Strategies own vectorization plus nearest-neighbor lookup. Similarities
    are floats; higher values mean more similar. Each strategy chooses its
    own scale (e.g. cosine in ``[-1, 1]``, Jaccard in ``[0, 1]``, hash
    collision in ``{0.0, 1.0}``); threshold semantics live in the orchestrators
    (``near_dedup`` / ``cross_dedup``), not the strategy itself.

    The shape mirrors the existing :class:`eval_toolkit.harness.Scorer`
    Protocol: structural typing, no ABC inheritance, runtime-checkable so
    ``isinstance(obj, SimilarityStrategy)`` returns True for any object that
    exposes the two methods.

    Notes
    -----
    Both methods return aligned ``(similarities, indices)`` arrays of shape
    ``(n_query, k_eff)`` where ``k_eff = min(k, n_reference)``. ``indices[i]``
    are positions into the reference list; for ``pairs_within`` the reference
    list IS the input list, so ``indices[i, 0] == i`` is conventional (each
    text is its own most-similar neighbor).

    See Also
    --------
    eval_toolkit.harness.Scorer : analogous Protocol pattern for predict_proba.
    """

    def pairs_within(
        self, texts: Sequence[str], k: int
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """k-NN of each text against the same list."""
        ...

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """k-NN of each query against ``reference_texts``."""
        ...


@dataclass(frozen=True, slots=True)
class TfidfCosineStrategy:
    """TF-IDF (n-gram) cosine similarity — the default lexical near-dedup.

    Wraps :class:`sklearn.feature_extraction.text.TfidfVectorizer` plus
    :class:`sklearn.neighbors.NearestNeighbors`. With default parameters,
    behavior is bit-for-bit equivalent to the inline implementation shipped
    in eval-toolkit v0.1.0.

    Parameters
    ----------
    ngram_range : tuple[int, int], optional
        ``(min_n, max_n)`` n-gram range. Default ``(1, 3)``.
    min_df : int, optional
        Minimum document frequency for a term. Default ``1``.
    lowercase : bool, optional
        Lowercase before vectorizing. Default ``True``.

    Examples
    --------
    >>> strat = TfidfCosineStrategy()
    >>> sims, idx = strat.pairs_within(
    ...     ["the quick fox", "the quick fox!", "lorem ipsum"], k=2
    ... )
    >>> sims.shape, idx.shape
    ((3, 2), (3, 2))
    >>> bool(idx[0, 0] == 0)  # self at slot 0
    True

    Notes
    -----
    TF-IDF cosine is a fast lexical signal — it will miss paraphrase
    duplicates that share no n-grams. For semantic-paraphrase dedup, see
    :class:`EmbeddingCosineStrategy`.

    Scaling: scales to ~100K+ texts on a single machine via sklearn's
    sparse vectorizer + approximate nearest neighbors. This is the
    recommended default for any non-trivial corpus.
    """

    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 1
    lowercase: bool = True

    def _vectorizer(self) -> TfidfVectorizer:
        """Construct a fresh TfidfVectorizer with the strategy's config."""
        return TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=self.lowercase,
        )

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """k-NN within ``texts`` via TF-IDF + cosine NearestNeighbors."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
        vec = self._vectorizer()
        tfidf = vec.fit_transform(list(texts))
        k_eff = min(k, n)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(tfidf)
        distances, indices = nn.kneighbors(tfidf)
        return 1.0 - distances, indices.astype(np.int64)

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """k-NN of ``query_texts`` against ``reference_texts``."""
        if not reference_texts or not query_texts:
            return (
                np.zeros((len(query_texts), 0), dtype=np.float64),
                np.zeros((len(query_texts), 0), dtype=np.int64),
            )
        vec = self._vectorizer()
        # Fit on union so vocabulary covers both sides; matches v0.1.0 behavior.
        vec.fit(list(reference_texts) + list(query_texts))
        tfidf_ref = vec.transform(list(reference_texts))
        tfidf_query = vec.transform(list(query_texts))
        k_eff = min(k, len(reference_texts))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(tfidf_ref)
        distances, indices = nn.kneighbors(tfidf_query)
        return 1.0 - distances, indices.astype(np.int64)


@dataclass(frozen=True, slots=True)
class ExactNormalizedHashStrategy:
    """SHA-256 hash-bucket dedup; similarities are exactly ``{0.0, 1.0}``.

    Reuses :func:`normalize_text_for_dedup` + :func:`sha256_text`. Two texts
    are similar iff they share a hash bucket — useful when the operational
    sense of "leakage" is exact paraphrase (after Unicode/whitespace/case
    normalization) rather than near-paraphrase.

    Parameters
    ----------
    normalize : bool, optional
        If ``True`` (default), apply :func:`normalize_text_for_dedup` before
        hashing. Set ``False`` to bucket on raw bytes only.

    Examples
    --------
    >>> strat = ExactNormalizedHashStrategy()
    >>> sims, idx = strat.pairs_within(["foo", "FOO", "bar"], k=2)
    >>> # Texts 0 and 1 collide after lowercasing; text 2 isolated.
    >>> bool(sims[0, 0] == 1.0 and idx[0, 0] == 0)  # self at slot 0
    True
    >>> bool(sims[0, 1] == 1.0 and idx[0, 1] == 1)  # collision with text 1
    True
    >>> bool(sims[2, 1] == 0.0)  # text 2 has no collision in the corpus
    True

    Notes
    -----
    Threshold ``t`` semantics for this strategy: any ``t`` in ``(0, 1]``
    catches all collisions (since collisions are sim=1.0); a threshold
    ``> 1.0`` catches nothing. The default ``DEFAULT_DEDUP_THRESHOLD = 0.9``
    works as expected.

    Scaling: O(n) hashing + O(n) bucket lookup — trivially scales to
    millions of texts. Use this strategy when "leakage" means *exact match
    after normalization*, not paraphrase. For lexical near-paraphrase use
    :class:`TfidfCosineStrategy`; for semantic near-paraphrase use
    :class:`EmbeddingCosineStrategy`.
    """

    normalize: bool = True

    def _hash(self, text: str) -> str:
        """SHA-256 of (optionally normalized) text."""
        return sha256_text(text, normalize=self.normalize)

    @staticmethod
    def _build_neighbors(
        n: int,
        k_eff: int,
        bucket_for_query: list[list[int]],
        ref_hashes: Sequence[str],
        query_hashes: Sequence[str],
        n_ref: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack each query's collision list (then arbitrary fillers) into (n, k)."""
        sims = np.zeros((n, k_eff), dtype=np.float64)
        indices = np.zeros((n, k_eff), dtype=np.int64)
        ref_index_set = set(range(n_ref))
        for i in range(n):
            collisions = bucket_for_query[i]
            non_collisions = sorted(ref_index_set - set(collisions))
            ranked = collisions + non_collisions
            chosen = ranked[:k_eff]
            for slot, j in enumerate(chosen):
                indices[i, slot] = j
                sims[i, slot] = 1.0 if ref_hashes[j] == query_hashes[i] else 0.0
        return sims, indices

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
        hashes = [self._hash(t) for t in texts]
        bucket: dict[str, list[int]] = {}
        for i, h in enumerate(hashes):
            bucket.setdefault(h, []).append(i)
        # Self at slot 0; then other collisions; then arbitrary fillers.
        bucket_for_query = [[i] + [j for j in bucket[hashes[i]] if j != i] for i in range(n)]
        k_eff = min(k, n)
        return self._build_neighbors(n, k_eff, bucket_for_query, hashes, hashes, n)

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q, n_r = len(query_texts), len(reference_texts)
        if n_q == 0 or n_r == 0:
            return (
                np.zeros((n_q, 0), dtype=np.float64),
                np.zeros((n_q, 0), dtype=np.int64),
            )
        ref_hashes = [self._hash(t) for t in reference_texts]
        q_hashes = [self._hash(t) for t in query_texts]
        ref_bucket: dict[str, list[int]] = {}
        for i, h in enumerate(ref_hashes):
            ref_bucket.setdefault(h, []).append(i)
        bucket_for_query = [list(ref_bucket.get(qh, [])) for qh in q_hashes]
        k_eff = min(k, n_r)
        return self._build_neighbors(n_q, k_eff, bucket_for_query, ref_hashes, q_hashes, n_r)


@dataclass(frozen=True, slots=True)
class EmbeddingCosineStrategy:
    """Cosine similarity on caller-supplied embeddings.

    The toolkit owns cosine + k-NN; the caller owns the embedder. This keeps
    the toolkit dep-free of any specific embedding library
    (sentence-transformers, OpenAI, local model, etc.) while still offering
    a turnkey strategy class.

    Parameters
    ----------
    embedder : Callable[[Sequence[str]], np.ndarray]
        Maps a sequence of texts to a 2-D array of shape ``(n, d)``. Caller
        is responsible for batching, GPU placement, model loading, etc.

    Examples
    --------
    >>> import numpy as np
    >>> def stub_embedder(ts):
    ...     # One-hot per text index → behaves like exact-match on identity.
    ...     return np.eye(len(ts))
    >>> strat = EmbeddingCosineStrategy(stub_embedder)
    >>> sims, idx = strat.pairs_within(["a", "b", "c"], k=3)
    >>> bool(sims.shape == (3, 3) and idx.shape == (3, 3))
    True
    >>> bool(idx[0, 0] == 0)  # self at slot 0
    True

    Notes
    -----
    The embedder is called once per ``pairs_*`` invocation. Persistent
    embedding caches are out of scope — callers should wrap their embedder
    in any cache they need.

    Scaling: dominated by the embedder. With sentence-transformers on GPU,
    100K+ texts is routine; with OpenAI / remote-API embedders, throughput
    is API-limited but k-NN itself remains fast (sklearn cosine NN). This
    is the recommended strategy whenever lexical TF-IDF signal is
    insufficient (paraphrase, multilingual, semantic-near-duplicate).
    """

    embedder: Callable[[Sequence[str]], np.ndarray]

    def _embed(self, texts: Sequence[str], name: str) -> np.ndarray:
        """Coerce embedder output to a 2-D float64 ndarray + validate shape."""
        emb = np.asarray(self.embedder(texts), dtype=np.float64)
        if emb.ndim != 2:
            raise ValueError(f"embedder must return 2-D array; {name} got shape {emb.shape}")
        if emb.shape[0] != len(texts):
            raise ValueError(
                f"embedder must return one row per text; {name} got shape "
                f"{emb.shape} for {len(texts)} texts"
            )
        return emb

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
        emb = self._embed(texts, "texts")
        k_eff = min(k, n)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(emb)
        distances, indices = nn.kneighbors(emb)
        return 1.0 - distances, indices.astype(np.int64)

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q, n_r = len(query_texts), len(reference_texts)
        if n_q == 0 or n_r == 0:
            return (
                np.zeros((n_q, 0), dtype=np.float64),
                np.zeros((n_q, 0), dtype=np.int64),
            )
        ref_emb = self._embed(reference_texts, "reference_texts")
        query_emb = self._embed(query_texts, "query_texts")
        # Cross-call dimension consistency — buggy embedders that return
        # different `d` for query vs reference would silently mis-align cosine.
        if ref_emb.shape[1] != query_emb.shape[1]:
            raise ValueError(
                f"embedder returned inconsistent feature dimensions: "
                f"reference_texts has d={ref_emb.shape[1]}, "
                f"query_texts has d={query_emb.shape[1]}"
            )
        k_eff = min(k, n_r)
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(ref_emb)
        distances, indices = nn.kneighbors(query_emb)
        return 1.0 - distances, indices.astype(np.int64)


@dataclass(frozen=True, slots=True)
class JaccardNgramStrategy:
    """Set-based n-gram Jaccard similarity — **diagnostic / small-corpus only**.

    .. warning::

        Brute-force pairwise — O(n²) in the corpus size for both memory and
        compute. Empirical scaling on a single core: ~1K texts in under a
        second; ~10K texts ≈ tens of seconds; ~100K texts will exhaust
        memory and is effectively unusable. **Do not use this strategy at
        production scale.** Prefer :class:`TfidfCosineStrategy` (default,
        sparse + ANN k-NN, scales to 100K+) or :class:`EmbeddingCosineStrategy`
        backed by a MinHash/LSH or sentence-transformer embedder.

    When this strategy is the right tool:

    - **Tiny diagnostic corpora** (< ~500 texts) where you want a clean
      mathematical match-counting interpretation of similarity.
    - **Token-order-invariant fingerprints** where lexical TF-IDF cosine
      over-weights position (e.g., SQL fragments, CLI-flag strings,
      shell-command normalization, JSON-key bag-of-words).
    - **Reference / sanity-check** implementation against which you want to
      validate a faster MinHash/LSH approximation.

    Parameters
    ----------
    n : int, optional
        N-gram size. Default ``3``. Must be ``≥ 1``.
    analyzer : {'char', 'word'}, optional
        N-gram unit. Default ``'char'``.

    Examples
    --------
    >>> strat = JaccardNgramStrategy(n=2, analyzer='char')
    >>> # bigrams("abc") = {ab, bc}; bigrams("abd") = {ab, bd};
    >>> # ∩ = {ab}; ∪ = {ab, bc, bd} → J = 1/3.
    >>> sims, idx = strat.pairs_within(["abc", "abd"], k=2)
    >>> bool(abs(float(sims[0, 1]) - 1 / 3) < 1e-9)
    True

    Notes
    -----
    Jaccard on *sets* of n-grams (order-invariant) gives a strict-match
    similarity in ``[0, 1]``: identical n-gram sets map to 1.0, disjoint
    n-gram sets to 0.0. For approximate Jaccard at production scale, see
    MinHash + LSH (Broder 1997, Indyk & Motwani 1998); the toolkit does not
    ship a MinHash strategy in v0.2 — wrap your preferred MinHash library in
    :class:`EmbeddingCosineStrategy` if you need it.

    References
    ----------
    .. [1] Broder, A. "On the resemblance and containment of documents."
           Compression and Complexity of Sequences, 1997.
    .. [2] Indyk, P. & Motwani, R. "Approximate nearest neighbors: Towards
           removing the curse of dimensionality." STOC 1998.
    """

    n: int = 3
    analyzer: Literal["char", "word"] = "char"

    def __post_init__(self) -> None:
        """Validate constructor arguments (frozen dataclass invariants)."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.analyzer not in ("char", "word"):
            raise ValueError(f"analyzer must be 'char' or 'word', got {self.analyzer!r}")

    def _ngrams(self, text: str) -> set[str]:
        """N-gram set for ``text`` per the configured analyzer."""
        if self.analyzer == "char":
            if len(text) < self.n:
                return {text} if text else set()
            return {text[i : i + self.n] for i in range(len(text) - self.n + 1)}
        tokens = text.split()
        if len(tokens) < self.n:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity ``|a ∩ b| / |a ∪ b|``; both empty → 1.0."""
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _pairwise(
        self,
        queries: Sequence[set[str]],
        references: Sequence[set[str]],
    ) -> np.ndarray:
        """Brute-force Jaccard matrix of shape ``(len(queries), len(references))``."""
        m = np.zeros((len(queries), len(references)), dtype=np.float64)
        for i, qs in enumerate(queries):
            for j, rs in enumerate(references):
                m[i, j] = self._jaccard(qs, rs)
        return m

    @staticmethod
    def _topk(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by descending similarity per row; stable on ties."""
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
        sims = np.take_along_axis(matrix, order, axis=1)
        return sims.astype(np.float64), order.astype(np.int64)

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
        ngs = [self._ngrams(t) for t in texts]
        k_eff = min(k, n)
        return self._topk(self._pairwise(ngs, ngs), k_eff)

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q, n_r = len(query_texts), len(reference_texts)
        if n_q == 0 or n_r == 0:
            return (
                np.zeros((n_q, 0), dtype=np.float64),
                np.zeros((n_q, 0), dtype=np.int64),
            )
        q_ngs = [self._ngrams(t) for t in query_texts]
        r_ngs = [self._ngrams(t) for t in reference_texts]
        k_eff = min(k, n_r)
        return self._topk(self._pairwise(q_ngs, r_ngs), k_eff)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def near_dedup(
    texts: list[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
    *,
    strategy: SimilarityStrategy | None = None,
) -> DedupReport:
    """Greedy near-deduplication via a pluggable similarity strategy.

    Forward-scan: for each kept text ``i``, drop all ``j > i`` with
    ``similarity(i, j) >= threshold`` per the active strategy. Default
    strategy is :class:`TfidfCosineStrategy` with literal v0.1.0 defaults
    (preserved bit-for-bit; existing callers keep working unchanged).

    Parameters
    ----------
    texts : list[str]
    threshold : float, optional
        Similarity threshold in (0, 1). Default 0.9. Strategy-specific scale:
        for cosine-style strategies (TF-IDF, embedding) similarities are in
        [-1, 1]; for hash-bucket dedup similarities are in {0.0, 1.0}; for
        Jaccard in [0, 1].
    k_neighbors : int, optional
        Maximum neighbors to consider per query. Default 20.
    strategy : SimilarityStrategy or None, optional
        Pluggable similarity backend. ``None`` (default) instantiates
        :class:`TfidfCosineStrategy` with constructor defaults.

    Returns
    -------
    DedupReport

    Raises
    ------
    TypeError
        If ``texts`` is not a list.
    ValueError
        If ``threshold`` is outside (0, 1).

    Examples
    --------
    Default strategy (TF-IDF cosine):

    >>> texts = ["the quick brown fox", "the quick brown fox!", "lorem ipsum"]
    >>> report = near_dedup(texts, threshold=0.8)
    >>> report.n_kept >= 2  # at most one near-duplicate dropped
    True
    >>> set(report.kept_indices) | {p[0] for p in report.dropped_pairs} == set(range(3))
    True

    Custom strategy (exact-match-after-normalization):

    >>> report_exact = near_dedup(
    ...     ["foo", "FOO", "bar"], threshold=0.5,
    ...     strategy=ExactNormalizedHashStrategy(),
    ... )
    >>> report_exact.n_kept  # "foo"/"FOO" collide; "bar" isolated
    2

    Notes
    -----
    The orchestrator is strategy-agnostic: it dispatches to
    ``strategy.pairs_within(texts, k_neighbors)`` and applies the threshold
    plus forward-scan greedy drop logic. Different "senses of leakage"
    (lexical, semantic, exact, n-gram-set) are encoded by swapping the
    strategy.

    **Order dependence**: forward-scan greedy is order-dependent — for any
    cluster of near-duplicates, the *first* occurrence is kept and later
    occurrences are dropped. To make dedup reproducible across re-runs
    that may permute the input, **sort inputs by a canonical key** (URL,
    document id, primary key) before calling. This is the canonical
    approach in modern dedup pipelines (Lee et al. 2022 ACL "NearDup";
    Penedo et al. 2023 RefinedWeb; Penedo et al. 2025 FineWeb2).

    References
    ----------
    .. [1] Lee, K., et al. "Deduplicating training data makes language
           models better." ACL 2022.
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts must be a list, got {type(texts).__name__}")
    n = len(texts)
    if n == 0:
        return DedupReport([], [], threshold, 0)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    active_strategy: SimilarityStrategy = (
        strategy if strategy is not None else TfidfCosineStrategy()
    )
    similarities, indices = active_strategy.pairs_within(texts, k_neighbors)

    kept_mask = np.ones(n, dtype=bool)
    dropped: list[tuple[int, int, float]] = []

    for i in range(n):
        if not kept_mask[i]:
            continue
        for sim, j in zip(similarities[i], indices[i], strict=True):
            j_int = int(j)
            if j_int == i or j_int < i:
                continue
            if float(sim) >= threshold and kept_mask[j_int]:
                kept_mask[j_int] = False
                dropped.append((j_int, i, float(sim)))

    kept_indices = np.where(kept_mask)[0].tolist()
    return DedupReport(kept_indices, dropped, threshold, n)


def cross_dedup(
    train_texts: list[str],
    eval_texts: list[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
    *,
    strategy: SimilarityStrategy | None = None,
) -> list[int]:
    """Return eval indices to KEEP (drop those near-duplicate to any train text).

    Used to scrub OOD eval slices of any train-pool leakage before reporting
    OOD metrics. The notion of "near-duplicate" is owned by ``strategy``.

    Parameters
    ----------
    train_texts, eval_texts : list[str]
    threshold : float, optional
        Similarity threshold in (0, 1). Default 0.9.
    k_neighbors : int, optional
        Maximum neighbors to consider per eval text. Default 20.
    strategy : SimilarityStrategy or None, optional
        Pluggable similarity backend. ``None`` (default) instantiates
        :class:`TfidfCosineStrategy` with constructor defaults.

    Returns
    -------
    list[int]
        Indices into ``eval_texts`` of rows that are NOT near-duplicate to
        any train text. Order preserved.

    Examples
    --------
    >>> train = ["the quick brown fox", "lorem ipsum dolor sit amet"]
    >>> eval_set = ["the quick brown fox!", "completely different text"]
    >>> kept = cross_dedup(train, eval_set, threshold=0.8)
    >>> 1 in kept  # second eval text has no train match
    True
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if not train_texts:
        return list(range(len(eval_texts)))
    if not eval_texts:
        return []

    active_strategy: SimilarityStrategy = (
        strategy if strategy is not None else TfidfCosineStrategy()
    )
    similarities, _indices = active_strategy.pairs_across(eval_texts, train_texts, k_neighbors)
    if similarities.size == 0:
        return list(range(len(eval_texts)))
    max_sim_per_eval = similarities.max(axis=1)
    keep_mask = max_sim_per_eval < threshold
    return [int(i) for i in np.where(keep_mask)[0]]
