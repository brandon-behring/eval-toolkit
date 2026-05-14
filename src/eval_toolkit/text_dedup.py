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
    "MinHashLSHStrategy",
    "SimilarityAuditFinding",
    "SimilarityAuditReport",
    "SimilarityStrategy",
    "TfidfCosineStrategy",
    "audit_source_label_similarity",
    "cross_dedup",
    "cross_dedup_pairs",
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


SimilarityRelation = Literal[
    "unspecified",
    "within_source",
    "cross_source",
    "same_label",
    "cross_label",
    "within_source_same_label",
    "within_source_cross_label",
    "cross_source_same_label",
    "cross_source_cross_label",
]


@dataclass(frozen=True, slots=True)
class SimilarityAuditFinding:
    """One high-similarity pair found during a non-dropping audit."""

    left_index: int
    right_index: int
    similarity: float
    relation: SimilarityRelation = "unspecified"
    left_source: str | None = None
    right_source: str | None = None
    left_label: object | None = None
    right_label: object | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        out: dict[str, object] = {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "similarity": self.similarity,
            "relation": self.relation,
        }
        if self.left_source is not None:
            out["left_source"] = self.left_source
        if self.right_source is not None:
            out["right_source"] = self.right_source
        if self.left_label is not None:
            out["left_label"] = self.left_label
        if self.right_label is not None:
            out["right_label"] = self.right_label
        return out


@dataclass(frozen=True, slots=True)
class SimilarityAuditReport:
    """Non-dropping source/label-aware similarity audit report."""

    findings: list[SimilarityAuditFinding]
    threshold: float
    n_input: int
    strategy: str
    k_neighbors: int

    @property
    def n_findings(self) -> int:
        """Number of high-similarity pairs in the report."""
        return len(self.findings)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "threshold": self.threshold,
            "n_input": self.n_input,
            "strategy": self.strategy,
            "k_neighbors": self.k_neighbors,
            "n_findings": self.n_findings,
        }


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
# MinHash + LSH: production-scale approximate Jaccard
# ---------------------------------------------------------------------------


# Mersenne prime > 2**32 — used for the universal hash family in MinHash.
_MINHASH_PRIME: int = (1 << 61) - 1
_MINHASH_MAX_HASH: int = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class MinHashLSHStrategy:
    r"""Approximate Jaccard via MinHash + LSH banding (Broder 1997, Indyk-Motwani 1998).

    Production-scale alternative to :class:`JaccardNgramStrategy`. Computes
    MinHash signatures of length ``num_perm`` for each text using a
    universal hash family, then Locality-Sensitive Hashing (LSH) with
    ``bands × rows = num_perm`` partitions the signature space so any two
    texts with high Jaccard similarity probably collide in at least one
    band. For each query, we expand its candidate set from the LSH
    buckets, compute exact Jaccard on the surviving candidates, and
    return the top-``k`` by similarity.

    .. note::

        This is an *approximate* Jaccard. Two texts with true Jaccard ≥
        threshold are *probably* (not certainly) discovered as
        candidates; the false-negative probability decreases with the
        ``num_perm`` × bands tuning. See Indyk & Motwani 1998 for the
        analysis of the LSH band-curve.

    Parameters
    ----------
    n : int, optional
        Character n-gram size for shingling. Default ``3``.
    num_perm : int, optional
        Number of MinHash permutations / signature length. Default
        ``128``. Larger num_perm → more accurate Jaccard estimate, more
        compute. ``num_perm`` must equal ``bands × rows_per_band``.
    bands : int, optional
        Number of LSH bands. Default ``16``. With default ``num_perm=128``
        this gives ``rows_per_band = 8``. The LSH probability of two
        texts with true Jaccard :math:`s` colliding in at least one band
        is :math:`1 - (1 - s^r)^b`. Tune ``(bands, rows_per_band)`` to
        the threshold you care about; e.g. ``(20, 5)`` flips around
        :math:`s = 0.5`, ``(16, 8)`` flips around :math:`s ≈ 0.74`.
    seed : int, optional
        Deterministic seed for the universal-hash coefficients. Default
        ``42``.

    Examples
    --------
    >>> import numpy as np
    >>> strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16, seed=0)
    >>> texts = ["the quick brown fox", "the quick brown fox!", "lorem ipsum"]
    >>> sims, idx = strat.pairs_within(texts, k=2)
    >>> sims.shape, idx.shape
    ((3, 2), (3, 2))

    Notes
    -----
    Scaling: ``num_perm × n_texts`` MinHash work for signatures, then
    LSH bucketing is O(bands · n_texts) hashing + O(candidates per query)
    Jaccard recomputation. For ``n_texts = 100K`` with default params,
    this is dramatically faster than :class:`JaccardNgramStrategy`'s
    O(n²) brute-force.

    The "fillers" returned for queries with fewer than ``k`` candidates
    are filled in with arbitrary indices and similarity ``0.0``;
    downstream callers (``near_dedup``, ``cross_dedup``) only act on
    similarities ``≥ threshold`` so the fillers are correctly ignored.

    References
    ----------
    .. [1] Broder, A. "On the resemblance and containment of documents."
           Compression and Complexity of Sequences, 1997.
    .. [2] Indyk, P. & Motwani, R. "Approximate nearest neighbors:
           Towards removing the curse of dimensionality." STOC 1998.
    """

    n: int = 3
    num_perm: int = 128
    bands: int = 16
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate constructor arguments."""
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")
        if self.num_perm < 8:
            raise ValueError(f"num_perm must be ≥ 8, got {self.num_perm}")
        if self.bands < 1 or self.bands > self.num_perm:
            raise ValueError(f"bands must be in [1, num_perm={self.num_perm}], got {self.bands}")
        if self.num_perm % self.bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by bands "
                f"({self.bands}) for evenly-sized rows"
            )

    @property
    def rows_per_band(self) -> int:
        """Rows per LSH band; equals ``num_perm // bands``."""
        return self.num_perm // self.bands

    def _hash_coefs(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (a, b) coefficients for the universal hash family.

        The hash is :math:`h_i(x) = ((a_i x + b_i) \\mod p) \\mod 2^{32}`,
        a 4-universal hash family per Carter & Wegman 1979.
        """
        rng = np.random.default_rng(self.seed)
        a = rng.integers(1, _MINHASH_PRIME, size=self.num_perm, dtype=np.int64)
        b = rng.integers(0, _MINHASH_PRIME, size=self.num_perm, dtype=np.int64)
        return a, b

    def _shingles(self, text: str) -> set[str]:
        """Char n-gram set for ``text`` (matches JaccardNgramStrategy.char-mode)."""
        if len(text) < self.n:
            return {text} if text else set()
        return {text[i : i + self.n] for i in range(len(text) - self.n + 1)}

    def _signature(self, shingle_set: set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """MinHash signature of length ``num_perm`` for one shingle set."""
        if not shingle_set:
            return np.full(self.num_perm, _MINHASH_MAX_HASH, dtype=np.int64)
        # Hash each shingle to a 32-bit integer (deterministic via SHA-256).
        shingle_hashes = np.fromiter(
            (
                int.from_bytes(
                    hashlib.sha256(s.encode("utf-8")).digest()[:4],
                    "big",
                )
                for s in shingle_set
            ),
            dtype=np.int64,
            count=len(shingle_set),
        )
        # For each of the num_perm permutations, take the min over the shingle hashes.
        # h_i(x) = ((a_i * x + b_i) mod prime) mod 2^32 — universal hash.
        permuted = (
            (np.outer(shingle_hashes, a) + b[np.newaxis, :])
            % _MINHASH_PRIME
            % (_MINHASH_MAX_HASH + 1)
        )
        sig: np.ndarray = permuted.min(axis=0).astype(np.int64)
        return sig

    def _signatures_for(self, texts: Sequence[str]) -> np.ndarray:
        """Stack of MinHash signatures of shape ``(len(texts), num_perm)``."""
        a, b = self._hash_coefs()
        n = len(texts)
        sigs = np.zeros((n, self.num_perm), dtype=np.int64)
        for i, t in enumerate(texts):
            sigs[i] = self._signature(self._shingles(t), a, b)
        return sigs

    def _build_lsh_index(self, sigs: np.ndarray) -> list[dict[bytes, list[int]]]:
        """Bucket signatures into ``self.bands`` band-keyed dicts."""
        rows = self.rows_per_band
        indices: list[dict[bytes, list[int]]] = [{} for _ in range(self.bands)]
        for i in range(sigs.shape[0]):
            for band in range(self.bands):
                start = band * rows
                key = sigs[i, start : start + rows].tobytes()
                indices[band].setdefault(key, []).append(i)
        return indices

    def _query_lsh(
        self,
        sigs_query: np.ndarray,
        sigs_ref: np.ndarray,
        ref_index: list[dict[bytes, list[int]]],
    ) -> list[set[int]]:
        """Per-query candidate set: ref indices sharing ≥ 1 band hash."""
        rows = self.rows_per_band
        candidates: list[set[int]] = [set() for _ in range(sigs_query.shape[0])]
        for i in range(sigs_query.shape[0]):
            for band in range(self.bands):
                start = band * rows
                key = sigs_query[i, start : start + rows].tobytes()
                if key in ref_index[band]:
                    candidates[i].update(ref_index[band][key])
        return candidates

    @staticmethod
    def _exact_jaccard(a: set[str], b: set[str]) -> float:
        """Standard Jaccard; matches :meth:`JaccardNgramStrategy._jaccard`."""
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)
        sigs = self._signatures_for(texts)
        index = self._build_lsh_index(sigs)
        shingles_per_text = [self._shingles(t) for t in texts]
        return self._compute_topk(
            list(range(n)),
            shingles_per_text,
            shingles_per_text,
            self._query_lsh(sigs, sigs, index),
            k_eff=min(k, n),
            n_ref=n,
            include_self=True,
        )

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
        sigs_ref = self._signatures_for(reference_texts)
        sigs_query = self._signatures_for(query_texts)
        index = self._build_lsh_index(sigs_ref)
        ref_shingles = [self._shingles(t) for t in reference_texts]
        query_shingles = [self._shingles(t) for t in query_texts]
        return self._compute_topk(
            list(range(n_q)),
            query_shingles,
            ref_shingles,
            self._query_lsh(sigs_query, sigs_ref, index),
            k_eff=min(k, n_r),
            n_ref=n_r,
            include_self=False,
        )

    def _compute_topk(
        self,
        query_idx: list[int],
        query_shingles: Sequence[set[str]],
        ref_shingles: Sequence[set[str]],
        candidate_sets: list[set[int]],
        k_eff: int,
        n_ref: int,
        include_self: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact Jaccard on LSH candidates and return top-k per query."""
        n_q = len(query_idx)
        sims = np.zeros((n_q, k_eff), dtype=np.float64)
        indices = np.zeros((n_q, k_eff), dtype=np.int64)
        for i, qi in enumerate(query_idx):
            cands = candidate_sets[i]
            if include_self:
                cands = cands | {qi}
            scored: list[tuple[float, int]] = [
                (self._exact_jaccard(query_shingles[i], ref_shingles[j]), j) for j in cands
            ]
            # Stable descending sort: highest similarity first; on ties, lower index first.
            scored.sort(key=lambda t: (-t[0], t[1]))
            chosen = scored[:k_eff]
            for slot, (sim, j) in enumerate(chosen):
                sims[i, slot] = sim
                indices[i, slot] = j
            # Pad with arbitrary fillers if fewer than k_eff candidates.
            if len(chosen) < k_eff:
                seen = {j for _, j in chosen}
                fillers = [j for j in range(n_ref) if j not in seen]
                for slot in range(len(chosen), k_eff):
                    if not fillers:
                        break
                    indices[i, slot] = fillers.pop(0)
                    # sims[i, slot] stays 0.0 — filler.
        return sims, indices


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


def audit_source_label_similarity(
    texts: Sequence[str],
    *,
    sources: Sequence[str] | None = None,
    labels: Sequence[object] | None = None,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
    strategy: SimilarityStrategy | None = None,
    include_within_source: bool = True,
    include_cross_source: bool = True,
    include_same_label: bool = True,
    include_cross_label: bool = True,
) -> SimilarityAuditReport:
    """Report high-similarity pairs without dropping rows.

    This is the evidence-first complement to :func:`near_dedup`. It uses the
    same pluggable similarity strategies but returns pair findings annotated
    with source/label relationships so consumers can decide whether duplicates
    are leakage, benign repeated labels, or label conflicts.
    """
    text_list = list(texts)
    n = len(text_list)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
    if sources is not None and len(sources) != n:
        raise ValueError("sources must have the same length as texts")
    if labels is not None and len(labels) != n:
        raise ValueError("labels must have the same length as texts")
    if n == 0:
        return SimilarityAuditReport([], threshold, 0, "none", k_neighbors)

    source_list = list(sources) if sources is not None else None
    label_list = list(labels) if labels is not None else None
    active_strategy: SimilarityStrategy = (
        strategy if strategy is not None else TfidfCosineStrategy()
    )
    similarities, indices = active_strategy.pairs_within(text_list, k_neighbors)

    findings: list[SimilarityAuditFinding] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for sim, j in zip(similarities[i], indices[i], strict=True):
            j_int = int(j)
            if j_int == i:
                continue
            left, right = sorted((i, j_int))
            pair = (left, right)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            similarity = float(sim)
            if similarity < threshold:
                continue
            if not _audit_pair_allowed(
                left,
                right,
                sources=source_list,
                labels=label_list,
                include_within_source=include_within_source,
                include_cross_source=include_cross_source,
                include_same_label=include_same_label,
                include_cross_label=include_cross_label,
            ):
                continue
            findings.append(
                SimilarityAuditFinding(
                    left_index=left,
                    right_index=right,
                    similarity=similarity,
                    relation=_similarity_relation(left, right, source_list, label_list),
                    left_source=source_list[left] if source_list is not None else None,
                    right_source=source_list[right] if source_list is not None else None,
                    left_label=label_list[left] if label_list is not None else None,
                    right_label=label_list[right] if label_list is not None else None,
                )
            )

    findings.sort(
        key=lambda finding: (-finding.similarity, finding.left_index, finding.right_index)
    )
    return SimilarityAuditReport(
        findings=findings,
        threshold=threshold,
        n_input=n,
        strategy=type(active_strategy).__name__,
        k_neighbors=k_neighbors,
    )


def _audit_pair_allowed(
    left: int,
    right: int,
    *,
    sources: Sequence[str] | None,
    labels: Sequence[object] | None,
    include_within_source: bool,
    include_cross_source: bool,
    include_same_label: bool,
    include_cross_label: bool,
) -> bool:
    """Return whether a candidate pair should be retained by audit filters."""
    if sources is not None:
        same_source = sources[left] == sources[right]
        if same_source and not include_within_source:
            return False
        if not same_source and not include_cross_source:
            return False
    if labels is not None:
        same_label = labels[left] == labels[right]
        if same_label and not include_same_label:
            return False
        if not same_label and not include_cross_label:
            return False
    return True


def _similarity_relation(
    left: int,
    right: int,
    sources: Sequence[str] | None,
    labels: Sequence[object] | None,
) -> SimilarityRelation:
    """Classify the source/label relationship for an audit finding."""
    if sources is None and labels is None:
        return "unspecified"
    same_source = sources[left] == sources[right] if sources is not None else None
    same_label = labels[left] == labels[right] if labels is not None else None
    if same_source is None:
        return "same_label" if same_label else "cross_label"
    if same_label is None:
        return "within_source" if same_source else "cross_source"
    if same_source and same_label:
        return "within_source_same_label"
    if same_source and not same_label:
        return "within_source_cross_label"
    if not same_source and same_label:
        return "cross_source_same_label"
    return "cross_source_cross_label"


def cross_dedup_pairs(
    train_texts: list[str],
    eval_texts: list[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
    *,
    strategy: SimilarityStrategy | None = None,
) -> list[tuple[int, int, float]]:
    """Return all (eval_idx, train_idx, similarity) tuples at or above threshold.

    Companion to :func:`cross_dedup` that exposes the matched train neighbor
    indices instead of only the eval-side keep set. Used by
    :class:`eval_toolkit.leakage.CrossSplitLeakageCheck` in
    ``label_aware`` mode (v0.17.0) to split near-duplicate hits into
    same-label and cross-label findings.

    Parameters mirror :func:`cross_dedup`. Returns ``[]`` when either side
    is empty.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if not train_texts or not eval_texts:
        return []
    active_strategy: SimilarityStrategy = (
        strategy if strategy is not None else TfidfCosineStrategy()
    )
    similarities, indices = active_strategy.pairs_across(eval_texts, train_texts, k_neighbors)
    if similarities.size == 0:
        return []
    out: list[tuple[int, int, float]] = []
    for eval_idx in range(similarities.shape[0]):
        for col in range(similarities.shape[1]):
            sim = float(similarities[eval_idx, col])
            if sim >= threshold:
                out.append((eval_idx, int(indices[eval_idx, col]), sim))
    return out


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
