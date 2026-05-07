"""Text deduplication: near-duplicate detection and cross-source leakage scrubbing.

Pure helpers:

- :func:`sha256_text` — canonical SHA-256 of (optionally normalized) text
- :func:`normalize_text_for_dedup` — Unicode/whitespace normalization

Algorithmic:

- :func:`near_dedup` — TF-IDF cosine forward-scan greedy near-deduplication
- :func:`cross_dedup` — drop eval rows near-duplicate to any train row
- :class:`DedupReport` — frozen audit-trail of which rows were dropped and why
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DedupReport",
    "cross_dedup",
    "near_dedup",
    "normalize_text_for_dedup",
    "sha256_text",
]

DEFAULT_DEDUP_THRESHOLD = 0.9


@dataclass(frozen=True, slots=True)
class DedupReport:
    """Outcome of a near-dedup pass.

    ``kept_indices`` are positions in the input ``texts`` to retain.
    ``dropped_pairs`` is a list of ``(dropped_idx, kept_idx, similarity)``
    triples for audit.
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


def near_dedup(
    texts: list[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
) -> DedupReport:
    """Greedy near-deduplication on TF-IDF (1-3-gram) cosine similarity.

    Forward-scan: for each kept text ``i``, drop all ``j > i`` with
    ``cosine(i, j) >= threshold``.

    Parameters
    ----------
    texts : list[str]
    threshold : float, optional
        Cosine-similarity threshold in (0, 1). Default 0.9.
    k_neighbors : int, optional
        Maximum neighbors to consider per query. Default 20.

    Returns
    -------
    DedupReport

    Raises
    ------
    TypeError
        If ``texts`` is not a list.
    ValueError
        If ``threshold`` is outside (0, 1).
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts must be a list, got {type(texts).__name__}")
    n = len(texts)
    if n == 0:
        return DedupReport([], [], threshold, 0)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True)
    tfidf = vec.fit_transform(texts)

    k = min(k_neighbors, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(tfidf)
    distances, indices = nn.kneighbors(tfidf)
    similarities = 1.0 - distances

    kept_mask = np.ones(n, dtype=bool)
    dropped: list[tuple[int, int, float]] = []

    for i in range(n):
        if not kept_mask[i]:
            continue
        for sim, j in zip(similarities[i], indices[i], strict=True):
            if j == i or j < i:
                continue
            if sim >= threshold and kept_mask[j]:
                kept_mask[j] = False
                dropped.append((int(j), i, float(sim)))

    kept_indices = np.where(kept_mask)[0].tolist()
    return DedupReport(kept_indices, dropped, threshold, n)


def cross_dedup(
    train_texts: list[str],
    eval_texts: list[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k_neighbors: int = 20,
) -> list[int]:
    """Return eval indices to KEEP (drop those near-duplicate to any train text).

    Used to scrub OOD eval slices of any train-pool leakage before reporting
    OOD metrics.

    Parameters
    ----------
    train_texts, eval_texts : list[str]
    threshold : float, optional
        Cosine-similarity threshold in (0, 1). Default 0.9.
    k_neighbors : int, optional
        Maximum neighbors to consider per eval text. Default 20.

    Returns
    -------
    list[int]
        Indices into ``eval_texts`` of rows that are NOT near-duplicate to
        any train text. Order preserved.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if not train_texts:
        return list(range(len(eval_texts)))
    if not eval_texts:
        return []

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True)
    vec.fit(train_texts + eval_texts)
    tfidf_train = vec.transform(train_texts)
    tfidf_eval = vec.transform(eval_texts)

    k = min(k_neighbors, len(train_texts))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(tfidf_train)
    distances, _ = nn.kneighbors(tfidf_eval)
    max_sim_per_eval = 1.0 - distances.min(axis=1)
    keep_mask = max_sim_per_eval < threshold
    return [int(i) for i in np.where(keep_mask)[0]]
