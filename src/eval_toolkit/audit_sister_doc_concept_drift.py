r"""Sister-doc concept-drift validator (embedding-similarity-based).

Catches the bug class where two linked sister docs both reference the
same concept token (e.g., ``T1``, ``manifest v3``, ``verified_disjoint``)
but the **surrounding-sentence definitions disagree**. Cross-doc drift
survives lychee (links resolve), anchor audits (anchors exist), and
numeric audits (numbers don't disagree because the prose is qualitative).

Motivating test case (from `prompt-injection-detection-submission`
audit, two reproducibility surfaces)::

    docs/REPRODUCIBILITY.md:85:
      T1 = "full canonical re-eval (GPU; A100 80GB): make headline-cloud
      re-runs ... ~7h wall-clock; ~$28 GPU spend"

    WRITEUP/reproducibility.md:33:
      T1 = "smoke (laptop, $0, ~10 min): `make smoke` verifies code health"

Both files cross-link as "Aggregator docs"; following the link lands a
reader on contradictory T1 definitions.

Algorithm
---------
1. For each ``concept_token``, scan all ``files`` for occurrences. Each
   occurrence captures the *surrounding sentence(s)* (configurable
   ``context_window_sentences``) — that's the candidate "definition".
2. Embed each surrounding-sentence string via the supplied ``embedder``
   (default: lazy :func:`eval_toolkit.embeddings.make_minilm_embedder`).
3. Cluster occurrences by single-linkage: two occurrences belong to the
   same cluster iff their cosine similarity is ``>= similarity_threshold``.
4. A concept_token with **>1 cluster** is a :class:`DriftCluster` — its
   occurrences split into semantically distinct definition groups.
5. A concept_token with **exactly 1 cluster** is consistent across all
   files.

Design (per ADR 0001 flat-module + ADR 0002 closed-config + ADR 0003
Tier 2 ADDITIVE on the ``[embeddings]`` optional extra surface):

- Consumer supplies the concept-token list + file glob; validator owns
  parsing + embedding + clustering + report assembly.
- Embedder is a callable ``Callable[[Sequence[str]], np.ndarray]`` —
  matches the existing :func:`~eval_toolkit.embeddings.make_minilm_embedder`
  factory contract. ``embedder=None`` defers to the canonical MiniLM
  recipe lazily (avoids forcing the ``[embeddings]`` extra import at
  module load time).
- Flat-module: ``eval_toolkit.audit_sister_doc_concept_drift.*`` (NOT a
  subpackage per ADR 0001 stay-flat-through-v1.x).

Closes upstream issue #72. v1.0.4. Completes the audit-validator family
of 3 (citation_alignment v1.0.1, value_bindings v1.0.3, sister_doc
concept_drift v1.0.4).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_toolkit._narrative import _line_starts, _position_to_line

__all__ = [
    "DriftCluster",
    "SisterDocDriftReport",
    "validate_sister_doc_concept_drift",
]


DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
DEFAULT_CONTEXT_WINDOW_SENTENCES: int = 1

# Sentence-ish splitter — markdown is not formal prose. Splits on
# ``.``, ``!``, ``?`` followed by whitespace or EOL. Imperfect but
# robust enough for cross-doc concept-drift detection (consumers
# tolerate boundary slop because clustering is the noise-tolerant
# downstream step).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d`])")


@dataclass(frozen=True, slots=True)
class DriftCluster:
    """A concept token whose occurrences split into >1 semantic cluster.

    Attributes
    ----------
    token : str
        The concept token (e.g., ``"T1"``, ``"manifest v3"``).
    sentences : tuple[tuple[Path, int, str], ...]
        Each occurrence as ``(file, line, surrounding_text)`` — line is
        1-indexed; surrounding_text is the ``context_window_sentences``-sized
        prose snippet that was embedded for clustering.
    divergence_score : float
        ``1 - min_inter_cluster_similarity`` for the worst-case pair
        between any two clusters. Range ``[0.0, 1.0]``; higher = stronger
        drift signal. ``0.0`` means clusters are barely distinguishable;
        ``1.0`` means orthogonal embeddings.
    """

    token: str
    sentences: tuple[tuple[Path, int, str], ...]
    divergence_score: float


@dataclass(frozen=True, slots=True)
class SisterDocDriftReport:
    """Result of :func:`validate_sister_doc_concept_drift`.

    Attributes
    ----------
    drift_clusters : tuple[DriftCluster, ...]
        Each concept_token whose occurrences split into >1 cluster.
        Empty tuple = all tokens consistent across the scanned files.
    consistent_tokens : tuple[str, ...]
        Concept tokens whose occurrences clustered to a single group
        (or had ≤1 occurrence total). Reported for completeness +
        coverage tracking.
    coverage : float
        Fraction of ``concept_tokens`` that produced ≥1 occurrence in
        the scanned files. Range ``[0.0, 1.0]``. ``1.0`` means every
        token was referenced; lower values flag stale tokens.
    """

    drift_clusters: tuple[DriftCluster, ...]
    consistent_tokens: tuple[str, ...]
    coverage: float


def validate_sister_doc_concept_drift(
    *,
    files: Sequence[Path | str],
    concept_tokens: Sequence[str],
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    context_window_sentences: int = DEFAULT_CONTEXT_WINDOW_SENTENCES,
) -> SisterDocDriftReport:
    """Validate cross-doc semantic consistency of concept token definitions.

    For each ``concept_token``, scan ``files`` for occurrences; extract
    the surrounding ``context_window_sentences``; embed each surrounding
    snippet; cluster by single-linkage cosine similarity at
    ``similarity_threshold``. Tokens that produce >1 cluster are flagged
    as drift.

    Parameters
    ----------
    files : Sequence[Path | str]
        Markdown files to scan. UTF-8 encoded.
    concept_tokens : Sequence[str]
        Seed list of concept tokens (e.g., ``["T0", "T1", "T3",
        "manifest v3", "verified_disjoint"]``). Each token is matched
        case-sensitively as a whole-word boundary regex
        (``\\b<token>\\b``).
    embedder : Callable[[Sequence[str]], np.ndarray] | None, optional
        Embedder callable returning ``(n, d)`` array. ``None`` (default)
        lazily routes to :func:`eval_toolkit.embeddings.make_minilm_embedder`
        — requires the ``[embeddings]`` optional extra
        (``pip install eval-toolkit[embeddings]``). Custom callables let
        consumers swap in any embedder (BGE, E5, OpenAI, mock for tests).
    similarity_threshold : float, optional
        Cosine-similarity threshold for single-linkage clustering.
        Default ``0.7``. Higher = stricter (more clusters; more drift
        flagged); lower = looser. ``0.7`` is the conservative default
        for ``all-MiniLM-L6-v2`` — semantic-near-paraphrase territory.
    context_window_sentences : int, optional
        Number of sentences to extract on each side of the token mention
        as the "definition" snippet (passed to the embedder). Default
        ``1`` (the sentence containing the token; longer windows mute
        token-specific signal with surrounding prose).

    Returns
    -------
    SisterDocDriftReport
        ``drift_clusters``, ``consistent_tokens``, ``coverage`` per the
        dataclass.

    Raises
    ------
    ImportError
        If ``embedder=None`` and ``sentence_transformers`` is not
        installed. Install via ``pip install eval-toolkit[embeddings]``.
    FileNotFoundError
        If any path in ``files`` does not exist. ``files`` is an
        explicit caller-supplied list, not a glob result, so a missing
        entry is a caller bug. Other ``OSError``\\ s and non-UTF-8
        files are warn-and-skipped (see Notes).

    Notes
    -----
    File-read policy (family-wide, #99): both regex validators and this
    one propagate ``FileNotFoundError`` (an explicit ``files`` list
    means a missing file is caller misconfiguration).
    ``audit_value_bindings`` additionally RAISES ``ValueError`` on a
    non-UTF-8 file (a binding audit must not silently drop a surface);
    this validator warn-and-skips non-UTF-8 / unreadable files instead
    — the embedding scan tolerates partial input, and a loud
    ``UserWarning`` keeps the skip visible.

    Clustering: single-linkage agglomerative on cosine similarity. Two
    occurrences land in the same cluster iff their similarity is
    ``>= similarity_threshold``. Transitive: ``a~b`` and ``b~c`` →
    ``a, b, c`` in one cluster even if ``cos(a, c) < threshold``. This
    is the canonical SBERT semantic-dedup recipe (see
    :class:`~eval_toolkit.text_dedup.EmbeddingCosineStrategy` for the
    sibling primitive at the inter-text-similarity level).

    Token matching is case-sensitive whole-word — ``"T1"`` matches
    ``"T1"`` but not ``"t1"`` or ``"T10"``. Adjust by passing
    pre-normalized token strings if case-insensitivity is desired.

    See Also
    --------
    eval_toolkit.audit_citation_alignment.validate_citations :
        Sibling validator (catches ADR-citation alignment drift).
    eval_toolkit.audit_value_bindings.validate_reader_value_bindings :
        Sibling validator (catches detector→value binding drift).
    eval_toolkit.embeddings.make_minilm_embedder :
        Default embedder factory.
    """
    files_resolved = tuple(Path(f) for f in files)
    tokens = tuple(concept_tokens)
    if not tokens:
        return SisterDocDriftReport(drift_clusters=(), consistent_tokens=(), coverage=0.0)

    # Resolve embedder lazily — defer the [embeddings] extra import
    # to call time so the module loads even when sentence_transformers
    # isn't installed (matches the EmbeddingCosineStrategy pattern in
    # text_dedup.py).
    if embedder is None:
        embedder = _default_embedder()

    # Pre-load all files (avoid re-reading per token).
    file_texts: dict[Path, str] = {}
    for path in files_resolved:
        try:
            file_texts[path] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # `files` is an explicit caller-supplied list (this validator
            # does no glob discovery): a missing file is caller
            # misconfiguration, not skippable noise. Propagate (STYLE §1
            # never fail silently; matches audit_value_bindings, which
            # also lets FileNotFoundError escape its read).
            raise
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError, not an OSError — without it a
            # single non-UTF-8 byte would crash the whole scan. Skip unreadable
            # or non-UTF-8 files, but warn so the skip is not silent (STYLE §1).
            warnings.warn(
                f"skipping unreadable file {path}: {exc}",
                stacklevel=2,
            )
            continue

    drift_clusters: list[DriftCluster] = []
    consistent_tokens: list[str] = []
    tokens_with_hits: set[str] = set()

    for token in tokens:
        occurrences = _collect_occurrences(token, file_texts, context_window_sentences)
        if not occurrences:
            continue
        tokens_with_hits.add(token)

        if len(occurrences) == 1:
            consistent_tokens.append(token)
            continue

        # Embed every surrounding snippet (one batch per token).
        snippets = [occ[2] for occ in occurrences]
        embeddings = np.asarray(embedder(snippets), dtype=np.float64)
        clusters = _single_linkage_clusters(embeddings, similarity_threshold)

        if len(clusters) == 1:
            consistent_tokens.append(token)
            continue

        # Compute divergence score from inter-cluster similarity.
        divergence = _divergence_score(embeddings, clusters)
        drift_clusters.append(
            DriftCluster(
                token=token,
                sentences=tuple(occurrences),
                divergence_score=divergence,
            )
        )

    coverage = len(tokens_with_hits) / len(tokens) if tokens else 0.0
    return SisterDocDriftReport(
        drift_clusters=tuple(drift_clusters),
        consistent_tokens=tuple(consistent_tokens),
        coverage=coverage,
    )


def _default_embedder() -> Callable[[Sequence[str]], np.ndarray]:
    """Lazy MiniLM embedder factory; raises ImportError with install hint."""
    try:
        from eval_toolkit.embeddings import make_minilm_embedder
    except ImportError as exc:  # pragma: no cover
        msg = (
            "audit_sister_doc_concept_drift requires the [embeddings] optional "
            "extra (sentence_transformers). Install via "
            "`pip install eval-toolkit[embeddings]` OR pass a custom embedder "
            "callable via the embedder= kwarg."
        )
        raise ImportError(msg) from exc
    return make_minilm_embedder()


def _collect_occurrences(
    token: str, file_texts: dict[Path, str], context_window_sentences: int
) -> list[tuple[Path, int, str]]:
    """Find every occurrence of ``token`` (whole-word) across files.

    Returns list of ``(file, line, surrounding_text)`` tuples where
    ``surrounding_text`` is the ``context_window_sentences`` window
    centered on the sentence containing the token.
    """
    occurrences: list[tuple[Path, int, str]] = []
    token_re = re.compile(rf"\b{re.escape(token)}\b")
    for path, text in file_texts.items():
        sentences = _split_sentences(text)
        for s_idx, sent in enumerate(sentences):
            if not token_re.search(sent.text):
                continue
            window_lo = max(0, s_idx - context_window_sentences)
            window_hi = min(len(sentences), s_idx + context_window_sentences + 1)
            surrounding = " ".join(sentences[i].text for i in range(window_lo, window_hi))
            occurrences.append((path, sent.line, surrounding))
    return occurrences


@dataclass(frozen=True, slots=True)
class _SentenceSpan:
    text: str
    line: int  # 1-indexed line of the sentence's start


def _split_sentences(text: str) -> list[_SentenceSpan]:
    """Split markdown text into sentence spans with line numbers.

    Imperfect: skips fenced code blocks (```) but otherwise treats every
    text region as prose. Good enough for concept-drift detection at the
    sentence-of-context-around-token granularity.
    """
    # Strip fenced code blocks (replace with spaces preserving newlines so
    # line numbers stay accurate).
    in_fence = False
    stripped_lines = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            stripped_lines.append(line)  # keep newline for line-number alignment
            continue
        stripped_lines.append(line if not in_fence else "\n")
    cleaned = "".join(stripped_lines)

    # Compute (line_start_pos -> line_no) map via the shared
    # `_narrative` positional helpers (#99 consolidation).
    line_starts = _line_starts(cleaned)

    # Split into rough sentences. Markdown headings + lists are
    # treated as standalone sentences.
    spans: list[_SentenceSpan] = []
    # Process line-by-line first so headings/bullets stay isolated.
    pos = 0
    for raw_line in cleaned.splitlines(keepends=True):
        line_text = raw_line.rstrip("\n").strip()
        line_start_pos = pos
        pos += len(raw_line)
        if not line_text:
            continue
        # If line starts with #, treat as a sentence on its own
        if line_text.startswith("#") or line_text.startswith("- ") or line_text.startswith("* "):
            spans.append(
                _SentenceSpan(text=line_text, line=_position_to_line(line_starts, line_start_pos))
            )
            continue
        # Else split on sentence-ish delimiters
        for piece in _SENTENCE_SPLIT_RE.split(line_text):
            piece = piece.strip()
            if piece:
                spans.append(
                    _SentenceSpan(text=piece, line=_position_to_line(line_starts, line_start_pos))
                )
    return spans


def _single_linkage_clusters(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Single-linkage agglomerative clustering on cosine similarity.

    Returns list of clusters, each a list of row indices into ``embeddings``.
    Two rows i, j are in the same cluster iff there exists a chain
    i = k_0 ~ k_1 ~ ... ~ k_n = j where each adjacent pair has
    ``cosine(k_i, k_{i+1}) >= threshold``.
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / safe_norms
    sim = normed @ normed.T

    # Union-find on edges where sim >= threshold
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _divergence_score(embeddings: np.ndarray, clusters: list[list[int]]) -> float:
    """``1 - min_inter_cluster_similarity`` across all cluster pairs.

    Higher = stronger drift. ``0.0`` means clusters are barely separated;
    ``1.0`` means orthogonal embeddings.
    """
    if len(clusters) < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / safe_norms
    sim = normed @ normed.T
    min_sim = 1.0
    for a_idx in range(len(clusters)):
        for b_idx in range(a_idx + 1, len(clusters)):
            a, b = clusters[a_idx], clusters[b_idx]
            # Min similarity between any pair across the two clusters
            sub = sim[np.ix_(a, b)]
            pair_min = float(sub.min())
            min_sim = min(min_sim, pair_min)
    return 1.0 - max(0.0, min_sim)
