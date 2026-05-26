r"""Reader-prose value-binding validator.

Catches the bug class where a reader-facing markdown surface pairs a
detector name with the **wrong** canonical value — both values are
present in the source-of-truth table, but the binding is misordered.

Motivating test case (from `prompt-injection-detection-prototype`
v1.3.1 audit-fix, ADR-080 patch closure 2026-05-22)::

    WRITEUP_NARRATIVE.md:38:
      "The TF-IDF + logistic regression baseline reaches 0.974 AUPRC
      on balanced direct-versus-benign validation."

Canonical: TF-IDF direct val AUPRC = 0.971; LoRA direct val AUPRC =
0.974. Both values exist in the bindings table; the bug is the wrong
(detector, value) pairing. The pre-existing ``audit_numbers.py``-style
primitive validates VALUES against source data; this validator
validates BINDINGS — that each prose-mentioned (detector_token,
metric_token, value) triple matches the canonical binding.

Design (per ADR 0001 flat-module + ADR 0002 closed-config + ADR 0003
Tier 1 STRICT public-API contract):

- Consumer supplies the canonical-binding table + value/metric/detector
  regex patterns; validator handles position-aware regex scan + binding
  lookup + report assembly.
- Flat-module: `eval_toolkit.audit_value_bindings.*` (NOT a subpackage
  per ADR 0001 stay-flat-through-v1.x).
- All Tier-1 STRICT public symbols (`validate_reader_value_bindings`,
  `Match`, `Violation`, `ValueBindingsReport`) re-exported at top level
  via `_EXPORTS` lazy resolver.

Closes upstream issue #71. v1.0.3.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

__all__ = [
    "BindingKey",
    "Match",
    "ValueBindingsReport",
    "Violation",
    "validate_reader_value_bindings",
]


_logger = logging.getLogger(__name__)


DEFAULT_VALUE_PATTERN: str = r"\d+\.\d{2,4}"
DEFAULT_MAX_DISTANCE_CHARS: int = 80
DEFAULT_SLICE_WINDOW_CHARS: int = 120
DEFAULT_TOLERANCE: float = 1e-4


# v1.2.0 context-aware narrative filters. Keyword lists are hardcoded
# module-level frozensets (per ADR 0005 §4: Tier-1 ADDITIVE — no new
# public kwargs; consumers can file an issue to extend the default
# lists if their prose surfaces missed patterns).
#
# _DELTA_KEYWORDS: case-insensitive whole-token markers indicating a
# value is a paired-delta or comparative magnitude, not a binding claim.
# T1 filter suppresses candidate values when any of these appears within
# ±30 chars of the value position (under scope="narrative").
_DELTA_KEYWORDS: frozenset[str] = frozenset(
    {
        # Unambiguous delta nouns/verbs (consumer prose patterns):
        "delta",
        "drop",
        "drops",
        "lift",
        "lifts",
        "gap",
        "margin",
        # Comparison verbs that signal "this is a relative magnitude":
        "regresses",
        "improves",
        "beats",
        "exceeds",
        "trails",
        "underperforms",
        # "vs"/"versus" intentionally INCLUDED — they're the canonical
        # delta separator in consumer prose ("AUPRC 0.556 vs 0.519").
        # The before-only window keeps these tight: "X vs Y" fires on
        # Y (preceded by "vs"), not X. T3 also catches the same-sentence
        # duplicate-binding flag separately.
        "vs",
        "versus",
        # Comparison directions — kept under before-only window so
        # "drops -0.071 below" suppresses -0.071 (sign also catches),
        # but "0.515 (delta -0.132)" doesn't suppress 0.515 ("delta"
        # is AFTER 0.515).
        # Excluded: "against", "above", "ahead", "behind" — too
        # ambiguous; common comparison prepositions that appear in
        # legitimate binding claims.
        "below",
    }
)

# _FLOOR_KEYWORDS: markers indicating a value is a random-baseline or
# floor reference, not a detector binding. T2 filter suppresses
# candidate values when any of these appears within −50 / +5 chars
# (asymmetric: floor mentions canonically precede the value, e.g.,
# "random AUPRC is 0.374").
#
# Intentionally narrow: "baseline", "prior", "majority" are EXCLUDED
# because they have legitimate non-floor senses ("TF-IDF baseline",
# "prior work", "majority of detectors"). The consumer's prose
# patterns with these words ("below the prevalence baseline of 0.374")
# are caught by T1 via "below"/"above" instead — the comparative
# preposition is the reliable signal, not the noun.
_FLOOR_KEYWORDS: frozenset[str] = frozenset(
    {
        "random",
        "floor",
        "chance",
        "trivial",
    }
)

# _ABBREV_BEFORE_DOT: tokens that should NOT trigger a sentence
# boundary when followed by `.`. The multi-letter pattern (e.g., i.e.,
# c.f.) is handled separately via letter-dot-letter detection.
_ABBREV_BEFORE_DOT: frozenset[str] = frozenset(
    {
        "vs",
        "etc",
        "cf",
        "fig",
        "eq",
        "pp",
        "viz",
        "ca",
    }
)


def _compile_keyword_pattern(keywords: frozenset[str]) -> re.Pattern[str]:
    """Compile case-insensitive word-boundary OR regex matching any keyword."""
    parts = sorted(re.escape(kw) for kw in keywords)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


_DELTA_PATTERN: re.Pattern[str] = _compile_keyword_pattern(_DELTA_KEYWORDS)
_FLOOR_PATTERN: re.Pattern[str] = _compile_keyword_pattern(_FLOOR_KEYWORDS)


@dataclass(frozen=True)
class BindingKey:
    """Canonical identity for a `(detector, metric, slice)` measurement.

    Frozen dataclass usable as a dict key. Forward-extensible: new
    identity axes (split, ci_kind, source_ref, ...) can be added as
    defaulted fields without breaking the `bindings`-dict-key schema.
    See ADR 0005 for the underlying rule
    ("structured keys over positional tuples for canonical-identity
    types in audit validators").

    Three input shapes are accepted by
    :func:`validate_reader_value_bindings` (all normalized internally
    to ``BindingKey``):

    1. Canonical: ``BindingKey(detector=..., metric=..., slice=...)``
       (recommended for new consumer code).
    2. Sugar 3-tuple: ``(detector, metric, slice)`` (concise dict literal).
    3. Legacy 2-tuple: ``(detector, metric)`` (treated as
       ``slice="any"``; preserves pre-v1.1 consumer code).

    Attributes
    ----------
    detector : str
        Canonical detector identifier (e.g., ``"tf-idf + lr"``).
    metric : str
        Canonical metric identifier (e.g., ``"direct_val_auprc"``).
    slice : str
        Canonical slice identifier (e.g., ``"direct_validation"``,
        ``"pooled_ood"``). Default ``"any"`` disables slice scoping
        (matches anywhere in the document, mirroring legacy 2-tuple
        behavior).
    """

    detector: str
    metric: str
    slice: str = "any"


@dataclass(frozen=True)
class Match:
    """A reader-prose (detector, metric, value) triple that matches the canonical binding.

    Attributes
    ----------
    file : Path
        File where the match was found.
    line : int
        1-indexed line number of the value occurrence.
    detector : str
        Canonical detector key from the ``bindings`` dict (NOT the
        regex-matched surface form).
    metric : str
        Canonical metric key from the ``bindings`` dict.
    value : float
        The numeric value found in the prose.
    """

    file: Path
    line: int
    detector: str
    metric: str
    value: float


@dataclass(frozen=True)
class Violation:
    """A reader-prose (detector, metric, value) triple where the value disagrees with the canonical binding.

    Attributes
    ----------
    file : Path
        File where the violation was found.
    line : int
        1-indexed line number of the offending value occurrence.
    detector : str
        Canonical detector key from the ``bindings`` dict (NOT the
        regex-matched surface form).
    metric : str
        Canonical metric key from the ``bindings`` dict.
    found_value : float
        The numeric value the prose claims.
    expected_value : float
        The canonical value from the ``bindings`` dict.
    surrounding_text : str
        Excerpt centered on the value (configurable window) for
        diagnostic display.
    """

    file: Path
    line: int
    detector: str
    metric: str
    found_value: float
    expected_value: float
    surrounding_text: str


@dataclass(frozen=True)
class ValueBindingsReport:
    """Result of :func:`validate_reader_value_bindings`.

    Attributes
    ----------
    violations : tuple[Violation, ...]
        Each detected (detector, metric) → wrong-value triple. Empty
        tuple if all reader-prose bindings match the canonical table.
    matched : tuple[Match, ...]
        Each detected (detector, metric, value) triple that matched
        the canonical binding. Useful for coverage analysis +
        regression-testing that the validator's regexes still fire.
    coverage : float
        Fraction of canonical bindings keys that produced at least
        one :class:`Match`. Range ``[0.0, 1.0]``. ``1.0`` means every
        binding was referenced in the scanned prose; lower values
        flag potentially un-cited bindings (which may be expected OR
        may indicate stale prose). For slice-aware bindings each
        ``BindingKey`` counts independently (same ``(detector,
        metric)`` across multiple slices each contribute to the
        denominator).
    unmatched_slice_count : int, default 0
        Count of (detector, metric, value) triples that matched the
        detector+metric+value triple BUT had a slice-scoped
        :class:`BindingKey` whose ``slice`` did not appear within
        ``slice_window_chars`` of the value. These triples are
        suppressed (no violation, no match) on the assumption that
        the prose context (e.g., a paired-delta cell or a
        random-floor mention) does not carry the slice
        determination. A nonzero count is informational — it flags
        prose where the value couldn't be slice-disambiguated; it is
        NOT a validation failure. Always ``0`` when no slice-scoped
        ``BindingKey`` is present in ``bindings`` (i.e., when all
        keys are legacy 2-tuples or have ``slice="any"``).
    """

    violations: tuple[Violation, ...]
    matched: tuple[Match, ...]
    coverage: float
    unmatched_slice_count: int = 0


def validate_reader_value_bindings(
    *,
    files: Sequence[Path | str],
    bindings: Mapping[BindingKey | tuple[str, str] | tuple[str, str, str], float],
    value_pattern: str = DEFAULT_VALUE_PATTERN,
    max_distance_chars: int = DEFAULT_MAX_DISTANCE_CHARS,
    metric_aliases: Mapping[str, Sequence[str]] = MappingProxyType({}),
    detector_aliases: Mapping[str, Sequence[str]] = MappingProxyType({}),
    slice_aliases: Mapping[str, Sequence[str]] | None = None,
    slice_window_chars: int = DEFAULT_SLICE_WINDOW_CHARS,
    scope: Literal["all", "narrative"] = "all",
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValueBindingsReport:
    """Validate (detector, metric, value) bindings in reader-prose markdown.

    For each ``BindingKey(detector, metric, slice) -> expected_value``
    entry in ``bindings``, scan each file for triples of (detector
    mention, metric mention, numeric value) within a
    ``max_distance_chars`` window. When the key's ``slice != "any"``,
    additionally require that one of the corresponding ``slice_aliases``
    appears within ``slice_window_chars`` of the value (slice
    disambiguation). Compare the found value to the expected value;
    emit a :class:`Violation` on mismatch, a :class:`Match` on
    agreement.

    Both the detector and the metric must appear within the window
    surrounding a candidate value for the triple to be considered —
    a value that has only a detector or only a metric nearby is
    ignored (those belong to a value-existence audit, not a binding
    audit). Triples where slice disambiguation is required but no
    slice alias is found in the slice window are silently suppressed
    and counted in ``ValueBindingsReport.unmatched_slice_count``
    (warn-only; not a violation).

    Parameters
    ----------
    files : Sequence[Path | str]
        Markdown files to scan. UTF-8 encoded.
    bindings : Mapping[BindingKey | tuple[str, str] | tuple[str, str, str], float]
        Canonical-binding table keyed by :class:`BindingKey` (or a
        2-tuple ``(detector, metric)`` shorthand for legacy
        ``slice="any"``, or a 3-tuple ``(detector, metric, slice)``
        sugar). Values are the expected numeric measurement. All key
        shapes are normalized to ``BindingKey`` internally; mixed
        shapes in a single dict are supported. See :class:`BindingKey`
        for the canonical form.
    value_pattern : str, optional
        Regex matching numeric values in prose. Default matches
        ``\\d+\\.\\d{2,4}`` (1+ integer part, 2-4 decimals).
    max_distance_chars : int, optional
        Maximum character distance allowed between a detector mention,
        a metric mention, and a numeric value for them to be treated
        as a triple. Default 80.
    metric_aliases : Mapping[str, Sequence[str]], optional
        ``metric_name -> [regex_alternatives, ...]``. Each canonical
        metric name in ``bindings`` may have multiple natural-language
        forms (e.g., ``"direct_val_auprc"`` matches both ``"direct .*?
        AUPRC"`` and ``"validation AUPRC"``). Missing keys default to
        the canonical name itself, escaped.
    detector_aliases : Mapping[str, Sequence[str]], optional
        Same shape as ``metric_aliases``, applied case-insensitively.
        Useful for ``"tf-idf + lr"`` → ``["TF-IDF", "TfIdf", "tfidf"]``.
    slice_aliases : Mapping[str, Sequence[str]] | None, optional
        Same shape as ``metric_aliases`` but for slice tokens
        (e.g., ``{"direct_validation": ["direct.*?validation",
        "in-pool"], "pooled_ood": ["pooled OOD", "cross-family"]}``).
        Only used when at least one ``BindingKey`` in ``bindings``
        has ``slice != "any"``. Missing slice keys default to the
        canonical slice name itself, escaped. ``None`` (the default)
        disables slice-aware matching entirely; if ``bindings``
        contains slice-scoped keys without aliases provided here, the
        canonical slice names themselves must appear verbatim in
        prose.
    slice_window_chars : int, optional
        Character window around a candidate value within which a
        slice alias must appear for the slice match to count.
        Default 120 (≈ same-sentence scope; tighter than the
        ``max_distance_chars`` detector/metric window because slice
        context can bleed across topic-introduction prose).
        Increase carefully — wider windows risk attributing a value
        to a setup-clause slice in the preceding sentence.
    scope : {"all", "narrative"}, optional
        Content-type filter for value matching. Default ``"all"``
        matches values anywhere in the file (legacy v1.0.x behavior;
        preserved for backward compat). ``"narrative"`` restricts
        matching to narrative-prose values, excluding:

        - **Markdown table rows** (lines starting with ``|``):
          structured data, audited via different mechanisms; cells
          typically contain multiple metrics per row that the
          validator's positional heuristics can't disambiguate.
        - **Bracketed expressions** ``[...]``: confidence intervals,
          reference markers, ranges. Bound values are not
          point-estimate claims.
        - **Fenced code blocks**: triple-backtick blocks contain code
          or literal data, not narrative claims.

        Use ``"narrative"`` to dramatically reduce false positives
        when scanning research writeups with dense statistical
        tables and CI bounds. The motivating misbinding bug class
        (V1.3.1 ADR-080) was in narrative prose, so ``"narrative"``
        does not lose recall on that bug class. Set to ``"all"``
        if you have a custom value/alias setup that intends to
        match values inside tables or brackets.
    tolerance : float, optional
        Absolute tolerance for float comparison. Default ``1e-4``
        (i.e., ``0.974`` and ``0.9740`` are considered equal).

    Returns
    -------
    ValueBindingsReport
        ``violations``, ``matched``, ``coverage``,
        ``unmatched_slice_count`` per the dataclass.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> import textwrap
    >>> with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
    ...     _ = f.write("TF-IDF + LR reaches 0.974 AUPRC on direct val.\\n")
    ...     path = Path(f.name)
    >>> report = validate_reader_value_bindings(
    ...     files=[path],
    ...     bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
    ...     detector_aliases={"tf-idf + lr": ["TF-IDF"]},
    ...     metric_aliases={"direct_val_auprc": ["direct val"]},
    ... )
    >>> len(report.violations)
    1
    >>> report.violations[0].found_value
    0.974
    >>> report.violations[0].expected_value
    0.971

    Notes
    -----
    The validator is **pure**: consumer-side scripts glob markdown
    files and parse canonical-binding tables (e.g., from a JSON
    results file); this function does the regex + window + comparison
    work and returns a structured report.

    Multiple candidate values within the same detector+metric window
    each produce their own Match / Violation entry. Coverage counts
    a (detector, metric) key as covered iff at least one Match was
    emitted for it (Violations don't count toward coverage — a
    misbound mention proves the binding was REACHED but disproves
    it was correct; the report makes both signals available).

    Case-sensitivity: detector and metric regexes are applied with
    ``re.IGNORECASE``. The canonical names in ``bindings`` are used
    verbatim in report keys regardless of how they were matched in
    prose.

    See Also
    --------
    eval_toolkit.audit_citation_alignment.validate_citations :
        Sibling validator catching ADR-citation alignment drift.
    """
    files_resolved = tuple(Path(f) for f in files)

    # Normalize all bindings keys (legacy 2-tuple, sugar 3-tuple, or
    # canonical BindingKey) to BindingKey. Mixed key shapes in a
    # single dict are supported; the per-key normalize raises
    # TypeError on unrecognized shapes (caught at function boundary,
    # not silently coerced).
    canonical_bindings: dict[BindingKey, float] = {
        _normalize_binding_key(k): v for k, v in bindings.items()
    }
    metric_aliases_dict = dict(metric_aliases)
    detector_aliases_dict = dict(detector_aliases)
    slice_aliases_dict: dict[str, Sequence[str]] = dict(slice_aliases) if slice_aliases else {}

    detector_keys = sorted({k.detector for k in canonical_bindings})
    metric_keys = sorted({k.metric for k in canonical_bindings})
    # Only compile slice patterns for non-"any" slice keys; "any"
    # signals legacy 2-tuple semantics (no slice scoping).
    slice_keys = sorted({k.slice for k in canonical_bindings if k.slice != "any"})

    detector_patterns: dict[str, re.Pattern[str]] = {
        d: _build_pattern(d, detector_aliases_dict.get(d, ()), case_insensitive=True)
        for d in detector_keys
    }
    metric_patterns: dict[str, re.Pattern[str]] = {
        m: _build_pattern(m, metric_aliases_dict.get(m, ()), case_insensitive=True)
        for m in metric_keys
    }
    slice_patterns: dict[str, re.Pattern[str]] = {
        s: _build_pattern(s, slice_aliases_dict.get(s, ()), case_insensitive=True)
        for s in slice_keys
    }
    value_re = re.compile(value_pattern)

    violations: list[Violation] = []
    matched: list[Match] = []
    matched_keys: set[BindingKey] = set()
    unmatched_slice_count = 0

    for file_path in files_resolved:
        text = file_path.read_text(encoding="utf-8")
        line_starts = _line_starts(text)

        # When scope="narrative", pre-compute the character ranges to
        # exclude (markdown tables, brackets, code blocks). Empty list
        # for scope="all" (legacy semantics; no exclusion).
        excluded_ranges = _build_exclusion_ranges(text, line_starts) if scope == "narrative" else []

        # v1.2.0 T3 + T4 (narrative-scope only): precompute sentence
        # boundaries once per file (paragraph-aware abbreviation guard).
        # T3 uses a per-(sentence, canonical_key) set to suppress
        # duplicate matches of the same binding within one sentence
        # (e.g., "0.556 vs 0.519" — the second value belongs to a
        # contrasting detector implicit in the prose). T4 uses the
        # boundaries to reject (detector, value) pairings that cross
        # a sentence terminator.
        sentence_positions: Sequence[int] = (
            _sentence_boundary_positions(text) if scope == "narrative" else ()
        )
        consumed_in_sentence: set[tuple[int, BindingKey]] = set()

        # Pre-collect ALL detector positions (across every canonical
        # detector key) so each value can be paired with its NEAREST
        # detector. This avoids cross-detector contamination — e.g.,
        # "TF-IDF achieves 0.971, while LoRA reaches 0.974" should
        # pair 0.971 with TF-IDF and 0.974 with LoRA, NOT pair the
        # 0.974 with TF-IDF's binding just because they happen to be
        # within max_distance_chars of each other.
        detector_positions: list[tuple[int, str]] = []  # (position, canonical_key)
        for det_key, det_re in detector_patterns.items():
            for det_match in det_re.finditer(text):
                detector_positions.append((det_match.start(), det_key))
        detector_positions.sort()

        # Pre-collect ALL slice positions (across every non-"any" slice
        # alias) so each value can be paired with its NEAREST slice
        # mention by the same last-before-first-after rule as detectors.
        # Mirrors the detector-pairing logic — required so prose like
        # "On direct+benign validation, X scored 0.971. On pooled OOD,
        # X scored 0.291." pairs 0.971 with direct_validation and 0.291
        # with pooled_ood, instead of any-within-window matching that
        # over-matches when slice mentions are close together.
        slice_positions: list[tuple[int, str]] = []  # (position, canonical_slice)
        for s_key, s_re in slice_patterns.items():
            for s_match in s_re.finditer(text):
                slice_positions.append((s_match.start(), s_key))
        slice_positions.sort()

        # For each canonical binding, look in each file for triples.
        for canonical_key, expected in canonical_bindings.items():
            det_key = canonical_key.detector
            met_key = canonical_key.metric
            slice_key = canonical_key.slice
            det_re = detector_patterns[det_key]
            met_re = metric_patterns[met_key]

            for det_match in det_re.finditer(text):
                window_start = max(0, det_match.start() - max_distance_chars)
                window_end = min(len(text), det_match.end() + max_distance_chars)
                window_text = text[window_start:window_end]
                window_offset = window_start

                # Both metric and a value must appear in the window.
                met_hits = list(met_re.finditer(window_text))
                if not met_hits:
                    continue

                for val_match in value_re.finditer(window_text):
                    # Skip values immediately adjacent to digits (avoid
                    # picking up e.g., "0.974" inside "10.974" or version
                    # strings like "1.0.974"). Simple heuristic: the
                    # character before the match (if any) must not be a
                    # digit or dot. v1.2.0 T1a (narrative-scope only):
                    # also skip values immediately preceded by `+` or
                    # `-` (delta-magnitude markers like "-0.071 AUPRC").
                    val_start_in_full = window_offset + val_match.start()
                    if val_start_in_full > 0:
                        prev_char = text[val_start_in_full - 1]
                        if prev_char.isdigit() or prev_char == ".":
                            continue
                        if scope == "narrative" and prev_char in "+-":
                            continue

                    val_str = val_match.group(0)
                    try:
                        found = float(val_str)
                    except ValueError:  # pragma: no cover
                        continue

                    # Narrative-scope exclusion: skip values inside
                    # markdown tables, brackets, or code blocks.
                    # No-op when scope="all" (empty exclusion list).
                    if excluded_ranges and _is_excluded(val_start_in_full, excluded_ranges):
                        continue

                    # v1.2.0 T1b (narrative-scope only): delta-keyword
                    # context filter. Skip values whose preceding 30
                    # chars contain a delta-marker token (e.g.,
                    # "delta", "drop", "lift", "vs", "below"). Window
                    # is BEFORE-only: delta keywords canonically
                    # introduce the delta magnitude ("delta -0.132",
                    # "drops -0.071"). Symmetric ±30 windows
                    # mis-fire on prose like "X scored 0.515 (delta
                    # -0.132)" where "delta" describes a DIFFERENT
                    # value (-0.132), not the preceding 0.515.
                    if scope == "narrative" and _has_keyword_in_window(
                        text, val_start_in_full, _DELTA_PATTERN, 30, 0
                    ):
                        continue

                    # v1.2.0 T2 (narrative-scope only): floor-keyword
                    # context filter. Skip values within −50/+5 chars of
                    # a floor-marker token (e.g., "random", "floor",
                    # "baseline"). Floor mentions canonically precede
                    # the value ("random AUPRC is 0.374"), hence the
                    # asymmetric window.
                    if scope == "narrative" and _has_keyword_in_window(
                        text, val_start_in_full, _FLOOR_PATTERN, 50, 5
                    ):
                        continue

                    # Cross-detector disambiguation: require the current
                    # det_key to be the detector paired with this value
                    # by the text-order rule (last detector before; else
                    # first detector after). Avoids cross-contamination
                    # on multi-detector prose like "TF-IDF achieves
                    # 0.971, while LoRA reaches 0.974".
                    detector_match = _nearest_canonical_key(
                        detector_positions, val_start_in_full, max_distance_chars
                    )
                    if detector_match is None or detector_match[0] != det_key:
                        continue
                    paired_det_pos = detector_match[1]

                    # v1.2.0 T4 (narrative-scope only): reject the
                    # detector-value pair if a sentence boundary lies
                    # between them. Prevents prose like "X scored
                    # 0.291. The random floor is 0.374" from pairing
                    # 0.374 with X across the `.` boundary.
                    if scope == "narrative" and _crosses_sentence_boundary(
                        paired_det_pos, val_start_in_full, sentence_positions
                    ):
                        continue

                    # Require the metric mention be within distance of the value too,
                    # not just within the detector window.
                    met_close = any(
                        abs(mh.start() - val_match.start()) <= max_distance_chars for mh in met_hits
                    )
                    if not met_close:
                        continue

                    # Slice disambiguation: when the canonical key is
                    # slice-scoped (slice != "any"), pair the value
                    # with the NEAREST slice mention by the same
                    # last-before-first-after rule used for detectors.
                    # Three cases:
                    #   (a) no slice mention within slice_window_chars
                    #       → ambiguous; suppress + count in
                    #       unmatched_slice_count (warn-only signal).
                    #   (b) paired slice != this binding's slice → this
                    #       triple belongs to a different slice's
                    #       binding; skip silently (the other binding
                    #       will pick it up on its own loop iteration).
                    #   (c) paired slice == this binding's slice →
                    #       fall through to value comparison.
                    if slice_key != "any":
                        slice_match = _nearest_canonical_key(
                            slice_positions,
                            val_start_in_full,
                            slice_window_chars,
                        )
                        if slice_match is None:
                            unmatched_slice_count += 1
                            _logger.warning(
                                "audit_value_bindings: no slice mention "
                                "within ±%d chars of %s=%s in %s; binding "
                                "key %r is slice-scoped and the prose "
                                "context is ambiguous",
                                slice_window_chars,
                                det_key,
                                val_str,
                                file_path,
                                canonical_key,
                            )
                            continue
                        paired_slice = slice_match[0]
                        if paired_slice != slice_key:
                            continue

                    # v1.2.0 T3 (narrative-scope only): suppress
                    # duplicate matches of the same binding within one
                    # sentence. After a Match is emitted for
                    # (canonical_key) at this sentence, subsequent
                    # candidate values in the same sentence for the
                    # same canonical_key are skipped. Catches dense
                    # multi-detector enumerations like "AUPRC 0.556 vs
                    # 0.519" where 0.519 is implicitly a contrasting
                    # detector's value.
                    if sentence_positions:
                        sent_id = _sentence_id_of(val_start_in_full, sentence_positions)
                        if (sent_id, canonical_key) in consumed_in_sentence:
                            continue
                    else:
                        sent_id = 0  # placeholder; not used when scope="all"

                    line_no = _position_to_line(line_starts, val_start_in_full)
                    if abs(found - expected) <= tolerance:
                        matched.append(
                            Match(
                                file=file_path,
                                line=line_no,
                                detector=det_key,
                                metric=met_key,
                                value=found,
                            )
                        )
                        matched_keys.add(canonical_key)
                        if sentence_positions:
                            consumed_in_sentence.add((sent_id, canonical_key))
                    else:
                        # Widen the surrounding context for diagnostic
                        # clarity. Center on the value but include
                        # ±60 chars to typically capture the detector
                        # mention.
                        ctx_start = max(0, val_start_in_full - 60)
                        ctx_end = min(len(text), val_start_in_full + len(val_str) + 60)
                        surrounding = text[ctx_start:ctx_end].replace("\n", " ").strip()
                        violations.append(
                            Violation(
                                file=file_path,
                                line=line_no,
                                detector=det_key,
                                metric=met_key,
                                found_value=found,
                                expected_value=expected,
                                surrounding_text=surrounding,
                            )
                        )

    coverage = len(matched_keys) / len(canonical_bindings) if canonical_bindings else 0.0
    return ValueBindingsReport(
        violations=tuple(violations),
        matched=tuple(matched),
        coverage=coverage,
        unmatched_slice_count=unmatched_slice_count,
    )


def _normalize_binding_key(
    key: BindingKey | tuple[str, str] | tuple[str, str, str],
) -> BindingKey:
    """Normalize a bindings dict key to canonical :class:`BindingKey`.

    Accepts a canonical ``BindingKey``, a legacy 2-tuple ``(detector,
    metric)`` (treated as ``slice="any"``), or a sugar 3-tuple
    ``(detector, metric, slice)``. Raises :class:`TypeError` on any
    other shape so misconfiguration surfaces loudly at the function
    boundary rather than silently producing zero matches downstream.

    Note: ``BindingKey`` is a frozen dataclass, not a tuple subclass,
    so the ``isinstance(key, BindingKey)`` check is checked first
    (and is independent of the ``isinstance(key, tuple)`` branch).
    """
    if isinstance(key, BindingKey):
        return key
    if isinstance(key, tuple):
        if len(key) == 2:
            return BindingKey(detector=key[0], metric=key[1])  # slice="any" default
        if len(key) == 3:
            return BindingKey(detector=key[0], metric=key[1], slice=key[2])
    raise TypeError(
        f"bindings key must be BindingKey, a 2-tuple (detector, metric), "
        f"or a 3-tuple (detector, metric, slice); got {type(key).__name__!r} "
        f"with value {key!r}"
    )


def _build_exclusion_ranges(
    text: str,
    line_starts: Sequence[int],
) -> list[tuple[int, int]]:
    """Compute sorted character ranges that ``scope="narrative"`` excludes.

    Excluded content types (per the lint-scope design discussion in
    pending ADR 0005):

    - **Markdown table rows**: lines starting with optional whitespace
      then ``|``. Tables are structured data audited via different
      mechanisms (e.g., direct results-table verification), not via
      narrative-prose binding-claim checks. Values in cells are
      typically inline statistics (multiple metrics per row), and the
      validator's positional heuristics can't disambiguate them.
    - **Bracketed expressions** ``[...]``: confidence intervals,
      reference markers, ranges. The numeric content inside brackets
      is not a point-estimate claim; the validator should not flag it.
    - **Fenced code blocks**: triple-backtick blocks contain code or
      literal data, not narrative claims.

    Returns a sorted list of ``(start, end)`` character intervals
    (half-open) for use with :func:`_is_excluded`.
    """
    excluded: list[tuple[int, int]] = []
    in_code_block = False
    code_block_start = 0
    n_lines = len(line_starts)
    for line_idx in range(n_lines):
        line_start = line_starts[line_idx]
        line_end = line_starts[line_idx + 1] if line_idx + 1 < n_lines else len(text)
        line = text[line_start:line_end]

        # Triple-backtick code-fence toggle. The fence line itself is
        # also part of the excluded range (so values aren't matched
        # from within the fence marker, though that's unlikely).
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_start = line_start
            else:
                in_code_block = False
                excluded.append((code_block_start, line_end))
            continue
        if in_code_block:
            # Lines inside a code block are folded into the outer
            # range emitted at the closing fence; no per-line emission.
            continue

        # Markdown table row.
        if stripped.startswith("|"):
            excluded.append((line_start, line_end))
            continue

        # Bracketed expressions on this line. Multiple `[...]` allowed.
        # Nested brackets are rare in measurement prose; first close
        # wins.
        i = 0
        while True:
            open_idx = line.find("[", i)
            if open_idx == -1:
                break
            close_idx = line.find("]", open_idx + 1)
            if close_idx == -1:
                break
            excluded.append((line_start + open_idx, line_start + close_idx + 1))
            i = close_idx + 1

    # Handle unterminated code block (defensive: treat rest of file as
    # excluded). Sort by start position.
    if in_code_block:
        excluded.append((code_block_start, len(text)))
    excluded.sort()
    return excluded


def _is_excluded(pos: int, excluded: Sequence[tuple[int, int]]) -> bool:
    """Return True if ``pos`` falls inside any excluded range.

    Uses binary search on the sorted ranges. Half-open semantics: a
    range ``(start, end)`` excludes positions ``start <= pos < end``.
    """
    if not excluded:
        return False
    # Find rightmost range with start <= pos.
    lo, hi = 0, len(excluded)
    while lo < hi:
        mid = (lo + hi) // 2
        if excluded[mid][0] <= pos:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return False
    start, end = excluded[lo - 1]
    return start <= pos < end


# ---------------------------------------------------------------------------
# v1.2.0 context-aware narrative filters.
# Helpers below implement T1 (delta/sign), T2 (floor), T3 (consume-on-match
# per-sentence), and T4 (sentence-boundary detector-pair reject) — all
# scoped to `scope="narrative"`. Per ADR 0005 §4, these are Tier-1
# ADDITIVE: legacy `scope="all"` callers see zero behavior change.
# ---------------------------------------------------------------------------


def _is_sentence_terminator_dot(text: str, dot_pos: int) -> bool:
    """Return True if the dot at ``dot_pos`` terminates a sentence.

    False positives the abbreviation guard catches:

    - Decimal numbers (digit-dot-digit): ``0.5``, ``§5.2``.
    - Letter-dot-letter-dot patterns: ``e.g.``, ``i.e.``, ``c.f.``.
    - Single-token abbreviations preceding the dot (whitespace- /
      punctuation-separated): ``vs.``, ``etc.``, ``cf.``, ``fig.``,
      ``eq.``, ``pp.``, ``viz.``, ``ca.``. See ``_ABBREV_BEFORE_DOT``.
    """
    n = len(text)
    prev_char = text[dot_pos - 1] if dot_pos > 0 else ""
    next_char = text[dot_pos + 1] if dot_pos + 1 < n else ""
    # Decimal: digit-dot-digit.
    if prev_char.isdigit() and next_char.isdigit():
        return False
    # Letter-dot-letter-dot pattern, dot is the SECOND dot in "x.y."
    if (
        dot_pos >= 3
        and prev_char.isalpha()
        and text[dot_pos - 2] == "."
        and text[dot_pos - 3].isalpha()
    ):
        return False
    # Letter-dot-letter-dot pattern, dot is the FIRST dot in "x.y."
    if dot_pos + 2 < n and next_char.isalpha() and text[dot_pos + 2] == ".":
        return False
    # Single-token abbreviation preceding the dot.
    j = dot_pos - 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    word = text[j + 1 : dot_pos].lower()
    return word not in _ABBREV_BEFORE_DOT


def _sentence_boundary_positions(text: str) -> list[int]:
    """Return sorted character positions where each sentence STARTS.

    Hard breaks (sentence terminators):

    - ``!`` and ``?`` always terminate.
    - ``.`` terminates unless the abbreviation guard
      (:func:`_is_sentence_terminator_dot`) returns False.
    - ``\\n\\n`` (paragraph break) terminates.

    Soft breaks (NOT sentence boundaries):

    - Single ``\\n`` (markdown line-wrap mid-sentence).
    - ``;`` (semicolons in dense list constructions).
    - ``:`` (colons preceding list items or definitions).

    The first sentence starts at position 0. Subsequent sentence starts
    are recorded at the first non-whitespace character after a hard
    break. Used by T3 (consume-on-match) and T4 (sentence-boundary
    detector-pair reject).
    """
    positions = [0]
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        boundary = False
        skip = 1
        if ch in "!?" or ch == "." and _is_sentence_terminator_dot(text, i):
            boundary = True
        elif ch == "\n" and i + 1 < n and text[i + 1] == "\n":
            boundary = True
            skip = 2
        if boundary:
            j = i + skip
            while j < n and text[j].isspace():
                j += 1
            if j < n and j > positions[-1]:
                positions.append(j)
            i = max(j, i + skip)
        else:
            i += 1
    return positions


def _sentence_id_of(pos: int, sentence_positions: Sequence[int]) -> int:
    """Return the zero-based sentence index containing ``pos``.

    Uses binary search over the sorted ``sentence_positions``. Returns
    ``0`` for any position before the first sentence start.
    """
    if not sentence_positions:
        return 0
    idx = bisect.bisect_right(sentence_positions, pos) - 1
    return max(0, idx)


def _crosses_sentence_boundary(pos_a: int, pos_b: int, sentence_positions: Sequence[int]) -> bool:
    """Return True if a sentence boundary lies strictly between ``pos_a`` and ``pos_b``.

    Sentence-boundary positions are derived from
    :func:`_sentence_boundary_positions`. Used by T4 to reject
    (detector, value) pairs whose detector mention is in a different
    sentence than the value.
    """
    if not sentence_positions:
        return False
    lo = min(pos_a, pos_b)
    hi = max(pos_a, pos_b)
    idx = bisect.bisect_right(sentence_positions, lo)
    return idx < len(sentence_positions) and sentence_positions[idx] <= hi


def _is_signed_value(text: str, val_start: int) -> bool:
    """True if the value at ``val_start`` is immediately preceded by ``+`` or ``-``.

    The sign marker indicates a paired-delta or comparative magnitude
    (e.g., ``-0.071`` AUPRC delta), not a binding claim. T1 filter
    skips these under ``scope="narrative"``.
    """
    return val_start > 0 and text[val_start - 1] in "+-"


def _has_keyword_in_window(
    text: str,
    val_start: int,
    pattern: re.Pattern[str],
    before_chars: int,
    after_chars: int,
) -> bool:
    """True if ``pattern`` matches anywhere in the character window around ``val_start``.

    Used by T1 (delta keywords) and T2 (floor keywords) to detect
    context cues near a candidate value. ``before_chars`` and
    ``after_chars`` control the asymmetric window — floor mentions
    typically PRECEDE the value (e.g., "random AUPRC is 0.374"),
    while delta mentions can be on either side.
    """
    start = max(0, val_start - before_chars)
    end = min(len(text), val_start + after_chars)
    return bool(pattern.search(text, start, end))


def _build_pattern(
    canonical: str,
    aliases: Sequence[str],
    *,
    case_insensitive: bool,
) -> re.Pattern[str]:
    """Build an OR-joined regex covering canonical name + aliases."""
    parts = [re.escape(canonical), *aliases]
    pattern = "|".join(f"(?:{p})" for p in parts)
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(pattern, flags)


def _line_starts(text: str) -> list[int]:
    """Return character positions where each line starts. line[i] starts at line_starts[i]."""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _nearest_canonical_key(
    positions: Sequence[tuple[int, str]],
    value_pos: int,
    max_distance: int,
) -> tuple[str, int] | None:
    """Return ``(key, position)`` paired with ``value_pos`` by text-order, or None.

    Pairing rule: pick the LAST canonical occurrence that appears
    BEFORE the value (text-order); if none is within ``max_distance``,
    fall back to the FIRST canonical occurrence that appears AFTER
    the value within the same range. Matches natural English prose
    pattern "<token> ... <value>" (subject-verb-object, predominant)
    with a fallback for the inverted "<value> ... by <token>" form.

    Used for DETECTOR pairing AND slice pairing. The
    "absolute-distance nearest" heuristic was rejected for detectors
    — it produces false positives on prose like "TF-IDF achieves
    0.971, while LoRA reaches 0.974" where 0.971 is closer to LoRA
    in raw distance even though it semantically belongs to TF-IDF.

    v1.2.0: now returns ``(key, position)`` instead of just ``key``
    so callers can apply position-dependent secondary checks (e.g.,
    T4 sentence-boundary detector-pair reject). The slice-pairing
    call site discards the position.
    """
    if not positions:
        return None
    # Look for the LAST position strictly before the value, within range.
    last_before: tuple[str, int] | None = None
    for pos, key in positions:
        if pos < value_pos and (value_pos - pos) <= max_distance:
            last_before = (key, pos)
        elif pos >= value_pos:
            break
    if last_before is not None:
        return last_before
    # Fall back: FIRST position after the value, within range.
    for pos, key in positions:
        if pos >= value_pos and (pos - value_pos) <= max_distance:
            return (key, pos)
    return None


def _nearest_slice_key_by_distance(
    positions: Sequence[tuple[int, str]],
    value_pos: int,
    max_distance: int,
) -> str | None:
    """Return the canonical slice key paired with ``value_pos`` by raw distance.

    Unlike :func:`_nearest_canonical_key` (which uses text-order to
    handle the subject-verb-object English prose pattern for
    detectors), slice context is a prepositional adjunct: "On
    <slice>, X scored Y" (pre-value) and "X scored Y on <slice>"
    (post-value) are both common. A raw-distance nearest rule
    handles both naturally: the slice mention closer to the value
    in characters wins.

    Mitigates the cross-paragraph slice-bleed false positive (e.g.,
    "cross-family distribution shift, finding that... reaches 0.971
    AUPRC on direct+benign validation" — the text-order rule would
    mis-attribute 0.971 to the "cross-family" topic introduced
    earlier; raw-distance correctly attributes to the
    "direct+benign validation" mention 30 chars after the value).
    """
    if not positions:
        return None
    best_key: str | None = None
    best_dist = max_distance + 1
    for pos, key in positions:
        dist = abs(pos - value_pos)
        if dist <= max_distance and dist < best_dist:
            best_key = key
            best_dist = dist
    return best_key


def _position_to_line(line_starts: list[int], pos: int) -> int:
    """Convert a 0-indexed character position to a 1-indexed line number."""
    # Binary-search-like; line_starts is sorted.
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1
