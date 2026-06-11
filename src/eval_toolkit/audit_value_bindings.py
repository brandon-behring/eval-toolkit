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

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

# v1.4.0 refactor (ADR 0007): shared narrative-prose helpers extracted
# to private flat module `_narrative.py` so audit_citation_alignment
# (v1.4.0+) can reuse them. All helpers preserve their pre-v1.4.0
# signatures; this is a signature-preserving refactor — all v1.3.0
# tests continue to pass unchanged.
from eval_toolkit._narrative import (
    _DELTA_PATTERN,
    _FLOOR_PATTERN,
    _GROUP_SUBJECT_PATTERN,
    _build_exclusion_ranges,
    _crosses_sentence_boundary,
    _has_keyword_in_window,
    _is_excluded,
    _sentence_boundary_positions,
    _sentence_id_of,
)

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


# v1.2.0 context-aware narrative filters (T1 delta, T2 floor, T3
# consume-on-match, T4 sentence-boundary) and v1.3.0 Layer 3 pairing
# rules (Patterns A/B/C/D per ADR 0006). The shared narrative helpers
# (keyword sets, exclusion ranges, sentence boundaries) live in the
# private flat module `_narrative.py` per ADR 0007's v1.4.0 refactor;
# they are imported at the top of this module.


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

    Raises
    ------
    ValueError
        If ``scope`` is not ``"all"`` or ``"narrative"`` (a typo would
        otherwise silently revert to legacy matching), or if any file in
        ``files`` is not valid UTF-8.
    TypeError
        If a ``bindings`` key is not a recognized shape (2-tuple,
        3-tuple, or :class:`BindingKey`).
    FileNotFoundError
        If any path in ``files`` does not exist (propagates from the
        underlying read; ``files`` is an explicit list, not a glob
        result).

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
    if scope not in {"all", "narrative"}:
        raise ValueError(f"scope must be 'all' or 'narrative', got {scope!r}")
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
    # v1.3.0 Pattern D (metric-axis nearest-pairing) requires knowing
    # ALL metrics that might appear in prose, not just bound metrics —
    # e.g., when prose mentions AUROC near a value but only AUPRC is
    # bound, Pattern D needs the AUROC pattern to correctly pair the
    # value with the right metric. Union of binding metrics +
    # consumer-supplied metric_aliases keys.
    metric_keys = sorted({k.metric for k in canonical_bindings} | set(metric_aliases_dict.keys()))
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

    # v1.3.0 Layer 3 pairing rules (per ADR 0006). Built per-call
    # because Patterns A and B depend on the consumer's detector
    # aliases. `None` when scope="all" (legacy; rules don't fire).
    postfix_pat: re.Pattern[str] | None = (
        _build_postfix_pattern(detector_aliases_dict, detector_keys)
        if scope == "narrative"
        else None
    )
    possessive_pat: re.Pattern[str] | None = (
        _build_possessive_pattern(detector_aliases_dict, detector_keys)
        if scope == "narrative"
        else None
    )

    # Inverse-alias index: alias-regex (string form) → canonical key.
    # Used to resolve a matched postfix/possessive alias-group back to
    # the canonical detector for override resolution. Each alias regex
    # is keyed verbatim; the resolution path tries each canonical key's
    # alias list + canonical-name fallback.
    def _resolve_canonical_from_alias_match(alias_text: str) -> str | None:
        """Return the canonical detector key whose pattern matched ``alias_text``.

        Iterates the per-detector patterns and tries to match the
        alias_text. Uses re.IGNORECASE for consistency with the
        outer postfix/possessive patterns. First-match wins (the
        OR-build above means there's only one canonical key per
        match anyway in practice).
        """
        for det_key in detector_keys:
            det_pat = detector_patterns[det_key]
            # det_pat is the alias OR pattern from _build_pattern,
            # case-insensitive. fullmatch on the alias_text checks
            # whether this alias belongs to det_key's set.
            if det_pat.fullmatch(alias_text):
                return det_key
        return None

    violations: list[Violation] = []
    matched: list[Match] = []
    matched_keys: set[BindingKey] = set()
    unmatched_slice_count = 0

    for file_path in files_resolved:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{file_path}: not valid UTF-8 ({exc}). "
                f"audit_value_bindings requires UTF-8-encoded markdown."
            ) from exc
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

        # v1.3.0 Pattern D — metric-axis nearest-pairing (Layer 3 per
        # ADR 0006, narrative-scope only). Pre-collect ALL metric
        # positions so each value can be paired with its NEAREST
        # metric mention (text-order). Catches the case where prose
        # mentions BOTH metrics ("AUPRC delta suggests: AUROC 0.383")
        # and the validator's window-based metric proximity check
        # picks up the wrong metric. Symmetric to detector pairing.
        metric_positions: list[tuple[int, str]] = []  # (position, canonical_metric)
        for m_key, m_re in metric_patterns.items():
            for m_match in m_re.finditer(text):
                metric_positions.append((m_match.start(), m_key))
        metric_positions.sort()

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

                    # v1.3.0 Pattern C — group-subject suppression
                    # (narrative-scope only). When prose says "for the
                    # {trained|frozen|baseline|all|both|other}
                    # detectors" within ±60 chars of the value AND on
                    # the same side of any sentence boundary, the
                    # value refers to a multi-detector group statement
                    # that doesn't bind to a single canonical detector.
                    # Suppress the candidate (v1.4.0+ may attempt
                    # multi-detector inference per ADR 0006).
                    if scope == "narrative":
                        gs_start = max(0, val_start_in_full - 60)
                        gs_end = min(len(text), val_start_in_full + len(val_str) + 60)
                        gs_match = _GROUP_SUBJECT_PATTERN.search(text, gs_start, gs_end)
                        if gs_match is not None and not _crosses_sentence_boundary(
                            gs_match.start(), val_start_in_full, sentence_positions
                        ):
                            continue

                    # v1.3.0 Pattern A / B — Layer 3 pairing-rule
                    # OVERRIDES (narrative-scope only). When a postfix
                    # or possessive explicitly names a detector, the
                    # override is AUTHORITATIVE — it confirms or
                    # rejects the binding without falling through to
                    # the proximity-based detector pairing below.
                    #
                    # - postfix_confirmed_pos / possessive_confirmed_pos:
                    #   the character position of the override match,
                    #   used as the effective "paired detector
                    #   position" for downstream T4 (sentence-
                    #   boundary) check.
                    # - If postfix/possessive_canonical == det_key:
                    #   confirmed; bypass proximity.
                    # - If != det_key AND is in bindings: skip (the
                    #   other detector's loop iteration claims it).
                    # - If doesn't resolve / no match: fall through
                    #   to proximity-based pairing.
                    pairing_confirmed_pos: int | None = None

                    # Pattern A — "for {detector}" postfix
                    if postfix_pat is not None:
                        val_end = val_start_in_full + len(val_str)
                        pf_match = postfix_pat.search(text, val_end, min(len(text), val_end + 50))
                        if pf_match is not None:
                            # Intervening-value guard: prose like
                            # "X 0.971 versus 0.293 for LoRA" — the
                            # "for LoRA" postfix belongs to 0.293,
                            # not 0.971. CI brackets like `[0.283,
                            # 0.298]` are excluded from intervening
                            # consideration via the existing
                            # excluded_ranges (v1.1.0 scope filter):
                            # values inside brackets aren't real
                            # binding-candidate intervening values.
                            intervening: re.Match[str] | None = None
                            for m in value_re.finditer(text, val_end, pf_match.start()):
                                if not (
                                    excluded_ranges and _is_excluded(m.start(), excluded_ranges)
                                ):
                                    intervening = m
                                    break
                            if intervening is None:
                                postfix_canonical = _resolve_canonical_from_alias_match(
                                    pf_match.group(1)
                                )
                                if postfix_canonical is not None:
                                    if postfix_canonical != det_key:
                                        continue
                                    pairing_confirmed_pos = pf_match.start()

                    # Pattern B — possessive `'s` (only if Pattern A
                    # didn't already confirm). Find the LAST possessive
                    # in the −80 char pre-window; if its end is within
                    # 30 chars of the value start, apply override.
                    if pairing_confirmed_pos is None and possessive_pat is not None:
                        ps_matches = list(
                            possessive_pat.finditer(
                                text, max(0, val_start_in_full - 80), val_start_in_full
                            )
                        )
                        if ps_matches:
                            ps_match = ps_matches[-1]
                            if val_start_in_full - ps_match.end() <= 30:
                                possessive_canonical = _resolve_canonical_from_alias_match(
                                    ps_match.group(1)
                                )
                                if possessive_canonical is not None:
                                    if possessive_canonical != det_key:
                                        continue
                                    pairing_confirmed_pos = ps_match.start()

                    # Detector pairing: when a Layer 3 override
                    # confirmed the binding (pairing_confirmed_pos
                    # set), skip the proximity check — the postfix /
                    # possessive is authoritative. Otherwise, fall
                    # back to the text-order proximity rule (last
                    # detector before; else first detector after).
                    if pairing_confirmed_pos is not None:
                        paired_det_pos = pairing_confirmed_pos
                    else:
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

                    # v1.3.0 Pattern D — metric-axis nearest-pairing
                    # (narrative-scope only). Require the NEAREST
                    # metric mention to the value (by text-order
                    # last-before-first-after) to be THIS binding's
                    # canonical metric. Catches prose like "than the
                    # AUPRC delta suggests: LoRA's pooled OOD AUROC
                    # is 0.383" where the AUPRC mention from the
                    # delta clause is within window of 0.383 but
                    # AUROC is the metric semantically owning it.
                    if scope == "narrative":
                        metric_match = _nearest_canonical_key(
                            metric_positions, val_start_in_full, max_distance_chars
                        )
                        if metric_match is not None and metric_match[0] != met_key:
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


def _build_postfix_pattern(
    detector_aliases: Mapping[str, Sequence[str]],
    detector_keys: Sequence[str],
) -> re.Pattern[str] | None:
    """Build a regex matching `"for {detector_alias}"` postfix constructs.

    v1.3.0 Pattern A (Layer 3 pairing rule per ADR 0006). Used to
    re-pair a candidate value with the detector named in a "for X"
    postfix (e.g., ``"0.291 [...] for TF-IDF + LR"`` binds 0.291 to
    TF-IDF + LR via the postfix, overriding proximity-based pairing).

    Each alias is paired with its canonical key in a single named-group
    OR pattern; the capture group reveals which detector matched. The
    canonical-key-as-fallback ensures the canonical name itself is
    matched even if no alias regex is provided for that detector.

    Returns None if there are no detectors to build patterns for
    (empty bindings).
    """
    if not detector_keys:
        return None
    alts: list[str] = []
    for det_key in detector_keys:
        # Canonical name as a literal alternative + all alias regexes
        # (which may themselves contain regex syntax like `\+`).
        parts = [re.escape(det_key)] + list(detector_aliases.get(det_key, ()))
        # Each detector's parts collapse into a non-capturing group.
        alts.append("(?:" + "|".join(parts) + ")")
    # The outer capture group reveals which detector token matched.
    # The text-order rule means the first alternative wins per Python
    # re semantics, which is fine for our use case.
    return re.compile(
        r"\bfor\s+(?:the\s+)?(" + "|".join(alts) + r")(?=[\s,;.)\]]|$)",
        re.IGNORECASE,
    )


def _build_possessive_pattern(
    detector_aliases: Mapping[str, Sequence[str]],
    detector_keys: Sequence[str],
) -> re.Pattern[str] | None:
    """Build a regex matching `"{detector_alias}'s"` possessive markers.

    v1.3.0 Pattern B (Layer 3 pairing rule per ADR 0006). The
    possessive ``'s`` construction is a strong binding signal that
    isn't captured by detector-alias regex matching directly (alias
    patterns don't typically include the apostrophe). Re-pairs the
    candidate value with the possessor detector.

    The pattern matches JUST the possessive marker (``{alias}'s``);
    binding-claim proximity is enforced at the call site (the
    inner loop's Pattern B block requires the LAST possessive
    in the pre-window to END within 30 chars of the value, which
    covers both `"frozen probe's 0.515"` (immediate) and
    `"LoRA's pooled OOD AUROC is 0.383"` (5-token clause).

    Returns None if there are no detectors (empty bindings).
    """
    if not detector_keys:
        return None
    alts: list[str] = []
    for det_key in detector_keys:
        parts = [re.escape(det_key)] + list(detector_aliases.get(det_key, ()))
        alts.append("(?:" + "|".join(parts) + ")")
    # Match `{alias}'s` (ASCII apostrophe or typographic ’s). Tight
    # — proximity to the value is enforced at the call site via
    # `match.end()` against the value position.
    return re.compile(
        r"(" + "|".join(alts) + r")[’']s\b",
        re.IGNORECASE,
    )


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

    A raw-distance variant for SLICE pairing (prepositional-adjunct
    grammar) was prototyped but never wired; it was removed in
    v1.11.0 (#99) — slice pairing deliberately shares this text-order
    rule, and the rejected alternative lives in git history.
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
