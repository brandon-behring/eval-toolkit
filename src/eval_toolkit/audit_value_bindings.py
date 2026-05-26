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
                    # digit or dot.
                    val_start_in_full = window_offset + val_match.start()
                    if val_start_in_full > 0:
                        prev_char = text[val_start_in_full - 1]
                        if prev_char.isdigit() or prev_char == ".":
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

                    # Cross-detector disambiguation: require the current
                    # det_key to be the detector paired with this value
                    # by the text-order rule (last detector before; else
                    # first detector after). Avoids cross-contamination
                    # on multi-detector prose like "TF-IDF achieves
                    # 0.971, while LoRA reaches 0.974".
                    paired_key = _nearest_canonical_key(
                        detector_positions, val_start_in_full, max_distance_chars
                    )
                    if paired_key != det_key:
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
                        paired_slice = _nearest_canonical_key(
                            slice_positions,
                            val_start_in_full,
                            slice_window_chars,
                        )
                        if paired_slice is None:
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
                        if paired_slice != slice_key:
                            continue

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
) -> str | None:
    """Return the canonical key paired with ``value_pos`` by text-order, or None.

    Pairing rule: pick the LAST canonical occurrence that appears
    BEFORE the value (text-order); if none is within ``max_distance``,
    fall back to the FIRST canonical occurrence that appears AFTER
    the value within the same range. Matches natural English prose
    pattern "<token> ... <value>" (subject-verb-object, predominant)
    with a fallback for the inverted "<value> ... by <token>" form.

    Used for DETECTOR pairing. The "absolute-distance nearest"
    heuristic was rejected for detectors — it produces false positives
    on prose like "TF-IDF achieves 0.971, while LoRA reaches 0.974"
    where 0.971 is closer to LoRA in raw distance even though it
    semantically belongs to TF-IDF.

    For slice pairing, use :func:`_nearest_slice_key_by_distance`
    instead — slice context is a prepositional adjunct that appears
    EITHER side of the value with no strong syntactic prior, and the
    text-order bias mis-attributes setup-clause slices to values in
    subsequent clauses.
    """
    if not positions:
        return None
    # Look for the LAST position strictly before the value, within range.
    last_before: str | None = None
    for pos, key in positions:
        if pos < value_pos and (value_pos - pos) <= max_distance:
            last_before = key
        elif pos >= value_pos:
            break
    if last_before is not None:
        return last_before
    # Fall back: FIRST position after the value, within range.
    for pos, key in positions:
        if pos >= value_pos and (pos - value_pos) <= max_distance:
            return key
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
