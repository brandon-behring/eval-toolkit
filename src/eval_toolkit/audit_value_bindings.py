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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "Match",
    "ValueBindingsReport",
    "Violation",
    "validate_reader_value_bindings",
]


DEFAULT_VALUE_PATTERN: str = r"\d+\.\d{2,4}"
DEFAULT_MAX_DISTANCE_CHARS: int = 80
DEFAULT_TOLERANCE: float = 1e-4


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
        Fraction of ``(detector, metric)`` keys in the ``bindings``
        dict that produced at least one :class:`Match`. Range
        ``[0.0, 1.0]``. ``1.0`` means every binding was referenced in
        the scanned prose; lower values flag potentially un-cited
        bindings (which may be expected OR may indicate stale prose).
    """

    violations: tuple[Violation, ...]
    matched: tuple[Match, ...]
    coverage: float


def validate_reader_value_bindings(
    *,
    files: Sequence[Path | str],
    bindings: Mapping[tuple[str, str], float],
    value_pattern: str = DEFAULT_VALUE_PATTERN,
    max_distance_chars: int = DEFAULT_MAX_DISTANCE_CHARS,
    metric_aliases: Mapping[str, Sequence[str]] = MappingProxyType({}),
    detector_aliases: Mapping[str, Sequence[str]] = MappingProxyType({}),
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValueBindingsReport:
    """Validate (detector, metric, value) bindings in reader-prose markdown.

    For each ``(detector_token, metric_token) -> expected_value`` entry
    in ``bindings``, scan each file for triples of (detector mention,
    metric mention, numeric value) within a ``max_distance_chars``
    window. Compare the found value to the expected value; emit a
    :class:`Violation` on mismatch, a :class:`Match` on agreement.

    Both the detector and the metric must appear within the window
    surrounding a candidate value for the triple to be considered —
    a value that has only a detector or only a metric nearby is
    ignored (those belong to a value-existence audit, not a binding
    audit).

    Parameters
    ----------
    files : Sequence[Path | str]
        Markdown files to scan. UTF-8 encoded.
    bindings : Mapping[tuple[str, str], float]
        Canonical (detector_name, metric_name) → expected_value table.
        Keys are the canonical *identifiers* used in the report — the
        regex patterns that match these in prose come from the
        ``*_aliases`` dicts (with the canonical name as a default
        fallback pattern).
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
    tolerance : float, optional
        Absolute tolerance for float comparison. Default ``1e-4``
        (i.e., ``0.974`` and ``0.9740`` are considered equal).

    Returns
    -------
    ValueBindingsReport
        ``violations``, ``matched``, ``coverage`` per the dataclass.

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

    bindings_dict = dict(bindings)
    metric_aliases_dict = dict(metric_aliases)
    detector_aliases_dict = dict(detector_aliases)

    detector_keys = sorted({d for d, _ in bindings_dict})
    metric_keys = sorted({m for _, m in bindings_dict})

    detector_patterns: dict[str, re.Pattern[str]] = {
        d: _build_pattern(d, detector_aliases_dict.get(d, ()), case_insensitive=True)
        for d in detector_keys
    }
    metric_patterns: dict[str, re.Pattern[str]] = {
        m: _build_pattern(m, metric_aliases_dict.get(m, ()), case_insensitive=True)
        for m in metric_keys
    }
    value_re = re.compile(value_pattern)

    violations: list[Violation] = []
    matched: list[Match] = []
    matched_keys: set[tuple[str, str]] = set()

    for file_path in files_resolved:
        text = file_path.read_text(encoding="utf-8")
        line_starts = _line_starts(text)

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

        # For each binding, look in each file for triples.
        for (det_key, met_key), expected in bindings_dict.items():
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

                    # Cross-detector disambiguation: require the current
                    # det_key to be the detector paired with this value
                    # by the text-order rule (last detector before; else
                    # first detector after). Avoids cross-contamination
                    # on multi-detector prose like "TF-IDF achieves
                    # 0.971, while LoRA reaches 0.974".
                    paired_key = _nearest_detector_key(
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
                        matched_keys.add((det_key, met_key))
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

    coverage = len(matched_keys) / len(bindings_dict) if bindings_dict else 0.0
    return ValueBindingsReport(
        violations=tuple(violations),
        matched=tuple(matched),
        coverage=coverage,
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


def _nearest_detector_key(
    detector_positions: Sequence[tuple[int, str]],
    value_pos: int,
    max_distance: int,
) -> str | None:
    """Return the canonical detector key paired with ``value_pos``, or None.

    Pairing rule: pick the LAST detector that appears BEFORE the value
    (text-order); if none is within ``max_distance``, fall back to the
    FIRST detector that appears AFTER the value within the same range.
    This matches natural English prose patterns "<detector> ...
    <value>" (predominant) and "<value> ... by <detector>" (rare).

    The previous "absolute-distance nearest" heuristic produced false
    positives on prose like "TF-IDF achieves 0.971, while LoRA reaches
    0.974" where 0.971 is closer to LoRA in raw distance even though
    it semantically belongs to TF-IDF.
    """
    if not detector_positions:
        return None
    # Look for the LAST detector strictly before the value, within range.
    last_before: str | None = None
    for pos, key in detector_positions:
        if pos < value_pos and (value_pos - pos) <= max_distance:
            last_before = key
        elif pos >= value_pos:
            break
    if last_before is not None:
        return last_before
    # Fall back: FIRST detector after the value, within range.
    for pos, key in detector_positions:
        if pos >= value_pos and (pos - value_pos) <= max_distance:
            return key
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
