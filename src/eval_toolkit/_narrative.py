"""Shared narrative-prose helpers for the `audit_*` validator family.

This private flat module hosts the Layer 2 (scope) + Layer 3 (pairing-
rule) building blocks that emerged from the v1.1.0 → v1.3.0 cycle of
``audit_value_bindings`` and are reused by ``audit_citation_alignment``
at v1.4.0+. Per ADR 0007, the three-layer correctness model (identity
+ scope + pairing) applies family-wide; this module is the canonical
home for the prose-pattern primitives that the scope + pairing layers
build on.

Design notes:

- **Private flat module** (underscore-prefixed name): matches ADR 0001's
  `_rng.py` / `_parallel.py` / `_sweep.py` precedent. Not in the
  package's public ``_EXPORTS`` resolver; consumers import via
  ``eval_toolkit.audit_*`` modules, which in turn import from here.
- **Helpers preserve their exact signatures from audit_value_bindings.py**
  (v1.1.0–v1.3.0 vintage) — extraction is a signature-preserving
  refactor. All 43 existing audit_value_bindings tests continue to
  pass unchanged.
- **Keyword frozensets are audit_value_bindings-specific** (delta /
  floor / group-subject keywords are about value-binding prose, not
  citation prose). Other validators that need similar lists define
  their own constants. The SHARED parts are the regex-compilation
  utility and the structural helpers (exclusion ranges, sentence
  boundaries) that are validator-agnostic.

Cross-references:
- ADR 0005 — identity layer (BindingKey)
- ADR 0006 — pairing layer (audit_value_bindings-specific)
- ADR 0007 — three-layer architecture, family-wide
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Module-level keyword sets and compiled patterns.
# Specific to audit_value_bindings' Layer 2 filters (T1/T2/C); kept here
# so the validator can `from eval_toolkit._narrative import ...` rather
# than maintaining two copies. Other audit_* validators define their
# own keyword sets analogously.
# ---------------------------------------------------------------------------


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


# Group-subject adjectives that introduce a multi-detector statement.
# When prose says "for the trained detectors", the following value
# refers to a GROUP (LoRA + TF-IDF + ... whatever bindings exist),
# not a single canonical detector. The validator can't infer which
# specific detectors own the group value with positional heuristics,
# so v1.3.0 suppresses the candidate rather than attempting multi-
# detector inference (a v1.4.0+ candidate per ADR 0006).
_GROUP_SUBJECT_KEYWORDS: frozenset[str] = frozenset(
    {
        "trained",
        "frozen",
        "baseline",
        "all",
        "both",
        "other",
    }
)

# Module-level: detector-independent group-subject regex. Matches
# "for the {trained|frozen|...} detectors" (with optional "the"; both
# singular and plural "detector"/"detectors" tolerated).
_GROUP_SUBJECT_PATTERN: re.Pattern[str] = re.compile(
    r"\bfor\s+(?:the\s+)?(?:"
    + "|".join(sorted(re.escape(kw) for kw in _GROUP_SUBJECT_KEYWORDS))
    + r")\s+detectors?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Layer 2: content-type filtering helpers (`scope="narrative"`).
# Used by audit_value_bindings (v1.1.0+) and audit_citation_alignment
# (v1.4.0+) to exclude markdown table rows, bracketed expressions, and
# fenced code blocks from candidate-value / candidate-citation matching.
# ---------------------------------------------------------------------------


def _build_exclusion_ranges(
    text: str,
    line_starts: Sequence[int],
) -> list[tuple[int, int]]:
    """Compute sorted character ranges that ``scope="narrative"`` excludes.

    Excluded content types (per the lint-scope design discussion in
    ADR 0005):

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
# Sentence-boundary detection (paragraph-aware, abbreviation-guarded).
# Used by v1.2.0 T3/T4 in audit_value_bindings and the v1.4.0 Layer 3
# rule γ in audit_citation_alignment (sentence-boundary respect for
# category-keyword window).
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
    break. Used by audit_value_bindings T3/T4 and
    audit_citation_alignment Layer 3 rule γ.
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
    :func:`_sentence_boundary_positions`. Used by audit_value_bindings
    T4 (reject (detector, value) pairs across a sentence boundary)
    and audit_citation_alignment Layer 3 rule γ (the category-keyword
    extraction window for an ADR citation must not cross a sentence
    boundary).
    """
    if not sentence_positions:
        return False
    lo = min(pos_a, pos_b)
    hi = max(pos_a, pos_b)
    idx = bisect.bisect_right(sentence_positions, lo)
    return idx < len(sentence_positions) and sentence_positions[idx] <= hi


# ---------------------------------------------------------------------------
# Value-context helpers (used by audit_value_bindings T1/T2; kept here
# for any future audit_* validator that wants the same primitives).
# ---------------------------------------------------------------------------


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

    Used by audit_value_bindings T1 (delta keywords) and T2 (floor
    keywords) to detect context cues near a candidate value.
    ``before_chars`` and ``after_chars`` control the asymmetric
    window — floor mentions typically PRECEDE the value (e.g.,
    "random AUPRC is 0.374"), while delta mentions can be on either
    side.
    """
    start = max(0, val_start - before_chars)
    end = min(len(text), val_start + after_chars)
    return bool(pattern.search(text, start, end))


# ---------------------------------------------------------------------------
# Positional helpers (line-number bookkeeping). Shared by all three
# audit_* validators since v1.11.0 (#99); previously triplicated.
# ---------------------------------------------------------------------------


def _line_starts(text: str) -> list[int]:
    """Return character positions where each line starts.

    Line ``i`` (0-indexed) starts at ``_line_starts(text)[i]``. Companion
    :func:`_position_to_line` converts a position back to a 1-indexed line.
    """
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _position_to_line(line_starts: Sequence[int], pos: int) -> int:
    """Convert a 0-indexed character position to a 1-indexed line number.

    Binary search over the sorted output of :func:`_line_starts`.
    """
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1
