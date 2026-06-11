r"""ADR-citation alignment validator.

Catches the bug class where a reader-facing markdown surface cites
"per ADR-NNN" but the cited ADR's actual subject doesn't match the
surrounding claim category.

Motivating test case (from `prompt-injection-detection-prototype` v1.3.2
audit, file `docs/REPRODUCIBILITY.md:76`)::

    "Two-tier reproduction (locked at Phase 0-07 via ADR-029):"

ADR-029 is the test-marker-strategy ADR (unit / smoke / integration /
network markers). The actual reproducibility-tier-lock ADR is ADR-034.
The citation is wrong by 5 digits, with surrounding context "Two-tier
reproduction" clearly in the *reproducibility* category, not the
*test_markers* category. :func:`validate_citations` flags this case.

Design (per ADR 0001 contract-first; ADR 0002 metric-spec style for the
configurable categories):

- The validator is **pure**: pass in markdown text + ADR frontmatter +
  a category-keyword map; get back a list of
  :class:`CitationMisalignment` records. No filesystem I/O inside the
  validator; the CLI wrapper handles globbing.
- Categories are **consumer-supplied**: this module ships no default
  category map. Consumers wire their project's claim taxonomy
  (reproducibility / cost / calibration / threshold / contamination /
  test_markers / leakage / etc.) into the validator.

References
----------
.. [1] Nygard, M. "Documenting Architecture Decisions." 2011.
    https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

# v1.4.0 (ADR 0007): shared narrative-prose helpers extracted in the
# v1.4.0 refactor. Same module used by audit_value_bindings for
# Layer 2 (scope='narrative') content-type filtering + Layer 3
# sentence-boundary respect.
from eval_toolkit._narrative import (
    _build_exclusion_ranges,
    _is_excluded,
    _line_starts,
    _sentence_boundary_positions,
)

# Default citation pattern: matches "per ADR-NNN", "via ADR-NNN", "by ADR-NNN",
# "under ADR-NNN" — case-insensitive on the citation phrase; ADR-NNN is
# 3-digit-zero-padded by Nygard convention.
DEFAULT_CITATION_PATTERN: Final[str] = r"(?i)(?:per|via|by|under)\s+ADR-(\d{3})"

# Sniff radius around a citation match for category-keyword matching.
# Locked at ±2 lines so the validator catches citations whose claim
# category is on the immediately-adjacent line (common in Markdown
# tables / bullet lists / wrapped prose).
DEFAULT_CONTEXT_LINES: Final[int] = 2


@dataclass(frozen=True, slots=True)
class ADRSubject:
    """Subject category of a single ADR.

    Parameters
    ----------
    adr_id : str
        3-digit-zero-padded ADR id, e.g. ``"029"``.
    title : str
        ADR title (from frontmatter ``title:`` field).
    slug : str
        ADR slug (from frontmatter ``slug:`` field). Often informative
        about the actual subject.
    category : str | None
        First-match claim-taxonomy category the ADR belongs to (e.g.
        ``"test_markers"``, ``"reproducibility"``, ``"cost"``). ``None``
        if no category matched the ADR's title/slug keywords. Under
        ``scope="narrative"`` :func:`validate_citations` SKIPS citations
        to a ``None``-category subject (no basis for comparison); under
        ``scope="all"`` it flags them whenever a claim category is
        inferred (legacy v1.0.1 behavior). Always the value displayed in
        :attr:`CitationMisalignment.adr_actual_category`.
    categories : frozenset[str] | None, optional
        ALL matching claim-taxonomy categories (added v1.11.0 as a
        strictly-appended trailing optional field — SemVer-MINOR per
        ADR 0003). Populate via :func:`extract_adr_subject_categories`
        so multi-topic ADRs are not collapsed to their first-match
        category. When not ``None`` (including the explicit empty set),
        it takes precedence over ``category`` for match decisions in
        :func:`validate_citations`. ``None`` (default) preserves
        pre-v1.11.0 scalar behavior exactly.
    """

    adr_id: str
    title: str
    slug: str
    category: str | None
    categories: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class CitationMisalignment:
    """A "per ADR-NNN" citation whose category doesn't match the cited ADR's subject.

    Parameters
    ----------
    file : Path
        Reader-facing markdown file the citation appears in.
    line : int
        1-indexed line number of the citation.
    cited_adr_id : str
        3-digit-zero-padded ADR id from the citation.
    surrounding_text : str
        ≤300 chars of context around the citation (for human review).
        Under ``scope="all"``: the ±``context_lines`` line window.
        Under ``scope="narrative"``: the citation's sentence, centered
        on the citation when the sentence exceeds 300 chars.
    claim_category : str | None
        Category inferred from the surrounding text (None if no
        category keyword matched).
    adr_actual_category : str | None
        Category inferred from the cited ADR's title+slug (None if no
        category keyword matched). Always the cited ADR's scalar
        ``category`` (first-match); when :attr:`ADRSubject.categories`
        is populated the full set lives on the subject record, not
        here.
    """

    file: Path
    line: int
    cited_adr_id: str
    surrounding_text: str
    claim_category: str | None
    adr_actual_category: str | None


def extract_adr_subject_category(
    title: str,
    slug: str,
    category_keywords: dict[str, list[str]],
) -> str | None:
    """Infer an ADR's claim-taxonomy category from its title + slug.

    Walks each ``(category, keywords)`` entry in ``category_keywords``
    and returns the first category whose keywords appear in the
    concatenated title+slug (case-insensitive).

    Parameters
    ----------
    title : str
        ADR title from frontmatter.
    slug : str
        ADR slug from frontmatter or filename.
    category_keywords : dict[str, list[str]]
        Map from category name to a list of keyword substrings. First
        keyword match wins; categories are tested in dict-insertion
        order, so the caller controls priority.

    Returns
    -------
    str | None
        Matching category name, or ``None`` if no keyword matched.

    Examples
    --------
    >>> extract_adr_subject_category(
    ...     title="Reproducibility tier - full ladder T0 + T1 + T3",
    ...     slug="reproducibility-tier-full-ladder",
    ...     category_keywords={
    ...         "test_markers": ["marker", "smoke marker"],
    ...         "reproducibility": ["reproduc", "tier"],
    ...     },
    ... )
    'reproducibility'

    See Also
    --------
    extract_adr_subject_categories : Plural sibling returning ALL
        matching categories (v1.11.0); this function is the
        insertion-order-first element of that set.
    """
    matches = extract_adr_subject_categories(title, slug, category_keywords)
    for category in category_keywords:
        if category in matches:
            return category
    return None


def extract_adr_subject_categories(
    title: str,
    slug: str,
    category_keywords: dict[str, list[str]],
) -> frozenset[str]:
    """Infer ALL of an ADR's claim-taxonomy categories from its title + slug.

    Plural sibling of :func:`extract_adr_subject_category` (v1.11.0,
    Tier-2 ADDITIVE per ADR 0003). Returns every category with at least
    one keyword hit in the concatenated title+slug (case-insensitive),
    so multi-topic ADRs (e.g., a reproducibility ADR that also locks the
    cost envelope) are not collapsed to a single first-match category.
    Populate :attr:`ADRSubject.categories` with the result.

    Parameters
    ----------
    title : str
        ADR title from frontmatter.
    slug : str
        ADR slug from frontmatter or filename.
    category_keywords : dict[str, list[str]]
        Map from category name to a list of keyword substrings.

    Returns
    -------
    frozenset[str]
        All matching category names; empty when nothing matched.
        Invariant: ``extract_adr_subject_category(...)`` equals the
        insertion-order-first element of this set (``None`` when empty).

    Examples
    --------
    >>> sorted(extract_adr_subject_categories(
    ...     title="Reproducibility tier lock and GPU cost envelope",
    ...     slug="reproducibility-tier-cost",
    ...     category_keywords={
    ...         "test_markers": ["marker"],
    ...         "reproducibility": ["reproduc", "tier"],
    ...         "cost": ["cost", "envelope"],
    ...     },
    ... ))
    ['cost', 'reproducibility']
    """
    haystack = f"{title} {slug}".lower()
    matches: set[str] = set()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in haystack:
                matches.add(category)
                break
    return frozenset(matches)


def _extract_context_text(
    lines: list[str],
    line_index: int,
    context_lines: int,
) -> str:
    """Return ≤300-char snippet of the ±`context_lines` window around `line_index` (1-indexed)."""
    start = max(0, line_index - 1 - context_lines)
    end = min(len(lines), line_index + context_lines)
    return " ".join(line.strip() for line in lines[start:end])[:300]


def _infer_claim_category(
    context: str,
    category_keywords: dict[str, list[str]],
) -> str | None:
    """Same first-match-wins keyword check as ADR subject extraction, on prose context."""
    haystack = context.lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in haystack:
                return category
    return None


def _extract_sentence_context(
    text: str,
    citation_pos: int,
    sentence_positions: Sequence[int],
) -> str:
    """Return a ≤300-char window from the sentence containing ``citation_pos``.

    v1.11.0 (#99): when the sentence exceeds 300 chars, the window is
    the 300-char slice CENTERED on the citation, clamped to the
    sentence bounds. The pre-v1.11.0 trailing ``[:300]`` truncation
    anchored at the sentence start, so a citation past char 300 of its
    own sentence fell OUTSIDE the window that classifies it (5 of the
    37 dogfood residuals). The no-sentence-positions fallback
    (defensive; unreachable from :func:`validate_citations`, which
    always passes a non-empty list) is reconciled to the same cap: a
    ±150-char window centered on the citation.

    v1.4.0 Layer 3 rule γ per ADR 0007: the category-keyword
    extraction window for an ADR citation must not cross sentence
    boundaries. Uses :func:`_narrative._sentence_boundary_positions`
    (paragraph-aware, abbreviation-guarded).

    Replaces ``_extract_context_text``'s ±N-line window when
    ``scope="narrative"``. The ±2-line window is too wide for
    multi-topic prose (e.g., the consumer's
    ``"per ADR-025 + ADR-021 + ADR-034 + ADR-045"`` Pattern α case)
    because it pulls in keywords from previous sentences that don't
    semantically apply to the citation.
    """
    if not sentence_positions:
        return text[max(0, citation_pos - 150) : citation_pos + 150].strip()
    sent_idx = max(0, bisect.bisect_right(sentence_positions, citation_pos) - 1)
    sent_start = sentence_positions[sent_idx] if sent_idx < len(sentence_positions) else 0
    next_idx = sent_idx + 1
    sent_end = sentence_positions[next_idx] if next_idx < len(sentence_positions) else len(text)
    # Measure the STRIPPED sentence, not the raw inter-sentence span:
    # `sent_end` is the next sentence's first char, so terminator +
    # blank-line whitespace inflates the raw span. A sentence whose
    # text fits in 300 chars must always return whole (review finding,
    # #99): the centered clamp below operates on the whitespace-trimmed
    # core so trailing padding can never clip the sentence head.
    raw = text[sent_start:sent_end]
    stripped = raw.strip()
    if len(stripped) <= 300:
        return stripped
    core_start = sent_start + (len(raw) - len(raw.lstrip()))
    core_end = core_start + len(stripped)
    start = max(core_start, min(citation_pos - 150, core_end - 300))
    return text[start : start + 300].strip()


def _infer_claim_categories_multi(
    context: str,
    category_keywords: dict[str, list[str]],
) -> set[str]:
    """Return the SET of all categories whose keywords appear in context.

    v1.4.0 Layer 3 rule α per ADR 0007: when prose surrounding a
    multi-ADR-citation list (e.g.,
    ``"per ADR-025 + ADR-021 + ADR-034 + ADR-045"``) contains MULTIPLE
    topic keywords, each ADR can legitimately address a DIFFERENT
    topic. Instead of first-match-wins (which incorrectly flags the
    other ADRs against the dominant topic), return the full set of
    matched categories; the validator then accepts the citation if
    the ADR's actual category is IN the set.

    First-match-wins per category (so categories don't double-count
    on multi-keyword matches), but multi-category aggregation across
    the context.
    """
    haystack = context.lower()
    matches: set[str] = set()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in haystack:
                matches.add(category)
                break
    return matches


def _effective_categories(subject: ADRSubject) -> frozenset[str] | None:
    """Resolve the subject's effective category set for match decisions.

    ``categories`` (plural, v1.11.0) wins when not ``None`` — including
    the explicit empty set, which means "the plural extractor ran and
    matched nothing" (the same epistemic state as scalar ``None``).
    Otherwise the scalar ``category`` is lifted to a singleton.
    ``None`` when neither field is populated.
    """
    if subject.categories is not None:
        return subject.categories
    if subject.category is not None:
        return frozenset({subject.category})
    return None


def validate_citations(
    *,
    markdown_text: str,
    markdown_path: Path,
    adr_subjects: dict[str, ADRSubject],
    category_keywords: dict[str, list[str]],
    citation_pattern: str = DEFAULT_CITATION_PATTERN,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    known_exempt_citations: Sequence[tuple[Path, int, str]] = (),
    scope: Literal["all", "narrative"] = "all",
) -> list[CitationMisalignment]:
    """Find "per ADR-NNN" citations whose category doesn't match the cited ADR.

    Parameters
    ----------
    markdown_text : str
        Body of the reader-facing markdown file.
    markdown_path : Path
        Path of the markdown file (for misalignment.file annotation).
    adr_subjects : dict[str, ADRSubject]
        Map from 3-digit ADR id to :class:`ADRSubject` records. Caller
        builds this by parsing each ADR's frontmatter; the
        ``ADRSubject.category`` field is populated via
        :func:`extract_adr_subject_category`, and (v1.11.0, optional)
        ``ADRSubject.categories`` via
        :func:`extract_adr_subject_categories` so multi-topic ADRs
        match on their full category set.
    category_keywords : dict[str, list[str]]
        Map from category name to substring-keyword list (used both for
        ADR subject inference and for surrounding-text category
        inference). Same map MUST be used for both directions.
    citation_pattern : str, optional
        Regex finding the citation surface. Group 1 must capture the
        3-digit ADR id. Default :data:`DEFAULT_CITATION_PATTERN`
        matches "per/via/by/under ADR-NNN".
    context_lines : int, optional
        Number of lines (±) around the citation to consider when
        inferring the claim category. Default
        :data:`DEFAULT_CONTEXT_LINES` (=2).
    known_exempt_citations : Sequence of (Path, int, str), optional
        ``(file, line, cited_adr_id)`` tuples to skip. Useful for
        consumers with known historical drift that's been accepted by
        policy (e.g., immutable ADR bodies with frozen-in errors that
        a superseding ADR has already addressed).
    scope : {"all", "narrative"}, optional
        v1.4.0 Layer 2 + Layer 3 opt-in (per ADR 0007). Default
        ``"all"`` preserves v1.0.1 / v1.3.0 behavior exactly
        (legacy line-window context, first-match category). Setting
        ``scope="narrative"`` activates:

        - **Layer 2 (Pattern β)**: skip citations inside markdown
          table rows, bracketed expressions, and fenced code blocks
          (the same content-type filter from v1.1.0
          ``audit_value_bindings``). Catches dense per-row ADR
          citations in spec tables (SPEC_SHEET.md and similar).
        - **Layer 3 rule γ**: extract the category-keyword window
          from the SENTENCE containing the citation (paragraph-aware
          abbreviation-guarded boundaries), not from a fixed
          ±line-window. Prevents the validator from pulling
          keywords across sentence boundaries.
        - **Layer 3 rule α**: when the citation's sentence matches
          MULTIPLE category keywords (multi-topic prose — common in
          dense multi-ADR lists like ``"per ADR-A + ADR-B + ADR-C"``,
          but applied to ANY multi-topic sentence regardless of
          citation count), the citation is accepted if the ADR's
          actual category set intersects the SET of matched
          categories, not just the first match.
        - **Layer 2 symmetric-``None`` skip**: citations to an ADR
          whose effective category set is empty (``categories`` is the
          empty set, or ``categories`` is ``None`` with ``category``
          ``None`` — see the ``categories`` precedence rule on
          :class:`ADRSubject`) are skipped — the consumer's
          ``category_keywords`` map gives no basis for comparison,
          mirroring the claim-side ``None`` skip described in Notes.
          Under ``scope="all"`` such citations are still flagged
          (legacy v1.0.1 behavior).

        Tier-1 ADDITIVE per ADR 0003; ``scope="all"`` (the default)
        guarantees byte-identical behavior to v1.3.x for subjects
        constructed without ``categories``.

    Returns
    -------
    list[CitationMisalignment]
        One :class:`CitationMisalignment` per misaligned citation.
        Empty if no misalignments OR no citations matched the pattern.

    Raises
    ------
    ValueError
        If ``scope`` is not ``"all"`` or ``"narrative"``. A typo would
        otherwise silently disable Layers 2+3 and fall through to the
        legacy line-window path.

    Notes
    -----
    A citation with ``claim_category=None`` (no category keyword
    matched the surrounding context) is **NOT** flagged as a
    misalignment. The validator defers to the caller's category map:
    if the caller's vocabulary doesn't cover the claim, there's no
    basis for saying the citation is misaligned. To force every
    citation to be flaggable, the caller should ensure their
    ``category_keywords`` has broad coverage.

    Examples
    --------
    >>> adr_subjects = {
    ...     "029": ADRSubject(
    ...         adr_id="029",
    ...         title="Test marker strategy",
    ...         slug="test-marker-strategy",
    ...         category="test_markers",
    ...     ),
    ... }
    >>> result = validate_citations(
    ...     markdown_text="Two-tier reproduction locked at Phase 0-07 via ADR-029.\\n",
    ...     markdown_path=Path("docs/REPRODUCIBILITY.md"),
    ...     adr_subjects=adr_subjects,
    ...     category_keywords={
    ...         "reproducibility": ["reproduc", "tier", "T0", "T1", "T3"],
    ...         "test_markers": ["marker"],
    ...     },
    ... )
    >>> len(result)
    1
    >>> result[0].cited_adr_id
    '029'
    >>> result[0].claim_category
    'reproducibility'
    >>> result[0].adr_actual_category
    'test_markers'
    """
    if scope not in {"all", "narrative"}:
        raise ValueError(f"scope must be 'all' or 'narrative', got {scope!r}")
    exempt_set = {(str(p), ln, adr) for (p, ln, adr) in known_exempt_citations}
    misalignments: list[CitationMisalignment] = []
    lines = markdown_text.splitlines()
    citation_re = re.compile(citation_pattern)

    # v1.4.0 narrative-scope precomputations (Layer 2 + Layer 3 per
    # ADR 0007). Empty / no-op when scope="all".
    line_starts_arr = _line_starts(markdown_text) if scope == "narrative" else []
    excluded_ranges = (
        _build_exclusion_ranges(markdown_text, line_starts_arr) if scope == "narrative" else []
    )
    sentence_positions = _sentence_boundary_positions(markdown_text) if scope == "narrative" else []

    for line_no, line in enumerate(lines, start=1):
        for match in citation_re.finditer(line):
            adr_id = match.group(1)
            if (str(markdown_path), line_no, adr_id) in exempt_set:
                continue
            subject = adr_subjects.get(adr_id)
            if subject is None:
                # Citation references an unknown ADR. Out of scope for
                # this validator (a different validator should check
                # "does ADR-NNN exist"). Skip.
                continue

            if scope == "narrative":
                # v1.4.0 narrative-scope filters per ADR 0007.
                abs_pos = line_starts_arr[line_no - 1] + match.start()

                # Pattern β (Layer 2): skip citations inside markdown
                # table rows, bracketed expressions, or fenced code
                # blocks.
                if _is_excluded(abs_pos, excluded_ranges):
                    continue

                # v1.4.0 Layer 2 symmetric-None skip (narrative-scope
                # only): if the cited ADR's subject category is None
                # (the consumer's category_keywords map doesn't
                # classify this ADR's title/slug), defer per the same
                # "no basis for comparison" principle that already
                # skips when claim_category is None. Symmetrizes the
                # existing v1.0.1 behavior; closes the dominant
                # residual under #82 dogfood where ADRs about
                # categories not in the consumer's map (e.g.,
                # "contamination", "training", "evaluation") get
                # flagged against every claim-side keyword match.
                # v1.11.0: keyed on the EFFECTIVE category set — an
                # explicit empty `categories` is the same epistemic
                # state as scalar None.
                effective_cats = _effective_categories(subject)
                if not effective_cats:
                    continue

                # Pattern γ (Layer 3): sentence-bounded context.
                context = _extract_sentence_context(markdown_text, abs_pos, sentence_positions)

                # Pattern α (Layer 3): multi-category-set membership
                # check. If the sentence containing the citation
                # matches MULTIPLE category keywords (multi-topic
                # prose), defer to the consumer's category map by
                # accepting the citation when the ADR's actual
                # category is in the set. Single-topic sentences
                # still use first-match-wins.
                #
                # v1.4.0 design refinement: extends Pattern α from
                # "multi-citation sentences only" (initial framing)
                # to "any multi-topic sentence." The underlying
                # principle — that single-keyword first-match is
                # wrong in multi-topic prose — applies regardless
                # of citation count.
                matched_set = _infer_claim_categories_multi(context, category_keywords)
                if not matched_set:
                    continue  # no category basis
                if len(matched_set) > 1:
                    # Multi-topic: accept if ANY of the ADR's categories
                    # intersects the sentence's matched set (v1.11.0 set
                    # intersection; identical to the scalar membership
                    # check when `categories` is unset).
                    if effective_cats & matched_set:
                        continue
                    # Else flag with first-match for diagnostic display
                    claim_category = _infer_claim_category(context, category_keywords)
                else:
                    # Single-topic: first-match-wins (legacy semantics)
                    claim_category = next(iter(matched_set))
                    if claim_category in effective_cats:
                        continue
            else:
                # Legacy scope='all' behavior (v1.3.x and prior), plus
                # the v1.11.0 opt-in: a populated `categories` set widens
                # the scalar equality to set membership. Byte-identical
                # to v1.3.x for every subject constructed without
                # `categories`.
                context = _extract_context_text(lines, line_no, context_lines)
                claim_category = _infer_claim_category(context, category_keywords)
                if claim_category is None:
                    continue
                effective_cats = _effective_categories(subject)
                if effective_cats is not None and claim_category in effective_cats:
                    continue

            misalignments.append(
                CitationMisalignment(
                    file=markdown_path,
                    line=line_no,
                    cited_adr_id=adr_id,
                    surrounding_text=context,
                    claim_category=claim_category,
                    adr_actual_category=subject.category,
                )
            )
    return misalignments
