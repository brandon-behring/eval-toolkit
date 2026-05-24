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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Default citation pattern: matches "per ADR-NNN", "via ADR-NNN", "by ADR-NNN",
# "under ADR-NNN" — case-insensitive on the citation phrase; ADR-NNN is
# 3-digit-zero-padded by Nygard convention.
DEFAULT_CITATION_PATTERN: Final[str] = r"(?i)(?:per|via|by|under)\s+ADR-(\d{3})"

# Sniff radius around a citation match for category-keyword matching.
# Locked at ±2 lines so the validator catches citations whose claim
# category is on the immediately-adjacent line (common in Markdown
# tables / bullet lists / wrapped prose).
DEFAULT_CONTEXT_LINES: Final[int] = 2


@dataclass(frozen=True)
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
        Claim-taxonomy category the ADR belongs to (e.g.
        ``"test_markers"``, ``"reproducibility"``, ``"cost"``). ``None``
        if no category matched the ADR's title/slug keywords (caller
        decides whether to treat ``None`` as a finding or skip).
    """

    adr_id: str
    title: str
    slug: str
    category: str | None


@dataclass(frozen=True)
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
        ≤120 chars of context around the citation (for human review).
    claim_category : str | None
        Category inferred from the surrounding text (None if no
        category keyword matched).
    adr_actual_category : str | None
        Category inferred from the cited ADR's title+slug (None if no
        category keyword matched).
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
    """
    haystack = f"{title} {slug}".lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in haystack:
                return category
    return None


def _extract_context_text(
    lines: list[str],
    line_index: int,
    context_lines: int,
) -> str:
    """Return ≤120-char snippet of context around `line_index` (1-indexed)."""
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


def validate_citations(
    *,
    markdown_text: str,
    markdown_path: Path,
    adr_subjects: dict[str, ADRSubject],
    category_keywords: dict[str, list[str]],
    citation_pattern: str = DEFAULT_CITATION_PATTERN,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    known_exempt_citations: Sequence[tuple[Path, int, str]] = (),
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
        :func:`extract_adr_subject_category`.
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

    Returns
    -------
    list[CitationMisalignment]
        One :class:`CitationMisalignment` per misaligned citation.
        Empty if no misalignments OR no citations matched the pattern.

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
    exempt_set = {(str(p), ln, adr) for (p, ln, adr) in known_exempt_citations}
    misalignments: list[CitationMisalignment] = []
    lines = markdown_text.splitlines()
    citation_re = re.compile(citation_pattern)

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
            context = _extract_context_text(lines, line_no, context_lines)
            claim_category = _infer_claim_category(context, category_keywords)
            if claim_category is None:
                # No category basis for comparison; skip per the deferral above.
                continue
            if claim_category == subject.category:
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
