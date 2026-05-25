"""Tests for eval_toolkit.audit_citation_alignment.

Seed motivating case: ``docs/REPRODUCIBILITY.md:76`` in
`brandon-behring/prompt-injection-detection-prototype` (v1.3.2 audit
finding P1-2 Part 2) cited "ADR-029" for a tier-lock claim, but ADR-029
is the test-marker-strategy ADR (not the tier-lock; that's ADR-034). The
validator must flag this misalignment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_toolkit.audit_citation_alignment import (
    ADRSubject,
    CitationMisalignment,
    extract_adr_subject_category,
    validate_citations,
)

# Common consumer-supplied fixtures, mirroring the seed case's
# `prompt-injection-detection-prototype` category taxonomy.
SEED_CATEGORY_KEYWORDS = {
    "reproducibility": ["reproduc", "tier", "T0", "T1", "T3"],
    "cost": ["cost", "envelope", "spend", "$"],
    "test_markers": ["marker"],
    "calibration": ["calibration", "ECE", "Brier"],
}

# ADRs implicated in the seed case.
SEED_ADR_SUBJECTS = {
    "029": ADRSubject(
        adr_id="029",
        title="Test marker strategy — ratify 4-marker stratification",
        slug="test-marker-strategy-four-marker-ratification",
        category="test_markers",
    ),
    "034": ADRSubject(
        adr_id="034",
        title="Reproducibility tier — full ladder T0 + T1 + T3",
        slug="reproducibility-tier-full-ladder",
        category="reproducibility",
    ),
    "020": ADRSubject(
        adr_id="020",
        title="Compute infrastructure and cost discipline",
        slug="compute-infrastructure-and-cost-discipline",
        category="cost",
    ),
}


# ---- extract_adr_subject_category ----


def test_extract_adr_subject_category_first_match_wins() -> None:
    """First matching keyword across categories wins (dict insertion order)."""
    out = extract_adr_subject_category(
        title="Reproducibility tier — full ladder",
        slug="reproducibility-tier-full-ladder",
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert out == "reproducibility"


def test_extract_adr_subject_category_no_match_returns_none() -> None:
    """Title with no category keywords returns None (caller decides)."""
    out = extract_adr_subject_category(
        title="Some unrelated decision",
        slug="some-unrelated",
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert out is None


def test_extract_adr_subject_category_case_insensitive() -> None:
    """Keyword match is case-insensitive (haystack lower'd)."""
    out = extract_adr_subject_category(
        title="REPRODUCIBILITY TIER",
        slug="reproducibility-tier",
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert out == "reproducibility"


# ---- validate_citations: the seed positive case ----


def test_validate_citations_seed_case_flags_misalignment() -> None:
    """The v1.3.2 P1-2 ADR-029 mis-citation is correctly flagged."""
    text = "Two-tier reproduction (locked at Phase 0-07 via ADR-029):\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("docs/REPRODUCIBILITY.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert len(result) == 1
    m = result[0]
    assert m.cited_adr_id == "029"
    assert m.claim_category == "reproducibility"
    assert m.adr_actual_category == "test_markers"
    assert "ADR-029" in m.surrounding_text or "reproduction" in m.surrounding_text.lower()


def test_validate_citations_correct_citation_no_flag() -> None:
    """A citation to the CORRECT tier-lock ADR (034) doesn't flag."""
    text = "Three-tier reproduction (locked at Phase 0-07 via ADR-034):\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("docs/REPRODUCIBILITY.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert result == []


# ---- validate_citations: edge cases ----


def test_validate_citations_unknown_adr_id_skipped() -> None:
    """Citation to an ADR not in `adr_subjects` is skipped (out of scope)."""
    text = "Some claim per ADR-999.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,  # 999 not present
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert result == []


def test_validate_citations_no_claim_category_skipped() -> None:
    """If surrounding context has no category keywords, no misalignment can be inferred."""
    text = "Generic prose with no category keywords per ADR-029.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    # ADR-029 is test_markers; context has no marker/reproduc/cost/calibration
    # keywords → no inferred claim category → skip.
    assert result == []


def test_validate_citations_context_lines_window() -> None:
    """Context window picks up keywords on adjacent lines (±2 default)."""
    text = (
        "## Reproducibility section\n\nSome prose.\nMore prose per ADR-029.\n\n## Other section\n"
    )
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    # "Reproducibility" header is line 1; citation on line 4. With default
    # context_lines=2, the window includes lines 2-6, which doesn't include
    # the header. So this case should NOT flag — the keyword must be within
    # the window.
    # If context_lines were 3, it would; verify by passing 3.
    assert result == []

    result_wide = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        context_lines=3,
    )
    assert len(result_wide) == 1


def test_validate_citations_known_exempt_skipped() -> None:
    """Caller can declare specific (file, line, adr_id) tuples as known-exempt."""
    text = "Two-tier reproduction per ADR-029.\n"
    md_path = Path("docs/REPRODUCIBILITY.md")
    result_no_exempt = validate_citations(
        markdown_text=text,
        markdown_path=md_path,
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert len(result_no_exempt) == 1

    result_exempt = validate_citations(
        markdown_text=text,
        markdown_path=md_path,
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        known_exempt_citations=[(md_path, 1, "029")],
    )
    assert result_exempt == []


def test_validate_citations_multiple_citations_on_one_line() -> None:
    """A line with two citations is processed correctly (both checked)."""
    text = "Two-tier reproduction per ADR-029 + cost cap per ADR-020.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    # ADR-029 (test_markers) cited in reproduc context → flagged
    # ADR-020 (cost) cited in reproduc context → also flagged because
    # reproducibility wins as the first-match keyword on the line.
    assert len(result) >= 1
    flagged_adrs = {m.cited_adr_id for m in result}
    assert "029" in flagged_adrs


def test_validate_citations_case_insensitive_citation_phrase() -> None:
    """`Per`, `Via`, `By`, `Under` all match (default pattern uses (?i))."""
    text = "Two-tier reproduction Via ADR-029.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert len(result) == 1
    assert result[0].cited_adr_id == "029"


def test_citation_misalignment_is_frozen_dataclass() -> None:
    """CitationMisalignment instances are immutable (hashable + safe to share)."""
    from dataclasses import FrozenInstanceError

    m = CitationMisalignment(
        file=Path("test.md"),
        line=1,
        cited_adr_id="029",
        surrounding_text="ctx",
        claim_category="reproducibility",
        adr_actual_category="test_markers",
    )
    with pytest.raises(FrozenInstanceError):
        m.line = 2  # type: ignore[misc]
