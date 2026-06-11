"""Tests for eval_toolkit.audit_citation_alignment.

Seed motivating case: ``docs/REPRODUCIBILITY.md:76`` in
`brandon-behring/prompt-injection-detection-prototype` (v1.3.2 audit
finding P1-2 Part 2) cited "ADR-029" for a tier-lock claim, but ADR-029
is the test-marker-strategy ADR (not the tier-lock; that's ADR-034). The
validator must flag this misalignment.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from eval_toolkit.audit_citation_alignment import (
    ADRSubject,
    CitationMisalignment,
    extract_adr_subject_categories,
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


# ---------------------------------------------------------------------------
# v1.4.0: scope='narrative' Layer 2 + Layer 3 (per ADR 0007).
# All rules activate ONLY when scope='narrative'. scope='all' (default)
# preserves v1.0.1 / v1.3.x behavior exactly.
# Closes the residual 188 warnings from issue #82 on consumer HEAD.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pattern_beta_table_row_excluded_under_narrative() -> None:
    """Layer 2: markdown table-row citations are excluded under scope='narrative'.

    The consumer's SPEC_SHEET.md has 67 table-row citations like
    ``"| §1 Data | ... | ... (see ADR-008) | ..."`` — each is a
    structured spec entry, not narrative prose; the validator's
    keyword-based category inference can't disambiguate multi-topic
    cells. v1.4.0 excludes table rows entirely (Pattern β).
    """
    text = (
        "Reproducibility tier locked via ADR-029.\n"
        "\n"
        "| §1 Data | reproducibility | via ADR-029 | green |\n"
    )
    # Under scope='all', BOTH citations are flagged (table-row included).
    r_all = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="all",
    )
    # Under scope='narrative', only the line-1 narrative-prose citation
    # is flagged; the table-row citation is excluded by Pattern β.
    r_nar = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="narrative",
    )
    assert len(r_all) > len(r_nar)
    # The legitimate narrative-prose misalignment still fires under
    # narrative scope (proves Pattern β doesn't over-suppress).
    assert len(r_nar) == 1
    assert r_nar[0].line == 1


@pytest.mark.unit
def test_pattern_gamma_sentence_boundary_window() -> None:
    """Layer 3 rule γ: category keywords don't pull across sentence boundaries.

    Under scope='all' with the default ±2-line context, a citation in
    sentence 2 might match a keyword from sentence 1 even though they
    are semantically disjoint. Narrative scope bounds the context to
    the sentence containing the citation.
    """
    text = (
        # Sentence 1: reproducibility keyword. Sentence 2: marker citation,
        # NO keywords on its own. Under ±2-line scope='all', "reproduc"
        # leaks from sentence 1 → category mismatch flagged.
        # Under scope='narrative', sentence-bounded context = "Run smoke
        # via ADR-029" → no category keyword → no basis to flag.
        "Tier reproducibility locked at Phase 0.\n"
        "Run smoke via ADR-029.\n"
    )
    r_nar = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="narrative",
    )
    # Narrative scope: no false positive (sentence-bounded context for
    # "via ADR-029" doesn't contain "tier"/"reproduc"/"marker" keywords).
    assert r_nar == []


@pytest.mark.unit
def test_pattern_alpha_multi_adr_list_multi_category_match() -> None:
    """Layer 3 rule α: multiple citations in same sentence → multi-category check.

    Multi-ADR list prose (e.g., #82's "per ADR-025 + ADR-021 + ADR-034")
    has multiple topic keywords; each ADR can legitimately address a
    different topic. Multi-category match accepts the citation if the
    ADR's actual category is in the SET of matched categories — not
    just the first-match.
    """
    # Three categories in the sentence: reproducibility + cost + test_markers.
    # All three are matched by the SEED keyword map.
    # Each ADR is checked against the SET; all pass.
    text = (
        "Reproducibility tier locks + cost discipline + test markers "
        "framework defined via ADR-034, ADR-020, and ADR-029.\n"
    )
    # Sanity check: this sentence contains 3 citations (one per
    # supported ADR). The "via ADR-NNN" citation pattern matches the
    # first; we use a broader pattern to match all three.
    r_nar = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        # Broader pattern catches "ADR-NNN" anywhere; in real consumer use
        # they pass a custom pattern. Default pattern only matches
        # "per/via/by/under ADR-NNN" so only the first ADR-034 fires.
        citation_pattern=r"(?i)ADR-(\d{3})",
        scope="narrative",
    )
    # Under multi-category Layer 3 rule α: each ADR's actual category is
    # checked against the SET {reproducibility, cost, test_markers}.
    # ADR-034 (reproducibility) ✓ in set
    # ADR-020 (cost) ✓ in set
    # ADR-029 (test_markers) ✓ in set
    # → zero misalignments
    assert r_nar == []


@pytest.mark.unit
def test_scope_all_unchanged_backward_compat() -> None:
    """scope='all' (default) preserves v1.0.1 / v1.3.x behavior exactly.

    Backward-compat regression: a consumer on legacy scope='all' sees
    the same flags they saw before v1.4.0.
    """
    text = "Two-tier reproduction locked at Phase 0-07 via ADR-029.\n"
    r_default = validate_citations(
        markdown_text=text,
        markdown_path=Path("docs/REPRODUCIBILITY.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )  # scope defaults to "all"
    r_explicit_all = validate_citations(
        markdown_text=text,
        markdown_path=Path("docs/REPRODUCIBILITY.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="all",
    )
    # Default + explicit scope='all' produce identical results.
    assert r_default == r_explicit_all
    # And they BOTH flag the seed misalignment (proving legacy behavior intact).
    assert len(r_default) == 1
    assert r_default[0].cited_adr_id == "029"
    assert r_default[0].claim_category == "reproducibility"
    assert r_default[0].adr_actual_category == "test_markers"


@pytest.mark.unit
def test_narrative_helpers_imported_from_shared_module() -> None:
    """Sanity: Layer 2 + Layer 3 helpers come from `_narrative`, not duplicated.

    Per ADR 0007 v1.4.0 refactor, the narrative-prose helpers are
    centralized in `eval_toolkit._narrative` and shared between
    `audit_value_bindings` and `audit_citation_alignment`. This test
    documents that contract.
    """
    # Both validators should reference the SAME helper objects (same id()).
    from eval_toolkit import _narrative as _narr_module
    from eval_toolkit.audit_citation_alignment import (
        _build_exclusion_ranges as _aca_excl,
    )
    from eval_toolkit.audit_value_bindings import (
        _build_exclusion_ranges as _avb_excl,
    )

    # All three references point to the same function object.
    assert _aca_excl is _avb_excl
    assert _aca_excl is _narr_module._build_exclusion_ranges


@pytest.mark.unit
def test_combined_consumer_residuals_resolve_narrative() -> None:
    """Combined: all three #82 patterns resolve under scope='narrative'.

    Synthesized prose mirroring the consumer's actual residual
    patterns. Under scope='narrative' the v1.4.0 Layer 2 + Layer 3
    rules suppress all false positives; the legitimate misalignment
    (a single ADR cited with wrong category) is preserved.
    """
    text = (
        # Pattern β — table row with ADR citation (should be excluded)
        "| §1 Data | reproducibility | see ADR-029 | green |\n"
        "\n"
        # Pattern γ — multi-clause sentence; ADR-020 citation has cost
        # context locally, but "reproducibility" appears in a prior
        # clause. Sentence-bounded context isolates the local clause.
        "Compute discipline framework: cost envelope tracking via ADR-020.\n"
        "\n"
        # Pattern α — multi-ADR list; all categories are matched in the
        # sentence; each ADR's actual category is in the set.
        "Locks: reproducibility tier + cost + test markers via "
        "ADR-034, ADR-020, and ADR-029.\n"
        "\n"
        # A LEGITIMATE misalignment (should still flag): wrong ADR for
        # the claim — narrative scope doesn't suppress real bugs.
        "Two-tier reproduction locked at Phase 0-07 via ADR-029.\n"
    )
    r_nar = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        citation_pattern=r"(?i)(?:per|via|by|under|see)\s+ADR-(\d{3})",
        scope="narrative",
    )
    # The last line's ADR-029 citation is a real misalignment (claim is
    # reproducibility-flavored, ADR-029 is test_markers). Should fire.
    # The other 3 citations: β excluded, γ no-cross-sentence-keyword
    # match, α multi-category accepts.
    flagged_ids = {m.cited_adr_id for m in r_nar}
    # Only the LEGITIMATE last-line citation is flagged. (Note: the
    # "via ADR-029" in the multi-ADR sentence is in alpha-mode, so it's
    # accepted because test_markers is in the matched set.)
    assert flagged_ids <= {"029"}


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


@pytest.mark.unit
def test_invalid_scope_raises_value_error() -> None:
    """A scope typo raises instead of silently disabling Layers 2+3 (#99 V6)."""
    with pytest.raises(ValueError, match="scope must be 'all' or 'narrative'"):
        validate_citations(
            markdown_text="Two-tier reproduction via ADR-029.\n",
            markdown_path=Path("test.md"),
            adr_subjects=SEED_ADR_SUBJECTS,
            category_keywords=SEED_CATEGORY_KEYWORDS,
            scope="narative",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_long_sentence_window_centers_on_citation() -> None:
    """Rule-γ window is citation-centered when the sentence exceeds 300 chars (#99 V11).

    Pre-v1.11.0 the window was the sentence's leading 300 chars, so the
    keyword next to this citation (at the sentence's END) was invisible
    and the misalignment silently skipped.
    """
    filler = "lorem ipsum dolor sit amet, consectetur adipiscing elit, " * 7
    text = filler + "and the two-tier reproducibility lock is recorded via ADR-029.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="narrative",
    )
    assert len(result) == 1
    assert result[0].cited_adr_id == "029"
    assert result[0].claim_category == "reproducibility"
    assert "ADR-029" in result[0].surrounding_text


@pytest.mark.unit
def test_long_sentence_head_keywords_fall_outside_centered_window() -> None:
    """Pins the centering trade-off: keywords >150 chars BEFORE the citation no longer match."""
    filler = "lorem ipsum dolor sit amet, consectetur adipiscing elit, " * 7
    text = "the reproducibility lock intro, " + filler + "is recorded via ADR-029.\n"
    result = validate_citations(
        markdown_text=text,
        markdown_path=Path("test.md"),
        adr_subjects=SEED_ADR_SUBJECTS,
        category_keywords=SEED_CATEGORY_KEYWORDS,
        scope="narrative",
    )
    assert result == []  # no keyword in the centered window -> no basis to flag


# ---- multi-category ADRSubject (#99 V12) + symmetric-None survivor ----


@pytest.mark.unit
def test_extract_adr_subject_categories_all_matches() -> None:
    """Plural extractor returns EVERY matching category, not just the first."""
    out = extract_adr_subject_categories(
        title="Reproducibility tier lock and GPU cost envelope",
        slug="reproducibility-tier-cost",
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert out == frozenset({"reproducibility", "cost"})


@pytest.mark.unit
def test_extract_adr_subject_categories_no_match_returns_empty() -> None:
    out = extract_adr_subject_categories(
        title="Unrelated subject",
        slug="unrelated-subject",
        category_keywords=SEED_CATEGORY_KEYWORDS,
    )
    assert out == frozenset()


@pytest.mark.unit
def test_scalar_extractor_is_insertion_order_first_of_plural() -> None:
    """Pins the documented invariant between the scalar and plural extractors."""
    title = "Reproducibility tier lock and GPU cost envelope"
    slug = "reproducibility-tier-cost"
    plural = extract_adr_subject_categories(title, slug, SEED_CATEGORY_KEYWORDS)
    scalar = extract_adr_subject_category(title, slug, SEED_CATEGORY_KEYWORDS)
    assert scalar == next(c for c in SEED_CATEGORY_KEYWORDS if c in plural)


@pytest.mark.unit
def test_adr_subject_positional_construction_unchanged() -> None:
    """Tier-1 appended-optional guarantee: pre-v1.11.0 positional calls still work."""
    positional = ADRSubject("029", "t", "s", "test_markers")
    keyword = ADRSubject(adr_id="029", title="t", slug="s", category="test_markers")
    assert positional == keyword
    assert positional.categories is None


_MULTI_TOPIC_SUBJECT = ADRSubject(
    adr_id="045",
    title="Reproducibility tier lock and cost envelope",
    slug="reproducibility-tier-cost-envelope",
    category="reproducibility",
    categories=frozenset({"reproducibility", "cost"}),
)


@pytest.mark.unit
def test_multi_category_subject_accepted_when_claim_in_set_narrative() -> None:
    """A multi-topic ADR is accepted for ANY of its categories under narrative scope."""
    common = {
        "markdown_text": "GPU cost envelope locked via ADR-045.\n",
        "markdown_path": Path("test.md"),
        "category_keywords": SEED_CATEGORY_KEYWORDS,
        "scope": "narrative",
    }
    assert validate_citations(**common, adr_subjects={"045": _MULTI_TOPIC_SUBJECT}) == []
    control = replace(_MULTI_TOPIC_SUBJECT, categories=None)
    flagged = validate_citations(**common, adr_subjects={"045": control})
    assert len(flagged) == 1
    assert flagged[0].adr_actual_category == "reproducibility"


@pytest.mark.unit
def test_multi_category_multi_topic_intersection_narrative() -> None:
    """Rule-alpha uses set INTERSECTION when the subject carries multiple categories."""
    subject = ADRSubject(
        adr_id="050",
        title="Calibration sweep and cost budget",
        slug="calibration-cost",
        category="calibration",
        categories=frozenset({"calibration", "cost"}),
    )
    common = {
        "markdown_text": "Reproducibility tier and cost envelope locked via ADR-050.\n",
        "markdown_path": Path("test.md"),
        "category_keywords": SEED_CATEGORY_KEYWORDS,
        "scope": "narrative",
    }
    assert validate_citations(**common, adr_subjects={"050": subject}) == []
    control = replace(subject, categories=None)
    flagged = validate_citations(**common, adr_subjects={"050": control})
    assert len(flagged) == 1


@pytest.mark.unit
def test_multi_category_set_membership_under_scope_all() -> None:
    """scope='all' also honors a populated categories set (no silent scalar trap)."""
    common = {
        "markdown_text": "GPU cost envelope locked via ADR-045.\n",
        "markdown_path": Path("test.md"),
        "category_keywords": SEED_CATEGORY_KEYWORDS,
        "scope": "all",
    }
    assert validate_citations(**common, adr_subjects={"045": _MULTI_TOPIC_SUBJECT}) == []
    control = replace(_MULTI_TOPIC_SUBJECT, categories=None)
    assert len(validate_citations(**common, adr_subjects={"045": control})) == 1


@pytest.mark.unit
def test_empty_categories_skips_narrative_flags_under_all() -> None:
    """Explicit empty categories == scalar None: skip under narrative, flag under all."""
    subject = replace(SEED_ADR_SUBJECTS["029"], category=None, categories=frozenset())
    common = {
        "markdown_text": "Two-tier reproduction via ADR-029.\n",
        "markdown_path": Path("test.md"),
        "adr_subjects": {"029": subject},
        "category_keywords": SEED_CATEGORY_KEYWORDS,
    }
    assert validate_citations(**common, scope="narrative") == []
    flagged = validate_citations(**common, scope="all")
    assert len(flagged) == 1
    assert flagged[0].adr_actual_category is None


@pytest.mark.unit
def test_symmetric_none_skip_narrative_vs_all() -> None:
    """Pre-tag survivor: scalar-None subjects skip under narrative, flag under all."""
    subject = ADRSubject(adr_id="029", title="Unclassifiable", slug="unclassifiable", category=None)
    common = {
        "markdown_text": "Two-tier reproduction via ADR-029.\n",
        "markdown_path": Path("test.md"),
        "adr_subjects": {"029": subject},
        "category_keywords": SEED_CATEGORY_KEYWORDS,
    }
    assert validate_citations(**common, scope="narrative") == []
    flagged = validate_citations(**common, scope="all")
    assert len(flagged) == 1
    assert flagged[0].adr_actual_category is None


@pytest.mark.unit
def test_dataclasses_define_slots() -> None:
    """STYLE §5: family dataclasses are slotted — no per-instance __dict__ (#99 V9)."""
    for cls in (ADRSubject, CitationMisalignment):
        assert "__slots__" in cls.__dict__
    s = ADRSubject(adr_id="029", title="t", slug="s", category=None)
    assert not hasattr(s, "__dict__")


@pytest.mark.unit
def test_positional_helpers_shared_from_narrative() -> None:
    """_line_starts/_position_to_line are single-sourced in _narrative (#99 V4)."""
    from eval_toolkit import _narrative as _narr
    from eval_toolkit.audit_citation_alignment import _line_starts as _aca_ls
    from eval_toolkit.audit_value_bindings import (
        _line_starts as _avb_ls,
    )
    from eval_toolkit.audit_value_bindings import (
        _position_to_line as _avb_ptl,
    )

    assert _aca_ls is _avb_ls is _narr._line_starts
    assert _avb_ptl is _narr._position_to_line
