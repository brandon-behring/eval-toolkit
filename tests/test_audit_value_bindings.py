"""Tests for eval_toolkit.audit_value_bindings.

Seed motivating case (consumer V1.3.1 audit-fix patch closure,
ADR-080 2026-05-22): WRITEUP_NARRATIVE.md:38 said
"The TF-IDF + logistic regression baseline reaches 0.974 AUPRC on
balanced direct-versus-benign validation." Canonical: TF-IDF direct
val AUPRC = 0.971; LoRA direct val AUPRC = 0.974. The validator must
flag this mis-binding.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from eval_toolkit.audit_value_bindings import (
    BindingKey,
    Match,
    ValueBindingsReport,
    Violation,
    _normalize_binding_key,
    validate_reader_value_bindings,
)

SEED_DETECTOR_ALIASES = {
    "tf-idf + lr": ["TF-IDF \\+ logistic regression", "TF-IDF \\+ LR", "TF-IDF"],
    "lora": ["LoRA"],
    "frozen probe": ["frozen probe", "frozen-probe"],
}

SEED_METRIC_ALIASES = {
    "direct_val_auprc": [
        "direct.*?AUPRC",
        "direct.*?benign.*?validation",
        "validation AUPRC",
    ],
    "pooled_ood_auprc": ["pooled OOD AUPRC", "OOD AUPRC"],
}

SEED_SLICE_ALIASES = {
    "direct_validation": [
        "direct.*?validation",
        "direct\\+benign validation",
        "balanced direct",
        "in-pool",
    ],
    "pooled_ood": [
        "pooled OOD",
        "pooled_ood",
        "cross-family",
    ],
}

SEED_BINDINGS = {
    ("tf-idf + lr", "direct_val_auprc"): 0.971,
    ("lora", "direct_val_auprc"): 0.974,
    ("frozen probe", "pooled_ood_auprc"): 0.364,
}

# 3-tuple sugar form (issue #80's proposed schema): (detector, metric, slice).
SEED_BINDINGS_3TUPLE = {
    ("tf-idf + lr", "direct_val_auprc", "direct_validation"): 0.971,
    ("tf-idf + lr", "direct_val_auprc", "pooled_ood"): 0.291,
    ("lora", "direct_val_auprc", "direct_validation"): 0.974,
}

# Canonical BindingKey form (recommended for new consumer code).
SEED_BINDINGS_STRUCTURED = {
    BindingKey("tf-idf + lr", "direct_val_auprc", "direct_validation"): 0.971,
    BindingKey("tf-idf + lr", "direct_val_auprc", "pooled_ood"): 0.291,
    BindingKey("lora", "direct_val_auprc", "direct_validation"): 0.974,
}


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_validate_seed_case_flags_misbinding(tmp_path: Path) -> None:
    """Verbatim WRITEUP_NARRATIVE-style prose with TF-IDF / 0.974 mis-binding.

    The motivating bug class from consumer v1.3.1 audit-fix patch. The
    validator must report 1 violation on (tf-idf + lr, direct_val_auprc)
    where prose says 0.974 but canonical is 0.971.
    """
    file = _write(
        tmp_path,
        "WRITEUP_NARRATIVE.md",
        (
            "The TF-IDF + logistic regression baseline reaches 0.974 AUPRC "
            "on balanced direct-versus-benign validation."
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings=SEED_BINDINGS,
        detector_aliases=SEED_DETECTOR_ALIASES,
        metric_aliases=SEED_METRIC_ALIASES,
    )
    assert isinstance(report, ValueBindingsReport)
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.detector == "tf-idf + lr"
    assert v.metric == "direct_val_auprc"
    assert v.found_value == 0.974
    assert v.expected_value == 0.971
    assert "TF-IDF" in v.surrounding_text or "tf-idf" in v.surrounding_text.lower()


@pytest.mark.unit
def test_validate_correct_binding_no_flag(tmp_path: Path) -> None:
    """Same prose shape with the canonical value → 0 violations + 1 match."""
    file = _write(
        tmp_path,
        "WRITEUP_OK.md",
        (
            "The TF-IDF + logistic regression baseline reaches 0.971 AUPRC "
            "on balanced direct-versus-benign validation."
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings=SEED_BINDINGS,
        detector_aliases=SEED_DETECTOR_ALIASES,
        metric_aliases=SEED_METRIC_ALIASES,
    )
    assert report.violations == ()
    assert len(report.matched) >= 1
    assert any(m.detector == "tf-idf + lr" and m.value == 0.971 for m in report.matched)


@pytest.mark.unit
def test_detector_alias_resolution(tmp_path: Path) -> None:
    """Different surface forms (TF-IDF / TfIdf / tfidf) all resolve to one canonical key."""
    file = _write(
        tmp_path,
        "MIX.md",
        (
            "Run 1: TfIdf scored 0.971 on direct validation AUPRC.\n"
            "Run 2: tfidf scored 0.971 on direct validation AUPRC.\n"
            "Run 3: TF-IDF + LR scored 0.971 on direct validation AUPRC.\n"
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TfIdf", "tfidf", "TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
    )
    assert report.violations == ()
    # All 3 surface forms should match the single canonical key.
    matched_lines = {m.line for m in report.matched}
    assert len(matched_lines) == 3


@pytest.mark.unit
def test_metric_alias_resolution(tmp_path: Path) -> None:
    """Different metric phrasings resolve to one canonical key."""
    file = _write(
        tmp_path,
        "METRICS.md",
        (
            "LoRA: 0.974 direct AUPRC.\n"
            "LoRA reported validation AUPRC of 0.974.\n"
            "LoRA hits 0.974 on direct-versus-benign validation.\n"
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("lora", "direct_val_auprc"): 0.974},
        detector_aliases={"lora": ["LoRA"]},
        metric_aliases={
            "direct_val_auprc": [
                "direct AUPRC",
                "validation AUPRC",
                "direct-versus-benign validation",
            ],
        },
    )
    assert report.violations == ()
    assert len(report.matched) >= 3


@pytest.mark.unit
def test_value_outside_distance_window_not_flagged(tmp_path: Path) -> None:
    """Value > max_distance_chars from detector mention is ignored."""
    # 200 chars of filler between detector and the value/metric.
    filler = "X" * 200
    file = _write(
        tmp_path,
        "FAR.md",
        f"TF-IDF detector. {filler} direct validation AUPRC = 0.974",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TF-IDF"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
        max_distance_chars=80,
    )
    # Detector + value are >200 chars apart; not paired.
    assert report.violations == ()
    assert report.matched == ()


@pytest.mark.unit
def test_detector_with_no_nearby_value_skipped(tmp_path: Path) -> None:
    """Detector mention without a nearby value is not a violation."""
    file = _write(
        tmp_path,
        "BARE.md",
        "The TF-IDF baseline performs well on the benchmark.",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TF-IDF"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
    )
    # No metric mention + no value → nothing emitted.
    assert report.violations == ()
    assert report.matched == ()


@pytest.mark.unit
def test_value_without_metric_in_window_skipped(tmp_path: Path) -> None:
    """A value near a detector but with no metric mention in the window is skipped."""
    file = _write(
        tmp_path,
        "VAL_ONLY.md",
        "TF-IDF baseline scored 0.971 on the test set.",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TF-IDF"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
    )
    # Metric "direct validation AUPRC" not present → no triple.
    assert report.violations == ()
    assert report.matched == ()


@pytest.mark.unit
def test_coverage_fraction(tmp_path: Path) -> None:
    """Coverage = fraction of bindings keys that produced at least one Match."""
    file = _write(
        tmp_path,
        "PARTIAL.md",
        (
            "TF-IDF + LR scored 0.971 on direct validation AUPRC.\n"
            "LoRA scored 0.974 on direct validation AUPRC.\n"
            # frozen probe not mentioned → coverage = 2/3
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings=SEED_BINDINGS,
        detector_aliases=SEED_DETECTOR_ALIASES,
        metric_aliases=SEED_METRIC_ALIASES,
    )
    assert report.violations == ()
    # 2 of 3 binding keys produced matches.
    assert report.coverage == pytest.approx(2.0 / 3.0)


@pytest.mark.unit
def test_tolerance_close_value_accepted(tmp_path: Path) -> None:
    """Values within tolerance of expected are matches, not violations."""
    file = _write(
        tmp_path,
        "TOL.md",
        "TF-IDF + LR scored 0.9710 on direct validation AUPRC.",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
        tolerance=0.001,
    )
    assert report.violations == ()
    assert len(report.matched) == 1


@pytest.mark.unit
def test_tolerance_close_value_below_tolerance_flagged(tmp_path: Path) -> None:
    """Values JUST outside tolerance are flagged."""
    file = _write(
        tmp_path,
        "TOL_BAD.md",
        "TF-IDF + LR scored 0.973 on direct validation AUPRC.",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971},
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
        tolerance=0.001,
    )
    assert len(report.violations) == 1
    assert report.violations[0].found_value == 0.973


@pytest.mark.unit
def test_multiple_detectors_in_paragraph(tmp_path: Path) -> None:
    """Each detector in the same paragraph gets its own triple."""
    file = _write(
        tmp_path,
        "BOTH.md",
        (
            "We compare two baselines on direct validation AUPRC: "
            "TF-IDF + LR achieves 0.971, while LoRA reaches 0.974."
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={
            ("tf-idf + lr", "direct_val_auprc"): 0.971,
            ("lora", "direct_val_auprc"): 0.974,
        },
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"], "lora": ["LoRA"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
    )
    assert report.violations == ()
    detector_keys = {m.detector for m in report.matched}
    assert detector_keys == {"tf-idf + lr", "lora"}


@pytest.mark.unit
def test_frozen_dataclass_invariants() -> None:
    """Match, Violation, ValueBindingsReport are immutable."""
    m = Match(file=Path("x"), line=1, detector="d", metric="me", value=0.5)
    v = Violation(
        file=Path("x"),
        line=1,
        detector="d",
        metric="me",
        found_value=0.5,
        expected_value=0.6,
        surrounding_text="ctx",
    )
    r = ValueBindingsReport(violations=(v,), matched=(m,), coverage=0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.value = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.found_value = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.coverage = 1.0  # type: ignore[misc]


@pytest.mark.unit
def test_empty_bindings_zero_coverage(tmp_path: Path) -> None:
    """Empty bindings → coverage=0.0, no violations, no matches."""
    file = _write(tmp_path, "ANY.md", "TF-IDF reaches 0.971 AUPRC")
    report = validate_reader_value_bindings(
        files=[file],
        bindings={},
    )
    assert report.violations == ()
    assert report.matched == ()
    assert report.coverage == 0.0
    assert report.unmatched_slice_count == 0


# ---------------------------------------------------------------------------
# v1.1.0: BindingKey + multi-shape input adapter + slice-aware matching
# Closes #80. Issue source: consumer reproducer at
# brandon-behring/prompt-injection-detection-prototype@v1.3.9 produced
# 96 warnings of which ~95 were false positives because
# (detector, metric) collapsed cross-slice values.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_binding_key_round_trip_bindingkey() -> None:
    """BindingKey input passes through normalization unchanged."""
    bk = BindingKey(detector="tf-idf + lr", metric="direct_val_auprc", slice="direct_validation")
    assert _normalize_binding_key(bk) is bk  # identity preserved for canonical input


@pytest.mark.unit
def test_normalize_binding_key_2tuple_defaults_to_slice_any() -> None:
    """Legacy 2-tuple → BindingKey with slice='any'."""
    bk = _normalize_binding_key(("tf-idf + lr", "direct_val_auprc"))
    assert bk == BindingKey("tf-idf + lr", "direct_val_auprc", "any")
    assert bk.slice == "any"


@pytest.mark.unit
def test_normalize_binding_key_3tuple_preserves_slice() -> None:
    """3-tuple sugar → BindingKey with the slice from position 3."""
    bk = _normalize_binding_key(("tf-idf + lr", "direct_val_auprc", "pooled_ood"))
    assert bk == BindingKey("tf-idf + lr", "direct_val_auprc", "pooled_ood")
    assert bk.slice == "pooled_ood"


@pytest.mark.unit
def test_normalize_binding_key_invalid_shape_raises_typeerror() -> None:
    """Malformed key shapes (4-tuple, set, string, ...) raise TypeError."""
    with pytest.raises(TypeError, match="bindings key must be BindingKey"):
        _normalize_binding_key(("a", "b", "c", "d"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bindings key must be BindingKey"):
        _normalize_binding_key("not-a-tuple")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bindings key must be BindingKey"):
        _normalize_binding_key({"a", "b"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bindings key must be BindingKey"):
        _normalize_binding_key(("only_one",))  # type: ignore[arg-type]


@pytest.mark.unit
def test_bindingkey_is_frozen_and_hashable() -> None:
    """BindingKey is frozen (immutable) and hashable (usable as dict key)."""
    bk = BindingKey("d", "m", "s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        bk.detector = "other"  # type: ignore[misc]
    # Usable as dict key:
    d = {bk: 0.5}
    assert d[BindingKey("d", "m", "s")] == 0.5  # equality / hash by value


@pytest.mark.unit
def test_bindingkey_input_equivalent_to_2tuple_for_slice_any(tmp_path: Path) -> None:
    """BindingKey(slice='any') input yields the same report as 2-tuple input."""
    file = _write(
        tmp_path,
        "EQUIV.md",
        "The TF-IDF + LR baseline scored 0.971 on direct validation AUPRC.",
    )
    common = {
        "files": [file],
        "detector_aliases": {"tf-idf + lr": ["TF-IDF \\+ LR"]},
        "metric_aliases": {"direct_val_auprc": ["direct validation AUPRC"]},
    }
    r_2tuple = validate_reader_value_bindings(
        bindings={("tf-idf + lr", "direct_val_auprc"): 0.971}, **common
    )
    r_bk = validate_reader_value_bindings(
        bindings={BindingKey("tf-idf + lr", "direct_val_auprc"): 0.971}, **common
    )
    assert r_2tuple.violations == r_bk.violations
    assert r_2tuple.coverage == r_bk.coverage
    assert len(r_2tuple.matched) == len(r_bk.matched) == 1


@pytest.mark.unit
def test_same_detector_metric_different_slices_no_cross_flag(tmp_path: Path) -> None:
    """Issue #80's core complaint: same (detector, metric) across slices must not cross-flag.

    Pre-v1.1, prose mentioning ``TF-IDF + LR ... 0.291 ... pooled OOD AUPRC``
    would have been flagged against ``(tf-idf + lr, direct_val_auprc): 0.971``
    because the 2-tuple key collapsed the slice axis. With BindingKey:
    the 0.291 mention is correctly associated with the ``pooled_ood`` slice
    and matched against ``BindingKey(..., slice="pooled_ood"): 0.291``.
    """
    file = _write(
        tmp_path,
        "RESULTS.md",
        (
            "On direct+benign validation, TF-IDF + LR scored 0.971 direct AUPRC.\n"
            "On pooled OOD, the same TF-IDF + LR model scored 0.291 direct AUPRC "
            "(pooled OOD setting).\n"
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings=SEED_BINDINGS_STRUCTURED,
        detector_aliases=SEED_DETECTOR_ALIASES,
        metric_aliases=SEED_METRIC_ALIASES,
        slice_aliases=SEED_SLICE_ALIASES,
    )
    # Both lines should match their respective slice-scoped bindings.
    # Zero violations: nothing cross-flagged.
    assert report.violations == ()
    matched_slices = {(m.detector, m.value) for m in report.matched}
    assert ("tf-idf + lr", 0.971) in matched_slices
    assert ("tf-idf + lr", 0.291) in matched_slices


@pytest.mark.unit
def test_3tuple_input_normalizes_and_matches(tmp_path: Path) -> None:
    """3-tuple sugar input is normalized to BindingKey and matched correctly."""
    file = _write(
        tmp_path,
        "SUGAR.md",
        "LoRA achieves 0.974 on direct validation AUPRC (direct+benign validation).",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={("lora", "direct_val_auprc", "direct_validation"): 0.974},
        detector_aliases={"lora": ["LoRA"]},
        metric_aliases={"direct_val_auprc": ["direct validation AUPRC"]},
        slice_aliases={"direct_validation": ["direct\\+benign validation"]},
    )
    assert report.violations == ()
    assert len(report.matched) == 1
    assert report.matched[0].value == 0.974


@pytest.mark.unit
def test_mixed_2tuple_and_3tuple_keys_in_same_bindings(tmp_path: Path) -> None:
    """Dict with mixed 2-tuple and 3-tuple keys is supported; each normalized per-key."""
    file = _write(
        tmp_path,
        "MIXED.md",
        (
            "The frozen probe baseline scored 0.364 on pooled OOD AUPRC.\n"
            "LoRA achieves 0.974 direct AUPRC on direct+benign validation.\n"
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={
            # legacy 2-tuple (slice="any" implied) — frozen probe binding:
            ("frozen probe", "pooled_ood_auprc"): 0.364,
            # 3-tuple sugar (slice-scoped) — LoRA direct validation:
            ("lora", "direct_val_auprc", "direct_validation"): 0.974,
        },
        detector_aliases={"frozen probe": ["frozen probe"], "lora": ["LoRA"]},
        metric_aliases={
            "pooled_ood_auprc": ["pooled OOD AUPRC"],
            "direct_val_auprc": ["direct AUPRC"],
        },
        slice_aliases={"direct_validation": ["direct\\+benign validation"]},
    )
    # Both bindings should match cleanly; no violations.
    assert report.violations == ()
    detectors = {m.detector for m in report.matched}
    assert detectors == {"frozen probe", "lora"}


@pytest.mark.unit
def test_unmatched_slice_increments_count_not_violations(tmp_path: Path) -> None:
    """Triple matches detector+metric+value but slice alias absent in window → suppressed.

    The triple is counted in ``unmatched_slice_count`` (warn-only signal)
    and NOT emitted as a violation. This is the v1.0.x false-positive
    failure mode that issue #80 fixes: ambiguous-slice prose should not
    pollute the violations list.
    """
    file = _write(
        tmp_path,
        "AMBIGUOUS.md",
        # Prose mentions the detector + metric + a value but does NOT name
        # the slice; the binding is slice-scoped, so the prose is ambiguous.
        "TF-IDF + LR achieved 0.291 on direct AUPRC in our evaluation.",
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings={
            ("tf-idf + lr", "direct_val_auprc", "pooled_ood"): 0.291,
        },
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct AUPRC"]},
        slice_aliases={"pooled_ood": ["pooled OOD", "cross-family"]},
    )
    assert report.violations == ()
    assert report.matched == ()
    assert report.unmatched_slice_count >= 1


@pytest.mark.unit
def test_narrative_scope_excludes_markdown_table_rows(tmp_path: Path) -> None:
    """scope='narrative' skips values inside markdown table rows.

    Issue: consumer writeups contain dense statistical tables with
    multiple metrics per row (`| LoRA | AUPRC 0.974, AUROC 0.993 |`).
    The validator's positional heuristics can't disambiguate these;
    scope='narrative' restricts matching to prose only.
    """
    file = _write(
        tmp_path,
        "TABLE.md",
        (
            "The TF-IDF + LR baseline reaches 0.971 AUPRC on direct validation.\n"
            "\n"
            "| Detector | AUPRC | AUROC |\n"
            "|---|---|---|\n"
            "| TF-IDF | 0.971 | 0.992 |\n"
            "| LoRA   | 0.974 | 0.993 |\n"
        ),
    )
    bindings = {("tf-idf + lr", "direct_val_auprc"): 0.971}
    aliases = {"tf-idf + lr": ["TF-IDF \\+ LR", "TF-IDF"]}
    metric_aliases = {"direct_val_auprc": ["direct validation", "AUPRC"]}

    r_all = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="all",
    )
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="narrative",
    )
    # 'all' picks up the 0.992 AUROC in the table row as a violation
    # (validator can't distinguish from AUPRC); 'narrative' filters it.
    assert len(r_all.violations) > len(r_nar.violations)
    # Narrative match should still find the prose 0.971 on line 1.
    assert any(m.value == 0.971 for m in r_nar.matched)


@pytest.mark.unit
def test_narrative_scope_excludes_bracketed_expressions(tmp_path: Path) -> None:
    """scope='narrative' skips values inside [...] (CI bounds, ranges)."""
    file = _write(
        tmp_path,
        "CI.md",
        # Same line: detector + metric + a point estimate AND a CI bound
        # in brackets. The CI bound should be excluded under narrative.
        "TF-IDF + LR scored 0.971 AUPRC on direct validation [CI 0.965, 0.978].",
    )
    bindings = {("tf-idf + lr", "direct_val_auprc"): 0.971}
    r = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation", "AUPRC"]},
        scope="narrative",
    )
    # Under narrative scope, only 0.971 (point estimate) matches.
    # 0.965 and 0.978 are inside [...] and excluded → not flagged.
    assert r.violations == ()
    matched_vals = {m.value for m in r.matched}
    assert 0.971 in matched_vals
    # The bracket-bound values 0.965 and 0.978 must NOT appear in
    # either violations or matches.
    assert 0.965 not in matched_vals
    assert 0.978 not in matched_vals


@pytest.mark.unit
def test_narrative_scope_excludes_fenced_code_blocks(tmp_path: Path) -> None:
    """scope='narrative' skips values inside ```...``` fenced code blocks."""
    file = _write(
        tmp_path,
        "CODE.md",
        (
            "TF-IDF + LR scored 0.971 AUPRC on direct validation.\n"
            "\n"
            "```python\n"
            "EXPECTED_AUPRC = 0.993  # not a narrative claim — code\n"
            "```\n"
        ),
    )
    bindings = {("tf-idf + lr", "direct_val_auprc"): 0.971}
    r = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation", "AUPRC"]},
        scope="narrative",
    )
    matched_vals = {m.value for m in r.matched}
    # Narrative match finds 0.971 in prose.
    assert 0.971 in matched_vals
    # 0.993 inside the code block is excluded → no violation, no match.
    assert 0.993 not in matched_vals
    assert all(v.found_value != 0.993 for v in r.violations)


@pytest.mark.unit
def test_narrative_scope_default_is_all_for_backward_compat(tmp_path: Path) -> None:
    """The default ``scope`` value is 'all' (legacy v1.0.x behavior)."""
    file = _write(
        tmp_path,
        "BACKWARD.md",
        "| TF-IDF | 0.974 |",  # table row with a value
    )
    bindings = {("tf-idf + lr", "direct_val_auprc"): 0.971}
    # Default call (scope omitted) matches v1.0.x behavior: should
    # find the table-cell value 0.974 and flag it against 0.971.
    r_default = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": ["TF-IDF"]},
        metric_aliases={"direct_val_auprc": ["AUPRC"]},
    )
    # The default behavior scans the table row. With no metric alias
    # matching in window, no violation is emitted either way; this
    # test asserts the scope parameter is not REQUIRED and that the
    # default value is the legacy-compatible "all".
    # (Functional difference is verified in the dedicated scope tests
    # above.)
    assert isinstance(r_default, ValueBindingsReport)


@pytest.mark.unit
def test_3tuple_keys_coverage_counts_each_slice_independently(tmp_path: Path) -> None:
    """Coverage denominator includes each (detector, metric, slice) BindingKey separately.

    Prior to v1.1, two slices for the same (detector, metric) collapsed
    to one coverage slot. With BindingKey, each slice variant is its
    own coverage unit — matching the canonical-binding semantics.
    """
    file = _write(
        tmp_path,
        "COV.md",
        (
            "On direct+benign validation, TF-IDF + LR scored 0.971 direct AUPRC.\n"
            "On pooled OOD, TF-IDF + LR scored 0.291 direct AUPRC.\n"
            # LoRA direct_validation binding present but NOT cited → covers 2/3.
        ),
    )
    report = validate_reader_value_bindings(
        files=[file],
        bindings=SEED_BINDINGS_STRUCTURED,
        detector_aliases=SEED_DETECTOR_ALIASES,
        metric_aliases=SEED_METRIC_ALIASES,
        slice_aliases=SEED_SLICE_ALIASES,
    )
    assert report.violations == ()
    # 2 of 3 binding keys produced matches → coverage = 2/3.
    assert report.coverage == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# v1.2.0: context-aware narrative filters (T1, T2, T3, T4).
# All four activate ONLY when scope="narrative". scope="all" preserves
# legacy v1.1.0 behavior (proves backward compat).
# Closes the residual 36 from the consumer's v1.3.11 dogfood per the
# pending Round 13 ledger. ADR 0005 §4 governs the design.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t1_delta_filter_suppresses_signed_values_in_narrative(tmp_path: Path) -> None:
    """T1 (narrative): values immediately preceded by + or - are deltas, not bindings."""
    file = _write(
        tmp_path,
        "DELTA_SIGN.md",
        # Three delta-magnitude mentions; all should be suppressed under narrative.
        "LoRA drops -0.071 AUPRC below frozen probe; gap is +0.073 AUPRC over tfidf-lr.",
    )
    bindings = {("lora", "direct_val_auprc"): 0.293}
    aliases = {"lora": ["LoRA"]}
    metric_aliases = {"direct_val_auprc": ["AUPRC"]}
    r_all = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="all",
    )
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="narrative",
    )
    # Legacy 'all' flags the signed deltas; narrative scope suppresses them.
    assert len(r_all.violations) >= 1
    assert r_nar.violations == ()


@pytest.mark.unit
def test_t1_delta_keyword_filter_suppresses_comparative_magnitudes(tmp_path: Path) -> None:
    """T1 (narrative): values near delta-keyword tokens (delta/drop/below/vs) are suppressed."""
    file = _write(
        tmp_path,
        "DELTA_KW.md",
        # "delta" keyword nearby; value is the magnitude of a comparison.
        "Frozen probe's pooled OOD AUROC delta is 0.132 relative to LoRA.",
    )
    bindings = {("frozen probe", "pooled_ood_auroc"): 0.515}
    aliases = {"frozen probe": ["frozen probe"]}
    metric_aliases = {"pooled_ood_auroc": ["AUROC"]}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="narrative",
    )
    # The 0.132 value should NOT be flagged as a (frozen probe, AUROC) violation
    # — the "delta" keyword nearby identifies it as a comparative magnitude.
    assert r_nar.violations == ()


@pytest.mark.unit
def test_t2_floor_filter_suppresses_random_baseline_values(tmp_path: Path) -> None:
    """T2 (narrative): values near floor-keyword tokens (random/floor/chance) are suppressed."""
    file = _write(
        tmp_path,
        "FLOOR.md",
        # The 0.374 IS the random floor, not a frozen probe binding claim.
        "For pooled OOD, the random AUPRC floor is 0.374. The frozen probe scored above it.",
    )
    bindings = {("frozen probe", "pooled_ood_auprc"): 0.364}
    aliases = {"frozen probe": ["frozen probe"]}
    metric_aliases = {"pooled_ood_auprc": ["AUPRC"]}
    r_all = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="all",
    )
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=aliases,
        metric_aliases=metric_aliases,
        scope="narrative",
    )
    # 'all' flags 0.374 against the 0.364 binding (false positive).
    # narrative suppresses because "random" and "floor" are within window.
    assert any(v.found_value == 0.374 for v in r_all.violations)
    assert r_nar.violations == ()


@pytest.mark.unit
def test_t3_consume_on_match_suppresses_duplicate_in_same_sentence(tmp_path: Path) -> None:
    """T3 (narrative): once a binding matches in a sentence, subsequent same-binding values are skipped.

    Catches dense multi-detector enumerations like "ProtectAI v2 improves
    over v1 on JBB: AUPRC 0.556 vs 0.519." — the 0.519 is implicitly v1's
    value, not v2's, but T3 only needs to suppress the duplicate-binding
    flag (cross-detector inference is out of scope per ADR 0005).
    """
    file = _write(
        tmp_path,
        "VS.md",
        # First value matches v2; second value (after "vs") is implicitly v1.
        "ProtectAI v2 improves over v1 on JBB-Behaviors: AUPRC 0.556 vs 0.519.",
    )
    bindings = {
        BindingKey("ProtectAI-v2", "AUPRC", "jbb"): 0.556,
    }
    detector_aliases = {"ProtectAI-v2": [r"ProtectAI v2"]}
    metric_aliases = {"AUPRC": [r"AUPRC"]}
    slice_aliases = {"jbb": [r"JBB-Behaviors", r"JBB"]}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=detector_aliases,
        metric_aliases=metric_aliases,
        slice_aliases=slice_aliases,
        scope="narrative",
    )
    # T3 suppresses the second-value flag (0.519 != 0.556) once 0.556 matches.
    assert len(r_nar.matched) == 1
    assert r_nar.matched[0].value == 0.556
    assert r_nar.violations == ()


@pytest.mark.unit
def test_t4_sentence_boundary_rejects_cross_sentence_pairing(tmp_path: Path) -> None:
    """T4 (narrative): detector and value in different sentences should not pair.

    Catches the consumer's "X scored 0.291. The random floor is 0.374"
    pattern where 0.374 was being paired with X across the `.` boundary.
    """
    file = _write(
        tmp_path,
        "CROSS.md",
        # First sentence: frozen probe with its own value (matches).
        # Second sentence: 0.374 is the floor value, NOT a frozen-probe binding.
        # T4 rejects the cross-sentence detector pairing for 0.374.
        # T2 would also catch it via "floor" keyword — this test asserts T4
        # is the relevant filter via a prose that doesn't trip T2.
        "Frozen probe achieved AUPRC 0.364 on the eval. Under direct val, LoRA reached 0.974 AUPRC.",
    )
    bindings = {
        ("frozen probe", "direct_val_auprc"): 0.653,
    }
    detector_aliases = {"frozen probe": ["frozen probe"]}
    metric_aliases = {"direct_val_auprc": ["AUPRC", "direct val"]}
    r_all = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=detector_aliases,
        metric_aliases=metric_aliases,
        scope="all",
    )
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=detector_aliases,
        metric_aliases=metric_aliases,
        scope="narrative",
    )
    # 'all' pairs 0.974 (in sentence 2) with "frozen probe" (in sentence 1)
    # — only detector aliased — and flags it as a violation (expected
    # 0.653, found 0.974). narrative scope T4 rejects the
    # cross-sentence pair entirely.
    assert any(v.found_value == 0.974 for v in r_all.violations)
    assert not any(v.found_value == 0.974 for v in r_nar.violations)


@pytest.mark.unit
def test_t4_same_sentence_still_matches(tmp_path: Path) -> None:
    """T4 regression: detector and value in the SAME sentence must still match.

    Asserts T4 doesn't over-reject when the legitimate detector mention
    is in the same sentence as the value.
    """
    file = _write(
        tmp_path,
        "SAME.md",
        "The TF-IDF + LR baseline reaches 0.971 AUPRC on direct validation.",
    )
    bindings = {("tf-idf + lr", "direct_val_auprc"): 0.971}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": ["TF-IDF \\+ LR"]},
        metric_aliases={"direct_val_auprc": ["direct validation", "AUPRC"]},
        scope="narrative",
    )
    assert len(r_nar.matched) == 1
    assert r_nar.matched[0].value == 0.971
    assert r_nar.violations == ()


@pytest.mark.unit
def test_narrative_combined_filters_dogfood_pattern(tmp_path: Path) -> None:
    """Combined: a prose pattern from the consumer's v1.3.11 dogfood.

    Real-world snippet (paraphrased from RESULTS.md:171) that exercises
    multiple filters: cross-sentence pairing, delta language, paired-CI
    brackets. Under narrative, none of these should produce a violation
    against the canonical (frozen probe, AUROC, pooled_ood) = 0.515
    binding.
    """
    file = _write(
        tmp_path,
        "DOGFOOD.md",
        # Multi-clause sentence with delta, CI bracket, and cross-sentence values.
        # The 0.132 is a delta magnitude (T1 via "delta" keyword);
        # the 0.286 / 0.301 are inside brackets (v1.1.0 scope filter);
        # the 0.515 IS the legitimate match.
        "LoRA's pooled OOD AUROC is 0.383 against frozen probe's 0.515 (delta -0.132; CI [0.286, 0.301]).",
    )
    bindings = {
        BindingKey("frozen_probe", "AUROC", "pooled_ood"): 0.515,
        BindingKey("LoRA", "AUROC", "pooled_ood"): 0.383,
    }
    detector_aliases = {
        "frozen_probe": ["frozen probe"],
        "LoRA": ["LoRA"],
    }
    metric_aliases = {"AUROC": ["AUROC"]}
    slice_aliases = {"pooled_ood": [r"pooled OOD"]}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases=detector_aliases,
        metric_aliases=metric_aliases,
        slice_aliases=slice_aliases,
        scope="narrative",
    )
    # Expect 2 matches (0.515 and 0.383), 0 violations.
    matched_values = {m.value for m in r_nar.matched}
    assert 0.515 in matched_values
    assert 0.383 in matched_values
    assert r_nar.violations == ()


@pytest.mark.unit
def test_sentence_boundary_helper_respects_abbreviations(tmp_path: Path) -> None:
    """Direct unit test for the sentence-boundary helper's abbreviation guard.

    v1.4.0 (ADR 0007): helpers live in eval_toolkit._narrative (private
    flat module); the import path moved during the refactor.
    """
    from eval_toolkit._narrative import (
        _is_sentence_terminator_dot,
        _sentence_boundary_positions,
    )

    # "e.g." and "i.e." should NOT split sentences.
    text = "Consider X, e.g. its variant. Now Y."
    positions = _sentence_boundary_positions(text)
    # Expected: 2 sentences: "Consider X, e.g. its variant." and "Now Y."
    # So positions = [0, 30] (start of "Now")
    assert len(positions) == 2
    # The first sentence INCLUDES "e.g." and ends at "variant."
    # Verify by checking _is_sentence_terminator_dot at the dot of "e.g."
    eg_dot_pos = text.find("e.g.") + 1  # the dot after "e"
    assert not _is_sentence_terminator_dot(text, eg_dot_pos)
    # Decimal numbers also don't split
    text2 = "Score 0.5 is the floor. Under it lies LoRA."
    pos2 = _sentence_boundary_positions(text2)
    assert len(pos2) == 2  # "Score 0.5 is the floor." and "Under it..."


# ---------------------------------------------------------------------------
# v1.3.0: Layer 3 pairing rules per ADR 0006 (Patterns A/B/C/D).
# Closes issue #81 — cross-detector list-grammar + metric-axis
# nearest-pairing. All four rules activate ONLY when scope="narrative".
# Consumer dogfood acceptance: 4 → 0 warnings on
# prompt-injection-detection-submission HEAD.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pattern_a_for_postfix_overrides_proximity(tmp_path: Path) -> None:
    """Pattern A: 'for {detector}' postfix re-pairs value to that detector.

    Catches consumer's WRITEUP_PAPER.md:304 pattern: 'versus 0.364 [...]
    for the frozen probe and 0.291 [...] for TF-IDF + LR' — under
    proximity-only pairing, 0.291 would attribute to frozen probe
    (nearer prior mention), but the 'for TF-IDF + LR' postfix is
    authoritative. Synthetic prose mirrors the consumer's structure:
    the postfix is the dominant binding signal; no delta keywords
    immediately adjacent to the value (which would otherwise be
    handled by T1 or T3).
    """
    file = _write(
        tmp_path,
        "POSTFIX.md",
        # Two values introduced by parallel postfix constructs. Under
        # proximity-only pairing, 0.293 attributes to TF-IDF (the
        # earlier detector mention). Pattern A's postfix override
        # re-pairs it to LoRA via "for the LoRA model".
        "Pooled OOD AUPRC: 0.364 for the TF-IDF baseline and 0.293 for the LoRA model.",
    )
    bindings = {
        BindingKey("tf-idf + lr", "direct_val_auprc", "any"): 0.364,
        BindingKey("lora", "direct_val_auprc", "any"): 0.293,
    }
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": [r"TF-IDF"], "lora": [r"LoRA"]},
        metric_aliases={"direct_val_auprc": [r"AUPRC"]},
        scope="narrative",
    )
    # Both values match their correct detectors via Pattern A override.
    assert r_nar.violations == ()
    matched_by_detector = {(m.detector, m.value) for m in r_nar.matched}
    assert ("tf-idf + lr", 0.364) in matched_by_detector
    assert ("lora", 0.293) in matched_by_detector


@pytest.mark.unit
def test_pattern_b_possessive_overrides_proximity(tmp_path: Path) -> None:
    """Pattern B: "{detector}'s" possessive re-pairs value to possessor.

    Catches consumer's RESULTS.md:171 pattern: "LoRA's pooled OOD AUROC
    is 0.383 against frozen probe's 0.515" — both possessives
    immediately precede their values; without Pattern B, the proximity
    rule would mis-attribute under cross-detector confusion.
    """
    file = _write(
        tmp_path,
        "POSSESSIVE.md",
        "LoRA's pooled OOD AUROC is 0.383 against frozen probe's 0.515.",
    )
    bindings = {
        BindingKey("LoRA", "AUROC", "pooled_ood"): 0.383,
        BindingKey("frozen_probe", "AUROC", "pooled_ood"): 0.515,
    }
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"LoRA": [r"LoRA"], "frozen_probe": [r"frozen probe"]},
        metric_aliases={"AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD"]},
        scope="narrative",
    )
    assert r_nar.violations == ()
    matched_by_detector = {(m.detector, m.value) for m in r_nar.matched}
    assert ("LoRA", 0.383) in matched_by_detector
    assert ("frozen_probe", 0.515) in matched_by_detector


@pytest.mark.unit
def test_pattern_c_group_subject_suppresses(tmp_path: Path) -> None:
    """Pattern C: "for the {trained|frozen|...} detectors" suppresses the value.

    Catches consumer's README.md:71 pattern: '0.38 AUROC for the trained
    detectors' — the value belongs to a multi-detector group, not a
    single canonical binding. v1.3.0 suppresses the candidate (v1.4.0+
    may attempt multi-detector inference per ADR 0006).
    """
    file = _write(
        tmp_path,
        "GROUP.md",
        # 0.38 follows "for the trained detectors" group-subject phrase.
        # The frozen probe binding (0.515) shouldn't be flagged against
        # 0.38 even though frozen probe is mentioned nearby.
        "Generalization gap: 0.38 AUROC for the trained detectors; frozen probe sits higher.",
    )
    bindings = {BindingKey("frozen_probe", "AUROC", "pooled_ood"): 0.515}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"frozen_probe": [r"frozen probe"]},
        metric_aliases={"AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD", r"cross-family"]},
        scope="narrative",
    )
    # No violations: 0.38 was suppressed by Pattern C.
    assert r_nar.violations == ()


@pytest.mark.unit
def test_pattern_d_metric_nearest_pairing(tmp_path: Path) -> None:
    """Pattern D: nearest metric mention overrides window-based metric proximity.

    Catches consumer's RESULTS.md:171 pattern: "than the AUPRC delta
    suggests: ... AUROC is 0.383" — AUPRC appears in the window but
    AUROC is the metric semantically owning 0.383. Pattern D pairs
    each value with the nearest metric mention by text-order.
    """
    file = _write(
        tmp_path,
        "METRIC.md",
        # The AUPRC binding (0.293) should NOT be flagged against 0.383
        # because the nearest metric to 0.383 is AUROC, not AUPRC.
        "Sharper than the AUPRC delta suggests: LoRA's pooled OOD AUROC is 0.383.",
    )
    bindings = {BindingKey("LoRA", "AUPRC", "pooled_ood"): 0.293}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"LoRA": [r"LoRA"]},
        metric_aliases={"AUPRC": [r"AUPRC"], "AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD"]},
        scope="narrative",
    )
    # No violation: 0.383's nearest metric is AUROC, not AUPRC.
    # Pattern D filters this out before the value comparison.
    assert r_nar.violations == ()


@pytest.mark.unit
def test_pattern_a_unknown_alias_falls_through(tmp_path: Path) -> None:
    """Pattern A regression: "for X" where X isn't a known detector → no override.

    The validator must fall back to proximity-based pairing when the
    postfix names an unknown alias. Confirms the override is opt-in
    via the alias dict, not blanket suppression.
    """
    file = _write(
        tmp_path,
        "UNKNOWN.md",
        "TF-IDF reaches 0.974 AUPRC for the calibration check.",
    )
    bindings = {BindingKey("tf-idf + lr", "direct_val_auprc", "any"): 0.971}
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"tf-idf + lr": [r"TF-IDF"]},
        metric_aliases={"direct_val_auprc": [r"AUPRC"]},
        scope="narrative",
    )
    # "calibration check" isn't a detector → Pattern A doesn't fire.
    # Proximity rule pairs 0.974 with TF-IDF → flagged as violation.
    assert len(r_nar.violations) == 1
    assert r_nar.violations[0].found_value == 0.974
    assert r_nar.violations[0].detector == "tf-idf + lr"


@pytest.mark.unit
def test_v130_pairing_rules_scope_all_unchanged(tmp_path: Path) -> None:
    """scope='all' regression: Layer 3 pairing rules MUST be inactive.

    Backward-compat assertion: a consumer on legacy scope='all' sees
    the same false-positive behavior they saw on v1.2.0. Uses the
    consumer's actual RESULTS.md:171 prose verbatim (cross-detector
    list-grammar + metric-axis confusion) — under scope='narrative'
    the v1.3.0 rules suppress the false positives; under scope='all'
    they fire.
    """
    file = _write(
        tmp_path,
        "ALL_MODE.md",
        # Verbatim consumer prose excerpt. Under scope='all', the
        # validator wrongly pairs 0.515 (frozen probe's AUROC) with the
        # frozen_probe AUPRC binding because metric proximity finds
        # "AUPRC" earlier in the line.
        (
            "Sharper than the AUPRC delta suggests: LoRA's pooled OOD AUROC "
            "is 0.383 against frozen probe's 0.515."
        ),
    )
    bindings = {BindingKey("frozen_probe", "AUPRC", "pooled_ood"): 0.364}
    r_all = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"frozen_probe": [r"frozen probe"]},
        metric_aliases={"AUPRC": [r"AUPRC"], "AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD"]},
        scope="all",
    )
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={"frozen_probe": [r"frozen probe"]},
        metric_aliases={"AUPRC": [r"AUPRC"], "AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD"]},
        scope="narrative",
    )
    # Under scope='all': layer-3 OFF — the validator's AUPRC alias
    # match (from "AUPRC delta") pairs 0.515 with the AUPRC binding,
    # producing a violation. Backward compat: same as v1.2.0.
    assert len(r_all.violations) > len(r_nar.violations)
    # Under scope='narrative': Pattern D (metric-axis nearest-pairing)
    # correctly identifies AUROC as the nearest metric for 0.515, so
    # the AUPRC binding doesn't flag.
    assert r_nar.violations == ()


@pytest.mark.unit
def test_combined_consumer_residuals_resolve(tmp_path: Path) -> None:
    """Combined dogfood: all 3 of #81's documented residual patterns resolve under v1.3.0.

    Synthesized excerpts from the consumer's HEAD prose; under
    scope='narrative' with the v1.3.0 Layer 3 rules, all canonical
    bindings should MATCH (not flag) the correct values.
    """
    file = _write(
        tmp_path,
        "DOGFOOD.md",
        (
            # Pattern A — "for X" postfix
            "Pooled OOD AUPRC: 0.364 for the frozen probe and 0.291 for TF-IDF + LR.\n\n"
            # Pattern B + D — possessive + metric-axis confusion
            "LoRA's pooled OOD AUROC is 0.383 against frozen probe's 0.515.\n\n"
            # Pattern C — group subject
            "Cross-family AUROC: 0.38 for the trained detectors; frozen probe gap is 0.91.\n"
        ),
    )
    bindings = {
        BindingKey("tf-idf + lr", "AUPRC", "pooled_ood"): 0.291,
        BindingKey("frozen_probe", "AUPRC", "pooled_ood"): 0.364,
        BindingKey("LoRA", "AUROC", "pooled_ood"): 0.383,
        BindingKey("frozen_probe", "AUROC", "pooled_ood"): 0.515,
    }
    r_nar = validate_reader_value_bindings(
        files=[file],
        bindings=bindings,
        detector_aliases={
            "tf-idf + lr": [r"TF-IDF \+ LR", r"TF-IDF"],
            "LoRA": [r"LoRA"],
            "frozen_probe": [r"frozen probe"],
        },
        metric_aliases={"AUPRC": [r"AUPRC"], "AUROC": [r"AUROC"]},
        slice_aliases={"pooled_ood": [r"pooled OOD", r"cross-family", r"Cross-family"]},
        scope="narrative",
    )
    # All 4 binding values should match cleanly with zero violations.
    matched_vals = {m.value for m in r_nar.matched}
    assert r_nar.violations == ()
    assert 0.291 in matched_vals
    assert 0.364 in matched_vals
    assert 0.383 in matched_vals
    assert 0.515 in matched_vals


@pytest.mark.unit
def test_non_utf8_file_raises_valueerror(tmp_path: Path) -> None:
    """A non-UTF-8 file raises a diagnostic ValueError (never a silent crash)."""
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"TF-IDF reaches 0.974 AUPRC \xff on direct val.\n")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        validate_reader_value_bindings(
            files=[bad],
            bindings=SEED_BINDINGS,
            detector_aliases=SEED_DETECTOR_ALIASES,
            metric_aliases=SEED_METRIC_ALIASES,
        )


@pytest.mark.unit
def test_invalid_scope_raises_value_error() -> None:
    """A scope typo raises instead of silently reverting to legacy matching (#99 V6)."""
    with pytest.raises(ValueError, match="scope must be 'all' or 'narrative'"):
        validate_reader_value_bindings(
            files=[], bindings={}, scope="narratives"  # type: ignore[arg-type]
        )
