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
    Match,
    ValueBindingsReport,
    Violation,
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

SEED_BINDINGS = {
    ("tf-idf + lr", "direct_val_auprc"): 0.971,
    ("lora", "direct_val_auprc"): 0.974,
    ("frozen probe", "pooled_ood_auprc"): 0.364,
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
