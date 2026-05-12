"""Tests for generic claim/evidence gates."""

from __future__ import annotations

import pytest

from eval_toolkit.claims import (
    ClaimSpec,
    evaluate_claims,
    headline_present_gate,
    low_fpr_feasibility_gate,
    metric_threshold_gate,
    minimum_slice_size_gate,
    no_leakage_errors_gate,
    no_scorer_errors_gate,
    paired_diff_present_gate,
    required_metric_gate,
    required_scorer_gate,
    required_slice_gate,
    source_role_gate,
)


@pytest.mark.unit
def test_basic_required_gates_pass() -> None:
    result = {
        "by_slice": {
            "test": {
                "n": 40,
                "n_positive": 20,
                "by_scorer": {"model": {"pr_auc": 0.8}},
                "paired_diffs": {"model_minus_base": {"delta": 0.1, "ci_95": [0.01, 0.2]}},
            }
        }
    }
    claim = ClaimSpec(
        name="model claim",
        gates=(
            required_slice_gate("test"),
            required_scorer_gate("test", "model"),
            required_metric_gate("test", "model", "pr_auc"),
            minimum_slice_size_gate("test", min_n=30, min_positive=10, min_negative=10),
            paired_diff_present_gate("test", "model_minus_base"),
        ),
    )
    report = evaluate_claims(result, [claim])
    assert not report.has_failures()


@pytest.mark.unit
def test_metric_threshold_gate_catches_hard_negative_fpr_failure() -> None:
    result = {
        "by_slice": {
            "external_diag": {
                "n": 100,
                "n_positive": 0,
                "by_scorer": {
                    "model": {
                        "transferred_operating_points": {
                            "calib": {"max_f1": {"fpr@threshold": 0.25}}
                        }
                    }
                },
            }
        }
    }
    claim = ClaimSpec(
        name="hard-negative screen",
        gates=(
            metric_threshold_gate(
                "external_diag",
                "model",
                "transferred_operating_points.calib.max_f1.fpr@threshold",
                op="<=",
                threshold=0.05,
            ),
        ),
    )
    report = evaluate_claims(result, [claim])
    assert report.has_failures()
    gate = report.claims["hard-negative screen"][0]
    assert gate.evidence["value"] == 0.25


@pytest.mark.unit
def test_low_fpr_feasibility_gate_fails_tiny_negative_slice() -> None:
    result = {"by_slice": {"holdout": {"n": 48, "n_positive": 23}}}
    claim = ClaimSpec(
        name="low-FPR claim",
        gates=(low_fpr_feasibility_gate("holdout", max_fpr=0.05),),
    )

    report = evaluate_claims(result, [claim])

    assert report.has_failures()
    gate = report.claims["low-FPR claim"][0]
    assert gate.evidence["n_negative"] == 25
    assert gate.evidence["empirical_fpr_step"] == pytest.approx(0.04)
    assert gate.evidence["best_case_fpr_ci_high"] > 0.05


@pytest.mark.unit
def test_low_fpr_feasibility_gate_passes_claim_sized_negative_slice() -> None:
    result = {"by_slice": {"holdout": {"n": 1100, "n_positive": 100}}}
    claim = ClaimSpec(
        name="low-FPR claim",
        gates=(low_fpr_feasibility_gate("holdout", max_fpr=0.005),),
    )

    report = evaluate_claims(result, [claim])

    assert not report.has_failures()
    gate = report.claims["low-FPR claim"][0]
    assert gate.evidence["n_negative"] == 1000
    assert gate.evidence["best_case_fpr_ci_high"] <= 0.005


@pytest.mark.unit
def test_low_fpr_feasibility_gate_fails_missing_or_bad_counts() -> None:
    missing = evaluate_claims(
        {},
        [ClaimSpec(name="missing", gates=(low_fpr_feasibility_gate("holdout", max_fpr=0.05),))],
    )
    bad_counts = evaluate_claims(
        {"by_slice": {"holdout": {"n": 10, "n_positive": 11}}},
        [ClaimSpec(name="bad", gates=(low_fpr_feasibility_gate("holdout", max_fpr=0.05),))],
    )

    assert missing.has_failures()
    assert bad_counts.has_failures()
    assert bad_counts.claims["bad"][0].evidence["n_negative"] == -1


@pytest.mark.unit
def test_no_scorer_and_leakage_error_gates_fail() -> None:
    result = {
        "config": {
            "leakage_report": {
                "findings": [
                    {
                        "check_name": "ExactDuplicateCheck",
                        "severity": "error",
                        "message": "duplicate",
                    }
                ]
            }
        },
        "by_slice": {
            "test": {
                "n": 2,
                "n_positive": 1,
                "by_scorer": {"model": {"error": "boom"}},
            }
        },
    }
    claim = ClaimSpec(
        name="clean run",
        gates=(no_scorer_errors_gate(), no_leakage_errors_gate()),
    )
    report = evaluate_claims(result, [claim])
    assert report.has_failures()
    assert [gate.passed for gate in report.claims["clean run"]] == [False, False]


@pytest.mark.unit
def test_v3_shaped_missing_headline_and_source_roles() -> None:
    result = {
        "headline": None,
        "by_slice": {
            "external_diag": {
                "n": 100,
                "n_positive": 0,
                "by_scorer": {"lr": {"fpr": 1.0}},
            }
        },
    }
    manifest = {
        "source_roles": [
            {"source": "email_pool", "role": "train"},
            {"source": "notinject", "role": "external_diagnostic"},
        ]
    }
    claim = ClaimSpec(
        name="claim-mode report",
        gates=(
            headline_present_gate(),
            minimum_slice_size_gate("external_diag", min_n=50),
            source_role_gate(("train", "external_diagnostic")),
        ),
    )
    report = evaluate_claims(result, [claim], manifest=manifest)
    assert report.has_failures()
    assert [gate.passed for gate in report.claims["claim-mode report"]] == [
        False,
        True,
        True,
    ]
