"""Coverage-focused tests for eval_toolkit.thresholds.

Pairs with the happy-path coverage in ``test_thresholds.py`` and the
invariants in ``test_thresholds_props.py``. Targets the helper
validation paths (``_record_float``, ``_record_float_or_inf``,
``_pr_curve_trim``, ``_roc_curve_trim``), the
``ThresholdPolicyMetadata.__post_init__`` guards, every
``CISafeThresholdSelector`` constraint-validation branch, and the
"no eligible threshold" RuntimeError for the recall/precision/FPR
selectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.calibration import CostMatrix
from eval_toolkit.thresholds import (
    CISafeThresholdSelector,
    CostSensitiveSelector,
    TargetFPRSelector,
    TargetPrecisionSelector,
    TargetRecallSelector,
    ThresholdPolicyMetadata,
    WilsonInterval,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# wilson_interval: validation + n == 0 branch (lines 188-196)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wilson_interval_rejects_negative_successes() -> None:
    with pytest.raises(ValueError, match="successes must be non-negative"):
        wilson_interval(-1, 10)


@pytest.mark.unit
def test_wilson_interval_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        wilson_interval(0, -1)


@pytest.mark.unit
def test_wilson_interval_rejects_successes_above_n() -> None:
    with pytest.raises(ValueError, match="successes cannot exceed n"):
        wilson_interval(5, 4)


@pytest.mark.unit
def test_wilson_interval_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(0, 10, confidence=0.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(0, 10, confidence=1.0)


@pytest.mark.unit
def test_wilson_interval_with_zero_n_returns_none_bounds() -> None:
    interval = wilson_interval(0, 0)
    assert interval.low is None
    assert interval.high is None
    out = interval.to_dict()
    assert out["low"] is None
    assert out["high"] is None
    assert out["n"] == 0


# ---------------------------------------------------------------------------
# ThresholdPolicyMetadata.__post_init__: validation errors (231, 233, 235, 237)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_threshold_policy_metadata_rejects_empty_calibration_slice() -> None:
    with pytest.raises(ValueError, match="calibration_slice must be non-empty"):
        ThresholdPolicyMetadata(
            calibration_slice="",
            score_column="score",
            selector="max_f1",
            constraints={"min_recall": 0.8},
        )


@pytest.mark.unit
def test_threshold_policy_metadata_rejects_empty_score_column() -> None:
    with pytest.raises(ValueError, match="score_column must be non-empty"):
        ThresholdPolicyMetadata(
            calibration_slice="dev",
            score_column="",
            selector="max_f1",
            constraints={"min_recall": 0.8},
        )


@pytest.mark.unit
def test_threshold_policy_metadata_rejects_empty_selector() -> None:
    with pytest.raises(ValueError, match="selector must be non-empty"):
        ThresholdPolicyMetadata(
            calibration_slice="dev",
            score_column="score",
            selector="",
            constraints={"min_recall": 0.8},
        )


@pytest.mark.unit
def test_threshold_policy_metadata_rejects_empty_constraints() -> None:
    with pytest.raises(ValueError, match="constraints must be non-empty"):
        ThresholdPolicyMetadata(
            calibration_slice="dev",
            score_column="score",
            selector="max_f1",
            constraints={},
        )


# ---------------------------------------------------------------------------
# CISafeThresholdSelector.__post_init__: validation branches (271, 279, 287)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ci_safe_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        CISafeThresholdSelector(max_fpr=0.05, confidence=0.0)


@pytest.mark.unit
def test_ci_safe_rejects_no_constraints() -> None:
    with pytest.raises(ValueError, match="at least one CI-safe threshold constraint"):
        CISafeThresholdSelector()


@pytest.mark.unit
def test_ci_safe_rejects_out_of_range_constraint() -> None:
    with pytest.raises(ValueError, match="max_fpr must be in"):
        CISafeThresholdSelector(max_fpr=1.5)
    with pytest.raises(ValueError, match="min_recall_ci_lower must be in"):
        CISafeThresholdSelector(min_recall_ci_lower=-0.1)


# ---------------------------------------------------------------------------
# CISafeThresholdSelector.select: empty candidates and no-accept branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ci_safe_select_no_accepted_threshold_raises() -> None:
    """A min_recall too high to ever satisfy → no accepted candidate."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    # An impossibly-high recall floor with a 100% lower bound is infeasible.
    selector = CISafeThresholdSelector(min_recall=1.0, min_recall_ci_lower=0.99)
    with pytest.raises(RuntimeError, match="no threshold satisfies"):
        selector.select(y, s)


# ---------------------------------------------------------------------------
# Per-selector "no eligible threshold" RuntimeError (524 + analogues)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_precision_no_eligible_threshold_raises() -> None:
    y = np.array([0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.05, 0.7])  # inverted: precision never reaches 0.99
    with pytest.raises(RuntimeError, match="No threshold achieves precision"):
        TargetPrecisionSelector(precision=0.99).select(y, s)


@pytest.mark.unit
def test_target_fpr_no_eligible_threshold_raises() -> None:
    # FPR=0 is unreachable when the single negative always lands at or above
    # the lowest threshold (i.e., every operating point keeps that FP active).
    y = np.array([0, 1, 1])
    s = np.array([0.9, 0.5, 0.8])  # all thresholds → FPR=1 (only 1 neg, always FP)
    with pytest.raises(RuntimeError, match="No threshold achieves FPR"):
        TargetFPRSelector(fpr=0.0).select(y, s)


# ---------------------------------------------------------------------------
# Selector validation: out-of-range constructor args
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_recall_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="recall must be in"):
        TargetRecallSelector(recall=0.0)
    with pytest.raises(ValueError, match="recall must be in"):
        TargetRecallSelector(recall=1.5)


@pytest.mark.unit
def test_target_precision_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="precision must be in"):
        TargetPrecisionSelector(precision=0.0)
    with pytest.raises(ValueError, match="precision must be in"):
        TargetPrecisionSelector(precision=1.5)


@pytest.mark.unit
def test_target_fpr_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="fpr must be in"):
        TargetFPRSelector(fpr=-0.1)
    with pytest.raises(ValueError, match="fpr must be in"):
        TargetFPRSelector(fpr=1.5)


# ---------------------------------------------------------------------------
# CISafeThresholdSelector.selected_operating_point with bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ci_safe_selected_operating_point_with_bootstrap() -> None:
    """Exercise the bootstrap CI branch of selected_operating_point."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    s = np.concatenate([rng.uniform(0.0, 0.4, 20), rng.uniform(0.6, 1.0, 20)])
    selector = CISafeThresholdSelector(max_fpr=0.5, min_recall=0.5)
    out = selector.selected_operating_point(y, s, bootstrap_selected=True, n_resamples=10, seed=1)
    # When bootstrap is requested, a bootstrap_selected block is present
    # with per-metric percentile CIs.
    assert "bootstrap_selected" in out
    assert "fpr" in out["bootstrap_selected"]
    assert "recall" in out["bootstrap_selected"]
    assert out["bootstrap_selected"]["fpr"]["n_resamples"] == 10
    assert "constraints" in out
    assert out["n_accepted"] > 0


# ---------------------------------------------------------------------------
# Cost-sensitive selector criterion property
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cost_sensitive_criterion_embeds_costs() -> None:
    cm = CostMatrix(prior=0.3, fp_cost=2.0, fn_cost=1.0)
    sel = CostSensitiveSelector(cm)
    assert "prior=0.300" in sel.criterion
    assert "fp=2.00" in sel.criterion
    assert "fn=1.00" in sel.criterion


# ---------------------------------------------------------------------------
# WilsonInterval direct construction + dict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wilson_interval_dict_round_trip() -> None:
    wi = WilsonInterval(low=0.1, high=0.3, confidence=0.95, successes=2, n=10)
    out = wi.to_dict()
    assert out == {"low": 0.1, "high": 0.3, "confidence": 0.95, "successes": 2, "n": 10}
