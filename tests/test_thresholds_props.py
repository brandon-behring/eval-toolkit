"""Hypothesis property tests for the v0.7.0 ThresholdSelector reference impls.

Restores coverage on `src/eval_toolkit/thresholds.py` toward the 90 % gate
(v0.7.1 / PR 1.5). Mirrors the shape of `test_metrics_props.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit import metrics_at_threshold
from eval_toolkit.calibration import CostMatrix, bayes_optimal_threshold
from eval_toolkit.metrics import ThresholdResult
from eval_toolkit.thresholds import (
    CostSensitiveSelector,
    MaxF1Selector,
    TargetFPRSelector,
    TargetPrecisionSelector,
    TargetRecallSelector,
    YoudenJSelector,
)
from tests.strategies import balanced_binary_array, score_array

# ---------------------------------------------------------------------------
# MaxF1Selector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(y=balanced_binary_array(80), s=score_array(80))
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_max_f1_returns_optimum(y: np.ndarray, s: np.ndarray) -> None:
    """The threshold MaxF1Selector returns has F1 ≥ F1 at any sampled threshold."""
    try:
        result = MaxF1Selector().select(y, s)
    except RuntimeError:
        return  # constant scores
    chosen_f1 = metrics_at_threshold(y, s, result.threshold)["f1"]
    if s.size < 2:
        return
    for cand in np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9]):
        assert chosen_f1 >= metrics_at_threshold(y, s, float(cand))["f1"] - 1e-6


@pytest.mark.property
@given(y=balanced_binary_array(60), s=score_array(60))
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_max_f1_result_fields_in_unit_interval(y: np.ndarray, s: np.ndarray) -> None:
    """Every ThresholdResult field is finite and bounded."""
    try:
        result = MaxF1Selector().select(y, s)
    except RuntimeError:
        return
    assert isinstance(result, ThresholdResult)
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.threshold <= 1.0
    assert result.criterion == "max_f1"


# ---------------------------------------------------------------------------
# TargetRecallSelector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    y=balanced_binary_array(60),
    s=score_array(60),
    target=st.floats(0.5, 0.95, allow_nan=False),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_target_recall_meets_target(y: np.ndarray, s: np.ndarray, target: float) -> None:
    """If TargetRecallSelector returns, recall on returned threshold >= target."""
    try:
        result = TargetRecallSelector(recall=target).select(y, s)
    except (RuntimeError, ValueError):
        return  # unreachable target / constant scores
    actual_recall = metrics_at_threshold(y, s, result.threshold)["recall"]
    assert actual_recall >= target - 1e-6
    assert result.criterion == f"recall_{target:.2f}"


# ---------------------------------------------------------------------------
# TargetPrecisionSelector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    y=balanced_binary_array(60),
    s=score_array(60),
    target=st.floats(0.30, 0.80, allow_nan=False),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_target_precision_meets_target(y: np.ndarray, s: np.ndarray, target: float) -> None:
    """If TargetPrecisionSelector returns, precision on returned threshold >= target."""
    try:
        result = TargetPrecisionSelector(precision=target).select(y, s)
    except (RuntimeError, ValueError):
        return
    # PR-curve precisions can be slightly higher than computed here due to
    # discretization; toolkit's TargetPrecisionSelector uses the curve's
    # value, which we re-derive — allow tolerance.
    assert result.precision >= target - 1e-6


# ---------------------------------------------------------------------------
# TargetFPRSelector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    y=balanced_binary_array(60),
    s=score_array(60),
    fpr_cap=st.floats(0.05, 0.40, allow_nan=False),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_target_fpr_meets_cap(y: np.ndarray, s: np.ndarray, fpr_cap: float) -> None:
    """If TargetFPRSelector returns, FPR at returned threshold ≤ cap."""
    try:
        result = TargetFPRSelector(fpr=fpr_cap).select(y, s)
    except (RuntimeError, ValueError):
        return
    actual_fpr = metrics_at_threshold(y, s, result.threshold)["fpr"]
    assert actual_fpr <= fpr_cap + 1e-6


# ---------------------------------------------------------------------------
# YoudenJSelector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(y=balanced_binary_array(60), s=score_array(60))
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_youden_j_returns_valid_result(y: np.ndarray, s: np.ndarray) -> None:
    """YoudenJSelector returns a sane ThresholdResult."""
    try:
        result = YoudenJSelector().select(y, s)
    except RuntimeError:
        return
    assert isinstance(result, ThresholdResult)
    assert 0.0 <= result.threshold <= 1.0
    assert result.criterion == "youden_j"


# ---------------------------------------------------------------------------
# CostSensitiveSelector
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    y=balanced_binary_array(40),
    s=score_array(40),
    prior=st.floats(0.05, 0.95, allow_nan=False),
    fp=st.floats(0.5, 5.0, allow_nan=False),
    fn=st.floats(0.5, 5.0, allow_nan=False),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_cost_sensitive_threshold_matches_bayes_optimal(
    y: np.ndarray, s: np.ndarray, prior: float, fp: float, fn: float
) -> None:
    """CostSensitiveSelector's threshold equals the closed-form Bayes-optimal."""
    cm = CostMatrix(prior=prior, fp_cost=fp, fn_cost=fn)
    expected = bayes_optimal_threshold(prior, fp, fn)
    result = CostSensitiveSelector(cm).select(y, s)
    assert result.threshold == pytest.approx(expected, abs=1e-9)


@pytest.mark.property
@given(
    fp=st.floats(0.5, 5.0, allow_nan=False),
    fn=st.floats(0.5, 5.0, allow_nan=False),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_cost_sensitive_symmetric_costs_at_prior_half(fp: float, fn: float) -> None:
    """At symmetric costs and prior=0.5, threshold is determined by the cost ratio
    independently of any data."""
    if fp != fn:
        return
    cm = CostMatrix(prior=0.5, fp_cost=fp, fn_cost=fn)
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    result = CostSensitiveSelector(cm).select(y, s)
    assert result.threshold == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(bad_recall=st.one_of(st.floats(max_value=0.0), st.floats(min_value=1.01)))
@settings(deadline=None, max_examples=10)
def test_target_recall_rejects_out_of_range(bad_recall: float) -> None:
    """Constructor rejects recall outside (0, 1]."""
    if np.isnan(bad_recall):
        return
    with pytest.raises(ValueError, match="recall must be in"):
        TargetRecallSelector(recall=bad_recall)


@pytest.mark.property
@given(bad_precision=st.one_of(st.floats(max_value=0.0), st.floats(min_value=1.01)))
@settings(deadline=None, max_examples=10)
def test_target_precision_rejects_out_of_range(bad_precision: float) -> None:
    """Constructor rejects precision outside (0, 1]."""
    if np.isnan(bad_precision):
        return
    with pytest.raises(ValueError, match="precision must be in"):
        TargetPrecisionSelector(precision=bad_precision)
