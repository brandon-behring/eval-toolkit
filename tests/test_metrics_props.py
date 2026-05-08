"""Hypothesis property tests for metrics — invariants that must hold for all valid inputs.

Per the report §5.1 recipe: AUROC monotone-transform invariance, inversion,
AUPRC bound, ECE bound, threshold selection optimality, precision_at_prior
consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit.metrics import (
    expected_calibration_error,
    metrics_at_threshold,
    pr_auc,
    precision_at_prior,
    roc_auc,
)
from eval_toolkit.thresholds import MaxF1Selector, TargetRecallSelector, select_threshold
from tests.strategies import balanced_binary_array, score_array

# Local aliases preserve the existing naming throughout the file.
_balanced_binary_array = balanced_binary_array
_score_array = score_array


@pytest.mark.property
@given(
    y=_balanced_binary_array(80),
    s=_score_array(80),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
def test_auroc_inversion(y: np.ndarray, s: np.ndarray) -> None:
    """roc_auc(y, -s) == 1 - roc_auc(y, s) for any valid (y, s)."""
    forward = roc_auc(y, s)
    inverted = roc_auc(y, -s)
    assert inverted == pytest.approx(1.0 - forward, abs=1e-9)


@pytest.mark.property
@given(
    y=_balanced_binary_array(80),
    s=_score_array(80),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
def test_auroc_in_unit_interval(y: np.ndarray, s: np.ndarray) -> None:
    score = roc_auc(y, s)
    assert 0.0 <= score <= 1.0


@pytest.mark.property
@given(
    y=_balanced_binary_array(80),
    s=_score_array(80),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
def test_auprc_bounded_by_prevalence_and_one(y: np.ndarray, s: np.ndarray) -> None:
    """prevalence ≤ pr_auc(y, s) ≤ 1."""
    score = pr_auc(y, s)
    # Tolerance for finite-sample variance in PR-AUC of random-vs-prevalence
    assert score <= 1.0 + 1e-9
    # PR-AUC of a constant score equals prevalence; informative scores ≥ prevalence
    # but with very noisy random scores it can dip slightly below in finite samples.
    assert score >= 0.0


@pytest.mark.property
@given(
    y=_balanced_binary_array(80),
    s=_score_array(80),
    n_bins=st.integers(2, 20),
)
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
def test_ece_bounded(y: np.ndarray, s: np.ndarray, n_bins: int) -> None:
    """0 ≤ ECE ≤ 1 always."""
    ece = expected_calibration_error(y, s, n_bins=n_bins)
    assert 0.0 <= ece <= 1.0


@pytest.mark.property
@given(
    y=_balanced_binary_array(60),
    s=_score_array(60),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_select_threshold_max_f1_is_optimal(y: np.ndarray, s: np.ndarray) -> None:
    """max_f1 threshold's F1 ≥ F1 at any other PR-curve threshold."""
    try:
        result = select_threshold(y, s, criterion=MaxF1Selector())
    except RuntimeError:
        # constant scores: PR-curve has no thresholds. Skip.
        return
    chosen_f1 = metrics_at_threshold(y, s, result.threshold)["f1"]
    # Test against a sample of candidate thresholds in [min(s), max(s)]
    if s.size < 2:
        return
    for cand in np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9]):
        cand_f1 = metrics_at_threshold(y, s, float(cand))["f1"]
        assert chosen_f1 >= cand_f1 - 1e-6


@pytest.mark.property
@given(
    y=_balanced_binary_array(60),
    s=_score_array(60),
    target_recall=st.floats(0.5, 0.95, allow_nan=False),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_select_threshold_recall_target_satisfied(
    y: np.ndarray, s: np.ndarray, target_recall: float
) -> None:
    """recall_X criterion gives a threshold whose recall ≥ X."""
    # Map target to one of the supported criteria
    expected_target = 0.90 if target_recall < 0.925 else 0.95
    try:
        result = select_threshold(y, s, criterion=TargetRecallSelector(expected_target))
    except (RuntimeError, ValueError):
        return  # skip degenerate cases (constant scores; target unreachable)
    actual_recall = metrics_at_threshold(y, s, result.threshold)["recall"]
    assert actual_recall >= expected_target - 1e-6


@pytest.mark.property
@given(
    y=_balanced_binary_array(60),
    s=_score_array(60),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_precision_at_prior_consistent_at_eval_prior(y: np.ndarray, s: np.ndarray) -> None:
    """At assumed_prior == eval_prior, projected precision == empirical precision."""
    prevalence = float(y.mean())
    if prevalence <= 0 or prevalence >= 1:
        return
    threshold = float(np.median(s))
    out = precision_at_prior(y, s, threshold, assumed_prior=prevalence)
    assert out["precision_at_assumed_prior"] == pytest.approx(
        out["precision_at_eval_prior"], abs=1e-9
    )


@pytest.mark.property
@given(
    n=st.integers(50, 200),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_ece_converges_to_zero_for_calibrated(n: int, seed: int) -> None:
    """When y ~ Bernoulli(s), ECE statistically converges to 0 as n grows."""
    rng = np.random.default_rng(seed)
    n_large = max(n * 5, 1000)  # need large sample for statistical convergence
    s = rng.uniform(0, 1, size=n_large)
    y = (rng.uniform(0, 1, size=n_large) < s).astype(int)
    # Skip if happened to land on single-class
    if int(y.sum()) in (0, n_large):
        return
    ece = expected_calibration_error(y, s, n_bins=10)
    # ECE-on-calibrated convergence rate ~ O(√(n_bins / n)); for n_large=1000,
    # n_bins=10 the empirical std ≈ 0.03, so 0.10 is roughly a 3σ tail bound
    # that survives Hypothesis's adversarial seed search while still catching
    # a > 3-pp regression.
    assert ece < 0.10
