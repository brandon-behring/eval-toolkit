"""Hypothesis property tests for bootstrap CIs.

Per the report §5.2 recipe: point-estimate fidelity, determinism, paired
symmetry, overlaps_zero correctness, MDE monotone in n.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_diff,
    paired_mde,
)
from eval_toolkit.metrics import pr_auc, roc_auc
from tests.strategies import balanced_binary_array, score_array

# Local aliases preserve the existing naming throughout the file.
_balanced_binary_array = balanced_binary_array
_score_array = score_array


@pytest.mark.property
@given(
    y=_balanced_binary_array(50),
    s=_score_array(50),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_bootstrap_point_estimate_fidelity(y: np.ndarray, s: np.ndarray, seed: int) -> None:
    """ci.point_estimate equals the metric value on the original sample."""
    ci = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="percentile", rng=seed)
    direct = pr_auc(y, s)
    assert ci.point_estimate == pytest.approx(direct, abs=1e-12)


@pytest.mark.property
@given(
    y=_balanced_binary_array(50),
    s=_score_array(50),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_bootstrap_determinism(y: np.ndarray, s: np.ndarray, seed: int) -> None:
    """Same seed → identical CI."""
    ci1 = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="percentile", rng=seed)
    ci2 = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="percentile", rng=seed)
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high
    assert ci1.point_estimate == ci2.point_estimate


@pytest.mark.property
@pytest.mark.slow
@given(
    y=_balanced_binary_array(50),
    s_a=_score_array(50),
    s_b=_score_array(50),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_paired_bootstrap_diff_anti_symmetry(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, seed: int
) -> None:
    """paired_diff(A, B) and paired_diff(B, A) give negated delta."""
    diff_ab = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=100, rng=seed)
    diff_ba = paired_bootstrap_diff(y, s_b, s_a, pr_auc, n_resamples=100, rng=seed)
    assert diff_ab.delta == pytest.approx(-diff_ba.delta, abs=1e-12)


@pytest.mark.property
@given(
    y=_balanced_binary_array(50),
    s=_score_array(50),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_paired_diff_zero_delta_for_identical_scorers(
    y: np.ndarray, s: np.ndarray, seed: int
) -> None:
    """Same scorer: delta is exactly zero."""
    diff = paired_bootstrap_diff(y, s, s, pr_auc, n_resamples=50, rng=seed)
    assert diff.delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.property
@given(
    y=_balanced_binary_array(50),
    s_a=_score_array(50),
    s_b=_score_array(50),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_overlaps_zero_correctness(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, seed: int
) -> None:
    """overlaps_zero == (ci_low ≤ 0 ≤ ci_high)."""
    diff = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=100, rng=seed)
    expected = bool(diff.ci_low <= 0 <= diff.ci_high)
    assert diff.overlaps_zero == expected


@pytest.mark.property
@given(
    y=_balanced_binary_array(80),
    s=_score_array(80),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_bootstrap_ci_contains_point_estimate(y: np.ndarray, s: np.ndarray) -> None:
    """ci_low ≤ point_estimate ≤ ci_high."""
    ci = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="percentile", rng=42)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.property
@given(
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_paired_mde_decreases_with_n(seed: int) -> None:
    """MDE shrinks as n grows (within bootstrap noise)."""
    rng = np.random.default_rng(seed)
    n_small, n_big = 100, 400
    y_small = rng.binomial(1, 0.3, size=n_small).astype(int)
    s_a_small = rng.uniform(0, 1, size=n_small)
    s_b_small = s_a_small + 0.2 * y_small
    s_b_small = np.clip(s_b_small, 0, 1)

    y_big = rng.binomial(1, 0.3, size=n_big).astype(int)
    s_a_big = rng.uniform(0, 1, size=n_big)
    s_b_big = s_a_big + 0.2 * y_big
    s_b_big = np.clip(s_b_big, 0, 1)

    mde_small = paired_mde(y_small, s_a_small, s_b_small, pr_auc, n_resamples=100, rng=seed)
    mde_big = paired_mde(y_big, s_a_big, s_b_big, pr_auc, n_resamples=100, rng=seed)
    # n=400 should give a smaller MDE than n=100, with some noise tolerance.
    assert mde_big.mde < mde_small.mde * 1.2  # 20% noise allowance


@pytest.mark.property
@given(
    y=_balanced_binary_array(60),
    s=_score_array(60),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_paired_diff_with_roc_auc_metric(y: np.ndarray, s: np.ndarray) -> None:
    """paired_bootstrap_diff is metric-agnostic — works for ROC-AUC too."""
    diff = paired_bootstrap_diff(y, s, s, roc_auc, n_resamples=50, rng=42)
    assert diff.delta == pytest.approx(0.0, abs=1e-12)
