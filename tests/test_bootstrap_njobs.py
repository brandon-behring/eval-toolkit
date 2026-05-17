"""Determinism + smoke tests for n_jobs wiring on the 5 bootstrap fns.

Per the v0.34.0 reproducibility contract (see
``docs/source/methodology/parallelism.md``):
- Same seed reproduces bit-for-bit-identical results regardless of n_jobs.
- n_jobs=-1 (all cores) completes without error.

These tests are the cross-fn guard against silent breakage of the
SeedSequence.spawn pattern when contributors touch the wired loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_diff,
    paired_bootstrap_ece_diff,
    paired_bootstrap_op_point_diff,
    paired_mde,
)
from eval_toolkit.metrics import expected_calibration_error, pr_auc


def _max_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Picklable threshold_fn for op-point tests (lambdas not allowed under n_jobs > 1)."""
    from eval_toolkit.thresholds import MaxF1Selector

    return float(MaxF1Selector().select(y_true, y_score).threshold)


def _f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Picklable metric_fn for op-point tests."""
    from eval_toolkit.metrics import metrics_at_threshold

    return float(metrics_at_threshold(y_true, y_score, threshold)["f1"])


def _make_inputs(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s_a = rng.normal(0, 1, size=n)
    s_b = y + rng.normal(0, 0.3, size=n)
    return y, s_a, s_b


@pytest.mark.unit
@pytest.mark.slow
def test_paired_bootstrap_diff_n_jobs_reproducibility() -> None:
    """Same seed produces identical CI regardless of n_jobs."""
    y, s_a, s_b = _make_inputs()
    r1 = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=200, seed=42, n_jobs=1)
    r2 = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=200, seed=42, n_jobs=2)
    assert (r1.delta, r1.ci_low, r1.ci_high) == (r2.delta, r2.ci_low, r2.ci_high)


@pytest.mark.unit
@pytest.mark.slow
def test_paired_bootstrap_diff_n_jobs_minus_one_runs() -> None:
    """n_jobs=-1 (all cores) completes; no speedup assertion."""
    y, s_a, s_b = _make_inputs()
    r = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=100, seed=42, n_jobs=-1)
    assert r.ci_low <= r.ci_high


@pytest.mark.unit
@pytest.mark.slow
def test_bootstrap_ci_studentized_n_jobs_reproducibility() -> None:
    """Studentized bootstrap-t reproduces identically across n_jobs.

    Uses balanced n=200 + n_resamples=100 so the >5% degenerate gate doesn't
    trip on small-sample single-class draws (studentized inner jackknife is
    sensitive to single-class resamples).
    """
    rng = np.random.default_rng(42)
    n = 200
    y = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    rng.shuffle(y)
    s = y + rng.normal(0, 0.3, size=n)
    r1 = bootstrap_ci(y, s, pr_auc, n_resamples=100, seed=42, method="studentized", n_jobs=1)
    r2 = bootstrap_ci(y, s, pr_auc, n_resamples=100, seed=42, method="studentized", n_jobs=2)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


@pytest.mark.unit
def test_bootstrap_ci_rejects_n_jobs_with_non_studentized() -> None:
    """n_jobs > 1 with method='BCa' or 'percentile' raises ValueError."""
    y, _, s = _make_inputs()
    for method in ("BCa", "percentile"):
        with pytest.raises(ValueError, match=r"n_jobs.*studentized"):
            bootstrap_ci(y, s, pr_auc, n_resamples=50, seed=42, method=method, n_jobs=2)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.slow
def test_paired_bootstrap_ece_diff_n_jobs_reproducibility() -> None:
    """ECE paired-diff reproduces identically across n_jobs."""
    y, s_a, s_b = _make_inputs()
    # Coerce scores to [0,1] so ECE is well-defined.
    s_a_cal = 1.0 / (1.0 + np.exp(-s_a))
    s_b_cal = 1.0 / (1.0 + np.exp(-s_b))
    r1 = paired_bootstrap_ece_diff(
        y, s_a_cal, s_b_cal, ece_fn=expected_calibration_error, n_resamples=100, seed=42, n_jobs=1
    )
    r2 = paired_bootstrap_ece_diff(
        y, s_a_cal, s_b_cal, ece_fn=expected_calibration_error, n_resamples=100, seed=42, n_jobs=2
    )
    assert (r1.delta, r1.ci_low, r1.ci_high) == (r2.delta, r2.ci_low, r2.ci_high)


@pytest.mark.unit
@pytest.mark.slow
def test_paired_bootstrap_op_point_diff_n_jobs_reproducibility() -> None:
    """Two-level op-point bootstrap reproduces identically across n_jobs."""
    rng = np.random.default_rng(42)
    n = 100
    val_y = rng.integers(0, 2, size=n)
    val_a = val_y + rng.normal(0, 0.5, size=n)
    val_b = val_y + rng.normal(0, 0.3, size=n)
    test_y = rng.integers(0, 2, size=n)
    test_a = test_y + rng.normal(0, 0.5, size=n)
    test_b = test_y + rng.normal(0, 0.3, size=n)
    r1 = paired_bootstrap_op_point_diff(
        val_y,
        val_a,
        val_b,
        test_y,
        test_a,
        test_b,
        threshold_fn=_max_f1_threshold,
        metric_fn=_f1_at_threshold,
        n_resamples=100,
        seed=42,
        n_jobs=1,
    )
    r2 = paired_bootstrap_op_point_diff(
        val_y,
        val_a,
        val_b,
        test_y,
        test_a,
        test_b,
        threshold_fn=_max_f1_threshold,
        metric_fn=_f1_at_threshold,
        n_resamples=100,
        seed=42,
        n_jobs=2,
    )
    assert (r1.delta, r1.ci_low, r1.ci_high) == (r2.delta, r2.ci_low, r2.ci_high)


@pytest.mark.unit
@pytest.mark.slow
def test_paired_mde_n_jobs_reproducibility() -> None:
    """paired_mde pass-through to paired_bootstrap_diff reproduces across n_jobs."""
    y, s_a, s_b = _make_inputs()
    r1 = paired_mde(y, s_a, s_b, pr_auc, n_resamples=200, seed=42, n_jobs=1)
    r2 = paired_mde(y, s_a, s_b, pr_auc, n_resamples=200, seed=42, n_jobs=2)
    assert (r1.mde, r1.sigma_delta, r1.delta_observed) == (
        r2.mde,
        r2.sigma_delta,
        r2.delta_observed,
    )
