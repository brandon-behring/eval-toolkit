"""Edge-case error-path coverage for ``eval_toolkit.bootstrap`` (v0.26.0).

Closes audit gaps from the v0.26.0 toolkit completeness sweep around
small-n and rare-positive bootstrap paths. Three categories:

1. **n < 10 guards**: ``bootstrap_ci``, ``paired_bootstrap_diff``,
   and ``paired_bootstrap_ece_diff`` all enforce a minimum sample
   size; existing tests cover ``bootstrap_ci`` but the paired
   variants' guards are unasserted.
2. **No usable resamples**: paired bootstraps catch metric-raising
   resamples (rare positives in PR-AUC / ECE diff) and refuse to
   compute a CI on > 5% degenerate resamples; the "all degenerate"
   path was untested.
3. **>5% degenerate resamples**: a partial-degeneracy sister case
   to (2) — the threshold-crossing path.

These tests use small-n + rare-positive constructions that
deterministically hit the relevant raise sites, and pin the error
message phrasing so a regression that silently changes the message
(used by downstream consumers parsing it) would fail the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_diff,
    paired_bootstrap_ece_diff,
    paired_bootstrap_op_point_diff,
)
from eval_toolkit.metrics import expected_calibration_error, pr_auc
from eval_toolkit.thresholds import MaxF1Selector


def test_bootstrap_ci_raises_on_n_lt_10() -> None:
    """``bootstrap_ci`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:256``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    with pytest.raises(ValueError, match="n=6 too small for bootstrap"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100)


def test_paired_bootstrap_diff_raises_on_n_lt_10() -> None:
    """``paired_bootstrap_diff`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:448``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    with pytest.raises(ValueError, match="n=6 too small for paired bootstrap"):
        paired_bootstrap_diff(y, a, b, pr_auc, n_resamples=100)


def test_paired_bootstrap_ece_diff_raises_on_n_lt_10() -> None:
    """``paired_bootstrap_ece_diff`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:547``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    with pytest.raises(ValueError, match="n=6 too small for paired bootstrap"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=expected_calibration_error, n_resamples=100)


def _strict_single_class_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """A metric that explicitly raises on single-class y_true (no NaN return).

    sklearn's pr_auc / roc_auc return 0 / NaN on single-class inputs
    rather than raising, so they don't trigger the bootstrap's
    "degenerate resample" counter. This helper raises the explicit
    ValueError that ``paired_bootstrap_diff`` catches and counts.
    """
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError("metric undefined on single-class input")
    # Trivial well-behaved metric on mixed-class inputs: mean of positive scores.
    return float(np.mean(y_score[y_true == 1]))


def test_paired_bootstrap_diff_raises_on_too_many_degenerate_resamples() -> None:
    """``paired_bootstrap_diff`` raises when > 5% of resamples raise.

    Covers ``bootstrap.py:467``. Construction: rare-positive data
    (n=20, 1 positive) + a metric that explicitly raises on single-
    class resamples. With only 1 positive in n=20, most bootstrap
    resamples will draw 0 positives → metric raises → degenerate
    counter increments past 5% → assertion fires.
    """
    rng = np.random.default_rng(42)
    n = 20
    y = np.zeros(n, dtype=int)
    y[0] = 1  # exactly one positive
    a = rng.uniform(0, 1, size=n)
    b = rng.uniform(0, 1, size=n)
    with pytest.raises(ValueError, match="degenerate|no usable resamples"):
        paired_bootstrap_diff(y, a, b, _strict_single_class_metric, n_resamples=200, rng=42)


def test_paired_bootstrap_diff_shape_mismatch_raises() -> None:
    """``paired_bootstrap_diff`` raises ValueError on shape mismatch. Covers ``bootstrap.py:445``."""
    y = np.array([0, 1] * 10)
    a = np.array([0.5] * 20)
    b = np.array([0.5] * 19)  # one short
    with pytest.raises(ValueError, match="shapes mismatch"):
        paired_bootstrap_diff(y, a, b, pr_auc, n_resamples=100)


def test_paired_bootstrap_ece_diff_shape_mismatch_raises() -> None:
    """``paired_bootstrap_ece_diff`` raises ValueError on shape mismatch. Covers ``bootstrap.py:544``."""
    y = np.array([0, 1] * 10)
    a = np.array([0.5] * 20)
    b = np.array([0.5] * 19)
    with pytest.raises(ValueError, match="shapes mismatch"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=expected_calibration_error, n_resamples=100)


def test_paired_bootstrap_op_point_diff_rejects_identical_val_test_arrays() -> None:
    """``paired_bootstrap_op_point_diff`` raises ValueError when ``val_y is test_y``.

    Round 5 audit finding R5-F6e (Codex): the two-level resampler draws
    ``val_idx`` and ``test_idx`` independently, so passing the same array
    object for val and test causes ~63.2% overlap and silently violates the
    independence assumption that lets the CI absorb threshold-selection
    variance honestly. The identity guard catches the most common
    foot-shoot pattern at the API boundary (style invariant 1: no silent
    failures).
    """
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=50)
    s_a = rng.uniform(0, 1, size=50)
    s_b = rng.uniform(0, 1, size=50)

    def threshold_fn(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(MaxF1Selector().select(yt, ys).threshold)

    def f1_at(yt: np.ndarray, ys: np.ndarray, t: float) -> float:
        from eval_toolkit import metrics_at_threshold

        return float(metrics_at_threshold(yt, ys, t)["f1"])

    with pytest.raises(ValueError, match="val_y and test_y are the same array"):
        paired_bootstrap_op_point_diff(
            val_y=y,
            val_score_a=s_a,
            val_score_b=s_b,
            test_y=y,
            test_score_a=s_a,
            test_score_b=s_b,
            threshold_fn=threshold_fn,
            metric_fn=f1_at,
            n_resamples=50,
            rng=42,
        )

    # Disjoint slices succeed — the guard is identity-only, not a content check.
    half = len(y) // 2
    out = paired_bootstrap_op_point_diff(
        val_y=y[:half],
        val_score_a=s_a[:half],
        val_score_b=s_b[:half],
        test_y=y[half:],
        test_score_a=s_a[half:],
        test_score_b=s_b[half:],
        threshold_fn=threshold_fn,
        metric_fn=f1_at,
        n_resamples=50,
        rng=42,
    )
    assert hasattr(out, "delta")


def test_bootstrap_ci_invalid_confidence_raises() -> None:
    """``bootstrap_ci`` raises ValueError when ``confidence`` is out of (0, 1).

    Covers ``bootstrap.py:258``.
    """
    y = np.array([0, 1] * 10)
    s = np.array([0.5] * 20)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100, confidence=1.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100, confidence=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ─────────────────────────────────────────────────────────────────────────────
# Silent-NaN hardening (#96, v1.9.0): NaN-returning metrics in the paired
# bootstraps.
#
# Pre-#96, a NaN-returning metric flowed into np.quantile, yielding NaN bounds
# and `overlaps_zero = NaN <= 0.0 <= NaN` == False — silently reading
# "statistically significant". Now: NaN resample deltas count as degenerate
# draws (>5% triggers the existing gate, ≤5% are filtered like raising draws),
# a non-finite full-data delta raises immediately, and a non-finite-CI-bounds
# raise remains in each constructor as an unreachable-by-construction backstop.
# ─────────────────────────────────────────────────────────────────────────────


def _paired_nan_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    return y, a, b


def test_paired_bootstrap_diff_nan_resamples_hit_degenerate_gate() -> None:
    y, a, b = _paired_nan_inputs()

    def metric(y_t: np.ndarray, y_s: np.ndarray) -> float:
        # Finite on the full arrays (delta_point), NaN on every row resample.
        for o in (a, b):
            if y_s.shape == o.shape and np.array_equal(y_s, o):
                return 0.5
        return float("nan")

    with pytest.raises(ValueError, match="degenerate"):
        paired_bootstrap_diff(y, a, b, metric, n_resamples=50, rng=0)


def test_paired_bootstrap_diff_non_finite_delta_point_raises() -> None:
    y, a, b = _paired_nan_inputs()

    def metric(y_t: np.ndarray, y_s: np.ndarray) -> float:
        # NaN on the full arrays only — the delta_point guard must fire.
        for o in (a, b):
            if y_s.shape == o.shape and np.array_equal(y_s, o):
                return float("nan")
        return 0.5

    with pytest.raises(ValueError, match="non-finite delta"):
        paired_bootstrap_diff(y, a, b, metric, n_resamples=50, rng=0)


def test_paired_bootstrap_ece_diff_nan_resamples_hit_degenerate_gate() -> None:
    y, a, b = _paired_nan_inputs()

    def ece_fn(y_t: np.ndarray, y_s: np.ndarray, n_bins: int) -> float:
        for o in (a, b):
            if y_s.shape == o.shape and np.allclose(y_s, o):
                return 0.5
        return float("nan")

    with pytest.raises(ValueError, match="degenerate"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=ece_fn, n_resamples=50, rng=0)


def test_paired_bootstrap_ece_diff_non_finite_delta_point_raises() -> None:
    y, a, b = _paired_nan_inputs()

    def ece_fn(y_t: np.ndarray, y_s: np.ndarray, n_bins: int) -> float:
        for o in (a, b):
            if y_s.shape == o.shape and np.allclose(y_s, o):
                return float("nan")
        return 0.5

    with pytest.raises(ValueError, match="non-finite delta"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=ece_fn, n_resamples=50, rng=0)


def test_paired_bootstrap_op_point_diff_nan_resamples_hit_degenerate_gate() -> None:
    val_y, val_a, val_b = _paired_nan_inputs()
    test_y, test_a, test_b = _paired_nan_inputs()

    def threshold_fn(y_t: np.ndarray, y_s: np.ndarray) -> float:
        return 0.5

    def metric_fn(y_t: np.ndarray, y_s: np.ndarray, thr: float) -> float:
        # Finite on the full test arrays (delta_point), NaN on every resample.
        for o in (test_a, test_b):
            if y_s.shape == o.shape and np.array_equal(y_s, o):
                return 0.5
        return float("nan")

    with pytest.raises(RuntimeError, match="degenerate"):
        paired_bootstrap_op_point_diff(
            val_y,
            val_a,
            val_b,
            test_y,
            test_a,
            test_b,
            threshold_fn,
            metric_fn,
            n_resamples=50,
            rng=0,
        )


def test_paired_bootstrap_op_point_diff_non_finite_delta_point_raises() -> None:
    val_y, val_a, val_b = _paired_nan_inputs()
    test_y, test_a, test_b = _paired_nan_inputs()

    def threshold_fn(y_t: np.ndarray, y_s: np.ndarray) -> float:
        return 0.5

    def metric_fn(y_t: np.ndarray, y_s: np.ndarray, thr: float) -> float:
        for o in (test_a, test_b):
            if y_s.shape == o.shape and np.array_equal(y_s, o):
                return float("nan")
        return 0.5

    with pytest.raises(ValueError, match="non-finite delta"):
        paired_bootstrap_op_point_diff(
            val_y,
            val_a,
            val_b,
            test_y,
            test_a,
            test_b,
            threshold_fn,
            metric_fn,
            n_resamples=50,
            rng=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# v1.9.0 pre-tag review completion: bootstrap_ci's own silent-NaN gaps
# (S1 studentized step, S2 method-gated bounds check, S3 unguarded point).
# ─────────────────────────────────────────────────────────────────────────────


def _bootstrap_ci_nan_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    s = rng.normal(size=40)
    return y, s


def test_bootstrap_ci_non_finite_point_estimate_raises() -> None:
    """S3: a metric that is NaN on the full data raises instead of a silent NaN point."""
    y, s = _bootstrap_ci_nan_inputs()

    def metric(y_t: np.ndarray, y_s: np.ndarray) -> float:
        if y_s.shape == s.shape and np.array_equal(y_s, s):
            return float("nan")
        return 0.5

    with pytest.raises(ValueError, match="non-finite point estimate"):
        bootstrap_ci(y, s, metric, n_resamples=50, rng=0)


def test_bootstrap_ci_percentile_nan_resamples_raise() -> None:
    """S2: percentile-method NaN bounds raise (previously only BCa was checked)."""
    y, s = _bootstrap_ci_nan_inputs()

    def metric(y_t: np.ndarray, y_s: np.ndarray) -> float:
        # Finite on the full arrays (point estimate), NaN on every resample.
        if y_s.shape == s.shape and np.array_equal(y_s, s):
            return 0.5
        return float("nan")

    with pytest.raises(ValueError, match="non-finite CI bounds"):
        bootstrap_ci(y, s, metric, n_resamples=50, rng=0, method="percentile")


def test_bootstrap_ci_studentized_nan_resamples_hit_degenerate_gate() -> None:
    """S1: NaN outer statistics count as degenerate draws in the studentized path.

    Pre-fix a NaN theta_b passed the 95%-valid gate and poisoned the pivots
    into a silent all-NaN BootstrapCI with zero warnings.
    """
    y, s = _bootstrap_ci_nan_inputs()

    def metric(y_t: np.ndarray, y_s: np.ndarray) -> float:
        if y_s.shape == s.shape and np.array_equal(y_s, s):
            return 0.5
        return float("nan")

    with pytest.raises(ValueError, match="degenerate|non-finite"):
        bootstrap_ci(y, s, metric, n_resamples=50, rng=0, method="studentized")
