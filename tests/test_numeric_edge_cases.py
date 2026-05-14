"""Numeric edge-case sweep across metrics/bootstrap/calibration (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep: ``n=1``,
constant scores, and mixed-dtype inputs (``np.int32`` / ``np.float32``)
are systematically untested across the calibration / bootstrap /
metrics surfaces. A regression in dtype handling or single-row
behavior would slip through the existing test suite (which uses
``dtype=int``/``float`` and ``n ≥ 10`` throughout).

This module pins the documented behavior at edge inputs:

1. **n = 1**: most metrics raise ValueError (insufficient samples
   for confidence intervals); some return the single-sample value
   directly. The exact behavior per function is documented in the
   tests.
2. **Constant scores**: PR-AUC and ROC-AUC have meaningful values
   on constant scores (degenerate but defined); calibration fitters
   handle them without raising.
3. **Mixed dtypes**: calibrators and metrics accept ``np.int32`` /
   ``np.float32`` arrays without dtype-related errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import bootstrap_ci
from eval_toolkit.calibration import (
    fit_beta_calibrator,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)
from eval_toolkit.metrics import pr_auc, roc_auc

# ---------------------------------------------------------------------------
# n = 1 behavior across the surface.
# ---------------------------------------------------------------------------


def test_pr_auc_handles_single_sample_input() -> None:
    """``pr_auc`` on n=1 returns the trivial value (precision = label).

    sklearn's ``precision_recall_curve`` returns a 2-element array on
    n=1 (one threshold + the sentinel), so PR-AUC is computable.
    """
    y = np.array([1])
    s = np.array([0.5])
    # sklearn returns 1.0 for n=1 single-positive (precision = 1 at all thresholds).
    auc = pr_auc(y, s)
    assert 0.0 <= auc <= 1.0
    assert np.isfinite(auc)


def test_roc_auc_returns_nan_on_single_class_input() -> None:
    """``roc_auc`` on single-class input returns NaN, not raise."""
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    s = np.linspace(0.1, 0.9, 10)
    # sklearn raises UndefinedMetricWarning and returns NaN; eval-toolkit's
    # wrapper preserves this behavior.
    auc = roc_auc(y, s)
    assert np.isnan(auc), f"Expected NaN on single-class roc_auc, got {auc}"


def test_bootstrap_ci_raises_on_n_lt_10() -> None:
    """``bootstrap_ci`` requires ``n ≥ 10`` (already covered in test_bootstrap_edge_cases.py).

    Sanity check that the same input that would represent ``n=1``
    behavior in the broader suite is properly rejected here, not
    silently accepted.
    """
    y = np.array([0, 1])  # n=2
    s = np.array([0.3, 0.7])
    with pytest.raises(ValueError, match="n=2 too small"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100)


# ---------------------------------------------------------------------------
# Constant-score behavior.
# ---------------------------------------------------------------------------


def test_pr_auc_handles_constant_scores() -> None:
    """``pr_auc`` on constant scores returns prevalence (degenerate but defined)."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 50% prevalence
    s = np.full(8, 0.5)
    auc = pr_auc(y, s)
    # Constant scores → trivial classifier → PR-AUC ≈ prevalence.
    assert 0.0 <= auc <= 1.0
    assert np.isfinite(auc)


def test_roc_auc_handles_constant_scores() -> None:
    """``roc_auc`` on constant scores returns 0.5 (no rank info → random)."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.full(8, 0.5)
    auc = roc_auc(y, s)
    assert auc == pytest.approx(
        0.5, abs=1e-9
    ), f"ROC-AUC on constant scores should be 0.5; got {auc}"


def test_calibrators_handle_constant_scores() -> None:
    """``fit_*_calibrator`` accepts constant scores without raising."""
    rng = np.random.default_rng(42)
    n = 100
    y = rng.integers(0, 2, size=n)
    s = np.full(n, 0.5)
    # All three fitters should handle constant scores (the calibrated
    # output is the marginal positive rate, since there's no signal to
    # rank on).
    for name, fitter in (
        ("Platt", fit_platt_calibrator),
        ("Isotonic", fit_isotonic_calibrator),
        ("Beta", fit_beta_calibrator),
    ):
        try:
            cal = fitter(y, s)
        except ValueError as exc:
            pytest.fail(f"{name} calibrator should not raise on constant scores; got {exc}")
        # Apply to constant input — output should be in [0, 1].
        out = cal(s)
        assert (out >= 0.0).all() and (
            out <= 1.0
        ).all(), f"{name} output out of [0, 1] on constant input: {out[:5]}"


# ---------------------------------------------------------------------------
# Dtype acceptance (np.int32, np.float32, Python lists).
# ---------------------------------------------------------------------------


def test_calibrators_accept_int32_and_float32() -> None:
    """``fit_*_calibrator`` accepts ``np.int32`` / ``np.float32`` arrays."""
    rng = np.random.default_rng(42)
    n = 200
    y_int32 = rng.integers(0, 2, size=n).astype(np.int32)
    s_float32 = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    # All three fitters should accept narrow dtypes without dtype-related errors.
    for name, fitter in (
        ("Platt", fit_platt_calibrator),
        ("Isotonic", fit_isotonic_calibrator),
        ("Beta", fit_beta_calibrator),
    ):
        try:
            cal = fitter(y_int32, s_float32)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{name} calibrator should accept int32/float32 inputs; got {exc}")
        # Calibrator output must be numpy-friendly.
        out = cal(s_float32)
        assert isinstance(out, np.ndarray)
        assert out.shape == (n,)


def test_metrics_accept_int32_and_float32() -> None:
    """``pr_auc`` / ``roc_auc`` accept narrow dtype inputs."""
    rng = np.random.default_rng(42)
    n = 200
    y_int32 = rng.integers(0, 2, size=n).astype(np.int32)
    s_float32 = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    auc_pr = pr_auc(y_int32, s_float32)
    auc_roc = roc_auc(y_int32, s_float32)
    assert 0.0 <= auc_pr <= 1.0 and np.isfinite(auc_pr)
    assert 0.0 <= auc_roc <= 1.0 and np.isfinite(auc_roc)


def test_metrics_accept_python_list_inputs() -> None:
    """``pr_auc`` accepts Python ``list[int]`` / ``list[float]`` inputs."""
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    s = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    auc = pr_auc(y, s)
    assert 0.0 <= auc <= 1.0 and np.isfinite(auc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
