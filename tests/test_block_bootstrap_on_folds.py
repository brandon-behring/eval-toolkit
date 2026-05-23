"""Tests for `eval_toolkit.bootstrap.block_bootstrap_on_folds` (closes #21).

CV-aware sibling primitive to `cv_clt_ci`. The block bootstrap resamples
K folds with replacement; the percentile-CI on the resample-means is more
*conservative* under fold-level non-exchangeability than the Bayle 2020
CV-CLT correction. Used by prompt-injection-detection-submission's A-008
sensitivity check (compares the half-widths of the two CIs; LODO
non-exchangeability dominates when ratio > 1.5).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit import block_bootstrap_on_folds, cv_clt_ci, mde_from_ci


@pytest.mark.unit
def test_block_bootstrap_on_folds_returns_correct_shape_and_method() -> None:
    """Returns a BootstrapCI with method='block_bootstrap' and supplied n_resamples."""
    folds = np.array([0.83, 0.81, 0.85, 0.79, 0.84])
    ci = block_bootstrap_on_folds(folds, n_resamples=1000, rng=42)
    assert ci.method == "block_bootstrap"
    assert ci.n_resamples == 1000
    assert ci.confidence == 0.95
    assert ci.point_estimate == pytest.approx(folds.mean())
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_block_bootstrap_on_folds_wider_when_fold_variance_high() -> None:
    """Higher per-fold variance produces a wider CI (monotonic relationship)."""
    low_var = np.array([0.80, 0.81, 0.80, 0.79, 0.80])
    high_var = np.array([0.60, 0.90, 0.50, 0.95, 0.70])
    ci_low = block_bootstrap_on_folds(low_var, n_resamples=2000, rng=42)
    ci_high = block_bootstrap_on_folds(high_var, n_resamples=2000, rng=42)
    width_low = ci_low.ci_high - ci_low.ci_low
    width_high = ci_high.ci_high - ci_high.ci_low
    assert width_high > width_low, (
        f"high-variance fold input should produce wider CI; "
        f"got width_low={width_low}, width_high={width_high}"
    )


@pytest.mark.unit
def test_block_bootstrap_on_folds_seed_reproducibility() -> None:
    """Same seed → bit-for-bit-identical CI."""
    folds = np.array([0.7, 0.8, 0.6, 0.9, 0.75])
    ci_a = block_bootstrap_on_folds(folds, n_resamples=500, rng=42)
    ci_b = block_bootstrap_on_folds(folds, n_resamples=500, rng=42)
    assert (ci_a.ci_low, ci_a.ci_high) == (ci_b.ci_low, ci_b.ci_high)


@pytest.mark.unit
def test_block_bootstrap_on_folds_rejects_k1() -> None:
    """K=1 (1-D array with one entry) raises ValueError."""
    with pytest.raises(ValueError, match=r"K ≥ 2"):
        block_bootstrap_on_folds(np.array([0.5]))


@pytest.mark.unit
def test_block_bootstrap_on_folds_rejects_non_1d() -> None:
    """Non-1D fold_metrics raises ValueError."""
    with pytest.raises(ValueError, match=r"1-D"):
        block_bootstrap_on_folds(np.array([[0.5, 0.6], [0.7, 0.8]]))


@pytest.mark.unit
def test_block_bootstrap_on_folds_rejects_non_finite() -> None:
    """NaN or Inf in fold_metrics raises ValueError."""
    with pytest.raises(ValueError, match=r"all-finite"):
        block_bootstrap_on_folds(np.array([0.5, np.nan, 0.7]))


@pytest.mark.unit
def test_block_bootstrap_on_folds_rejects_invalid_confidence() -> None:
    """confidence outside (0, 1) raises ValueError."""
    folds = np.array([0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match=r"confidence"):
        block_bootstrap_on_folds(folds, confidence=1.0)
    with pytest.raises(ValueError, match=r"confidence"):
        block_bootstrap_on_folds(folds, confidence=0.0)


@pytest.mark.unit
def test_block_bootstrap_on_folds_round_trips_with_mde_from_ci() -> None:
    """End-to-end chain: block_bootstrap_on_folds → mde_from_ci works.

    This is the v0.34.0 'generalized mde_from_ci' use case — accepts the
    BootstrapCI from block_bootstrap_on_folds via the new Union type.
    """
    folds = np.array([0.83, 0.81, 0.85, 0.79, 0.84])
    ci = block_bootstrap_on_folds(folds, n_resamples=2000, rng=42)
    mde = mde_from_ci(ci, alpha=0.05, power=0.80)
    assert mde.mde > 0
    assert mde.sigma_delta > 0
    assert mde.delta_observed == pytest.approx(ci.point_estimate)


@pytest.mark.unit
def test_block_bootstrap_on_folds_vs_cv_clt_ci_both_bracket_mean() -> None:
    """Both sibling primitives bracket the same mean on the same input.

    Both are statistically valid CIs under different assumptions (CV-CLT:
    fold exchangeability; block bootstrap: more conservative, non-parametric).
    Asserting both bracket the mean is the cross-primitive sanity check.
    """
    folds = np.array([0.7, 0.8, 0.65, 0.85, 0.75])
    ci_block = block_bootstrap_on_folds(folds, n_resamples=2000, rng=42)
    ci_clt = cv_clt_ci(folds, confidence=0.95)
    point = folds.mean()
    assert ci_block.ci_low <= point <= ci_block.ci_high
    assert ci_clt.ci_low <= point <= ci_clt.ci_high
