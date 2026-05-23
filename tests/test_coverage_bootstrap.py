"""Coverage-targeted tests for bootstrap error paths and defensive code.

Extracted from the v0.27.x-era ``test_coverage_gap.py`` during the
v0.30.1 hygiene split — every assertion preserved verbatim; only the
file boundary changed.

Pairs with the happy-path coverage in ``test_bootstrap_unit.py``, the
invariants in ``test_bootstrap_props.py``, and the edge cases in
``test_bootstrap_edge_cases.py``. Targets input-validation error
branches, ``to_dict`` schema lock-ins, and the v0.4/v0.5 additions
(studentized bootstrap-t, cv_clt_ci, cross_validate_metric).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    BootstrapCI,
    PairedBootstrapCI,
    bootstrap_ci,
    mde_from_ci,
    paired_bootstrap_diff,
    paired_bootstrap_ece_diff,
    paired_bootstrap_op_point_diff,
    paired_mde,
)
from eval_toolkit.metrics import (
    expected_calibration_error,
    pr_auc,
)


@pytest.fixture
def informative() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = np.clip(y + rng.normal(0, 0.3, size=200), 0, 1)
    return y, s


# ---------------------------------------------------------------------------
# bootstrap: validation paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_ci_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        bootstrap_ci(np.zeros(20, dtype=int), np.zeros(10), pr_auc)


@pytest.mark.unit
def test_bootstrap_ci_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        bootstrap_ci(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), pr_auc)


@pytest.mark.unit
def test_bootstrap_ci_rejects_invalid_confidence() -> None:
    y = np.array([0, 1] * 10)
    s = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(y, s, pr_auc, confidence=1.5)


@pytest.mark.unit
def test_paired_bootstrap_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        paired_bootstrap_diff(
            np.zeros(20, dtype=int),
            np.zeros(10),
            np.zeros(20),
            pr_auc,
        )


@pytest.mark.unit
def test_paired_bootstrap_diff_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        paired_bootstrap_diff(
            np.array([0, 1, 0]),
            np.array([0.1, 0.5, 0.2]),
            np.array([0.2, 0.6, 0.3]),
            pr_auc,
        )


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        paired_bootstrap_ece_diff(
            np.zeros(20, dtype=int),
            np.zeros(10),
            np.zeros(20),
            ece_fn=expected_calibration_error,
        )


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        paired_bootstrap_ece_diff(
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            ece_fn=expected_calibration_error,
        )


@pytest.mark.unit
def test_paired_bootstrap_op_point_diff_rejects_shape_mismatch(
    informative: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = informative
    val_y = y[:100]
    val_a = s[:100]
    with pytest.raises(ValueError, match="val shape mismatch"):
        paired_bootstrap_op_point_diff(
            val_y=val_y,
            val_score_a=val_a,
            val_score_b=val_a[:50],
            test_y=y[100:],
            test_score_a=s[100:],
            test_score_b=s[100:],
            threshold_fn=lambda yt, ys: 0.5,
            metric_fn=lambda yt, ys, t: 0.0,
        )


@pytest.mark.unit
def test_mde_from_ci_rejects_zero_width() -> None:
    fake = PairedBootstrapCI(
        delta=0.0,
        ci_low=0.05,
        ci_high=0.05,
        overlaps_zero=False,
        confidence=0.95,
        n_resamples=100,
    )
    with pytest.raises(RuntimeError, match="non-positive"):
        mde_from_ci(fake)


@pytest.mark.unit
def test_paired_mde_returns_estimate(
    informative: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = informative
    s_b = s + 0.05 * y.astype(float)
    est = paired_mde(y, s, s_b, pr_auc, n_resamples=50, rng=0)
    assert est.mde >= 0
    assert est.n == len(y)


@pytest.mark.unit
def test_bootstrap_ci_to_dict_schema() -> None:
    """v0.48 §5B: schema renamed from {ci_95: [l, h]} to {low: l, high: h}."""
    ci = BootstrapCI(0.5, 0.4, 0.6, 0.95, 100, "BCa")
    d = ci.to_dict()
    assert set(d.keys()) == {"point", "low", "high", "confidence", "n_resamples", "method"}


@pytest.mark.unit
def test_paired_bootstrap_ci_to_dict_schema() -> None:
    """v0.48 §5B: PairedBootstrapCI gets the same rewrite."""
    pci = PairedBootstrapCI(0.05, 0.02, 0.08, False, 0.95, 100)
    d = pci.to_dict()
    assert set(d.keys()) == {"delta", "low", "high", "overlaps_zero", "confidence", "n_resamples"}


# ---------------------------------------------------------------------------
# v0.4.0 C2: studentized bootstrap-t
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.slow
def test_bootstrap_ci_studentized_runs() -> None:
    """method='studentized' returns a valid BootstrapCI."""
    rng = np.random.default_rng(0)
    n = 60  # smaller n so jackknife is fast
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    ci = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="studentized", rng=42)
    assert ci.method == "studentized"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high
    assert ci.ci_high - ci.ci_low > 0  # non-degenerate


@pytest.mark.unit
@pytest.mark.slow
def test_bootstrap_ci_studentized_deterministic() -> None:
    """Same seed → identical studentized CI."""
    rng = np.random.default_rng(0)
    n = 60
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    ci1 = bootstrap_ci(y, s, pr_auc, n_resamples=80, method="studentized", rng=7)
    ci2 = bootstrap_ci(y, s, pr_auc, n_resamples=80, method="studentized", rng=7)
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high


# ---------------------------------------------------------------------------
# v0.4.0 C3: cv_clt_ci helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cv_clt_ci_known_value() -> None:
    """CV-CLT CI for fixed inputs equals the closed-form Bayle 2020 formula."""
    from eval_toolkit.bootstrap import cv_clt_ci

    # 5-fold CV PR-AUC: mean=0.82, std (ddof=1)≈0.0245, z_{0.975}=1.96
    folds = np.array([0.83, 0.81, 0.85, 0.79, 0.82])
    ci = cv_clt_ci(folds, confidence=0.95)
    assert ci.method == "cv_clt"
    assert ci.n_resamples == 5
    assert ci.point_estimate == pytest.approx(0.82, abs=1e-9)
    expected_margin = 1.959963984540054 * float(np.std(folds, ddof=1)) / np.sqrt(5)
    assert ci.ci_low == pytest.approx(0.82 - expected_margin, abs=1e-9)
    assert ci.ci_high == pytest.approx(0.82 + expected_margin, abs=1e-9)


@pytest.mark.unit
def test_cv_clt_ci_validates() -> None:
    from eval_toolkit.bootstrap import cv_clt_ci

    with pytest.raises(ValueError, match="≥ 2 entries"):
        cv_clt_ci(np.array([0.5]))
    with pytest.raises(ValueError, match="NaN or inf"):
        cv_clt_ci(np.array([0.5, np.nan, 0.6]))
    with pytest.raises(ValueError, match="confidence"):
        cv_clt_ci(np.array([0.5, 0.6, 0.7]), confidence=0.0)


@pytest.mark.unit
def test_cv_clt_ci_widens_with_variance() -> None:
    """Higher across-fold variance → wider CI at fixed K."""
    from eval_toolkit.bootstrap import cv_clt_ci

    tight = cv_clt_ci(np.array([0.80, 0.81, 0.79, 0.80, 0.81]))
    wide = cv_clt_ci(np.array([0.70, 0.90, 0.60, 0.95, 0.80]))
    assert (wide.ci_high - wide.ci_low) > (tight.ci_high - tight.ci_low)


# ---------------------------------------------------------------------------
# v0.5.0 C1: cross_validate_metric eval-only orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cross_validate_metric_returns_per_fold_values() -> None:
    from eval_toolkit.bootstrap import cross_validate_metric

    rng = np.random.default_rng(42)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.3, n), 0, 1)
    folds = cross_validate_metric(y, s, metric=pr_auc, k=5, rng=42)
    assert folds.shape == (5,)
    valid = folds[~np.isnan(folds)]
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


@pytest.mark.unit
def test_cross_validate_metric_pairs_with_cv_clt_ci() -> None:
    """End-to-end: cross_validate_metric → cv_clt_ci."""
    from eval_toolkit.bootstrap import cross_validate_metric, cv_clt_ci

    rng = np.random.default_rng(0)
    n = 300
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    folds = cross_validate_metric(y, s, metric=pr_auc, k=5, rng=0)
    valid = folds[~np.isnan(folds)]
    assert valid.size >= 2  # Need ≥ 2 folds for cv_clt_ci
    ci = cv_clt_ci(valid)
    assert ci.method == "cv_clt"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_cross_validate_metric_validates() -> None:
    from eval_toolkit.bootstrap import cross_validate_metric

    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="shape"):
        cross_validate_metric(y, np.array([0.5]), metric=pr_auc)
    with pytest.raises(ValueError, match="k must be"):
        cross_validate_metric(y, s, metric=pr_auc, k=1)
    with pytest.raises(ValueError, match="exceeds n"):
        cross_validate_metric(y, s, metric=pr_auc, k=10)
