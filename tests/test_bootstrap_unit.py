"""Unit tests for bootstrap CIs.

Adapted from prompt_injection_detector/tests/test_bootstrap.py.
Covers BCa + percentile per-condition CIs, paired-difference CIs,
two-level operating-point bootstrap, MDE estimates, and the DI
contract for paired_bootstrap_ece_diff.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    BootstrapCI,
    MDEEstimate,
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
    metrics_at_threshold,
    pr_auc,
    select_threshold,
)


@pytest.fixture
def informative_signal() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.3, size=300)
    y_score = y_true + rng.normal(0, 0.3, size=300)
    y_score = np.clip(y_score, 0, 1)
    return y_true.astype(int), y_score


@pytest.mark.unit
def test_bootstrap_ci_contains_point_estimate(
    informative_signal: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_score = informative_signal
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, method="BCa", seed=42)
    assert isinstance(ci, BootstrapCI)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_bootstrap_ci_width_shrinks_with_n() -> None:
    rng = np.random.default_rng(0)
    big_y = rng.binomial(1, 0.3, size=2000)
    big_s = big_y + rng.normal(0, 0.3, size=2000)
    big_s = np.clip(big_s, 0, 1)
    small_y = big_y[:60]
    small_s = big_s[:60]
    ci_big = bootstrap_ci(big_y, big_s, pr_auc, n_resamples=200, seed=42)
    ci_small = bootstrap_ci(small_y, small_s, pr_auc, n_resamples=200, seed=42)
    assert (ci_big.ci_high - ci_big.ci_low) < (ci_small.ci_high - ci_small.ci_low)


@pytest.mark.unit
def test_paired_bootstrap_diff_detects_real_lift() -> None:
    """When B is genuinely better than A, the CI excludes zero."""
    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.3, size=400).astype(int)
    score_a = rng.uniform(0, 1, size=400)
    score_b = score_a + 0.4 * y_true
    score_b = np.clip(score_b, 0, 1)
    diff = paired_bootstrap_diff(y_true, score_a, score_b, pr_auc, n_resamples=300, seed=42)
    assert isinstance(diff, PairedBootstrapCI)
    assert diff.delta > 0
    assert not diff.overlaps_zero


@pytest.mark.unit
def test_paired_bootstrap_diff_overlaps_zero_when_no_lift() -> None:
    """Identical scorers give Δ ≈ 0."""
    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.3, size=400).astype(int)
    score = y_true + rng.normal(0, 0.3, size=400)
    score = np.clip(score, 0, 1)
    diff = paired_bootstrap_diff(y_true, score, score, pr_auc, n_resamples=300, seed=42)
    assert diff.delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_paired_bootstrap_handles_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes"):
        paired_bootstrap_diff(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0.5, 0.5]),
            pr_auc,
        )


@pytest.mark.unit
def test_bootstrap_too_small_n_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        bootstrap_ci(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.5]), pr_auc, seed=42)


@pytest.mark.unit
def test_percentile_method_works(informative_signal: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = informative_signal
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, method="percentile", seed=42)
    assert ci.method == "percentile"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_paired_bootstrap_overlaps_zero_field() -> None:
    rng = np.random.default_rng(11)
    y_true = rng.binomial(1, 0.5, size=200).astype(int)
    score_a = rng.uniform(0, 1, size=200)
    score_b = score_a + rng.normal(0, 0.05, size=200)
    score_b = np.clip(score_b, 0, 1)
    diff = paired_bootstrap_diff(y_true, score_a, score_b, pr_auc, n_resamples=200, seed=42)
    expected = bool(diff.ci_low < 0 < diff.ci_high)
    assert diff.overlaps_zero == expected


@pytest.mark.unit
def test_bootstrap_ci_to_dict_schema(informative_signal: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = informative_signal
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, seed=42)
    d = ci.to_dict()
    assert set(d.keys()) == {"mean", "ci_95", "confidence", "n_resamples", "method"}
    assert isinstance(d["ci_95"], list)


@pytest.mark.unit
def test_paired_bootstrap_diff_to_dict_schema() -> None:
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = y + rng.normal(0, 0.3, size=200)
    diff = paired_bootstrap_diff(y, s, s, pr_auc, n_resamples=100, seed=42)
    d = diff.to_dict()
    assert set(d.keys()) == {"delta", "ci_95", "overlaps_zero", "confidence", "n_resamples"}


@pytest.mark.unit
def test_paired_bootstrap_op_point_diff_runs(
    informative_signal: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two-level bootstrap returns a CI on operating-point Δ."""
    y_val, s_val = informative_signal
    y_test, s_test = informative_signal

    def threshold_fn(yt: np.ndarray, ys: np.ndarray) -> float:
        return select_threshold(yt, ys, criterion="max_f1").threshold

    def metric_fn(yt: np.ndarray, ys: np.ndarray, t: float) -> float:
        return float(metrics_at_threshold(yt, ys, t)["f1"])

    diff = paired_bootstrap_op_point_diff(
        y_val,
        s_val,
        s_val,
        y_test,
        s_test,
        s_test,
        threshold_fn=threshold_fn,
        metric_fn=metric_fn,
        n_resamples=100,
        seed=42,
    )
    # Same scorer paired with itself: delta should be 0.
    assert diff.delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_di_contract(
    informative_signal: tuple[np.ndarray, np.ndarray],
) -> None:
    """paired_bootstrap_ece_diff accepts any ece_fn callable."""
    y_true, y_score = informative_signal

    diff = paired_bootstrap_ece_diff(
        y_true,
        y_score,
        y_score,
        ece_fn=expected_calibration_error,
        n_resamples=100,
        seed=42,
    )
    assert isinstance(diff, PairedBootstrapCI)
    # Same scorer: delta should be 0.
    assert diff.delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_with_custom_ece_fn(
    informative_signal: tuple[np.ndarray, np.ndarray],
) -> None:
    """Custom ece_fn callable can be injected (DI working as documented)."""
    y_true, y_score = informative_signal
    call_count = [0]

    def my_ece(y: np.ndarray, s: np.ndarray, n_bins: int) -> float:
        call_count[0] += 1
        return float(expected_calibration_error(y, s, n_bins))

    diff = paired_bootstrap_ece_diff(
        y_true,
        y_score,
        y_score,
        ece_fn=my_ece,
        n_resamples=50,
        seed=42,
    )
    assert call_count[0] > 0
    assert isinstance(diff, PairedBootstrapCI)


@pytest.mark.unit
def test_paired_mde_runs() -> None:
    """paired_mde returns a sensible MDE for non-identical scorers."""
    rng = np.random.default_rng(7)
    n = 400
    y_true = rng.binomial(1, 0.3, size=n).astype(int)
    score_a = rng.uniform(0, 1, size=n)
    score_b = score_a + 0.3 * y_true  # B is genuinely better
    score_b = np.clip(score_b, 0, 1)
    est = paired_mde(y_true, score_a, score_b, pr_auc, n_resamples=200, seed=42)
    assert isinstance(est, MDEEstimate)
    assert est.alpha == 0.05
    assert est.power == 0.80
    assert est.n == n
    assert est.mde > 0


@pytest.mark.unit
def test_mde_from_ci_validates() -> None:
    """mde_from_ci rejects invalid alpha/power."""
    fake = PairedBootstrapCI(
        delta=0.1,
        ci_low=0.05,
        ci_high=0.15,
        overlaps_zero=False,
        confidence=0.95,
        n_resamples=1000,
    )
    with pytest.raises(ValueError, match="alpha"):
        mde_from_ci(fake, alpha=0.0)
    with pytest.raises(ValueError, match="power"):
        mde_from_ci(fake, alpha=0.05, power=1.0)
