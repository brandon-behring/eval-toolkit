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
    cross_validate_metric,
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
)
from eval_toolkit.thresholds import MaxF1Selector


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
    assert set(d.keys()) == {"point_estimate", "ci_95", "confidence", "n_resamples", "method"}
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
        return MaxF1Selector().select(yt, ys).threshold

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


@pytest.mark.unit
def test_paired_bootstrap_overlaps_zero_inclusive_on_degenerate_ci() -> None:
    """A zero-width CI at zero must report overlaps_zero=True (inclusive bounds).

    Regression test for the case where two scorers produce identical scores
    (delta=0 always, ci_low=ci_high=0). The semantic claim "0 ∈ [ci_low, ci_high]"
    is True and must round-trip through the dataclass.
    """
    n = 50
    y = np.array([0] * 5 + [1] * (n - 5), dtype=int)
    s_const = np.zeros(n, dtype=float)
    diff = paired_bootstrap_diff(y, s_const, s_const, pr_auc, n_resamples=100, seed=0)
    assert diff.delta == 0.0
    assert diff.ci_low == 0.0
    assert diff.ci_high == 0.0
    assert diff.overlaps_zero is True


@pytest.mark.unit
def test_paired_bootstrap_overlaps_zero_at_lower_boundary() -> None:
    """overlaps_zero is True when ci_low == 0 exactly (inclusive boundary)."""
    fake = PairedBootstrapCI(
        delta=0.05,
        ci_low=0.0,
        ci_high=0.10,
        overlaps_zero=(0.0 <= 0.0 <= 0.10),
        confidence=0.95,
        n_resamples=1000,
    )
    assert fake.overlaps_zero is True


# ---------------------------------------------------------------------------
# v0.8.1 + v0.8.2: bootstrap-diagnostic regression tests
# ---------------------------------------------------------------------------
# These cover the `first_failure` capture that replaced silent
# contextlib.suppress in `_bootstrap_t_ci` and `cross_validate_metric`
# (bootstrap.py:336 and :1116). A guard-rail raise should always quote
# the underlying exception so users can fix the real upstream problem.


@pytest.mark.unit
def test_cross_validate_metric_quotes_underlying_failure() -> None:
    """v0.8.1: >50% degenerate-folds raise must quote the first underlying exc."""

    def evil_metric(y: np.ndarray, s: np.ndarray) -> float:
        raise RuntimeError("synthetic-failure-XYZ")

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    s = rng.uniform(0, 1, size=40)
    with pytest.raises(ValueError) as excinfo:
        cross_validate_metric(y, s, metric=evil_metric, k=5, stratified=True)
    msg = str(excinfo.value)
    assert "5/5 folds raised" in msg
    assert "first underlying failure" in msg
    assert "RuntimeError: synthetic-failure-XYZ" in msg


@pytest.mark.unit
def test_bootstrap_t_quotes_underlying_failure() -> None:
    """v0.8.1: studentized bootstrap raise must quote the underlying exc.

    Realistic case: rare-positive small slice → most bootstrap resamples are
    all-negative, and a `pr_auc`-style metric raises on single-class data.
    """

    def picky_metric(y: np.ndarray, s: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            raise RuntimeError("single-class-slice-XYZ")
        return float(s.mean())

    n = 15
    y = np.zeros(n, dtype=int)
    y[0] = 1
    s = np.linspace(0.0, 1.0, n)
    with pytest.raises(ValueError) as excinfo:
        bootstrap_ci(y, s, metric=picky_metric, n_resamples=200, method="studentized")
    msg = str(excinfo.value)
    assert "degenerate" in msg
    assert "first underlying failure" in msg
    assert "RuntimeError: single-class-slice-XYZ" in msg


@pytest.mark.unit
def test_bootstrap_t_inner_loo_failure_surfaced() -> None:
    """v0.8.2: cover the *inner* LOO first_failure capture (bootstrap.py:343-345).

    Construct a metric that succeeds on the FULL resample but fails on every
    leave-one-out subset → valid.sum() < 2, theta_stars[b] stays NaN, eventually
    triggering the n_valid raise with `first_failure` surfaced from the inner
    loop (not the outer try/except).
    """
    n = 20

    def n_strict_metric(y: np.ndarray, s: np.ndarray) -> float:
        if len(y) < n:
            raise RuntimeError("inner-loo-failure-DEF")
        return float(s.mean())

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n)
    s = rng.uniform(0, 1, size=n)
    with pytest.raises(ValueError) as excinfo:
        bootstrap_ci(y, s, metric=n_strict_metric, n_resamples=50, method="studentized")
    msg = str(excinfo.value)
    assert "degenerate" in msg
    assert "first underlying failure" in msg
    # Crucially: the INNER LOO exception (not the outer one) is what's surfaced.
    assert "RuntimeError: inner-loo-failure-DEF" in msg
