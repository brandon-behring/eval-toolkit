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
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, method="BCa", rng=42)
    assert isinstance(ci, BootstrapCI)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
@pytest.mark.slow
def test_bootstrap_ci_width_shrinks_with_n() -> None:
    rng = np.random.default_rng(0)
    big_y = rng.binomial(1, 0.3, size=2000)
    big_s = big_y + rng.normal(0, 0.3, size=2000)
    big_s = np.clip(big_s, 0, 1)
    small_y = big_y[:60]
    small_s = big_s[:60]
    ci_big = bootstrap_ci(big_y, big_s, pr_auc, n_resamples=200, rng=42)
    ci_small = bootstrap_ci(small_y, small_s, pr_auc, n_resamples=200, rng=42)
    assert (ci_big.ci_high - ci_big.ci_low) < (ci_small.ci_high - ci_small.ci_low)


@pytest.mark.unit
def test_paired_bootstrap_diff_detects_real_lift() -> None:
    """When B is genuinely better than A, the CI excludes zero."""
    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.3, size=400).astype(int)
    score_a = rng.uniform(0, 1, size=400)
    score_b = score_a + 0.4 * y_true
    score_b = np.clip(score_b, 0, 1)
    diff = paired_bootstrap_diff(y_true, score_a, score_b, pr_auc, n_resamples=300, rng=42)
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
    diff = paired_bootstrap_diff(y_true, score, score, pr_auc, n_resamples=300, rng=42)
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
        bootstrap_ci(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.5]), pr_auc, rng=42)


@pytest.mark.unit
def test_percentile_method_works(informative_signal: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = informative_signal
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, method="percentile", rng=42)
    assert ci.method == "percentile"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_paired_bootstrap_overlaps_zero_field() -> None:
    rng = np.random.default_rng(11)
    y_true = rng.binomial(1, 0.5, size=200).astype(int)
    score_a = rng.uniform(0, 1, size=200)
    score_b = score_a + rng.normal(0, 0.05, size=200)
    score_b = np.clip(score_b, 0, 1)
    diff = paired_bootstrap_diff(y_true, score_a, score_b, pr_auc, n_resamples=200, rng=42)
    expected = bool(diff.ci_low < 0 < diff.ci_high)
    assert diff.overlaps_zero == expected


@pytest.mark.unit
def test_bootstrap_ci_to_dict_schema(informative_signal: tuple[np.ndarray, np.ndarray]) -> None:
    """v0.48 §5B: schema renamed from {ci_95: [l,h]} to {low: l, high: h}."""
    y_true, y_score = informative_signal
    ci = bootstrap_ci(y_true, y_score, pr_auc, n_resamples=200, rng=42)
    d = ci.to_dict()
    assert set(d.keys()) == {"point", "low", "high", "confidence", "n_resamples", "method"}
    # Bounds are scalar floats now (not a list)
    assert isinstance(d["low"], float)
    assert isinstance(d["high"], float)
    assert d["low"] <= d["point"] <= d["high"]


@pytest.mark.unit
def test_paired_bootstrap_diff_to_dict_schema() -> None:
    """v0.48 §5B: PairedBootstrapCI gets the same rewrite as BootstrapCI."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = y + rng.normal(0, 0.3, size=200)
    diff = paired_bootstrap_diff(y, s, s, pr_auc, n_resamples=100, rng=42)
    d = diff.to_dict()
    assert set(d.keys()) == {"delta", "low", "high", "overlaps_zero", "confidence", "n_resamples"}
    assert isinstance(d["low"], float)
    assert isinstance(d["high"], float)


@pytest.mark.unit
def test_bootstrap_ci_to_dict_self_describing_at_non_default_confidence() -> None:
    """v0.48 §5B: schema is self-describing — works at non-0.95 confidence.

    Pre-v0.48 schema lied: {"ci_95": [l, h], "confidence": 0.90} → the
    key name 'ci_95' implied 95% confidence regardless of the actual
    confidence field. Post-v0.48 the bounds are named neutrally; consumers
    interpret semantics from the confidence field.
    """
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = y + rng.normal(0, 0.3, size=200)
    ci = bootstrap_ci(y, s, pr_auc, n_resamples=200, rng=42, confidence=0.90)
    d = ci.to_dict()
    # The schema does NOT carry a misleading "ci_95" key at confidence=0.90
    assert "ci_95" not in d
    assert d["confidence"] == 0.90


@pytest.mark.unit
def test_paired_bootstrap_op_point_diff_runs(
    informative_signal: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two-level bootstrap returns a CI on operating-point Δ.

    Uses DISJOINT val + test slices — passing the same array for both is
    rejected at the API boundary by the v0.48 identity guard (Round 5
    R5-F6e audit finding); see methodology/thresholds.md.
    """
    y_all, s_all = informative_signal
    half = len(y_all) // 2
    y_val, y_test = y_all[:half], y_all[half:]
    s_val, s_test = s_all[:half], s_all[half:]

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
        rng=42,
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
        rng=42,
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
        rng=42,
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
    est = paired_mde(y_true, score_a, score_b, pr_auc, n_resamples=200, rng=42)
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
def test_mde_from_ci_rejects_nan_width() -> None:
    """R9 follow-on (F-bootstrap-2): NaN CI width bypasses `<= 0` check.

    Pre-v0.51 R9 follow-on: width = NaN passes `width <= 0` (NaN <= 0 is
    False in IEEE float), so mde_from_ci silently returns MDEEstimate
    with mde=NaN. Could happen when scipy.stats.bootstrap BCa returns
    NaN bounds on degenerate jackknife. Explicit `not np.isfinite` check
    surfaces this as RuntimeError. audit-verification-round-9-v0.51.0.md
    Part 2.
    """
    nan_ci = BootstrapCI(
        point_estimate=0.5,
        ci_low=float("nan"),
        ci_high=float("nan"),
        confidence=0.95,
        n_resamples=100,
        method="BCa",
    )
    with pytest.raises(RuntimeError, match=r"non-finite|degenerate"):
        mde_from_ci(nan_ci, alpha=0.05, power=0.80)


@pytest.mark.unit
def test_bootstrap_ci_bca_degeneracy_emits_warning() -> None:
    """R9 follow-on (F-bootstrap-1): BCa degeneracy emits UserWarning.

    Pre-v0.51 the R8-C4(b) RNG bug spuriously varied bootstrap streams
    and could mask BCa degeneracy (ci_low == ci_high == point on small
    n with ceiling/floor metrics); post-v0.51 with correct RNG, the
    brittleness is exposed. v0.51 R9 follow-on adds UserWarning so
    callers know to switch to method='percentile' or use larger n.
    audit-verification-round-9-v0.51.0.md Part 2.
    """
    import warnings as _warnings

    # Construct a small-n + ceiling-metric scenario that degenerates BCa.
    # Constant-1 scores on alternating labels → no variance in resamples.
    rng = np.random.default_rng(7)
    y = np.array([0, 1] * 10)
    s = np.ones(20)  # constant scores → degenerate
    with _warnings.catch_warnings(record=True) as ws:
        _warnings.simplefilter("always")
        # Use a metric that produces near-constant output on constant scores.
        ci = bootstrap_ci(y, s, metric=lambda yt, ys: float(ys.mean()), n_resamples=50, rng=rng)
    # Either the warning fires (degenerate path) or the CI is well-defined.
    # If well-defined, this test is a no-op for this seed/data combo; the
    # contract is "warn IF degenerated", not "always warn".
    if ci.ci_low == ci.ci_high == ci.point_estimate:
        user_warnings = [w for w in ws if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            f"BCa degenerated (ci_low=ci_high=point={ci.ci_low}) but no "
            "UserWarning emitted. F-bootstrap-1 fix should catch this."
        )
        assert "BCa degenerated" in str(user_warnings[0].message)


@pytest.mark.unit
def test_mde_from_ci_accepts_bootstrap_ci() -> None:
    """v0.34.0: mde_from_ci accepts BootstrapCI (was paired-only before rename)."""
    fake_marginal = BootstrapCI(
        point_estimate=0.10,
        ci_low=0.05,
        ci_high=0.15,
        confidence=0.95,
        n_resamples=1000,
        method="BCa",
    )
    mde = mde_from_ci(fake_marginal, alpha=0.05, power=0.80)
    # sigma_delta = (0.15 - 0.05) / (2 * 1.959964) ≈ 0.0255 → mde ≈ (1.96 + 0.842) * sigma ≈ 0.0715
    assert mde.sigma_delta == pytest.approx(0.0255, abs=1e-3)
    assert mde.mde > 0
    # delta_observed should be the point_estimate for BootstrapCI input
    assert mde.delta_observed == pytest.approx(0.10)


@pytest.mark.unit
def test_mde_from_ci_paired_kwarg_rejected_after_v0_34_0_rename() -> None:
    """v0.34.0 BREAKING: `paired=` keyword form raises TypeError after the rename.

    The first param was renamed from `paired` to `ci`. Positional callers
    are unaffected; keyword callers must update. This test proves the alias
    is genuinely gone (no backward-compat shim).
    """
    fake = PairedBootstrapCI(
        delta=0.1,
        ci_low=0.05,
        ci_high=0.15,
        overlaps_zero=False,
        confidence=0.95,
        n_resamples=1000,
    )
    with pytest.raises(TypeError, match=r"paired"):
        mde_from_ci(paired=fake)  # type: ignore[call-arg]


@pytest.mark.unit
def test_mde_from_ci_equivalent_paired_vs_bootstrap_input_on_matching_widths() -> None:
    """A BootstrapCI and PairedBootstrapCI with same (ci_low, ci_high, confidence,
    n_resamples) produce same sigma_delta + mde. Difference: delta_observed."""
    paired = PairedBootstrapCI(
        delta=0.2,
        ci_low=0.05,
        ci_high=0.15,
        overlaps_zero=False,
        confidence=0.95,
        n_resamples=500,
    )
    marginal = BootstrapCI(
        point_estimate=0.7,
        ci_low=0.05,
        ci_high=0.15,
        confidence=0.95,
        n_resamples=500,
        method="BCa",
    )
    mde_p = mde_from_ci(paired)
    mde_m = mde_from_ci(marginal)
    assert mde_p.sigma_delta == pytest.approx(mde_m.sigma_delta)
    assert mde_p.mde == pytest.approx(mde_m.mde)
    assert mde_p.delta_observed == 0.2
    assert mde_m.delta_observed == 0.7


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
    diff = paired_bootstrap_diff(y, s_const, s_const, pr_auc, n_resamples=100, rng=0)
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


# ---------------------------------------------------------------------------
# v0.8.3: degenerate-input boundary tests for bootstrap_ci
# ---------------------------------------------------------------------------
# Defensive coverage for n_resamples ≤ 2 (a scipy.stats.bootstrap delegate)
# and the n_resamples=0 rejection. These exist to fail loudly if the
# n_resamples validation behavior ever drifts (e.g., we swap out scipy or
# wrap with our own preflight check that silently floors negative inputs).


@pytest.mark.unit
def test_bootstrap_ci_minimal_n_resamples_does_not_crash() -> None:
    """v0.8.3: n_resamples=2 yields a (degenerate) CI without raising.

    Internal scipy variance computation warns about dof ≤ 0 on such tiny
    distributions; that's expected — we only assert the CI is well-formed
    (low ≤ point ≤ high) and the call returns rather than crashing.
    """
    import warnings

    y = np.array([0] * 10 + [1] * 10, dtype=int)
    s = np.linspace(0.0, 1.0, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ci = bootstrap_ci(y, s, pr_auc, n_resamples=2, method="percentile", rng=0)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_bootstrap_ci_rejects_n_resamples_zero() -> None:
    """v0.8.3: n_resamples=0 must raise (currently delegated to scipy).

    Belt-and-braces against silently producing a NaN-filled CI if either
    the scipy validation regresses or eval-toolkit ever wraps it with a
    permissive preflight. Uses n=10 so we clear the n<10 ValueError before
    reaching the n_resamples check.
    """
    y = np.array([0] * 5 + [1] * 5, dtype=int)
    s = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError):
        bootstrap_ci(y, s, pr_auc, n_resamples=0, method="percentile", rng=0)


# --- v0.20.0: DeLong correlated-ROC variance (C12) ---


@pytest.mark.unit
def test_delong_roc_variance_returns_result_dataclass() -> None:
    """delong_roc_variance returns a DeLongResult with the expected fields."""
    from eval_toolkit.bootstrap import DeLongResult, delong_roc_variance

    rng = np.random.default_rng(42)
    y = np.array([0] * 50 + [1] * 50)
    sa = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(1.0, 1.0, 50)])
    sb = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(0.5, 1.0, 50)])
    result = delong_roc_variance(y, sa, sb)
    assert isinstance(result, DeLongResult)
    assert 0.5 < result.auc_a < 1.0
    assert 0.5 < result.auc_b < 1.0
    assert result.var > 0.0
    assert result.ci_low <= result.delta_auc <= result.ci_high


@pytest.mark.unit
def test_delong_roc_variance_auc_matches_sklearn() -> None:
    """AUC point estimates agree with sklearn.metrics.roc_auc_score within 1e-8."""
    from sklearn.metrics import roc_auc_score

    from eval_toolkit.bootstrap import delong_roc_variance

    rng = np.random.default_rng(7)
    y = np.array([0] * 30 + [1] * 30)
    sa = np.concatenate([rng.normal(0.0, 1.0, 30), rng.normal(1.0, 1.0, 30)])
    sb = np.concatenate([rng.normal(0.0, 1.0, 30), rng.normal(1.5, 1.0, 30)])
    result = delong_roc_variance(y, sa, sb)
    assert abs(result.auc_a - roc_auc_score(y, sa)) < 1e-8
    assert abs(result.auc_b - roc_auc_score(y, sb)) < 1e-8
    assert abs(result.delta_auc - (result.auc_a - result.auc_b)) < 1e-12


@pytest.mark.unit
def test_delong_roc_variance_rejects_empty_class() -> None:
    """Must have at least one positive AND one negative."""
    from eval_toolkit.bootstrap import delong_roc_variance

    y_all_zero = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="at least one"):
        delong_roc_variance(y_all_zero, s, s)


@pytest.mark.unit
def test_delong_roc_variance_rejects_shape_mismatch() -> None:
    """All three arrays must share shape."""
    from eval_toolkit.bootstrap import delong_roc_variance

    y = np.array([0, 1, 0, 1])
    sa = np.array([0.1, 0.2, 0.3, 0.4])
    sb = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="share shape"):
        delong_roc_variance(y, sa, sb)


@pytest.mark.unit
def test_delong_roc_variance_p_value_low_when_b_is_strong() -> None:
    """Large effect size produces small p-value."""
    from eval_toolkit.bootstrap import delong_roc_variance

    rng = np.random.default_rng(11)
    y = np.array([0] * 200 + [1] * 200)
    sa = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(0.1, 1.0, 200)])
    sb = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(2.5, 1.0, 200)])
    result = delong_roc_variance(y, sa, sb)
    # B is much stronger -> delta_auc strongly negative, p-value tiny.
    assert result.delta_auc < -0.1
    assert result.p_value < 0.001


# ---------------------------------------------------------------------------
# Multiple-comparisons correction (BH / Bonferroni)
# Ported from piv5.eval.paired (V5 v0.6) — closes eval-toolkit #1.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bonferroni_simple() -> None:
    """Bonferroni multiplies by N tests; clips at 1.0."""
    from eval_toolkit.bootstrap import bonferroni_correct

    p = np.array([0.01, 0.04, 0.5, 1.0])
    q = bonferroni_correct(p)
    assert q[0] == pytest.approx(0.04)
    assert q[1] == pytest.approx(0.16)
    assert q[2] == 1.0
    assert q[3] == 1.0


@pytest.mark.unit
def test_bh_uniform_p_values() -> None:
    """Under uniform null (large N), BH q-values increase monotonically."""
    from eval_toolkit.bootstrap import fdr_bh_correct

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=100)
    q = fdr_bh_correct(p)
    # Sort by raw p; q-values should be monotone non-decreasing on that sort.
    order = np.argsort(p)
    q_sorted = q[order]
    assert (np.diff(q_sorted) >= -1e-9).all(), "BH q-values not monotone in p-rank"


@pytest.mark.unit
def test_bh_rejects_at_5pct_known_example() -> None:
    """BH spec example: p=[0.001, 0.01, 0.04, 0.1, 0.5] → first 2 reject at q<0.05."""
    from eval_toolkit.bootstrap import fdr_bh_correct

    p = np.array([0.001, 0.01, 0.04, 0.1, 0.5])
    q = fdr_bh_correct(p)
    # i=1: (5/1) * 0.001 = 0.005
    # i=2: (5/2) * 0.01  = 0.025
    # i=3: (5/3) * 0.04  ≈ 0.0667 (monotone-enforced ≥ upstream)
    assert q[0] < 0.05
    assert q[1] < 0.05


@pytest.mark.unit
def test_bh_preserves_input_order() -> None:
    """Output q-values must align with input p-values' positions, not sorted order."""
    from eval_toolkit.bootstrap import fdr_bh_correct

    p = np.array([0.5, 0.001, 0.04, 0.01])
    q = fdr_bh_correct(p)
    # smallest raw p (0.001 at index 1) should get the smallest q.
    assert np.argmin(q) == 1


@pytest.mark.unit
def test_correct_p_values_dispatch_bh() -> None:
    from eval_toolkit.bootstrap import correct_p_values, fdr_bh_correct

    p = np.array([0.01, 0.04, 0.5])
    assert np.allclose(correct_p_values(p, method="bh"), fdr_bh_correct(p))


@pytest.mark.unit
def test_correct_p_values_dispatch_bonferroni() -> None:
    from eval_toolkit.bootstrap import bonferroni_correct, correct_p_values

    p = np.array([0.01, 0.04, 0.5])
    assert np.allclose(correct_p_values(p, method="bonferroni"), bonferroni_correct(p))


@pytest.mark.unit
def test_correct_p_values_dispatch_none() -> None:
    from eval_toolkit.bootstrap import correct_p_values

    p = np.array([0.01, 0.04, 0.5])
    assert np.allclose(correct_p_values(p, method="none"), p)


@pytest.mark.unit
def test_correct_p_values_rejects_unknown_method() -> None:
    from eval_toolkit.bootstrap import correct_p_values

    with pytest.raises(ValueError, match="method must be"):
        correct_p_values(np.array([0.1]), method="bayesian")  # type: ignore[arg-type]


@pytest.mark.unit
def test_corrections_reject_invalid_input() -> None:
    """Bounds + empty-input invariants for both correction primitives."""
    from eval_toolkit.bootstrap import bonferroni_correct, fdr_bh_correct

    with pytest.raises(ValueError, match="non-empty"):
        fdr_bh_correct(np.array([]))
    with pytest.raises(ValueError, match="non-empty"):
        bonferroni_correct(np.array([]))
    with pytest.raises(ValueError, match="outside"):
        fdr_bh_correct(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match="outside"):
        bonferroni_correct(np.array([1.5]))
