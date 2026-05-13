"""Unit tests for calibration: reliability curves, Bayes thresholds, calibrators, T-scaling.

Adapted from prompt_injection_detector/tests/test_calibration.py and
test_temperature_scaling.py — merged into one file per Decision 2 layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.calibration import (
    CostMatrix,
    bayes_optimal_threshold,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    fit_temperature,
    fit_temperature_oracle,
    maximum_calibration_error,
    reliability_curve,
)


@pytest.fixture
def well_separated() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, n), 0, 1)
    return y, s


@pytest.mark.unit
def test_reliability_curve_schema(well_separated: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = well_separated
    out = reliability_curve(y, s, n_bins=10, strategy="uniform")
    expected = {
        "n",
        "n_positive",
        "n_bins",
        "strategy",
        "prob_true",
        "prob_pred",
        "bin_edges",
        "n_per_bin",
        "ece_equal_mass",
        "ece_equal_width",
    }
    assert expected.issubset(out.keys())
    assert len(out["bin_edges"]) == 11  # n_bins + 1
    assert 0.0 <= out["ece_equal_width"] <= 1.0
    assert 0.0 <= out["ece_equal_mass"] <= 1.0


@pytest.mark.unit
def test_reliability_curve_skips_single_class() -> None:
    y_all_neg = np.zeros(20, dtype=int)
    s = np.linspace(0.1, 0.9, 20)
    out = reliability_curve(y_all_neg, s)
    assert "skipped" in out
    assert out["n_positive"] == 0


@pytest.mark.unit
def test_maximum_calibration_error_in_unit_interval(
    well_separated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = well_separated
    mce = maximum_calibration_error(y, s, n_bins=10, strategy="quantile")
    assert mce is not None
    assert 0.0 <= mce <= 1.0


@pytest.mark.unit
def test_maximum_calibration_error_single_class_returns_none() -> None:
    s = np.linspace(0.1, 0.9, 20)
    assert maximum_calibration_error(np.zeros(20, dtype=int), s) is None
    assert maximum_calibration_error(np.ones(20, dtype=int), s) is None


@pytest.mark.unit
def test_maximum_calibration_error_validates() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="shape mismatch"):
        maximum_calibration_error(y, s[:3])
    with pytest.raises(ValueError, match="empty"):
        maximum_calibration_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="n_bins"):
        maximum_calibration_error(y, s, n_bins=1)
    with pytest.raises(ValueError, match="strategy"):
        maximum_calibration_error(y, s, strategy="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_maximum_calibration_error_agrees_with_reliability_curve_max_gap(
    well_separated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Reference-impl agreement: MCE = max |prob_true - prob_pred| over populated bins.

    The standalone metric must equal the max-gap reconstructed from
    :func:`reliability_curve`'s ``prob_true`` / ``prob_pred`` arrays.
    """
    y, s = well_separated
    for strategy in ("uniform", "quantile"):
        mce = maximum_calibration_error(y, s, n_bins=10, strategy=strategy)
        rc = reliability_curve(y, s, n_bins=10, strategy=strategy)
        expected = float(
            np.abs(np.asarray(rc["prob_true"]) - np.asarray(rc["prob_pred"])).max()
        )
        assert mce == pytest.approx(expected), f"strategy={strategy}"


@pytest.mark.unit
def test_maximum_calibration_error_ge_expected_calibration_error(
    well_separated: tuple[np.ndarray, np.ndarray],
) -> None:
    """MCE ≥ ECE: max of |gaps| dominates any weighted mean of |gaps|.

    Holds for both binning strategies; max ≥ weighted_mean is identity.
    """
    y, s = well_separated
    for strategy in ("uniform", "quantile"):
        mce = maximum_calibration_error(y, s, n_bins=10, strategy=strategy)
        rc = reliability_curve(y, s, n_bins=10, strategy=strategy)
        ece_key = "ece_equal_width" if strategy == "uniform" else "ece_equal_mass"
        ece = float(rc[ece_key])
        assert mce is not None
        assert mce + 1e-12 >= ece, f"strategy={strategy}: MCE {mce} < ECE {ece}"


@pytest.mark.unit
def test_bayes_optimal_threshold_symmetric_costs() -> None:
    """At symmetric costs c_fp=c_fn, the threshold equals 1-π (Elkan 2001).

    Derivation: t* = c_fp(1-π) / (c_fp(1-π) + c_fn·π).
    With c_fp = c_fn = c, this simplifies to (1-π) since c factors out
    of numerator and denominator: t* = (1-π) / ((1-π) + π) = 1-π.
    """
    for π in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert bayes_optimal_threshold(π, c_fp=1.0, c_fn=1.0) == pytest.approx(1.0 - π)


@pytest.mark.unit
def test_bayes_optimal_threshold_extremes() -> None:
    assert bayes_optimal_threshold(0.0, c_fp=1.0, c_fn=1.0) == 1.0
    assert bayes_optimal_threshold(1.0, c_fp=1.0, c_fn=1.0) == 0.0


@pytest.mark.unit
def test_bayes_optimal_threshold_validates() -> None:
    with pytest.raises(ValueError, match="prior"):
        bayes_optimal_threshold(-0.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="c_fp"):
        bayes_optimal_threshold(0.5, 0.0, 1.0)
    with pytest.raises(ValueError, match="c_fn"):
        bayes_optimal_threshold(0.5, 1.0, 0.0)


@pytest.mark.unit
def test_cost_matrix_bayes_threshold_consistency() -> None:
    cm = CostMatrix(prior=0.01, fp_cost=1.0, fn_cost=10.0)
    assert cm.bayes_threshold == bayes_optimal_threshold(0.01, 1.0, 10.0)


@pytest.mark.unit
def test_cost_matrix_validates() -> None:
    with pytest.raises(ValueError, match="prior"):
        CostMatrix(prior=1.5)
    with pytest.raises(ValueError, match="fp_cost"):
        CostMatrix(fp_cost=-1.0)
    with pytest.raises(ValueError, match="fn_cost"):
        CostMatrix(fn_cost=0.0)


@pytest.mark.unit
def test_fit_isotonic_calibrator_monotone(well_separated: tuple[np.ndarray, np.ndarray]) -> None:
    """Output is monotone in input."""
    y, s = well_separated
    g = fit_isotonic_calibrator(y, s)
    test_inputs = np.linspace(0.0, 1.0, 50)
    out = g(test_inputs)
    assert all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))


@pytest.mark.unit
def test_fit_isotonic_clips_to_unit_interval(
    well_separated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = well_separated
    g = fit_isotonic_calibrator(y, s)
    out = g(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_fit_platt_in_unit_interval(well_separated: tuple[np.ndarray, np.ndarray]) -> None:
    """Canonical Platt outputs are in [0, 1].

    Note: canonical Platt (Lin 2007) does NOT clip; large |z| values saturate
    σ(z) → 0 or 1 in floating-point. Previous (≤ v0.2.0) implementation
    used L2-regularized LogisticRegression which kept the slope smaller and
    avoided saturation; the saturation here is canonical-Platt-correct.
    """
    y, s = well_separated
    g = fit_platt_calibrator(y, s)
    out = g(np.linspace(-2.0, 3.0, 50))
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_fit_platt_monotonic(well_separated: tuple[np.ndarray, np.ndarray]) -> None:
    """Platt is a strictly monotonic transform on the score."""
    y, s = well_separated
    g = fit_platt_calibrator(y, s)
    grid = np.linspace(-2.0, 3.0, 50)
    out = g(grid)
    diffs = np.diff(out)
    # Either monotone increasing or monotone decreasing, depending on data direction.
    assert (diffs >= 0).all() or (diffs <= 0).all()


@pytest.mark.unit
def test_fit_temperature_oracle_runs(
    well_separated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = well_separated
    s_clipped = np.clip(s, 0.01, 0.99)
    T_opt, apply = fit_temperature_oracle(y, s_clipped)
    assert T_opt > 0
    out = apply(s_clipped)
    assert (out > 0.0).all() and (out < 1.0).all()


@pytest.mark.unit
def test_fit_temperature_recovers_known_temperature() -> None:
    """When labels are sampled at T_true=3, fit_temperature should recover ~3."""
    rng = np.random.default_rng(42)
    n = 2000
    base = rng.normal(size=(n, 2))
    T_true = 3.0
    # Labels generated at the true T (i.e., sharp logits / T_true)
    sharpened = base / T_true
    probs = np.exp(sharpened) / np.exp(sharpened).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=n) < probs[:, 1]).astype(int)
    # Now: present logits as `base` (overconfident) — fit T should recover T_true.
    result = fit_temperature(base, labels)
    # Tolerance: ±50% on T because synthetic noise is finite.
    assert 0.5 * T_true < result["temperature"] < 1.5 * T_true


@pytest.mark.unit
def test_fit_temperature_improves_nll(well_separated: tuple[np.ndarray, np.ndarray]) -> None:
    """fit_temperature never increases NLL (T=1 is a feasible point)."""
    rng = np.random.default_rng(0)
    n = 500
    logits = rng.normal(size=(n, 2)) * 3  # overconfident
    labels = (logits[:, 1] > logits[:, 0]).astype(int)
    result = fit_temperature(logits, labels)
    assert result["nll_post"] <= result["nll_pre"] + 1e-9
    assert result["improvement"] >= -1e-9


@pytest.mark.unit
def test_fit_temperature_validates() -> None:
    with pytest.raises(ValueError, match="must be \\(n, 2\\)"):
        fit_temperature(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError, match="must be binary"):
        fit_temperature(np.zeros((5, 2)), np.array([0, 1, 2, 0, 1]))
    with pytest.raises(ValueError, match="length mismatch"):
        fit_temperature(np.zeros((5, 2)), np.zeros(7, dtype=int))


@pytest.mark.unit
def test_calibrator_rejects_single_class() -> None:
    y_all_pos = np.ones(50, dtype=int)
    s = np.linspace(0.1, 0.9, 50)
    with pytest.raises(ValueError, match="both classes"):
        fit_isotonic_calibrator(y_all_pos, s)
    with pytest.raises(ValueError, match="both classes"):
        fit_platt_calibrator(y_all_pos, s)


@pytest.mark.unit
def test_fit_platt_matches_sklearn_canonical() -> None:
    """fit_platt_calibrator now matches sklearn's _SigmoidCalibration to <1e-6.

    Canonical Platt + Lin 2007 is implemented in
    sklearn.calibration._SigmoidCalibration; v0.3.0 reproduces it directly
    rather than wrapping LogisticRegression.
    """
    from sklearn.calibration import _SigmoidCalibration  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 300
    y = rng.binomial(1, 0.3, size=n).astype(int)  # imbalanced; Lin 2007 matters most here
    s = y + rng.normal(0, 1.0, size=n)

    ours = fit_platt_calibrator(y, s)
    sk_cal = _SigmoidCalibration().fit(s, y)

    grid = np.linspace(s.min(), s.max(), 100)
    ours_out = ours(grid)
    sk_out = sk_cal.predict(grid)
    np.testing.assert_allclose(ours_out, sk_out, atol=1e-6, rtol=1e-6)
