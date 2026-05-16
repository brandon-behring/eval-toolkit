"""Coverage-targeted tests for calibration error paths and defensive code.

Extracted from the v0.27.x-era ``test_coverage_gap.py`` during the
v0.30.1 hygiene split — every assertion preserved verbatim; only the
file boundary changed.

Pairs with the happy-path coverage in ``test_calibration_unit.py``,
the determinism checks in ``test_calibration_determinism.py``, and
the optimization-failure paths in
``test_calibration_optimization_failures.py``. Targets input-validation
error branches for calibrators (Platt, isotonic, temperature, Beta) and
the v0.3.0 cost-matrix additions.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.calibration import (
    bayes_optimal_threshold,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    fit_temperature,
    fit_temperature_oracle,
    reliability_curve,
)

# ---------------------------------------------------------------------------
# calibration: validation paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bayes_optimal_threshold_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="prior"):
        bayes_optimal_threshold(1.5, c_fp=1.0, c_fn=1.0)
    with pytest.raises(ValueError, match="c_fp"):
        bayes_optimal_threshold(0.5, c_fp=0.0, c_fn=1.0)
    with pytest.raises(ValueError, match="c_fn"):
        bayes_optimal_threshold(0.5, c_fp=1.0, c_fn=-1.0)


@pytest.mark.unit
def test_reliability_curve_handles_single_class() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    out = reliability_curve(y, s, n_bins=5)
    assert "skipped" in out


@pytest.mark.unit
def test_reliability_curve_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        reliability_curve(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        reliability_curve(np.array([], dtype=int), np.array([], dtype=float))
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(np.array([0, 1]), np.array([0.4, 0.6]), n_bins=1)
    with pytest.raises(ValueError, match="strategy"):
        reliability_curve(np.array([0, 1]), np.array([0.4, 0.6]), strategy="bogus")  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_isotonic_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_isotonic_calibrator(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        fit_isotonic_calibrator(np.array([], dtype=int), np.array([], dtype=float))
    with pytest.raises(ValueError, match="NaN"):
        fit_isotonic_calibrator(np.array([0, 1]), np.array([np.nan, 0.5]))
    with pytest.raises(ValueError, match="both classes"):
        fit_isotonic_calibrator(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


@pytest.mark.unit
def test_fit_isotonic_apply_rejects_nan_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0, 1)
    g = fit_isotonic_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))


@pytest.mark.unit
def test_fit_platt_apply_rejects_nan_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = y + rng.normal(0, 0.3, 50)
    g = fit_platt_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))


@pytest.mark.unit
def test_fit_temperature_validates_logits_shape() -> None:
    with pytest.raises(ValueError, match="must be"):
        fit_temperature(np.zeros((10,)), np.zeros(10, dtype=int))


@pytest.mark.unit
def test_fit_temperature_validates_length_match() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        fit_temperature(np.zeros((10, 2)), np.zeros(5, dtype=int))


@pytest.mark.unit
def test_fit_temperature_validates_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        fit_temperature(np.zeros((10, 2)), np.array([0, 1, 2] + [0] * 7))


@pytest.mark.unit
def test_fit_temperature_oracle_apply_rejects_nan() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0.01, 0.99)
    _, apply = fit_temperature_oracle(y, s)
    with pytest.raises(ValueError, match="NaN"):
        apply(np.array([np.nan, 0.5]))


@pytest.mark.unit
def test_fit_temperature_rejects_single_class() -> None:
    """fit_temperature is now consistent with peer calibrators on single-class input.

    (v0.3.0 C1 validation hardening — moved here from the metrics section
    of the original test_coverage_gap.py, since it tests calibration.py.)
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 2))
    labels = np.zeros(50, dtype=int)  # single-class
    with pytest.raises(ValueError, match="both classes"):
        fit_temperature(logits, labels)


# ---------------------------------------------------------------------------
# v0.3.0 C6: expected_cost + Beta calibration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cost_matrix_expected_cost_known_value() -> None:
    """expected_cost on a fixed scenario."""
    from eval_toolkit.calibration import CostMatrix

    cm = CostMatrix(prior=0.5, fp_cost=1.0, fn_cost=10.0)
    y = np.array([0, 1, 0, 1])
    s = np.array([0.6, 0.4, 0.1, 0.9])
    # At threshold=0.5: pred = [1, 0, 0, 1]; FP at idx 0, FN at idx 1
    # Cost = (1*1.0 + 1*10.0) / 4 = 2.75
    assert cm.expected_cost(y, s, threshold=0.5) == 2.75


@pytest.mark.unit
def test_cost_matrix_expected_cost_uses_bayes_threshold_by_default() -> None:
    from eval_toolkit.calibration import CostMatrix

    cm = CostMatrix(prior=0.01, fp_cost=1.0, fn_cost=10.0)
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.01, size=200)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, 200), 0, 1)
    cost_default = cm.expected_cost(y, s)
    cost_explicit = cm.expected_cost(y, s, threshold=cm.bayes_threshold)
    assert cost_default == cost_explicit


@pytest.mark.unit
def test_fit_beta_calibrator_returns_unit_interval() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300).astype(int)
    s = (y + rng.normal(0, 0.4, 300)).clip(0.01, 0.99)
    g = fit_beta_calibrator(y, s)
    out = g(s)
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_fit_beta_calibrator_validates_single_class() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    y = np.zeros(50, dtype=int)
    s = np.linspace(0.1, 0.9, 50)
    with pytest.raises(ValueError, match="both classes"):
        fit_beta_calibrator(y, s)


@pytest.mark.unit
def test_fit_beta_calibrator_apply_rejects_nan() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0.01, 0.99)
    g = fit_beta_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))
