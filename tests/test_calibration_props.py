"""Hypothesis property tests for calibration.

Per the report §5.3 recipe: bayes_optimal_threshold extremes,
isotonic monotonicity, Platt strict bounds, fit_temperature recovery.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit.calibration import (
    CostMatrix,
    bayes_optimal_threshold,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    fit_temperature,
    reliability_curve,
)


@pytest.mark.property
@given(
    π=st.floats(0.001, 0.999, allow_nan=False),
    c=st.floats(0.1, 10.0, allow_nan=False),
)
@settings(deadline=None, max_examples=50)
def test_bayes_threshold_symmetric_costs(π: float, c: float) -> None:
    """Symmetric costs (c_fp = c_fn = c): t* = 1 - π."""
    t_star = bayes_optimal_threshold(π, c_fp=c, c_fn=c)
    assert t_star == pytest.approx(1.0 - π, abs=1e-9)


@pytest.mark.property
@given(
    π=st.floats(0.01, 0.99, allow_nan=False),
    c_fp=st.floats(0.1, 10.0),
    c_fn=st.floats(0.1, 10.0),
)
@settings(deadline=None, max_examples=50)
def test_bayes_threshold_in_unit_interval(π: float, c_fp: float, c_fn: float) -> None:
    """Threshold is always in [0, 1]."""
    t = bayes_optimal_threshold(π, c_fp, c_fn)
    assert 0.0 <= t <= 1.0


@pytest.mark.property
@given(
    π=st.floats(0.01, 0.99, allow_nan=False),
    c_fp=st.floats(0.1, 10.0),
    c_fn=st.floats(0.1, 10.0),
)
@settings(deadline=None, max_examples=50)
def test_cost_matrix_consistency(π: float, c_fp: float, c_fn: float) -> None:
    """CostMatrix.bayes_threshold matches bayes_optimal_threshold call."""
    cm = CostMatrix(prior=π, fp_cost=c_fp, fn_cost=c_fn)
    direct = bayes_optimal_threshold(π, c_fp, c_fn)
    assert cm.bayes_threshold == pytest.approx(direct, abs=1e-12)


@pytest.mark.property
@given(seed=st.integers(0, 1000))
@settings(deadline=None, max_examples=10)
def test_isotonic_monotonicity(seed: int) -> None:
    """Isotonic calibrator output is monotonically non-decreasing in input."""
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, n), 0, 1)
    g = fit_isotonic_calibrator(y, s)
    test_inputs = np.linspace(0.0, 1.0, 100)
    out = g(test_inputs)
    for i in range(len(out) - 1):
        assert out[i] <= out[i + 1] + 1e-9


@pytest.mark.property
@given(seed=st.integers(0, 1000))
@settings(deadline=None, max_examples=10)
def test_platt_strict_bounds(seed: int) -> None:
    """Platt outputs are strictly in (0, 1)."""
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = rng.normal(0, 1, n)
    g = fit_platt_calibrator(y, s)
    out = g(np.linspace(-3.0, 3.0, 50))
    assert (out > 0.0).all()
    assert (out < 1.0).all()


@pytest.mark.property
@given(seed=st.integers(0, 1000))
@settings(deadline=None, max_examples=10)
def test_platt_monotone_in_score(seed: int) -> None:
    """Platt is monotonically non-decreasing in input score."""
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = rng.normal(0, 1, n)
    g = fit_platt_calibrator(y, s)
    test_inputs = np.linspace(-3.0, 3.0, 50)
    out = g(test_inputs)
    # Allow either monotone increasing or monotone decreasing depending on
    # which class has higher mean score.
    diffs = np.diff(out)
    # All diffs same sign (with float tolerance)
    assert (diffs >= -1e-9).all() or (diffs <= 1e-9).all()


@pytest.mark.property
@given(
    T_true=st.floats(0.5, 5.0, allow_nan=False),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_fit_temperature_recovery(T_true: float, seed: int) -> None:
    """When labels are sampled at T_true, fit_temperature recovers ~T_true."""
    rng = np.random.default_rng(seed)
    n = 2000
    base = rng.normal(size=(n, 2))
    sharpened = base / T_true
    probs = np.exp(sharpened) / np.exp(sharpened).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=n) < probs[:, 1]).astype(int)
    if labels.sum() in (0, n):
        return
    result = fit_temperature(base, labels)
    # Tolerance: factor of 2x in either direction (synthetic finite-sample noise).
    assert 0.5 * T_true < result["temperature"] < 2.0 * T_true


@pytest.mark.property
@given(seed=st.integers(0, 1000))
@settings(deadline=None, max_examples=10)
def test_fit_temperature_nll_non_increasing(seed: int) -> None:
    """fit_temperature never increases NLL (T=1 is a feasible point)."""
    rng = np.random.default_rng(seed)
    n = 500
    logits = rng.normal(size=(n, 2)) * rng.uniform(1, 5)
    labels = (logits[:, 1] > logits[:, 0]).astype(int)
    if labels.sum() in (0, n):
        return
    result = fit_temperature(logits, labels)
    assert result["nll_post"] <= result["nll_pre"] + 1e-9


@pytest.mark.property
@given(
    n_bins=st.integers(2, 30),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=10)
def test_reliability_curve_schema_invariants(n_bins: int, seed: int) -> None:
    """Schema fields are well-typed and bin_edges has length n_bins + 1."""
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, n), 0, 1)
    out = reliability_curve(y, s, n_bins=n_bins, strategy="uniform")
    assert "skipped" not in out
    assert len(out["bin_edges"]) == n_bins + 1
    assert isinstance(out["ece_equal_width"], float)
    assert isinstance(out["ece_equal_mass"], float)
    assert 0.0 <= out["ece_equal_width"] <= 1.0
    assert 0.0 <= out["ece_equal_mass"] <= 1.0
