"""Tests for fit_platt_binary + fit_beta_binary (v0.40.0, closes #43).

Both are scalar-prob binary-class adapters mirroring fit_temperature_binary.
Returns ``(params_tuple, apply)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit import (
    fit_beta_binary,
    fit_beta_calibrator,
    fit_platt_binary,
    fit_platt_calibrator,
    fit_temperature_binary,
)


@pytest.fixture
def synthetic_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated synthetic binary classification: y_true + miscalibrated y_score."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.binomial(1, 0.3, size=n).astype(int)
    # Discriminative but biased high so calibration matters
    p = np.clip(0.7 + 0.2 * (y - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99)
    return y, p


# ---------- fit_platt_binary ----------


@pytest.mark.unit
def test_platt_binary_returns_params_and_apply(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    (a, b), apply = fit_platt_binary(y, p)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert callable(apply)


@pytest.mark.unit
def test_platt_binary_apply_returns_same_shape(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    _, apply = fit_platt_binary(y, p)
    test = np.array([0.1, 0.5, 0.9])
    out = apply(test)
    assert out.shape == test.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_platt_binary_params_match_underlying_fit(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """fit_platt_binary should expose the same (a, b) as fit_platt_calibrator."""
    y, p = synthetic_binary_data
    (a, b), _ = fit_platt_binary(y, p)
    canonical_fit = fit_platt_calibrator(y, p)
    assert a == pytest.approx(canonical_fit.a)
    assert b == pytest.approx(canonical_fit.b)


@pytest.mark.unit
def test_platt_binary_apply_matches_underlying_transform(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """apply(p) should match fit_platt_calibrator(...).transform(p)."""
    y, p = synthetic_binary_data
    _, apply = fit_platt_binary(y, p)
    canonical_fit = fit_platt_calibrator(y, p)
    test = np.linspace(0.05, 0.95, 20)
    np.testing.assert_allclose(apply(test), canonical_fit.transform(test))


@pytest.mark.unit
def test_platt_binary_well_separated_positive_slope(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Well-separated data should yield positive Platt slope."""
    y, p = synthetic_binary_data
    (a, _), _ = fit_platt_binary(y, p)
    assert a > 0.0


@pytest.mark.unit
def test_platt_binary_rejects_single_class() -> None:
    y_single = np.zeros(50, dtype=int)
    p = np.random.default_rng(0).uniform(0.0, 1.0, 50)
    with pytest.raises(ValueError):
        fit_platt_binary(y_single, p)


# ---------- fit_beta_binary ----------


@pytest.mark.unit
def test_beta_binary_returns_three_params_and_apply(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    (a, b, c), apply = fit_beta_binary(y, p)
    assert all(isinstance(x, float) for x in (a, b, c))
    assert callable(apply)


@pytest.mark.unit
def test_beta_binary_apply_returns_same_shape(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    _, apply = fit_beta_binary(y, p)
    test = np.array([0.1, 0.5, 0.9])
    out = apply(test)
    assert out.shape == test.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_beta_binary_apply_matches_underlying_calibrator(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """fit_beta_binary apply should match fit_beta_calibrator output (same fit)."""
    y, p = synthetic_binary_data
    _, apply = fit_beta_binary(y, p)
    canonical_apply = fit_beta_calibrator(y, p)
    test = np.linspace(0.05, 0.95, 20)
    np.testing.assert_allclose(apply(test), canonical_apply(test), rtol=1e-6)


@pytest.mark.unit
def test_beta_binary_rejects_single_class() -> None:
    y_single = np.ones(50, dtype=int)
    p = np.random.default_rng(0).uniform(0.0, 1.0, 50)
    with pytest.raises(ValueError):
        fit_beta_binary(y_single, p)


@pytest.mark.unit
def test_beta_binary_rejects_non_finite_scores() -> None:
    y = np.array([0, 1, 0, 1, 0, 1] * 10)
    p_bad = np.full(60, np.nan)
    with pytest.raises(ValueError):
        fit_beta_binary(y, p_bad)


@pytest.mark.unit
def test_beta_binary_apply_rejects_non_finite_test_scores(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    _, apply = fit_beta_binary(y, p)
    with pytest.raises(ValueError, match="NaN or inf"):
        apply(np.array([0.5, np.nan, 0.8]))


# ---------- consistency across the calibrator family ----------


@pytest.mark.unit
def test_all_four_binary_adapters_have_consistent_shape(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """temperature_binary, platt_binary, beta_binary all return ``(params, apply)``.

    Documents the audit-battery contract for the 4-calibrator pattern.
    """
    y, p = synthetic_binary_data
    # All return (params, apply); apply is a callable taking (n,) -> (n,)
    for name, fitter in [
        ("temperature", fit_temperature_binary),
        ("platt", fit_platt_binary),
        ("beta", fit_beta_binary),
    ]:
        params, apply = fitter(y, p)
        assert callable(apply), f"{name}: apply not callable"
        out = apply(np.array([0.1, 0.5, 0.9]))
        assert out.shape == (3,), f"{name}: apply output shape mismatch"
        assert (out >= 0.0).all() and (out <= 1.0).all(), f"{name}: output not in [0,1]"


@pytest.mark.unit
def test_platt_binary_params_are_pair(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    params, _ = fit_platt_binary(y, p)
    assert len(params) == 2


@pytest.mark.unit
def test_beta_binary_params_are_triple(
    synthetic_binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = synthetic_binary_data
    params, _ = fit_beta_binary(y, p)
    assert len(params) == 3
