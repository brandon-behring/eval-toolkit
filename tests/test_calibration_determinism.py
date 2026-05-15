"""Calibration determinism tests across label-imbalance regimes.

Calibration fitters (Platt, isotonic) are deterministic in principle:
identical inputs should produce bit-identical outputs across runs. This
holds even though L-BFGS-B optimizer initializes from a fixed start
(no RNG) and isotonic regression is a closed-form algorithm.

This test suite verifies that determinism explicitly across three
prevalence regimes (1%, 50%, 99%) so that a future refactor that
introduces nondeterminism (e.g., adding a randomized restart, switching
to a stochastic solver, or accidentally seeding from system time)
fails immediately.

Property tests in ``test_calibration_props.py`` cover monotonicity,
strictly-proper-scoring-rule properties, etc. — these are focused on the
narrower (but load-bearing) determinism question.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.calibration import fit_isotonic_calibrator, fit_platt_calibrator


def _make_data(prevalence: float, *, n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build a (y, score) pair with the specified positive-class rate."""
    rng = np.random.default_rng(seed)
    n_pos = max(2, int(n * prevalence))
    n_pos = min(n_pos, n - 2)  # leave at least 2 negatives
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    # Discriminative-but-noisy scores
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)
    return y, s


@pytest.mark.unit
@pytest.mark.parametrize("prevalence", [0.01, 0.5, 0.99])
def test_platt_fit_is_deterministic_across_runs(prevalence: float) -> None:
    """Same (y, score) → bit-identical PlattFit.a / .b across two independent fits.

    Catches: switch to a stochastic solver, accidental randomized init,
    or order-of-operations drift that introduces floating-point reordering.
    """
    y, s = _make_data(prevalence)
    fit1 = fit_platt_calibrator(y, s)
    fit2 = fit_platt_calibrator(y, s)
    assert fit1.a == fit2.a, f"prevalence={prevalence}: PlattFit.a drifted across runs"
    assert fit1.b == fit2.b, f"prevalence={prevalence}: PlattFit.b drifted across runs"


@pytest.mark.unit
@pytest.mark.parametrize("prevalence", [0.01, 0.5, 0.99])
def test_platt_transform_is_deterministic(prevalence: float) -> None:
    """Calling the fitted Platt transform twice on identical input → identical output.

    Catches: nondeterministic numerical kernel under the transform
    (e.g., scipy.special.expit version drift wouldn't surface here, but
    a buggy fast-path with reordered float ops would).
    """
    y, s = _make_data(prevalence)
    fit = fit_platt_calibrator(y, s)
    out1 = fit(s)
    out2 = fit(s)
    np.testing.assert_array_equal(
        out1, out2, err_msg=f"prevalence={prevalence}: Platt transform output drifted"
    )


@pytest.mark.unit
@pytest.mark.parametrize("prevalence", [0.01, 0.5, 0.99])
def test_isotonic_fit_is_deterministic_across_runs(prevalence: float) -> None:
    """Same (y, score) → identical isotonic transform output across runs.

    Isotonic regression's PAVA algorithm is closed-form deterministic
    given a stable sort. This test asserts that determinism holds in
    practice across our 3 imbalance regimes.
    """
    y, s = _make_data(prevalence)
    fit1 = fit_isotonic_calibrator(y, s)
    fit2 = fit_isotonic_calibrator(y, s)
    # IsotonicCalibrator's transform output is the comparable interface
    out1 = fit1(s)
    out2 = fit2(s)
    np.testing.assert_array_equal(
        out1,
        out2,
        err_msg=f"prevalence={prevalence}: isotonic transform output drifted across runs",
    )


@pytest.mark.unit
def test_platt_fits_differ_across_imbalances() -> None:
    """Sanity check: different prevalence inputs produce different Platt fits.

    Without this, a determinism test passing wouldn't guarantee the
    fitter is actually responsive to its input (e.g., if it returned a
    constant default). This is the negative-space anchor for the
    parametrized determinism tests above.
    """
    y_balanced, s_balanced = _make_data(0.5)
    y_imbalanced, s_imbalanced = _make_data(0.01)

    fit_balanced = fit_platt_calibrator(y_balanced, s_balanced)
    fit_imbalanced = fit_platt_calibrator(y_imbalanced, s_imbalanced)

    # The fits should differ — Platt's intercept absorbs the base-rate shift
    assert fit_balanced.a != fit_imbalanced.a or fit_balanced.b != fit_imbalanced.b, (
        "Platt fits should differ across prevalence regimes; the determinism "
        "tests above are passing for the wrong reason if these are identical."
    )
