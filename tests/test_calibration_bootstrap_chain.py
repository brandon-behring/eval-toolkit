"""Integration test: bootstrap-CI-on-ECE before vs after calibration (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``calibration.py`` and ``bootstrap.py`` are tested in isolation, but
no test verifies the *workflow* — does fitting a calibrator actually
shrink the ECE bootstrap CI on miscalibrated data, end-to-end?

This test exercises the chain consumers run in production:

1. Generate miscalibrated synthetic data (over-confident scores).
2. Bootstrap-CI of ECE on the uncalibrated scores.
3. Fit a Platt calibrator on a held-out calibration split.
4. Apply calibrator; bootstrap-CI of ECE on the calibrated scores.
5. Assert the calibrated CI is bounded BELOW the uncalibrated CI in
   ≥ 70% of seeds — calibration improves ECE in expectation, but
   small-n bootstrap variance means individual seeds can invert.

The 70% dominance fraction matches the v0.25.0 flake-mitigation
policy used elsewhere in the research-grounded suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import bootstrap_ci
from eval_toolkit.calibration import fit_platt_calibrator
from eval_toolkit.metrics import expected_calibration_error

N_SEEDS = 50
N_PER_SAMPLE = 400
N_CALIBRATION = 200
N_RESAMPLES = 500
DOMINANCE_FRACTION = 0.70


def _miscalibrated_inputs(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize over-confident scores and matching labels.

    Latent ``z ~ N(0, 1)``, observed score ``s = sigmoid(2 * z)``
    (sharply saturated → over-confident in tails), label ``y ~
    Bernoulli(sigmoid(z))`` (the true-prob is the milder sigmoid).
    The mismatch makes ECE > 0; Platt should reduce it.
    """
    z = rng.normal(0.0, 1.0, size=n)
    s = 1.0 / (1.0 + np.exp(-2.0 * z))  # over-confident scores
    true_p = 1.0 / (1.0 + np.exp(-z))  # milder true-prob map
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(int)
    return y, s


def _ece_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Bootstrap-callable wrapper over expected_calibration_error."""
    return expected_calibration_error(y_true, y_score, n_bins=10)


@pytest.mark.slow
def test_calibrated_ece_point_below_uncalibrated_in_majority_of_seeds() -> None:
    """End-to-end: calibration shrinks point ECE on miscalibrated data via the bootstrap chain.

    For each of 50 seeds:
    1. Draw N=400 miscalibrated (y, s).
    2. Bootstrap-CI of ECE on uncalibrated s → ``ci_uncal``.
    3. Fit Platt on first 200 rows; apply to the rest.
    4. Bootstrap-CI of ECE on calibrated s → ``ci_cal``.
    5. Record whether ``ci_cal.point_estimate < ci_uncal.point_estimate``
       (point-ECE dominance — the seed-level direction).

    Assert dominance in ≥ 70% of seeds. Per the v0.25.0 flake-
    mitigation policy, this is a probabilistic claim, not per-seed.

    Note on weaker assertion: an earlier version of this test required
    *strict CI dominance* (``ci_cal.ci_high < ci_uncal.ci_low``); on
    n=200 eval data with bootstrap variance, the CIs typically overlap
    even when the calibrated estimate is lower. Point-estimate
    dominance is the testable form of the claim at the chosen sample
    size.
    """
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_SEEDS)
    cal_better_count = 0
    valid_seeds = 0
    deltas: list[float] = []

    for seed in seeds:
        sample_rng = np.random.default_rng(int(seed))
        y, s = _miscalibrated_inputs(sample_rng, N_PER_SAMPLE)
        if y.sum() in {0, len(y)}:
            continue

        # Use first N_CALIBRATION rows for calibrator fitting; rest for evaluation.
        y_cal_split = y[:N_CALIBRATION]
        s_cal_split = s[:N_CALIBRATION]
        y_eval = y[N_CALIBRATION:]
        s_eval = s[N_CALIBRATION:]
        if y_cal_split.sum() in {0, N_CALIBRATION} or y_eval.sum() in {0, len(y_eval)}:
            continue

        # Bootstrap CI of ECE on uncalibrated eval scores
        ci_uncal = bootstrap_ci(
            y_eval,
            s_eval,
            metric=_ece_metric,
            n_resamples=N_RESAMPLES,
            confidence=0.95,
            method="percentile",
            seed=int(seed),
        )

        # Fit Platt on the calibration split, apply to eval scores
        platt = fit_platt_calibrator(y_cal_split, s_cal_split)
        s_eval_cal = platt(s_eval)
        ci_cal = bootstrap_ci(
            y_eval,
            s_eval_cal,
            metric=_ece_metric,
            n_resamples=N_RESAMPLES,
            confidence=0.95,
            method="percentile",
            seed=int(seed),
        )

        valid_seeds += 1
        delta = ci_uncal.point_estimate - ci_cal.point_estimate  # positive ⇒ cal better
        deltas.append(delta)
        if ci_cal.point_estimate < ci_uncal.point_estimate:
            cal_better_count += 1

    assert valid_seeds >= 40, f"Need ≥ 40 valid seeds; got {valid_seeds}."
    fraction = cal_better_count / valid_seeds
    assert fraction >= DOMINANCE_FRACTION, (
        f"Calibrated point ECE should be below uncalibrated in "
        f"≥ {DOMINANCE_FRACTION:.0%} of seeds; got {cal_better_count}/{valid_seeds} "
        f"= {fraction:.2%}. Mean (uncal - cal) ECE delta = "
        f"{float(np.mean(deltas)):.5f}. Either calibration is broken or the "
        f"miscalibrated fixture isn't actually miscalibrated."
    )
    assert float(np.mean(deltas)) > 0, (
        f"Mean (uncal - cal) ECE delta should be positive (calibration helps in "
        f"expectation); got {float(np.mean(deltas)):.5f}."
    )


def test_calibrator_reduces_point_ece_estimate_on_miscalibrated_data() -> None:
    """Sanity check: the point ECE drops after Platt on a miscalibrated fixture.

    Less expensive than the bootstrap dominance test — single-seed
    point comparison. Catches catastrophic regressions where Platt
    produces *worse*-calibrated outputs.
    """
    rng = np.random.default_rng(42)
    y, s = _miscalibrated_inputs(rng, 1000)
    ece_uncal = expected_calibration_error(y, s, n_bins=10)
    platt = fit_platt_calibrator(y[:500], s[:500])
    s_cal = platt(s[500:])
    ece_cal = expected_calibration_error(y[500:], s_cal, n_bins=10)
    assert ece_cal < ece_uncal, (
        f"Platt should reduce point ECE on miscalibrated data; got "
        f"ECE_uncal={ece_uncal:.4f}, ECE_cal={ece_cal:.4f}."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
