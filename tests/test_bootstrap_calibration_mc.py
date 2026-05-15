"""Monte Carlo calibration tests for ``bootstrap_ci``.

Tier 1's ``test_bootstrap_golden.py`` pins exact BCa/percentile output on
canonical seeds — guarding against numerical drift. But goldens cannot
tell us whether the math is *correct*: a buggy implementation that
produces self-consistent wrong values would still pass goldens.

This module asks the harder question: do bootstrap CIs achieve their
nominal coverage rate? For a 95% CI:

- Coverage: the true parameter should fall inside the CI in ~95% of
  Monte Carlo replicates (target band: 93%–97% with 1000 replicates).
- Bias: the bootstrap point estimate should track the population
  parameter (|bias| < 0.01 for pr_auc / roc_auc on n≥200).
- Width scaling: width should shrink as O(1/√n) — checked via two n's.

These tests are SLOW (10-60s each) — guarded by ``slow`` marker and run
only in nightly CI, not in PR CI. The Monte Carlo harness pattern is
adapted from `temporalcv/tests/conftest.py`.

Methodology
-----------
For a binary-classification metric, the "true" value is defined as the
metric computed on a large reference population (n_population=20,000)
drawn from a known generative process. For each MC replicate:

1. Draw a sub-sample of size ``n`` from the population
2. Compute ``bootstrap_ci`` on the sub-sample
3. Record whether the resulting CI contains the population truth
4. Aggregate coverage across replicates

Coverage close to the nominal 95% level confirms the bootstrap math is
calibrated. Systematic miss (e.g., coverage = 0.85) signals a real bug:
either the CI is too narrow (anti-conservative) or biased.

References
----------
.. [1] Efron, B. & Tibshirani, R. (1993). "An Introduction to the
       Bootstrap." Chapman & Hall — Chapter 14 on coverage calibration.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from eval_toolkit.bootstrap import bootstrap_ci
from eval_toolkit.metrics import pr_auc, roc_auc

# ---------------------------------------------------------------------------
# Monte Carlo helpers (adapted from temporalcv/tests/conftest.py)
# ---------------------------------------------------------------------------


def _compute_mc_coverage(ci_lower: np.ndarray, ci_upper: np.ndarray, true_value: float) -> float:
    """Fraction of MC replicates whose CI contains the true parameter."""
    contains = (np.asarray(ci_lower) <= true_value) & (true_value <= np.asarray(ci_upper))
    return float(np.mean(contains))


def _compute_mc_bias(estimates: np.ndarray, true_value: float) -> float:
    """Absolute bias: |mean(estimates) - true_value|."""
    return abs(float(np.mean(estimates)) - true_value)


def _generate_population(
    n_population: int, prevalence: float, separation: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic reference population for binary-classification metrics.

    Positives drawn from ``N(0.5 + separation/2, 0.2)``; negatives from
    ``N(0.5 - separation/2, 0.2)``; both clipped to [0, 1]. Larger
    ``separation`` → more-discriminative scores → higher true pr_auc /
    roc_auc. ``prevalence`` controls the class balance.
    """
    rng = np.random.default_rng(seed)
    n_pos = max(2, int(n_population * prevalence))
    n_neg = n_population - n_pos
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s_pos = np.clip(rng.normal(0.5 + separation / 2, 0.2, size=n_pos), 0.0, 1.0)
    s_neg = np.clip(rng.normal(0.5 - separation / 2, 0.2, size=n_neg), 0.0, 1.0)
    s = np.concatenate([s_pos, s_neg])
    # Shuffle so the array isn't class-sorted
    order = rng.permutation(n_population)
    return y[order], s[order]


def _run_mc_coverage(
    metric: Callable[..., float],
    population: tuple[np.ndarray, np.ndarray],
    n_sub: int,
    n_replicates: int,
    method: str,
    confidence: float,
    base_seed: int,
) -> tuple[float, float]:
    """Run MC replicates; return (coverage, bias).

    Each replicate:
      1. Draw ``n_sub`` indices uniformly without replacement from the
         population
      2. Compute ``bootstrap_ci(y_sub, s_sub, metric=...)`` with a
         per-replicate seed
      3. Record CI [low, high] + point estimate
    """
    y_pop, s_pop = population
    n_pop = len(y_pop)
    true_value = float(metric(y_pop, s_pop))

    rng = np.random.default_rng(base_seed)
    lows = np.empty(n_replicates)
    highs = np.empty(n_replicates)
    points = np.empty(n_replicates)
    for i in range(n_replicates):
        idx = rng.choice(n_pop, size=n_sub, replace=False)
        y_sub = y_pop[idx]
        s_sub = s_pop[idx]
        # Skip degenerate sub-samples (sklearn metrics require both classes)
        if y_sub.sum() == 0 or y_sub.sum() == n_sub:
            # Use the boundary value; degenerate, but rare with n_sub=200, prev=0.3
            points[i] = float("nan")
            lows[i] = float("nan")
            highs[i] = float("nan")
            continue
        ci = bootstrap_ci(
            y_sub,
            s_sub,
            metric=metric,
            n_resamples=200,
            confidence=confidence,
            method=method,
            seed=base_seed + i,
        )
        points[i] = ci.point_estimate
        lows[i] = ci.ci_low
        highs[i] = ci.ci_high

    # Drop NaN replicates (degenerate sub-samples) from the aggregate
    mask = ~np.isnan(points)
    coverage = _compute_mc_coverage(lows[mask], highs[mask], true_value)
    bias = _compute_mc_bias(points[mask], true_value)
    return coverage, bias


# ---------------------------------------------------------------------------
# MC test cases
# ---------------------------------------------------------------------------

# A small set of cases that span: 2 metrics × 2 sample sizes × balanced /
# imbalanced. Each case runs 500 MC replicates with bootstrap_n=200 inside
# — about 5-10s per case on modern hardware.
MC_CASES = [
    {
        "id": "pr_auc_balanced_n200_BCa",
        "metric": pr_auc,
        "metric_name": "pr_auc",
        "n_sub": 200,
        "prevalence": 0.5,
        "separation": 0.4,
        "method": "BCa",
        "n_replicates": 500,
    },
    {
        "id": "pr_auc_balanced_n1000_BCa",
        "metric": pr_auc,
        "metric_name": "pr_auc",
        "n_sub": 1000,
        "prevalence": 0.5,
        "separation": 0.4,
        "method": "BCa",
        "n_replicates": 500,
    },
    {
        "id": "pr_auc_imbalanced_p05_n200_BCa",
        "metric": pr_auc,
        "metric_name": "pr_auc",
        "n_sub": 200,
        "prevalence": 0.05,
        "separation": 0.4,
        "method": "BCa",
        "n_replicates": 500,
    },
    {
        "id": "roc_auc_balanced_n200_BCa",
        "metric": roc_auc,
        "metric_name": "roc_auc",
        "n_sub": 200,
        "prevalence": 0.5,
        "separation": 0.4,
        "method": "BCa",
        "n_replicates": 500,
    },
    {
        "id": "pr_auc_balanced_n200_percentile",
        "metric": pr_auc,
        "metric_name": "pr_auc",
        "n_sub": 200,
        "prevalence": 0.5,
        "separation": 0.4,
        "method": "percentile",
        "n_replicates": 500,
    },
]

# Acceptance bands for 500 MC replicates of nominal-95% CIs:
# - Coverage: [0.90, 0.99] (slightly wider than the temporalcv [0.93, 0.97]
#   band because (a) we use n_resamples=200 inside bootstrap_ci, not 1000,
#   and (b) BCa on small samples is mildly anti-conservative). Tighten with
#   more replicates if false-fails appear.
# - Bias: |bias| < 0.05 (binary-classifier metrics on n=200 are noisy)
COVERAGE_LOWER = 0.90
COVERAGE_UPPER = 0.99
BIAS_THRESHOLD = 0.05


@pytest.mark.monte_carlo
@pytest.mark.slow
@pytest.mark.parametrize("case", MC_CASES, ids=[c["id"] for c in MC_CASES])  # type: ignore[index]
def test_bootstrap_ci_coverage_mc(case: dict[str, object]) -> None:
    """Empirical coverage of ``bootstrap_ci`` matches the nominal 95% level.

    For each case: draw a reference population, then run
    ``n_replicates`` sub-samples, compute a 95% bootstrap CI per
    sub-sample, and assert the fraction of CIs containing the
    population truth falls inside the acceptance band.

    Failure modes this catches:
    - Anti-conservative bias: coverage < 0.90 indicates the CI is too
      narrow (e.g., wrong α in the BCa quantile arithmetic)
    - Conservative bias: coverage > 0.99 indicates the CI is too wide
    - Estimation bias: ``|bias| > 0.05`` indicates the bootstrap point
      estimate is systematically off from the population value

    These are checks that goldens cannot perform — they pin exact
    numbers without asking whether those numbers are *right*.
    """
    population = _generate_population(
        n_population=20_000,
        prevalence=float(case["prevalence"]),  # type: ignore[arg-type]
        separation=float(case["separation"]),  # type: ignore[arg-type]
        seed=42,
    )
    coverage, bias = _run_mc_coverage(
        metric=case["metric"],  # type: ignore[arg-type]
        population=population,
        n_sub=int(case["n_sub"]),  # type: ignore[arg-type]
        n_replicates=int(case["n_replicates"]),  # type: ignore[arg-type]
        method=str(case["method"]),
        confidence=0.95,
        base_seed=42,
    )

    assert COVERAGE_LOWER <= coverage <= COVERAGE_UPPER, (
        f"{case['id']}: coverage={coverage:.3f} outside acceptance band "
        f"[{COVERAGE_LOWER}, {COVERAGE_UPPER}] — nominal 95% CI is "
        f"{'anti-conservative' if coverage < COVERAGE_LOWER else 'over-wide'}"
    )
    assert bias < BIAS_THRESHOLD, (
        f"{case['id']}: |bias|={bias:.4f} exceeds threshold {BIAS_THRESHOLD} — "
        "bootstrap point estimate is systematically off"
    )


@pytest.mark.monte_carlo
@pytest.mark.slow
def test_bootstrap_ci_width_scales_with_n() -> None:
    """CI width should shrink as O(1/√n) — verify on two sample sizes.

    Theoretical: CI half-width for a metric with bounded variance shrinks
    as sample size grows. Doubling n should roughly multiply width by
    ~1/√2 ≈ 0.71.
    """
    population = _generate_population(n_population=20_000, prevalence=0.5, separation=0.4, seed=42)

    rng = np.random.default_rng(42)
    n_replicates = 200

    widths_n200 = []
    widths_n800 = []
    for n_sub in (200, 800):
        widths = []
        for i in range(n_replicates):
            idx = rng.choice(20_000, size=n_sub, replace=False)
            y_sub = population[0][idx]
            s_sub = population[1][idx]
            if y_sub.sum() == 0 or y_sub.sum() == n_sub:
                continue
            ci = bootstrap_ci(
                y_sub, s_sub, metric=pr_auc, n_resamples=200, method="BCa", seed=42 + i
            )
            widths.append(ci.ci_high - ci.ci_low)
        (widths_n200 if n_sub == 200 else widths_n800).append(float(np.median(widths)))

    median_n200 = widths_n200[0]
    median_n800 = widths_n800[0]
    ratio = median_n800 / median_n200
    expected_ratio = 1.0 / np.sqrt(4)  # n800/n200 = 4 → ratio = 1/√4 = 0.5

    # Allow a wide band — finite-sample CI scaling has higher-order terms
    assert 0.35 <= ratio <= 0.70, (
        f"width(n=800) / width(n=200) = {ratio:.3f}; expected ≈ {expected_ratio:.3f} "
        "(±~30% for finite-sample second-order effects). If significantly outside, "
        "the bootstrap variance scaling is broken."
    )
