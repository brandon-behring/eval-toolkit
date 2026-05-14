"""Research-grounded threshold-selector tests (v0.26.0).

Validates the central methodological claim cited in
``docs/research/papers/inference/_dossier/05_thresholds_power_foundations.md``
§ E1 (Lipton, Elkan, Naryanaswamy 2014, "Thresholding classifiers to
maximize F1") against ``eval_toolkit.thresholds.MaxF1Selector``.

**Lipton 2014 Theorem 1**: For probability scores that are *well-
calibrated*, the F1-optimal threshold ``t*`` satisfies the closed-form
relationship ``t* ≈ F1(t*) / 2``. Equivalent statement: at the F1-
optimal point, ``threshold = (precision + recall - 1) / 2`` collapses
to ``F1 / 2`` when ``precision = recall``.

The intuition: F1 is the harmonic mean of precision and recall, so the
optimal trade-off equates the marginal gain in precision with the
marginal loss in recall. For calibrated probabilities this point lies
analytically at the threshold equal to the F1 score itself, halved.

We verify the relationship empirically: on synthetic well-calibrated
fixtures, the threshold returned by ``MaxF1Selector`` should be within
one grid step of ``F1(t) / 2``.

References
----------
- Lipton, Z. C., Elkan, C., & Naryanaswamy, B. "Thresholding
  classifiers to maximize F1 score." ECML PKDD 2014. arXiv:1402.1892.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.thresholds import MaxF1Selector

N_SEEDS = 10
N_PER_SAMPLE = 2000  # Large sample so the empirical PR curve is dense
# Tolerance: difference between t and F1/2 should be small. Lipton 2014's
# closed-form holds asymptotically; finite-n calibration noise gives a
# floor. The tolerance below tracks the inverse of the effective
# threshold-grid resolution.
LIPTON_TOLERANCE = 0.10


def _well_calibrated_scores(
    rng: np.random.Generator, n: int, prevalence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize perfectly-calibrated scores at a given base rate.

    Construction:

    1. Draw scores ``s ~ Beta(α, β)`` shaped so the marginal mean is
       roughly ``prevalence``.
    2. Draw labels ``y ~ Bernoulli(s)`` — by construction, the true
       calibration map IS the identity.

    This is the cleanest possible "well-calibrated" fixture; any
    deviation from ``t* = F1*/2`` is finite-n noise.
    """
    # Pick (α, β) so the Beta marginal mean = prevalence with mild dispersion.
    # Beta(2, 2*(1-p)/p) has mean p/(1+(1-p)/p · 1) — but a cleaner choice:
    # use uniform-on-[0,1] scores with rejection sampling so the empirical
    # marginal label rate matches `prevalence`.
    s = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < s).astype(int)
    # Adjust prevalence by re-mapping through Beta CDF if requested.
    # For simplicity, just return uniform-with-Bernoulli-labels (effective
    # prevalence ≈ 0.5). The `prevalence` argument is documented for
    # future extension.
    _ = prevalence
    return y, s


def _f1_at(precision: float, recall: float) -> float:
    """Closed-form F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@pytest.mark.parametrize("seed_offset", list(range(N_SEEDS)))
def test_lipton_2014_f1_optimal_threshold_closed_form(seed_offset: int) -> None:
    """MaxF1Selector returns threshold ≈ F1*/2 on well-calibrated scores (Lipton 2014 Thm 1).

    For each parametrized seed:

    1. Synthesize n=2000 well-calibrated (score, label) pairs.
    2. Run ``MaxF1Selector().select(y, s)``.
    3. Compute ``F1*`` from the returned ``ThresholdResult.f1``.
    4. Assert ``|threshold - F1*/2| < LIPTON_TOLERANCE``.

    The tolerance is set generously (0.10) to accommodate finite-n
    discreteness of the PR curve plus the small calibration noise from
    the Bernoulli label sampling at n=2000. A real failure would
    indicate either (a) the F1-argmax computation is broken or (b)
    ``ThresholdResult.f1`` no longer matches the F1 at the returned
    threshold.

    Parametrized over 10 seeds rather than aggregated because Lipton's
    closed-form is per-realization, not on-average.
    """
    rng = np.random.default_rng(42 + seed_offset)
    y, s = _well_calibrated_scores(rng, N_PER_SAMPLE, prevalence=0.5)
    # Need both classes for MaxF1Selector to find an interior optimum.
    if y.sum() in {0, N_PER_SAMPLE}:
        pytest.skip("Single-class draw; Lipton's closed-form requires both classes.")

    result = MaxF1Selector().select(y, s)
    expected_t = result.f1 / 2.0
    diff = abs(result.threshold - expected_t)
    assert diff < LIPTON_TOLERANCE, (
        f"Lipton 2014 Thm 1 closed-form: t* should ≈ F1*/2 on calibrated probs. "
        f"Got threshold={result.threshold:.4f}, F1={result.f1:.4f}, "
        f"F1/2={expected_t:.4f}, |diff|={diff:.4f} > tolerance={LIPTON_TOLERANCE}."
    )


def test_lipton_2014_aggregate_relationship_across_seeds() -> None:
    """Aggregate check: mean(|t - F1/2|) ≪ mean(|t|) on calibrated scores (Lipton 2014).

    Companion to the per-seed test that confirms the closed-form is
    *systematically close* (not just within tolerance on each seed).
    Asserts:

    - Mean(|t - F1/2|) < 0.05 across seeds (tight aggregate bound).
    - Mean(|t - F1/2|) < 0.5 · mean(t) (the diff is small relative
      to the threshold itself — guards against a degenerate case
      where t and F1/2 happen to be small numbers but their ratio is
      far from 1).
    """
    diffs: list[float] = []
    thresholds: list[float] = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(42 + seed)
        y, s = _well_calibrated_scores(rng, N_PER_SAMPLE, prevalence=0.5)
        if y.sum() in {0, N_PER_SAMPLE}:
            continue
        result = MaxF1Selector().select(y, s)
        diffs.append(abs(result.threshold - result.f1 / 2.0))
        thresholds.append(result.threshold)
    assert len(diffs) >= 8, f"Need ≥ 8 valid seeds; got {len(diffs)}."
    mean_diff = float(np.mean(diffs))
    mean_threshold = float(np.mean(thresholds))
    assert mean_diff < 0.05, (
        f"Lipton 2014 aggregate: mean(|t - F1/2|) = {mean_diff:.4f} should be < 0.05 "
        f"on calibrated probabilities."
    )
    assert mean_diff < 0.5 * mean_threshold, (
        f"Lipton 2014 aggregate: mean(|t - F1/2|) = {mean_diff:.4f} should be small "
        f"relative to mean(t) = {mean_threshold:.4f} (ratio < 0.5)."
    )


def test_max_f1_selector_returns_argmax_over_threshold_grid() -> None:
    """Sanity check: MaxF1Selector's returned threshold actually maximizes F1.

    Independent of Lipton 2014's closed-form: just verifies that the
    returned ``threshold`` produces an F1 ≥ the F1 at any other PR-
    curve threshold the selector could have chosen. Catches a buggy
    ``np.argmax`` index computation.
    """
    from sklearn.metrics import precision_recall_curve

    rng = np.random.default_rng(0)
    y, s = _well_calibrated_scores(rng, N_PER_SAMPLE, prevalence=0.5)
    if y.sum() in {0, N_PER_SAMPLE}:
        pytest.skip("Single-class draw.")

    result = MaxF1Selector().select(y, s)
    # Recompute F1 across all PR-curve thresholds; verify result.f1 is the max.
    precisions, recalls, _ = precision_recall_curve(y, s)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
    max_f1_grid = float(np.max(f1s))
    # Allow tiny numerical slack since result.f1 is computed at the matched
    # PR-curve point while max_f1_grid scans the full curve.
    assert result.f1 >= max_f1_grid - 1e-9, (
        f"MaxF1Selector.f1 = {result.f1:.5f} should be ≥ max_f1 over PR grid = "
        f"{max_f1_grid:.5f} (within numerical tolerance)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
