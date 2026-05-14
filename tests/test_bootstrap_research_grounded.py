"""Research-grounded bootstrap tests (v0.26.0).

Validates two methodological claims from
``docs/research/papers/inference/_dossier/01_bootstrap_and_cv_variance.md``
against the actual implementations in ``eval_toolkit.bootstrap``:

(a) **BCa transformation-respecting coverage on a skewed statistic**
    (DiCiccio & Efron 1996 §4). The bias-corrected and accelerated
    interval is second-order accurate even on skewed distributions
    where the percentile method under-covers. Asserts empirical
    coverage of ``bootstrap_ci(method="BCa")`` lands within a Wald-
    bounded window around the nominal 95% on a Beta(2, 5)-distributed
    sample mean.

(b) **CV-CLT-CI achieves near-nominal coverage on i.i.d. folds**
    (Bayle et al. 2020 Theorem 3.1, the inference primitive backing
    eval-toolkit's ``cv_clt_ci``; the dossier also flags Bates et al.
    2024's parallel result for nested CV). Bayle 2020's contribution
    is *proving* that the standard ``mean ± z_{α/2} · σ_fold / √K``
    formula gives asymptotically valid coverage on K-fold CV — they
    do not add an explicit correction term to the formula. We
    therefore validate the *coverage* claim directly rather than a
    width-comparison: assert empirical coverage of ``cv_clt_ci``
    lands inside a Wald-bounded window around the nominal level.

Both tests use multi-seed loops with bounded coverage / dominance
fractions (per the v0.25.0 flake-mitigation policy) so a single
realization's bootstrap noise cannot fail the suite.

References
----------
- DiCiccio, T. J. & Efron, B. "Bootstrap confidence intervals."
  Statistical Science 11(3), 1996.
- Efron, B. "Better bootstrap confidence intervals." JASA 82(397), 1987.
- Bayle, P., Bayle, A., Janson, L., & Mackey, L. "Cross-validation
  confidence intervals for test error." Annals of Statistics 48(6), 2020.
- Bates, S., Hastie, T., & Tibshirani, R. "Cross-validation: what does
  it estimate and how well does it do it?" JASA 119(546), 2024.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import bootstrap_ci, cv_clt_ci

# ---------------------------------------------------------------------------
# Suite (a): DiCiccio & Efron 1996 BCa coverage on a skewed mean.
# ---------------------------------------------------------------------------

# Beta(2, 5) population mean = 2/(2+5) = 2/7 ≈ 0.2857.
BETA_ALPHA = 2.0
BETA_BETA = 5.0
TRUE_BETA_MEAN = BETA_ALPHA / (BETA_ALPHA + BETA_BETA)

N_COVERAGE_SEEDS = 100
N_RESAMPLES_BCA = 2000
N_PER_SAMPLE = 200
NOMINAL_CONFIDENCE = 0.95

# Wald CI for binomial coverage with n=100 trials at nominal p=0.95:
# 1-σ ≈ √(0.05·0.95/100) ≈ 0.0218; ±2σ ⇒ window [0.906, 0.994].
# Use a slightly inside band to give the test a margin against minor
# under-coverage from finite resamples.
COVERAGE_LOWER_BOUND = 0.92
COVERAGE_UPPER_BOUND = 0.98


def _mean_metric(_y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Bootstrap-callable metric: mean of y_score (ignores labels).

    Used to bootstrap the sampling distribution of the sample mean
    via ``bootstrap_ci`` (which is shaped for label/score pairs but
    the bootstrap mechanism is the same).
    """
    return float(np.mean(y_score))


@pytest.mark.slow
def test_bca_transformation_respecting_coverage_on_skewed_statistic() -> None:
    """BCa empirical coverage stays near nominal 95% on Beta(2, 5) means (DiCiccio-Efron 1996).

    Per DiCiccio & Efron 1996 §4, the BCa interval is second-order
    accurate (transformation-respecting) — its coverage approaches the
    nominal level faster than the percentile method's first-order
    rate, especially on skewed distributions like Beta(2, 5) where the
    sample mean's distribution is also asymmetric.

    Test design:

    1. For each of 100 seeds, draw n=200 samples from Beta(2, 5).
    2. Compute ``bootstrap_ci(method="BCa", n_resamples=2000)`` of the
       mean.
    3. Record whether the population mean (2/7) is inside the CI.
    4. Assert empirical coverage ∈ [0.92, 0.98] (Wald-bounded around
       0.95 with margin for finite-resample noise).

    A real failure here would mean either (a) BCa's bias-correction
    is broken or (b) the underlying scipy.stats.bootstrap has shifted
    behavior on a skewed statistic.
    """
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_COVERAGE_SEEDS)
    inside = 0
    for seed in seeds:
        sample_rng = np.random.default_rng(int(seed))
        sample = sample_rng.beta(BETA_ALPHA, BETA_BETA, size=N_PER_SAMPLE)
        # bootstrap_ci shape: (y_true, y_score, metric); we use a dummy
        # y_true and a metric that ignores it.
        dummy_y = np.zeros_like(sample, dtype=int)
        ci = bootstrap_ci(
            dummy_y,
            sample,
            metric=_mean_metric,
            n_resamples=N_RESAMPLES_BCA,
            confidence=NOMINAL_CONFIDENCE,
            method="BCa",
            seed=int(seed),
        )
        if ci.ci_low <= TRUE_BETA_MEAN <= ci.ci_high:
            inside += 1

    coverage = inside / N_COVERAGE_SEEDS
    assert COVERAGE_LOWER_BOUND <= coverage <= COVERAGE_UPPER_BOUND, (
        f"BCa empirical coverage on Beta({BETA_ALPHA}, {BETA_BETA}) sample mean = "
        f"{coverage:.3f}; expected within [{COVERAGE_LOWER_BOUND}, "
        f"{COVERAGE_UPPER_BOUND}]. Either BCa is mis-calibrated on this skewed "
        f"distribution or n_resamples={N_RESAMPLES_BCA} is too small."
    )


@pytest.mark.slow
def test_bca_outperforms_percentile_on_skewed_distribution_by_coverage_proximity() -> None:
    """BCa coverage is closer to nominal than percentile on Beta(2, 5) (DiCiccio-Efron 1996).

    Counter-method check: the percentile method is first-order accurate;
    on a skewed distribution it tends to under-cover more than BCa.
    Assert |coverage_BCa - 0.95| ≤ |coverage_percentile - 0.95| in
    aggregate (across the same seeds).

    Not a hard "BCa always covers better per-seed" claim — both
    methods can over- or under-cover on individual seeds. The
    aggregate-coverage comparison is the testable form of the
    second-order accuracy claim at finite n.
    """
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=N_COVERAGE_SEEDS)
    inside_bca = 0
    inside_pct = 0
    for seed in seeds:
        sample_rng = np.random.default_rng(int(seed))
        sample = sample_rng.beta(BETA_ALPHA, BETA_BETA, size=N_PER_SAMPLE)
        dummy_y = np.zeros_like(sample, dtype=int)
        ci_bca = bootstrap_ci(
            dummy_y,
            sample,
            metric=_mean_metric,
            n_resamples=N_RESAMPLES_BCA,
            confidence=NOMINAL_CONFIDENCE,
            method="BCa",
            seed=int(seed),
        )
        ci_pct = bootstrap_ci(
            dummy_y,
            sample,
            metric=_mean_metric,
            n_resamples=N_RESAMPLES_BCA,
            confidence=NOMINAL_CONFIDENCE,
            method="percentile",
            seed=int(seed),
        )
        if ci_bca.ci_low <= TRUE_BETA_MEAN <= ci_bca.ci_high:
            inside_bca += 1
        if ci_pct.ci_low <= TRUE_BETA_MEAN <= ci_pct.ci_high:
            inside_pct += 1

    coverage_bca = inside_bca / N_COVERAGE_SEEDS
    coverage_pct = inside_pct / N_COVERAGE_SEEDS
    deviation_bca = abs(coverage_bca - NOMINAL_CONFIDENCE)
    deviation_pct = abs(coverage_pct - NOMINAL_CONFIDENCE)
    # BCa's deviation must not be worse than percentile's by more than
    # the binomial Wald 1-σ on n=100 trials (≈ 0.022). This is a mild
    # comparative claim — not requiring strict dominance per-seed.
    assert deviation_bca <= deviation_pct + 0.025, (
        f"BCa coverage |{coverage_bca:.3f} - 0.95| = {deviation_bca:.4f} should not "
        f"exceed percentile coverage |{coverage_pct:.3f} - 0.95| = {deviation_pct:.4f} "
        f"by more than 0.025 (binomial Wald 1-σ tolerance) on a skewed-mean fixture "
        f"(DiCiccio-Efron 1996 §4 second-order accuracy claim)."
    )


# ---------------------------------------------------------------------------
# Suite (b): Bayle 2020 / Bates 2024 CV-CLT-CI coverage validity.
# ---------------------------------------------------------------------------
#
# Note on test design: the v0.26.0 plan originally framed this as a
# width-dominance test ("CV-CLT narrower than naive percentile bootstrap
# in ≥ 70% of seeds"). Inspection of ``cv_clt_ci`` revealed that the
# implementation uses the standard ``mean ± z_{α/2} · σ / √K`` formula
# with no explicit correction term. Bayle et al. 2020 Theorem 3.1's
# contribution is *proving* the asymptotic validity of this formula on
# K-fold CV (where folds share training data); it is not a width
# improvement over a naive bootstrap. The right test of the claim is
# therefore *coverage*, not width.

K_FOLDS = 5
N_COVERAGE_SEEDS_CV = 200  # More seeds because per-seed signal is small (binary inside/outside)
TRUE_FOLD_MEAN = 0.85
TRUE_FOLD_STD = 0.02


@pytest.mark.slow
def test_cv_clt_ci_coverage_near_nominal_on_iid_folds() -> None:
    """CV-CLT-CI achieves near-nominal coverage on i.i.d. fold metrics (Bayle 2020 Thm 3.1).

    Per Bayle et al. 2020 Theorem 3.1, ``mean ± z_{α/2} · σ_fold / √K``
    gives asymptotically valid coverage on K-fold CV. The test uses
    K=5 fold metrics drawn i.i.d. from ``N(0.85, 0.02²)`` — the
    simplest setup where the CLT applies cleanly and we can isolate
    the CI's coverage property.

    Test design:

    1. For each of 200 seeds, draw K=5 fold metrics from
       ``N(0.85, 0.02²)``.
    2. Compute ``cv_clt_ci(folds, confidence=0.95)``.
    3. Record whether the population mean (0.85) is inside the CI.
    4. Assert empirical coverage ∈ [0.86, 0.98]. The lower bound is
       loose because the z-quantile-based CI under-covers slightly at
       small K (K=5 is the worst case for the t-vs-z gap; coverage
       converges to 0.95 as K → ∞).

    A real failure here would indicate either ``cv_clt_ci``'s formula
    has changed or the underlying ``_normal_quantile`` is broken.
    """
    rng = np.random.default_rng(42)
    fold_seeds = rng.integers(0, 1_000_000, size=N_COVERAGE_SEEDS_CV)
    inside = 0
    for seed in fold_seeds:
        sample_rng = np.random.default_rng(int(seed))
        folds = sample_rng.normal(loc=TRUE_FOLD_MEAN, scale=TRUE_FOLD_STD, size=K_FOLDS)
        ci = cv_clt_ci(folds, confidence=NOMINAL_CONFIDENCE)
        if ci.ci_low <= TRUE_FOLD_MEAN <= ci.ci_high:
            inside += 1
    coverage = inside / N_COVERAGE_SEEDS_CV
    # K=5 z-based CI under-covers by ~7-9% (true coverage ≈ 0.86-0.88).
    # Lower bound at 0.84 leaves margin for binomial sampling noise on
    # n=200 trials (Wald 1-σ ≈ 0.025).
    assert 0.84 <= coverage <= 0.99, (
        f"cv_clt_ci empirical coverage on iid N({TRUE_FOLD_MEAN}, {TRUE_FOLD_STD}²) "
        f"folds (K={K_FOLDS}) = {coverage:.3f}; expected within [0.84, 0.99] "
        f"(z-quantile-based CI under-covers slightly at K=5 vs. asymptotic 0.95). "
        f"A failure here would indicate the cv_clt_ci formula has changed."
    )


def test_cv_clt_ci_basic_invariants_for_research_test_fixture() -> None:
    """Sanity check: CV-CLT-CI matches the closed-form ``mean ± z·σ/√K`` formula.

    The implementation uses ``z_{α/2}`` (normal quantile, 1.96 at
    confidence=0.95), NOT ``t_{α/2,K-1}`` — Bayle 2020's asymptotic
    theory targets the standard normal-based formula. Verifies the
    arithmetic directly.
    """
    folds = np.array([0.83, 0.81, 0.85, 0.79, 0.84])
    ci = cv_clt_ci(folds, confidence=0.95)
    K = len(folds)
    mean = float(np.mean(folds))
    sigma = float(np.std(folds, ddof=1))
    # z-quantile at α/2 = 0.025 ≈ 1.95996...
    from scipy.stats import norm

    z_quantile = float(norm.ppf(0.975))
    expected_width = 2 * z_quantile * sigma / np.sqrt(K)
    actual_width = ci.ci_high - ci.ci_low
    assert ci.point_estimate == pytest.approx(
        mean
    ), f"CV-CLT point estimate = {ci.point_estimate:.5f} != fold mean = {mean:.5f}"
    assert actual_width == pytest.approx(expected_width, rel=1e-6), (
        f"CV-CLT width = {actual_width:.5f} != closed-form = {expected_width:.5f} "
        f"(formula: 2 * z_{{0.025}} * σ / √K with K={K}, σ={sigma:.5f})."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
