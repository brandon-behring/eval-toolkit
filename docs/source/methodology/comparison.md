# Comparison & confidence intervals

> **Background** *(skip if you've internalized this)*. A point estimate of
> a metric without a confidence interval is a vibe. Two-model comparisons
> that don't account for paired data (same eval rows, two models) lose
> power. "Model B is better than A" is a *statistical* claim that
> requires a CI on the difference, not just two separate CIs that happen
> to look apart. This chapter covers the bootstrap-based machinery
> eval-toolkit ships and when to reach for things it doesn't (McNemar,
> DeLong).

## Setup

```python
import numpy as np
from eval_toolkit import (
    bootstrap_ci, paired_bootstrap_diff, mde_from_ci, cv_clt_ci,
    pr_auc, roc_auc,
)
```

Two synthetic scorers: A (random) and B (informative).

```python
rng = np.random.default_rng(42)
y = rng.binomial(1, 0.3, size=300).astype(int)
s_a = rng.uniform(0, 1, size=300)
s_b = np.clip(0.6 * y + rng.normal(0, 0.25, size=300), 0, 1)
```

(comparison-bca-vs-percentile)=
## Single-condition CI: BCa vs percentile
A bootstrap CI for a metric resamples paired `(y_true, y_score)` indices
with replacement, computes the metric per resample, and reports a
quantile-based interval over the resampled distribution.

eval-toolkit's
[`bootstrap_ci`](../api/bootstrap.md) supports three
methods:

- **BCa** (bias-corrected, accelerated) — the default. Adjusts the
  quantile cuts for skew in the resampled distribution. Most accurate
  for moderate sample sizes (n ≥ 30); the standard recommendation in
  Efron & Tibshirani (1993).
- **percentile** — naive quantile cut, no bias correction. Slightly
  optimistic on skewed distributions but the recommended fallback when
  BCa's jackknife step is degenerate (very small n, constant scores).
- **studentized** — bootstrap-t. Rare in ML eval; ships for completeness.

```python
ci = bootstrap_ci(y, s_b, pr_auc, n_resamples=1000, method="BCa", seed=42)
print(f"BCa: PR-AUC={ci.point_estimate:.3f} CI [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")

ci_p = bootstrap_ci(y, s_b, pr_auc, n_resamples=1000, method="percentile", seed=42)
print(f"pct: PR-AUC={ci_p.point_estimate:.3f} CI [{ci_p.ci_low:.3f}, {ci_p.ci_high:.3f}]")
```

> **What NOT to do.** Don't fall back to percentile by default. BCa is
> almost always the right choice for ML eval; reserve percentile for
> the explicit fallback case (`n < 30`, constant scores, BCa raises).

(comparison-paired-bootstrap)=
## Paired bootstrap for two-model comparison
When comparing two models on the same eval set, the metric difference
$\Delta = M_B - M_A$ has *less variance* than each metric individually
because the resampling noise cancels — the same row that's hard for A
is also hard for B. Use paired bootstrap to exploit this.

eval-toolkit's
[`paired_bootstrap_diff`](../api/bootstrap.md) shares
resample indices across both scorers — the same rows enter each
resample for both A and B.

```python
diff = paired_bootstrap_diff(
    y, s_a, s_b, pr_auc, n_resamples=1000, seed=42,
)
print(f"Δ PR-AUC: {diff.delta:.3f}  CI [{diff.ci_low:.3f}, {diff.ci_high:.3f}]")
print(f"  overlaps zero: {diff.overlaps_zero}")
```

The `overlaps_zero` field is the headline finding: when the CI doesn't
contain 0, you have evidence that B is better (or worse) than A at the
chosen confidence level. When it does, you don't.

> **Why paired CIs are tighter.** If you computed two unpaired CIs
> independently and just visually compared them, you would *under-
> reject*: there's a well-known result that non-overlapping CIs imply
> a significant difference, but *overlapping* CIs do not imply
> non-significance. Always compute the paired CI explicitly.

(ece-differences)=
## ECE differences
For ECE differences, use
[`paired_bootstrap_ece_diff`](../api/bootstrap.md)
which threads `n_bins` through the metric correctly. ECE is a binned
estimator, so the resample's bin assignments need to match the
condition's bin definitions:

```python
from eval_toolkit import expected_calibration_error_l2_debiased, paired_bootstrap_ece_diff

# Using two probability-score scorers in [0, 1].
diff_ece = paired_bootstrap_ece_diff(
    y, s_a, s_b,
    ece_fn=expected_calibration_error_l2_debiased,
    n_bins=10, n_resamples=1000, seed=42,
)
print(f"Δ ECE: {diff_ece.delta:.4f}  CI [{diff_ece.ci_low:.4f}, {diff_ece.ci_high:.4f}]")
```

(two-level-bootstrap)=
## Operating-point differences (two-level bootstrap)
When the metric is *operating-point-dependent* (F1, precision at fixed
recall) and your deployment refits the threshold per data batch, use
the two-level paired bootstrap from
[thresholds.md §"When to refit threshold per resample"](thresholds.md#bootstrap-refit).

The interval is wider — it captures both metric variance AND
threshold-selection variance — which is the honest story when the
deployment isn't fixing the threshold once and forever.

(comparison-mde)=
## MDE: "we couldn't detect a difference" claims
A wide CI that overlaps zero is *not* evidence the two models perform
the same — it's evidence you don't have enough data to tell. Quantify
this with the **minimum detectable effect (MDE)**: the smallest true
difference that your bootstrap-CI procedure would have detected with
80 % power.

```python
mde = mde_from_ci(diff, alpha=0.05, power=0.80)
print(f"MDE @ 80 % power: {mde.mde:.4f}")
print(f"interpretation: differences smaller than ~{mde.mde:.3f} would not "
      "have shown up as significant with this n.")
```

If the MDE is comparable to or larger than the differences you care
about, you need more eval data — running a different statistical test
won't help.

(comparison-cv-ci)=
## CV-CI: confidence intervals from K-fold
For K-fold CV results, use
[`cv_clt_ci`](../api/bootstrap.md): a CLT-corrected
confidence interval over per-fold metric values. This is what
[`evaluate_folded`](../api/harness.md) auto-computes
and stores in `RunResult.fold_summary`.

```python
fold_metrics = np.array([0.74, 0.76, 0.71, 0.78, 0.73])
ci = cv_clt_ci(fold_metrics, confidence=0.95)
print(f"CV mean: {ci.point_estimate:.3f}  CI [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")
```

The CLT correction (Bates et al. 2024) accounts for the fact that
per-fold metrics are *not* independent — they share training data.
Naive Student's-t CIs over fold metrics are anti-conservative.

(comparison-out-of-scope)=
## What's NOT in eval-toolkit (and why)
Two classical paired tests are *deliberately* out of scope:

- **McNemar's test.** Compares the proportion of *disagreements* between
  two binary classifiers — A right / B wrong vs A wrong / B right. Use
  when you have hard predictions, not probability scores. Compute via
  [`scipy.stats.contingency`](https://docs.scipy.org/doc/scipy/reference/stats.contingency.html)
  + the McNemar `2×2` table.
- **DeLong's test.** Compares ROC-AUC between two scorers using the
  Mann-Whitney form's variance. Specific to ROC-AUC; doesn't generalize
  to PR-AUC or threshold metrics. Several Python implementations (e.g.,
  `pyroc-utils`, manual
  [DeLong implementations](https://github.com/yandexdataschool/roc_comparison)
  on GitHub).

Neither pays rent in eval-toolkit because:

1. **Bootstrap covers the same ground.** `paired_bootstrap_diff` gives
   a CI on any metric difference; McNemar and DeLong are special cases
   for binary predictions and ROC-AUC respectively.
2. **They don't generalize.** DeLong is ROC-AUC-only; McNemar is hard-
   prediction-only. The toolkit's bootstrap framework is metric-
   agnostic.
3. **Multiple-testing correction.** When comparing K > 2 models,
   bootstrap-CI on every pair is straightforward; McNemar / DeLong
   require explicit Bonferroni / FDR corrections.

If you need them anyway, both are fine to compute alongside
eval-toolkit — they'll generally agree with the bootstrap result on
informative data.

(comparison-pitfalls)=
## Pitfalls / Common mistakes
- **Comparing two unpaired CIs visually.** "B's CI starts above A's CI
  ceiling, so B is better." Mathematically: non-overlap implies
  significance, but overlap does NOT imply non-significance. Always
  compute the paired CI on the difference.
- **Treating "overlaps zero" as "no difference".** Wide CIs are about
  insufficient power, not equivalence. Report MDE alongside.
- **Bootstrapping accuracy on small slices.** BCa's jackknife step
  becomes degenerate when n is small (n < 30 is the toolkit's
  bright-line). The toolkit emits an error in that regime; fall back
  to `method="percentile"` and document the choice.
- **Comparing ECE bootstrap CIs across runs with different n_bins.** As
  in [calibration.md](calibration.md), ECE depends on bin count;
  bootstrap inherits that sensitivity.
- **Reporting bootstrap mean instead of point estimate.** The toolkit's
  `BootstrapCI.point_estimate` is the metric on the *original* data, not
  the resample mean. Report `point_estimate ± CI`, not `resample_mean
  ± CI`.

## Putting it all together

Full A-vs-B comparison report:

```python
ci_a = bootstrap_ci(y, s_a, pr_auc, n_resamples=1000, seed=42)
ci_b = bootstrap_ci(y, s_b, pr_auc, n_resamples=1000, seed=42)
diff = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=1000, seed=42)
mde = mde_from_ci(diff, alpha=0.05, power=0.80)

print(f"A: PR-AUC={ci_a.point_estimate:.3f} CI [{ci_a.ci_low:.3f}, {ci_a.ci_high:.3f}]")
print(f"B: PR-AUC={ci_b.point_estimate:.3f} CI [{ci_b.ci_low:.3f}, {ci_b.ci_high:.3f}]")
print(f"Δ: {diff.delta:.3f} CI [{diff.ci_low:.3f}, {diff.ci_high:.3f}]"
      f"  overlaps_zero={diff.overlaps_zero}")
print(f"MDE @ 80 % power: {mde.mde:.4f}")
```

## Further reading

- Efron, B. & Tibshirani, R. *An Introduction to the Bootstrap.*
  Chapman & Hall, 1993 — the canonical reference; BCa derived in §14.
- DiCiccio, T. & Efron, B. *Bootstrap confidence intervals.*
  Statistical Science 11(3), 1996 — comparison of CI methods.
- Bates, S., Hastie, T., & Tibshirani, R. *Cross-validation: what
  does it estimate and how well does it do it?* JASA 2024 — the basis
  for `cv_clt_ci`'s CLT correction.
- DeLong, E. R. et al. *Comparing the areas under two or more
  correlated ROC curves: a nonparametric approach.* Biometrics 44,
  1988.

See also: [thresholds.md](thresholds.md) (two-level bootstrap),
[calibration.md](calibration.md) (ECE differences),
[testing.md](testing.md) (property-test patterns for invariants).
