# Bootstrap

> **Background** *(skip if you've internalized this)*. The bootstrap
> turns a point estimate into a confidence interval by *resampling*
> rows from the eval set with replacement, recomputing the metric, and
> reading quantiles off the resampled distribution. It works for any
> metric, makes no parametric assumption, and gives honest CIs that
> reflect finite-sample variance. The cost: O(n × n_resamples) compute
> per CI. Modern toolkits use 1 000–10 000 resamples; eval-toolkit
> defaults to 1 000.

This chapter covers the four primitives in
`eval_toolkit.bootstrap`:
[`bootstrap_ci`](../api/bootstrap.md) (single
condition), [`paired_bootstrap_diff`](../api/bootstrap.md)
(two-condition difference), [`paired_bootstrap_op_point_diff`](../api/bootstrap.md)
(two-level: refit threshold per resample), and
[`cv_clt_ci`](../api/bootstrap.md) (K-fold CV-CI). For
a higher-level overview of model comparison see
[comparison.md](comparison.md); this chapter goes deeper on the
resampling theory.

## Setup

```python
import numpy as np
from eval_toolkit import (
    bootstrap_ci, paired_bootstrap_diff, mde_from_ci, cv_clt_ci, pr_auc,
)
```

A 200-row fixture used throughout:

```python
rng = np.random.default_rng(42)
y = rng.binomial(1, 0.3, size=200).astype(int)
s_a = rng.uniform(0, 1, size=200)
s_b = np.clip(0.6 * y + rng.normal(0, 0.25, size=200), 0, 1)
```

## BCa vs percentile {#bca-vs-percentile}

Two CI-construction methods ship in `bootstrap_ci`:

- **`method="BCa"`** (default) — Efron 1987: bias-corrected,
  accelerated. Adjusts quantile cuts for skew via a jackknife.
  Most accurate for moderate samples (n ≥ 30) and the
  recommended default in Efron & Tibshirani 1993.
- **`method="percentile"`** — naive `(α/2, 1-α/2)` quantile cut.
  Slightly optimistic on skewed distributions but the recommended
  fallback when BCa's jackknife is degenerate (very small n,
  constant scores).
- **`method="studentized"`** — bootstrap-t. Rare in ML eval; ships for
  completeness.

```python
ci_bca = bootstrap_ci(y, s_b, pr_auc, n_resamples=500, method="BCa", seed=42)
ci_pct = bootstrap_ci(y, s_b, pr_auc, n_resamples=500, method="percentile", seed=42)
print(f"BCa:        {ci_bca.point_estimate:.3f}  CI [{ci_bca.ci_low:.3f}, {ci_bca.ci_high:.3f}]")
print(f"Percentile: {ci_pct.point_estimate:.3f}  CI [{ci_pct.ci_low:.3f}, {ci_pct.ci_high:.3f}]")
```

The interval midpoints match (both methods point-estimate the original
data), but BCa shifts the bounds asymmetrically when the resampled
distribution is skewed. For roughly-symmetric metrics on moderate n
the difference is < 1 %; for highly-skewed cases (small n, rare-positive)
BCa's correction matters.

## Paired bootstrap for two-model comparison {#paired-bootstrap}

When comparing two models on the *same* eval rows, the metric
difference Δ has *less variance* than each metric individually because
the resampling noise cancels: rows that are hard for A are also hard
for B. Paired bootstrap exploits this by sharing resample indices.

```python
diff = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=500, seed=42)
print(f"Δ PR-AUC: {diff.delta:.3f}  CI [{diff.ci_low:.3f}, {diff.ci_high:.3f}]")
print(f"  overlaps zero: {diff.overlaps_zero}")
```

Compare with computing two separate CIs and eyeballing them:

```python
ci_a = bootstrap_ci(y, s_a, pr_auc, n_resamples=500, seed=42)
ci_b = bootstrap_ci(y, s_b, pr_auc, n_resamples=500, seed=42)
print(f"A CI: [{ci_a.ci_low:.3f}, {ci_a.ci_high:.3f}]")
print(f"B CI: [{ci_b.ci_low:.3f}, {ci_b.ci_high:.3f}]")
print(f"unpaired-CI width sum: {(ci_a.ci_high - ci_a.ci_low) + (ci_b.ci_high - ci_b.ci_low):.3f}")
print(f"paired-Δ width:        {diff.ci_high - diff.ci_low:.3f}")
```

The paired CI is reliably tighter — typically 30–50 % narrower for
informative score pairs.

## Two-level paired bootstrap (refit per resample) {#two-level}

When the metric is *operating-point-dependent* (F1, precision-at-recall)
and the deployment refits the threshold per data batch, the
single-level bootstrap underestimates uncertainty. The two-level
variant ([`paired_bootstrap_op_point_diff`](../api/bootstrap.md))
resamples a *validation* slice, refits the threshold on that
resample, then computes the metric on the *test* resample.

The CI captures both metric variance AND threshold-selection variance.
Empirically 1.5–2× wider than single-level. Reporting the narrower
single-level CI when the deployment refits is overconfident — see
[thresholds.md §"When to refit threshold per resample"](thresholds.md#bootstrap-refit)
for the operational decision.

## Minimum detectable effect (MDE) {#mde}

A wide CI overlapping zero isn't evidence the two models perform the
same — it's evidence you don't have power to tell. Quantify with MDE:
the smallest true difference your bootstrap procedure would detect at
80 % power.

```python
mde = mde_from_ci(diff, alpha=0.05, power=0.80)
print(f"MDE @ 80 % power: {mde.mde:.4f}")
```

If your MDE is 0.03 and the difference you care about is 0.01, you
need more eval data — running a different statistical test won't
help.

## CV-CI: K-fold bootstrap {#cv-ci}

[`cv_clt_ci`](../api/bootstrap.md) computes a
CLT-corrected confidence interval over per-fold metric values. Per-fold
metrics are *not* independent — they share training data, so naive
Student's-t CIs over fold metrics are anti-conservative. The
correction (Bates et al. 2024) accounts for this.

```python
fold_metrics = np.array([0.74, 0.76, 0.71, 0.78, 0.73])
ci_cv = cv_clt_ci(fold_metrics, confidence=0.95)
print(f"CV mean: {ci_cv.point_estimate:.3f}  CI [{ci_cv.ci_low:.3f}, {ci_cv.ci_high:.3f}]")
```

`evaluate_folded(...)` auto-computes this for every (slice, scorer,
metric) triple and stores it in `RunResult.fold_summary` — see
[splits.md §"K-fold cross-validation"](splits.md#stratified-kfold).

## Resample budget guidance {#budget}

| n_resamples | Use case |
|---|---|
| 200 | Quick sanity check during development; CIs accurate to ~3 % |
| 1 000 | **Default**; CIs accurate to ~1 % at the 95 % level |
| 5 000 | Publication-grade; use when reporting lift to a paper or external stakeholder |
| 10 000+ | Diminishing returns; only if the metric is very expensive to compute (LLM-judge) and you want maximum CI precision |

The resampling cost is O(n × n_resamples × metric_cost). For
LLM-judge scorers where the metric implicitly requires LLM calls, this
explodes — pre-compute scores once, then resample on the score arrays
(the toolkit's pattern: caller produces `(y_true, y_score)` arrays
externally and feeds them in).

## Pitfalls / Common mistakes {#pitfalls}

- **Treating non-overlapping CIs as significance.** Non-overlap implies
  significance, but overlap does NOT imply non-significance. Always
  compute the paired CI on the difference, not two separate CIs.
- **Reporting bootstrap mean instead of point estimate.**
  `BootstrapCI.point_estimate` is the metric on the *original* data,
  not the resample mean. Report `point_estimate ± CI`, not
  `resample_mean ± CI`.
- **Bootstrapping accuracy on n < 30.** BCa's jackknife is degenerate.
  The toolkit emits an error in that regime; fall back to
  `method="percentile"` and document the choice.
- **Not seeding.** `bootstrap_ci(..., seed=42)` makes runs reproducible.
  An unseeded run will give a slightly different CI on every
  invocation — annoying for golden-test discipline and CI flakiness.
- **Comparing ECE bootstrap CIs across runs with different `n_bins`.**
  ECE depends on bin count; bootstrap inherits that sensitivity. Pin
  `n_bins` in your project config.
- **Bootstrapping on the train set.** The bootstrap quantifies sample
  variance of *your eval set*, not generalization to a new population.
  For OOD claims you still need a held-out test set — see
  [splits.md §"When CV alone is insufficient"](splits.md#cv-and-ood).

## Further reading

- Efron, B. & Tibshirani, R. *An Introduction to the Bootstrap.*
  Chapman & Hall, 1993. **The canonical reference.** §14 derives BCa.
- DiCiccio, T. & Efron, B. *Bootstrap confidence intervals.*
  Statistical Science 11(3), 1996. — comparison of CI methods + when
  each fails.
- Bates, S., Hastie, T., & Tibshirani, R. *Cross-validation: what
  does it estimate and how well does it do it?* JASA 2024. — basis
  for `cv_clt_ci`'s CLT correction.
- Davison, A. C. & Hinkley, D. V. *Bootstrap Methods and their
  Application.* Cambridge, 1997. — alternate canonical text;
  good complement to Efron & Tibshirani.

See also: [comparison.md](comparison.md) (higher-level model
comparison framing), [splits.md](splits.md) (K-fold context for
`cv_clt_ci`), [calibration.md](calibration.md) (paired-ECE-difference
specifics).
