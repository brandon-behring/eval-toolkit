---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Worked example: paired bootstrap comparison

> **What this shows.** Compare two scorers on the same slice via
> `paired_bootstrap_diff`. The output's CI tells you whether one
> scorer is *statistically* better than the other on this slice —
> not just "the point estimate is higher."
>
> **Runtime:** ~2 s. Pure-numpy/scipy core; no optional deps.

## Why paired (not independent)

When two scorers are evaluated on the **same** y_true samples, their
scores are correlated. An unpaired comparison treats them as if they
came from independent data — wider CIs, statistical power wasted.
`paired_bootstrap_diff` resamples *indices* (instead of separate
arrays per scorer), preserving the within-sample correlation.

The classic illustration: if scorer B agrees with scorer A on 99 of
100 samples but differs on the one tricky one, the paired test
correctly attributes the small difference; the unpaired test sees
two near-identical distributions and concludes "no signal."

## Setup

```{code-cell}
import numpy as np
from eval_toolkit import paired_bootstrap_diff, set_global_seeds
from eval_toolkit.metrics import pr_auc
set_global_seeds(42)
```

## Synthetic data: two scorers on the same labels

Build a scenario where scorer B is *slightly* better than A (small but
real signal). The labels are shared; only the score arrays differ:

```{code-cell}
rng = np.random.default_rng(42)
n = 200
y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
rng.shuffle(y_true)

# Scorer A: discriminative-but-noisy
s_a = np.clip(0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)

# Scorer B: same shape but with stronger signal
s_b = np.clip(0.5 + 0.4 * (y_true - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)

print(f"pr_auc(A) = {pr_auc(y_true, s_a):.3f}")
print(f"pr_auc(B) = {pr_auc(y_true, s_b):.3f}")
print(f"point delta = {pr_auc(y_true, s_b) - pr_auc(y_true, s_a):.3f}")
```

## Paired bootstrap CI on the difference

`paired_bootstrap_diff(y, s_a, s_b, metric=pr_auc, n_resamples=...)`
returns a `PairedBootstrapCI` with the **delta**'s CI bounds. If the CI
excludes zero, the difference is significant at the configured
confidence level (default 95%):

```{code-cell}
result = paired_bootstrap_diff(
    y_true, s_a, s_b, metric=pr_auc, n_resamples=500, seed=42,
)
print(f"delta = {result.delta:.3f}  [95% CI: {result.ci_low:.3f}, {result.ci_high:.3f}]")
significant = result.ci_low > 0 or result.ci_high < 0
print(f"Significant at 95% confidence: {significant}")
```

## Reading the output

- **`delta` = `pr_auc(B) - pr_auc(A)`** point estimate (sign convention:
  positive means B is better)
- **`ci_low`, `ci_high`** are the BCa-style 95% CI bounds on the delta
- If `ci_low > 0`, B is significantly better (at α=0.05)
- If `ci_high < 0`, A is significantly better
- If the CI straddles zero, the difference is not significant — even if
  the point estimate suggests one direction

## MDE: how much delta could you have detected?

`mde_from_ci` complements `paired_bootstrap_diff` by reporting the
**minimum detectable effect** given the CI width — useful when the
result is "no significant difference" and you want to claim "we would
have caught at least a delta of X if there were one":

```{code-cell}
from eval_toolkit import mde_from_ci
mde = mde_from_ci(result, alpha=0.05, power=0.80)
print(f"MDE at 80% power: {mde.mde:.3f}")
print(f"  (vs observed delta: {mde.delta_observed:.3f})")
```

If `mde.mde = 0.05`, you had power to detect a 5pp improvement in
pr_auc; smaller real differences would have been underpowered. Pair
this with the CI from `paired_bootstrap_diff` to make claims like "no
significant difference, and we had power to detect ≥ X".

## Pre-1.0 design note

`paired_bootstrap_diff` defaults to BCa quantile arithmetic for the
delta's CI. Asymptotic normality is *not* assumed — bootstrap handles
skewed distributions natively. For multi-comparison settings (e.g.,
testing scorer B vs A across 6 slices), apply Bonferroni / BH-FDR
correction to the resulting p-values; see issue
[#1](https://github.com/brandon-behring/eval-toolkit/issues/1).

## See also

- [`bootstrap.py` reference](../api/bootstrap.md) —
  `paired_bootstrap_diff`, `paired_mde`, `mde_from_ci`,
  `paired_bootstrap_op_point_diff` for operating-point comparisons.
- [`metrics.py` reference](../api/metrics.md) — any metric with the
  `(y, s) -> float` signature works as the `metric` argument.
- [Metrics + bootstrap example](metrics_and_bootstrap.md) — the
  single-scorer baseline.
- [Evaluate harness example](evaluate_harness.md) — pass
  `paired_diffs=[("A", "B")]` to `evaluate(...)` to get this automatically
  per slice.
