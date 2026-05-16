# Worked example: metrics + bootstrap CIs

> **What this shows.** Compute `pr_auc` / `roc_auc` / `brier_score` on a
> synthetic binary-classification fixture, then attach a 95% bootstrap CI
> via `bootstrap_ci`. The minimal entry point into the toolkit.
>
> **Runtime:** ~1 s on a laptop. Pure numpy/scipy/sklearn core — no
> optional dependencies.

## Setup

```python
import numpy as np
from eval_toolkit import (
    pr_auc, roc_auc, brier_score,
    bootstrap_ci, set_global_seeds,
)
set_global_seeds(42)
```

## Synthetic data: 200-row balanced binary classifier

A toy ground-truth labels + scores from a discriminative-but-noisy
model. The signal is `+0.3` on the positives, plus Gaussian noise.

```python
rng = np.random.default_rng(42)
n = 200
y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
rng.shuffle(y_true)
y_score = np.clip(
    0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.2, size=n),
    0.0, 1.0,
)
```

## Point estimates: pr_auc, roc_auc, brier_score

Each is a single function call returning a float:

```python
ap = pr_auc(y_true, y_score)
auc = roc_auc(y_true, y_score)
bs = brier_score(y_true, y_score)
assert 0.0 <= ap <= 1.0
assert 0.0 <= auc <= 1.0
assert 0.0 <= bs <= 1.0
print(f"pr_auc={ap:.3f}  roc_auc={auc:.3f}  brier={bs:.3f}")
```

The signal-to-noise here gives ~0.85 AUC / ~0.85 AP. Brier ~0.09 (good
calibration on this fixture because the scores are well-spread).

## 95% bootstrap CI

`bootstrap_ci` wraps `scipy.stats.bootstrap` with BCa as the default
method and produces a `BootstrapCI` dataclass with `point_estimate`,
`ci_low`, `ci_high`:

```python
ci_ap = bootstrap_ci(y_true, y_score, metric=pr_auc, n_resamples=200, seed=42)
print(f"pr_auc = {ci_ap.point_estimate:.3f}  [95% CI: {ci_ap.ci_low:.3f}, {ci_ap.ci_high:.3f}]")
assert ci_ap.ci_low <= ci_ap.point_estimate <= ci_ap.ci_high
assert ci_ap.confidence == 0.95
assert ci_ap.method == "BCa"
```

## When to use percentile instead of BCa

BCa is the default. Fall back to `method="percentile"` for very small
samples where BCa's jackknife step can degenerate:

```python
ci_perc = bootstrap_ci(
    y_true, y_score, metric=pr_auc,
    n_resamples=200, method="percentile", seed=42,
)
print(f"pr_auc (percentile) = [{ci_perc.ci_low:.3f}, {ci_perc.ci_high:.3f}]")
```

The percentile CI is symmetric around the point estimate; BCa's bias-
correction (`a-hat`) makes it asymmetric when the bootstrap distribution
is skewed. For most well-conditioned cases the two methods agree to
within ~0.01.

## Pre-1.0 design note

`bootstrap_ci` rejects `n < 10` with `ValueError` (too few samples for
meaningful bootstrap variance). NaN/Inf scores are rejected by the
underlying metrics — see
[NaN/Inf rejection tests](https://github.com/brandon-behring/eval-toolkit/blob/main/tests/test_metrics_props.py)
for the full input-validation contract.

## See also

- [`metrics.py` reference](../api/metrics.md) — full list of available
  metrics (PR-AUC, ROC-AUC, Brier, ECE variants, `headline_metrics`
  bundle).
- [`bootstrap.py` reference](../api/bootstrap.md) — `paired_bootstrap_diff`
  for two-scorer comparisons, `cv_clt_ci` for cross-validated CIs.
- [Evaluate harness example](evaluate_harness.md) — same metrics applied
  via the slice-aware orchestrator.
