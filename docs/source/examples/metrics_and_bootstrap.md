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

# Worked example: metrics + bootstrap CIs

> **What this shows.** Compute PR-AUC / ROC-AUC / Brier on a synthetic
> binary-classification fixture via the v1.0 primary surface —
> `scorecard()` with `metric_specs` — which returns each cell with its
> 95% bootstrap CI attached. Then drop down to `bootstrap_ci` for
> bespoke CI configurations.
>
> **Runtime:** ~1 s on a laptop. Pure numpy/scipy/sklearn core — no
> optional dependencies.

## Setup

```{code-cell}
import numpy as np
from eval_toolkit import (
    scorecard,
    metric_specs as ms,
    bootstrap_ci,
    set_global_seeds,
)
from eval_toolkit.metrics import pr_auc  # internal API; needed for bespoke CIs below
set_global_seeds(42)
```

## Synthetic data: 200-row balanced binary classifier

A toy ground-truth labels + scores from a discriminative-but-noisy
model. The signal is `+0.3` on the positives, plus Gaussian noise.

```{code-cell}
rng = np.random.default_rng(42)
n = 200
y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
rng.shuffle(y_true)
y_score = np.clip(
    0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.2, size=n),
    0.0, 1.0,
)
```

## Primary surface: `scorecard()` with bootstrap CIs

The v1.0 entry point is `scorecard(y_true, y_score, metrics=[...])`,
which returns a `Mapping[str, MetricResult]`. Each `MetricResult`
carries `value`, `status`, an optional `BootstrapCI`, and (per ADR 0002)
is the **stable** way to reach metric values:

```{code-cell}
r = scorecard(
    y_true, y_score,
    metrics=[ms.pr_auc, ms.roc_auc, ms.brier],
    n_resamples=200,
    seed=42,
)
for name in ("pr_auc", "roc_auc", "brier"):
    cell = r[name]
    print(
        f"{name}={cell.value:.3f}  "
        f"[95% CI: {cell.ci.ci_low:.3f}, {cell.ci.ci_high:.3f}]"
    )
assert all(r[k].status == "ok" for k in r)
```

The signal-to-noise here gives ~0.85 AUC / ~0.85 AP. Brier ~0.09 (good
calibration on this fixture because the scores are well-spread).

## Bespoke CIs via `bootstrap_ci`

`bootstrap_ci` is the lower-level entry point used by `scorecard()`
under the hood. Reach for it when you need a metric that isn't shipped
in `metric_specs`, or when you want non-default CI knobs (e.g., a
`percentile` fallback for tiny samples). The scalar metric functions
live under `eval_toolkit.metrics` per ADR 0002:

```{code-cell}
ci_ap = bootstrap_ci(y_true, y_score, metric=pr_auc, n_resamples=200, seed=42)
print(f"pr_auc = {ci_ap.point_estimate:.3f}  [95% CI: {ci_ap.ci_low:.3f}, {ci_ap.ci_high:.3f}]")
assert ci_ap.ci_low <= ci_ap.point_estimate <= ci_ap.ci_high
assert ci_ap.confidence == 0.95
assert ci_ap.method == "BCa"
```

## When to use percentile instead of BCa

BCa is the default. Fall back to `method="percentile"` for very small
samples where BCa's jackknife step can degenerate:

```{code-cell}
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
