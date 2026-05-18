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

# Worked example: `plot_roc_curve` — annotated ROC with threshold marker

> **What this shows.** ROC curve rendering for a binary classifier, with
> an optional decision-threshold marker + baseline overlay. Sibling
> primitive to `plot_pr_curve`. Shipped in v0.33.0 (closes upstream
> issue #14).
>
> **Runtime:** <1 s. Requires `[plotting]` extra.

## Setup

```{code-cell}
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for docs build
import matplotlib.pyplot as plt
from eval_toolkit import plot_roc_curve, set_global_seeds
set_global_seeds(42)
```

## Synthetic data: discriminative scorer

```{code-cell}
rng = np.random.default_rng(42)
n = 200
y = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
rng.shuffle(y)
# Signal-with-noise: positives have a 0.5-shift over negatives.
score = y * 0.5 + rng.uniform(0, 0.5, size=n)
```

## Basic ROC

```{code-cell}
fig = plot_roc_curve(y, score, title="Synthetic discriminative scorer")
# fig is a matplotlib.figure.Figure — caller owns lifecycle.
assert fig is not None
plt.close(fig)
```

The diagonal chance line is drawn automatically (per `plot_pr_curve`'s
prevalence-line convention).

## With threshold marker

The `threshold=` kwarg marks the (FPR, TPR) pair closest to the requested
score-threshold — useful for showing where a deployed decision boundary
sits on the ROC.

```{code-cell}
fig = plot_roc_curve(
    y, score,
    threshold=0.5,  # deployment threshold
    title="ROC with decision threshold @ 0.5",
)
plt.close(fig)
```

## With baseline overlay

For comparing against a baseline scorer's ROC (e.g., last release vs
current), pass `baseline_curve=(fpr, tpr)`:

```{code-cell}
from sklearn.metrics import roc_curve as sklearn_roc

# A weaker baseline scorer for comparison.
baseline_score = y * 0.2 + rng.uniform(0, 0.5, size=n)
fpr_base, tpr_base, _ = sklearn_roc(y, baseline_score)

fig = plot_roc_curve(
    y, score,
    label="current",
    baseline_curve=(fpr_base, tpr_base),
    baseline_label="prior_release",
    title="Current vs prior-release ROC",
)
plt.close(fig)
```

The baseline is drawn in a muted color (the `PALETTE["baseline"]` slot);
the current scorer in the accent color.

## With caller-managed `ax`

For composite figures (e.g., side-by-side ROC + PR), pass an existing
`Axes`:

```{code-cell}
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
plot_roc_curve(y, score, ax=ax_roc, title="ROC")
# (plot_pr_curve(y, score, ax=ax_pr) — analogous)
plt.close(fig)
```

`plot_roc_curve` returns the parent `Figure` regardless of the `ax` path
(consistent with `plot_pr_curve` and the other v0.33.0+ plotting fns —
all 6 accept `ax=` except `plot_confusion_matrix_grid` which is
intrinsically grid-shaped).

## Common pitfalls

- **Single-class `y_true`**: ROC is undefined when `y_true` is all 0s or
  all 1s. The fn raises `ValueError` — guard upstream for slices that
  might collapse to a single class.
- **Score scale**: `y_score` must be monotone-equivalent to "higher =
  more positive". If your scores are inverted (lower = more positive),
  negate them before plotting.

## See also

- {func}`~eval_toolkit.plotting.plot_pr_curve` ([API](../api/plotting.md))
  — sibling primitive for precision-recall curves
- {func}`~eval_toolkit.plotting.plot_reliability_diagram` for the
  calibration story
