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

# Worked example: `plot_pareto_frontier` — cost vs performance

> **What this shows.** Cost-vs-performance scatter with the Pareto
> frontier (non-dominated points) overlaid. Dominated points are shown
> in a muted color. Use case: multi-rung evaluation where each rung has a
> different (compute_cost, AUPRC) tradeoff and you need to surface the
> defensible operating points. Shipped in v0.33.0 (closes upstream
> issue #15).
>
> **Runtime:** <1 s. Requires `[plotting]` extra.

## Setup

```{code-cell}
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval_toolkit import plot_pareto_frontier
```

## Synthetic data: rungs with cost/performance tradeoff

Build a scenario where 8 model rungs trade off compute cost against AUPRC
— some dominated, some on the frontier:

```{code-cell}
# (cost, auprc) per rung — cost in arbitrary units.
cost = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0])
auprc = np.array([0.60, 0.70, 0.71, 0.78, 0.82, 0.81, 0.85, 0.86])
labels = [f"rung_{i}" for i in range(len(cost))]
```

Rungs 2 (cost=3, auprc=0.71) and 5 (cost=10, auprc=0.81) are *dominated* by
neighbors: rung 1 (cost=2, auprc=0.70) gives nearly the same AUPRC at
lower cost; rung 4 (cost=8, auprc=0.82) gives higher AUPRC at lower cost
than rung 5. Pareto frontier highlights this.

## Basic frontier

```{code-cell}
fig = plot_pareto_frontier(
    cost, auprc,
    point_labels=labels,
    title="AUPRC vs compute cost (per-rung)",
)
plt.close(fig)
```

The frontier line connects non-dominated points; dominated points are
plotted in `PALETTE["baseline"]` (muted) — the visual contrast makes the
defensible-points obvious at a glance.

## With caller-managed `ax`

```{code-cell}
fig, ax = plt.subplots(figsize=(7, 5))
plot_pareto_frontier(
    cost, auprc,
    point_labels=labels,
    ax=ax,
    title="Cost-vs-AUPRC with frontier overlay",
)
plt.close(fig)
```

## Lower-metric-is-better case

By default `higher_metric_is_better=True` (higher metric = preferred).
For metrics where lower is better (latency, calibration error, etc.) flip
the flag:

```{code-cell}
latency_ms = np.array([10.0, 15.0, 25.0, 50.0, 80.0])
ece = np.array([0.08, 0.05, 0.04, 0.03, 0.04])  # lower ECE = better calibration

fig = plot_pareto_frontier(
    latency_ms, ece,
    point_labels=[f"calib_v{i}" for i in range(5)],
    higher_metric_is_better=False,  # lower ECE = better
    title="ECE vs latency (lower = better both axes)",
)
plt.close(fig)
```

## Common pitfalls

- **Cost is always lower-is-better**: the function's signature is
  `(cost, metric)` — cost is always treated as "lower is better". To
  invert the cost axis, negate the input array.
- **`point_labels` length must match arrays**: a `ValueError` is raised
  if `len(point_labels) != len(cost)`. Pass `None` to omit annotations.
- **Tied points**: when two rungs have identical `(cost, metric)`, the
  frontier-membership tie-break is index order (first-wins). Visually
  imperceptible; semantically: both are on the frontier.

## See also

- {func}`~eval_toolkit.plotting.plot_metric_bars` for the per-rung
  headline comparison (when cost is not the secondary axis)
- The consumer's Phase 4 F1 figure (per ADR-046 Q6) is the prototypical
  use case driving this primitive
