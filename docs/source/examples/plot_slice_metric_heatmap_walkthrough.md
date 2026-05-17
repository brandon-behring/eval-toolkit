# Worked example: `plot_slice_metric_heatmap` — stratified metric grid

> **What this shows.** A 2-D heatmap of `(row_label × col_label → metric)`
> values with colorbar + optional per-cell annotations. Use case: rung ×
> OOD-slice AUPRC grid; model × dataset accuracy matrix; method × fold
> performance table. Shipped in v0.33.0 (closes upstream issue #16).
>
> **Runtime:** <1 s. Requires `[plotting]` extra.

## Setup

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval_toolkit import plot_slice_metric_heatmap
```

## Synthetic data: 5 rungs × 6 OOD slices

Build an AUPRC grid where some rungs are uniformly strong, some are
slice-specific:

```python
rung_labels = ["baseline", "minilm", "deberta", "gpt4_zs", "gpt4_fs"]
slice_labels = ["ood_a", "ood_b", "ood_c", "ood_d", "ood_e", "id_holdout"]

# Random-but-plausible AUPRC values; in production this comes from
# eval-toolkit's evaluate() per (scorer, slice) pair.
rng = np.random.default_rng(42)
grid = rng.uniform(0.55, 0.95, size=(len(rung_labels), len(slice_labels)))
# Make one rung (gpt4_fs) uniformly strong:
grid[4] = np.maximum(grid[4], 0.85)
# Make one slice (ood_e) uniformly hard:
grid[:, 4] = np.minimum(grid[:, 4], 0.65)
```

## Basic heatmap

```python
fig = plot_slice_metric_heatmap(
    grid,
    row_labels=rung_labels,
    col_labels=slice_labels,
    metric_name="AUPRC",
    title="Per-rung × per-slice AUPRC",
)
plt.close(fig)
```

Per-cell annotations (the AUPRC values) are drawn by default
(`annotate=True`); the colormap defaults to `viridis`. The colorbar uses
the supplied `metric_name` for its label.

## Without per-cell annotations

For dense grids (e.g., 20 rungs × 30 slices = 600 cells) annotations get
visually busy. Disable them:

```python
big_grid = rng.uniform(0.5, 0.95, size=(20, 30))
fig = plot_slice_metric_heatmap(
    big_grid,
    row_labels=[f"r{i}" for i in range(20)],
    col_labels=[f"s{i}" for i in range(30)],
    metric_name="AUPRC",
    annotate=False,  # too dense to annotate readably
    figsize=(12, 6),
)
plt.close(fig)
```

## With NaN cells (intentional gaps)

Some `(rung, slice)` pairs may be intentionally un-evaluated (e.g., rung
doesn't apply to certain slice types). Pass `np.nan` for those cells; the
heatmap masks them in a neutral color:

```python
grid_with_gaps = grid.copy()
grid_with_gaps[0, 5] = np.nan  # baseline rung not evaluated on id_holdout
grid_with_gaps[1, 4] = np.nan  # minilm not evaluated on ood_e
fig = plot_slice_metric_heatmap(
    grid_with_gaps,
    row_labels=rung_labels,
    col_labels=slice_labels,
    metric_name="AUPRC",
    title="With intentional gaps (NaN cells)",
)
plt.close(fig)
```

## With caller-managed `ax`

```python
fig, ax = plt.subplots(figsize=(8, 5))
plot_slice_metric_heatmap(
    grid,
    row_labels=rung_labels,
    col_labels=slice_labels,
    ax=ax,
    metric_name="AUPRC",
)
plt.close(fig)
```

## Common pitfalls

- **Shape mismatch**: the function raises `ValueError` if
  `grid.shape != (len(row_labels), len(col_labels))`. Validate upstream.
- **Annotation format**: `annot_fmt="{:.3f}"` is the default; use
  `"{:.0%}"` for percentages or `"{:.2g}"` for compact scientific. Numbers
  outside the colormap's perceptual range (e.g., `1e9`) just won't render
  legibly — pick a `cmap=` whose midpoint matches your value range.
- **Colormap choice**: `"viridis"` is the default (perceptually uniform,
  colorblind-safe). For diverging metrics (e.g., delta-AUPRC vs baseline,
  where 0 is the reference), pass `cmap="RdBu_r"` and center the
  normalisation.

## See also

- {func}`~eval_toolkit.plotting.plot_metric_bars` for the per-slice
  (1-D) view — use when you only have one row dimension
- {func}`~eval_toolkit.plotting.plot_confusion_matrix_grid` for the
  confusion-matrix-shaped equivalent (the only v0.33.0+ plotting fn that
  doesn't accept `ax=` because it's intrinsically grid-shaped)
