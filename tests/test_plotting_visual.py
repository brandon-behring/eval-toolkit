"""pytest-mpl visual-regression baselines for the plot helpers.

Each test renders a deterministic figure and is compared (pixel-wise via
``pytest-mpl``) against a baseline PNG checked in at
``tests/baseline/test_plotting_visual/<test_name>.png``.

Run with:
- Generate baselines: ``pytest --mpl-generate-path=tests/baseline tests/test_plotting_visual.py``
- Validate against baselines: ``pytest --mpl tests/test_plotting_visual.py``

Without the ``--mpl`` flag, these tests just verify the helpers run end-to-end
and return a valid Figure (smoke-test fallback).

Closes audit gap #27.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from matplotlib.figure import Figure

from eval_toolkit.plotting import (
    plot_bootstrap_distribution,
    plot_confusion_matrix_grid,
    plot_lift_ci,
    plot_metric_bars,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_score_histograms,
)


@dataclass(frozen=True, slots=True)
class _StubCI:
    """Duck-typed CI for plot_lift_ci."""

    point_estimate: float
    ci_low: float
    ci_high: float


def _deterministic_data(seed: int = 0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible (y, s) pair for visual-regression tests."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    s = np.clip(y + rng.normal(0, 0.3, size=n), 0.0, 1.0).astype(np.float64)
    return y, s


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_pr_curve.png",
    tolerance=15,
)
def test_plot_pr_curve_baseline() -> Figure:
    y, s = _deterministic_data()
    return plot_pr_curve(y, s, label="model", threshold=0.5, prevalence=0.5)


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_reliability_diagram.png",
    tolerance=15,
)
def test_plot_reliability_diagram_baseline() -> Figure:
    y, s = _deterministic_data()
    return plot_reliability_diagram(y, s, n_bins=10)


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_confusion_matrix_grid.png",
    tolerance=15,
)
def test_plot_confusion_matrix_grid_baseline() -> Figure:
    matrices = {
        "model_a": np.array([[40, 5], [3, 12]]),
        "model_b": np.array([[42, 3], [4, 11]]),
    }
    return plot_confusion_matrix_grid(matrices, class_names=("negative", "positive"))


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_metric_bars.png",
    tolerance=15,
)
def test_plot_metric_bars_baseline() -> Figure:
    values = {"slice_a": 0.85, "slice_b": 0.72, "slice_c": 0.91}
    return plot_metric_bars(values, ylabel="PR-AUC")


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_score_histograms.png",
    tolerance=15,
)
def test_plot_score_histograms_baseline() -> Figure:
    rng = np.random.default_rng(0)
    scores = {
        "slice_x": rng.uniform(0, 1, 100),
        "slice_y": rng.uniform(0, 1, 80),
    }
    return plot_score_histograms(scores, scorer_name="model")


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_lift_ci.png",
    tolerance=15,
)
def test_plot_lift_ci_baseline() -> Figure:
    estimates = {
        "A_vs_B": _StubCI(point_estimate=0.05, ci_low=0.01, ci_high=0.09),
        "C_vs_B": _StubCI(point_estimate=-0.02, ci_low=-0.05, ci_high=0.01),
        "D_vs_B": _StubCI(point_estimate=0.03, ci_low=0.00, ci_high=0.06),
    }
    return plot_lift_ci(estimates, zero_line=True)


@pytest.mark.smoke
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline/test_plotting_visual",
    filename="plot_bootstrap_distribution.png",
    tolerance=15,
)
def test_plot_bootstrap_distribution_baseline() -> Figure:
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.02, size=500)
    return plot_bootstrap_distribution(
        deltas, ci_low=0.01, ci_high=0.09, title="lift bootstrap distribution"
    )
