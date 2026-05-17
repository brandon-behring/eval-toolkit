"""Edge-case + validation-branch tests for eval_toolkit.plotting.

Pairs with the visual-regression snapshot suite in
``test_plotting_visual.py``. These tests don't compare pixels; they
exercise the input validation branches, optional-argument paths, and
non-default rendering modes (prevalence line, threshold marker,
bin_counts bubble encoding, zero-line forest plot, CI overlay).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_toolkit.plotting import (  # noqa: E402
    plot_bootstrap_distribution,
    plot_confusion_matrix_grid,
    plot_lift_ci,
    plot_metric_bars,
    plot_pareto_frontier,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_roc_curve,
    plot_score_histograms,
    plot_slice_metric_heatmap,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures created by a test to keep matplotlib state clean."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _validate_pair branches (lines 303, 308, 310)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_pr_curve_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        plot_pr_curve(
            y_true=np.array([0, 1, 0]),
            y_score=np.array([0.1, 0.9]),
        )


@pytest.mark.unit
def test_plot_pr_curve_rejects_empty_y_true() -> None:
    with pytest.raises(ValueError, match="y_true is empty"):
        plot_pr_curve(
            y_true=np.array([], dtype=int),
            y_score=np.array([], dtype=float),
        )


@pytest.mark.unit
def test_plot_pr_curve_rejects_nan_in_y_score() -> None:
    with pytest.raises(ValueError, match="y_score contains NaN or inf"):
        plot_pr_curve(
            y_true=np.array([0, 1]),
            y_score=np.array([0.5, np.nan]),
        )


# ---------------------------------------------------------------------------
# _resolve_axes branch + plot_pr_curve optional paths (412, 421, 440)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_pr_curve_with_prevalence_threshold_baseline_and_title() -> None:
    """Hits the prevalence axhline, threshold star, baseline curve, and title."""
    fig, ax = plt.subplots()
    out = plot_pr_curve(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.4, 0.6, 0.9]),
        prevalence=0.5,
        threshold=0.6,
        baseline_curve=(np.array([0, 1]), np.array([0.5, 0.5])),
        title="With prevalence",
        ax=ax,
    )
    assert out is fig  # same Figure returned when an Axes was provided


# ---------------------------------------------------------------------------
# plot_reliability_diagram: bin_counts branches (486-490, 509-511)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reliability_rejects_bad_bin_counts_shape() -> None:
    with pytest.raises(ValueError, match="bin_counts must have shape"):
        plot_reliability_diagram(
            y_true=np.array([0, 1, 0, 1]),
            y_prob=np.array([0.1, 0.9, 0.3, 0.7]),
            n_bins=10,
            bin_counts=np.array([1, 2, 3]),  # wrong shape
        )


@pytest.mark.unit
def test_reliability_rejects_negative_bin_counts() -> None:
    with pytest.raises(ValueError, match="bin_counts must be non-negative"):
        plot_reliability_diagram(
            y_true=np.array([0, 1, 0, 1]),
            y_prob=np.array([0.1, 0.9, 0.3, 0.7]),
            n_bins=3,
            bin_counts=np.array([1, -1, 2]),
        )


@pytest.mark.unit
def test_reliability_with_bubble_encoding_and_title() -> None:
    """Non-None bin_counts triggers the bubble-size code path."""
    fig = plot_reliability_diagram(
        y_true=np.array([0, 0, 0, 1, 1, 1, 1, 0]),
        y_prob=np.array([0.1, 0.2, 0.15, 0.6, 0.7, 0.85, 0.95, 0.4]),
        n_bins=4,
        bin_counts=np.array([2, 1, 1, 4]),
        title="Reliability",
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# plot_confusion_matrix_grid: shape + title branches (580, 643)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confusion_matrix_rejects_non_2x2() -> None:
    with pytest.raises(ValueError, match="matrices must be 2x2"):
        plot_confusion_matrix_grid({"m": np.zeros((3, 3))})


@pytest.mark.unit
def test_confusion_matrix_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        plot_confusion_matrix_grid({})


@pytest.mark.unit
def test_confusion_matrix_with_title() -> None:
    fig = plot_confusion_matrix_grid(
        {"m": np.array([[5, 1], [2, 4]])},
        title="ablation grid",
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# plot_metric_bars: empty + title branch (693)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metric_bars_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        plot_metric_bars({})


@pytest.mark.unit
def test_metric_bars_with_title_and_ylabel() -> None:
    fig = plot_metric_bars(
        {"a": 0.7, "b": 0.85, "c": 0.6},
        ylabel="PR-AUC",
        title="model comparison",
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# plot_score_histograms: validation + title-derivation (774->776, 776->778)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_score_histograms_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        plot_score_histograms({})


@pytest.mark.unit
def test_score_histograms_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        plot_score_histograms({"dev": np.zeros((2, 3))})


@pytest.mark.unit
def test_score_histograms_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="is empty"):
        plot_score_histograms({"dev": np.array([], dtype=float)})


@pytest.mark.unit
def test_score_histograms_rejects_nan_input() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        plot_score_histograms({"dev": np.array([0.5, np.inf])})


@pytest.mark.unit
def test_score_histograms_auto_title_from_scorer_name() -> None:
    """Title derives from scorer_name when title is None (line 775)."""
    fig = plot_score_histograms(
        {"dev": np.array([0.1, 0.2, 0.3, 0.9])},
        scorer_name="model_v1",
    )
    # The figure's only Axes title should mention the scorer.
    ax = fig.axes[0]
    assert "model_v1" in ax.get_title()


@pytest.mark.unit
def test_score_histograms_explicit_title_wins() -> None:
    fig = plot_score_histograms(
        {"dev": np.array([0.1, 0.2, 0.3, 0.9])},
        title="Custom",
        scorer_name="ignored",
    )
    assert fig.axes[0].get_title() == "Custom"


# ---------------------------------------------------------------------------
# plot_lift_ci: validation + zero_line + title (819, 834-844, 861)
# ---------------------------------------------------------------------------


@dataclass
class _DuckCI:
    point_estimate: float
    ci_low: float
    ci_high: float


@pytest.mark.unit
def test_lift_ci_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one comparison"):
        plot_lift_ci({})


@pytest.mark.unit
def test_lift_ci_rejects_non_bootstrap_ci_like() -> None:
    """Object without point_estimate/ci_low/ci_high attributes is rejected."""

    class _Bad:
        ci_low = 0.0
        ci_high = 1.0  # missing point_estimate

    with pytest.raises(TypeError, match="BootstrapCI-like"):
        plot_lift_ci({"k": _Bad()})  # type: ignore[arg-type]


@pytest.mark.unit
def test_lift_ci_with_zero_line_and_title() -> None:
    fig = plot_lift_ci(
        {
            "model_a": _DuckCI(0.05, -0.01, 0.10),
            "model_b": _DuckCI(0.20, 0.05, 0.35),
        },
        zero_line=True,
        title="Lift over baseline",
    )
    assert fig is not None


@pytest.mark.unit
def test_lift_ci_without_zero_line() -> None:
    fig = plot_lift_ci(
        {"only": _DuckCI(0.1, 0.05, 0.15)},
        zero_line=False,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# plot_bootstrap_distribution: validation + CI lines + title (923, 941, 953)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_distribution_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="deltas must be 1-D"):
        plot_bootstrap_distribution(np.zeros((3, 2)))


@pytest.mark.unit
def test_bootstrap_distribution_rejects_empty() -> None:
    with pytest.raises(ValueError, match="deltas is empty"):
        plot_bootstrap_distribution(np.array([], dtype=float))


@pytest.mark.unit
def test_bootstrap_distribution_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        plot_bootstrap_distribution(np.array([0.1, np.nan, 0.2]))


@pytest.mark.unit
def test_bootstrap_distribution_rejects_partial_ci() -> None:
    with pytest.raises(ValueError, match="both be supplied or both None"):
        plot_bootstrap_distribution(np.array([0.1, 0.2, 0.3]), ci_low=0.05)


@pytest.mark.unit
def test_bootstrap_distribution_with_ci_overlay_and_title() -> None:
    """Hits the CI-overlay axvline branch and the title-setting branch."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.02, size=200)
    fig = plot_bootstrap_distribution(
        deltas,
        ci_low=float(np.quantile(deltas, 0.025)),
        ci_high=float(np.quantile(deltas, 0.975)),
        title="Δ distribution",
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# plot_roc_curve — validation branches + ax= path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_roc_curve_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        plot_roc_curve(
            y_true=np.array([0, 1, 0, 1]),
            y_score=np.array([0.1, 0.9, 0.4, 0.6]),
            threshold=1.5,
        )


@pytest.mark.unit
def test_plot_roc_curve_rejects_baseline_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="baseline_curve fpr and tpr must have same shape"):
        plot_roc_curve(
            y_true=np.array([0, 1, 0, 1]),
            y_score=np.array([0.1, 0.9, 0.4, 0.6]),
            baseline_curve=(np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])),
        )


@pytest.mark.unit
def test_plot_roc_curve_rejects_baseline_not_tuple() -> None:
    with pytest.raises(ValueError, match="baseline_curve must be a"):
        plot_roc_curve(
            y_true=np.array([0, 1, 0, 1]),
            y_score=np.array([0.1, 0.9, 0.4, 0.6]),
            baseline_curve=[np.array([0.0]), np.array([0.0])],  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_plot_roc_curve_with_threshold_baseline_and_title_uses_ax() -> None:
    """Hits threshold star + baseline overlay + title + ax= branch."""
    fig, ax = plt.subplots()
    out = plot_roc_curve(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
        baseline_curve=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        baseline_label="ref",
        title="ROC with overlays",
        ax=ax,
    )
    assert out is fig


# ---------------------------------------------------------------------------
# plot_pareto_frontier — validation branches + ax= path + alt direction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_pareto_frontier_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must have same shape"):
        plot_pareto_frontier(np.array([1.0, 2.0]), np.array([0.5, 0.7, 0.8]))


@pytest.mark.unit
def test_plot_pareto_frontier_rejects_2d_inputs() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        plot_pareto_frontier(np.array([[1.0]]), np.array([[0.5]]))


@pytest.mark.unit
def test_plot_pareto_frontier_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_pareto_frontier(np.array([]), np.array([]))


@pytest.mark.unit
def test_plot_pareto_frontier_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="must contain finite"):
        plot_pareto_frontier(np.array([1.0, 2.0]), np.array([0.5, np.nan]))


@pytest.mark.unit
def test_plot_pareto_frontier_rejects_label_length_mismatch() -> None:
    with pytest.raises(ValueError, match="point_labels length"):
        plot_pareto_frontier(
            np.array([1.0, 2.0]),
            np.array([0.5, 0.7]),
            point_labels=["only-one"],
        )


@pytest.mark.unit
def test_plot_pareto_frontier_lower_is_better_path_uses_ax() -> None:
    """Hits higher_metric_is_better=False branch + labels + ax= branch."""
    fig, ax = plt.subplots()
    cost = np.array([1.0, 2.0, 3.0, 2.5])
    err = np.array([0.5, 0.3, 0.25, 0.4])  # lower-better; index 3 dominated
    out = plot_pareto_frontier(
        cost,
        err,
        point_labels=["a", "b", "c", "d"],
        higher_metric_is_better=False,
        xlabel="cost",
        ylabel="error",
        title="lower-better frontier",
        ax=ax,
    )
    assert out is fig


# ---------------------------------------------------------------------------
# plot_slice_metric_heatmap — validation branches + ax= path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_slice_metric_heatmap_rejects_1d_grid() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        plot_slice_metric_heatmap(
            np.array([0.5, 0.7]),
            row_labels=("a",),
            col_labels=("x", "y"),
        )


@pytest.mark.unit
def test_plot_slice_metric_heatmap_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_slice_metric_heatmap(
            np.zeros((0, 3)),
            row_labels=(),
            col_labels=("a", "b", "c"),
        )


@pytest.mark.unit
def test_plot_slice_metric_heatmap_rejects_row_label_mismatch() -> None:
    with pytest.raises(ValueError, match="row_labels length"):
        plot_slice_metric_heatmap(
            np.array([[0.5, 0.7]]),
            row_labels=("a", "b"),
            col_labels=("x", "y"),
        )


@pytest.mark.unit
def test_plot_slice_metric_heatmap_rejects_col_label_mismatch() -> None:
    with pytest.raises(ValueError, match="col_labels length"):
        plot_slice_metric_heatmap(
            np.array([[0.5, 0.7]]),
            row_labels=("a",),
            col_labels=("x",),
        )


@pytest.mark.unit
def test_plot_slice_metric_heatmap_no_annotate_uses_ax() -> None:
    """Hits annotate=False branch + ax= branch + NaN cells get masked."""
    fig, ax = plt.subplots()
    grid = np.array([[0.5, np.nan], [0.7, 0.8]])
    out = plot_slice_metric_heatmap(
        grid,
        row_labels=("r1", "r2"),
        col_labels=("c1", "c2"),
        annotate=False,
        title="no annotations + NaN cells",
        ax=ax,
    )
    assert out is fig


# ---------------------------------------------------------------------------
# ax= parity backfill — plot_metric_bars + plot_score_histograms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_metric_bars_with_ax_returns_same_figure() -> None:
    fig, ax = plt.subplots()
    out = plot_metric_bars(
        {"slice_a": 0.85, "slice_b": 0.72, "slice_c": 0.91},
        ylabel="PR-AUC",
        ax=ax,
    )
    assert out is fig


@pytest.mark.unit
def test_plot_score_histograms_with_ax_returns_same_figure() -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    out = plot_score_histograms(
        {"slice_x": rng.uniform(0, 1, 50), "slice_y": rng.uniform(0, 1, 40)},
        scorer_name="model",
        ax=ax,
    )
    assert out is fig
