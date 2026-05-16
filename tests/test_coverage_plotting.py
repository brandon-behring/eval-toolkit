"""Coverage-targeted tests for plotting error paths and defensive code.

Extracted from the v0.27.x-era ``test_coverage_gap.py`` during the
v0.30.1 hygiene split — every assertion preserved verbatim; only the
file boundary changed.

Pairs with the happy-path coverage in ``test_plotting_smoke.py``, the
visual regression suite in ``test_plotting_visual.py``, and the edge
cases in ``test_plotting_edge.py``. Targets input-validation error
branches across the plotting API + the v0.3.0 bootstrap-distribution
+ pdf/svg + immutable-PALETTE additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eval_toolkit.plotting import (
    plot_confusion_matrix_grid,
    plot_lift_ci,
    plot_metric_bars,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_score_histograms,
    save_figure,
)


@dataclass(frozen=True, slots=True)
class _StubCI:
    """Duck-typed CI for plot_lift_ci tests."""

    point_estimate: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# plotting: validation + branch coverage
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_save_figure_rejects_non_positive_dpi(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, tmp_path / "x.png", dpi=0)


@pytest.mark.smoke
def test_save_figure_skip_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_TOOLKIT_SKIP_SAVEFIG", "1")
    fig, _ = plt.subplots()
    target = tmp_path / "skipped.png"
    out = save_figure(fig, target)
    assert out == target.resolve()
    assert not target.exists()


@pytest.mark.smoke
def test_plot_pr_curve_with_baseline_curve_and_threshold() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100).astype(np.int64)
    s = (y + rng.normal(0, 0.3, 100)).clip(0, 1)
    bl = (np.linspace(0, 1, 50), np.linspace(0.5, 0.5, 50))
    fig = plot_pr_curve(
        y, s, threshold=0.5, prevalence=0.3, baseline_curve=bl, baseline_label="rand"
    )
    assert fig.axes


@pytest.mark.smoke
def test_plot_pr_curve_rejects_invalid_threshold() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="threshold"):
        plot_pr_curve(y, s, threshold=1.5)


@pytest.mark.smoke
def test_plot_pr_curve_rejects_invalid_prevalence() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="prevalence"):
        plot_pr_curve(y, s, prevalence=1.5)


@pytest.mark.smoke
def test_plot_pr_curve_rejects_bad_baseline_curve() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="baseline_curve"):
        plot_pr_curve(y, s, baseline_curve=(np.array([0.1]),))  # type: ignore[arg-type]


@pytest.mark.smoke
def test_plot_pr_curve_baseline_shape_mismatch_raises() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    bad = (np.array([0.0, 1.0]), np.array([0.5]))
    with pytest.raises(ValueError, match="same shape"):
        plot_pr_curve(y, s, baseline_curve=bad)


@pytest.mark.smoke
def test_plot_reliability_rejects_single_class() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="single class"):
        plot_reliability_diagram(y, s)


@pytest.mark.smoke
def test_plot_confusion_matrix_grid_validates() -> None:
    with pytest.raises(ValueError):
        plot_confusion_matrix_grid({})


@pytest.mark.smoke
def test_plot_metric_bars_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_metric_bars({})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_score_histograms({})


@pytest.mark.smoke
def test_plot_score_histograms_validates_array_shape() -> None:
    with pytest.raises(ValueError, match="1D"):
        plot_score_histograms({"x": np.zeros((3, 3))})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_score_histograms({"x": np.array([], dtype=float)})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="NaN"):
        plot_score_histograms({"x": np.array([0.1, np.nan, 0.5])})


@pytest.mark.smoke
def test_plot_lift_ci_handles_empty_dict() -> None:
    with pytest.raises(ValueError):
        plot_lift_ci({})


@pytest.mark.smoke
def test_plot_lift_ci_renders_zero_line() -> None:
    cis = {
        "A": _StubCI(0.05, 0.02, 0.08),
        "B": _StubCI(-0.01, -0.05, 0.03),
    }
    fig = plot_lift_ci(cis)
    assert fig.axes


# ---------------------------------------------------------------------------
# v0.3.0 C8: plot_bootstrap_distribution + pdf/svg + immutable PALETTE
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_plot_bootstrap_distribution_runs() -> None:
    from eval_toolkit.plotting import plot_bootstrap_distribution

    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.02, size=500)
    fig = plot_bootstrap_distribution(deltas, ci_low=0.01, ci_high=0.09, title="lift dist")
    assert fig.axes


@pytest.mark.smoke
def test_plot_bootstrap_distribution_validates() -> None:
    from eval_toolkit.plotting import plot_bootstrap_distribution

    with pytest.raises(ValueError, match="empty"):
        plot_bootstrap_distribution(np.array([], dtype=float))
    with pytest.raises(ValueError, match="NaN"):
        plot_bootstrap_distribution(np.array([0.1, np.nan, 0.2]))
    with pytest.raises(ValueError, match="must both"):
        plot_bootstrap_distribution(np.array([0.1, 0.2]), ci_low=0.0)


@pytest.mark.smoke
def test_palette_is_immutable() -> None:
    from eval_toolkit.plotting import PALETTE

    with pytest.raises(TypeError):
        PALETTE["new_role"] = "#000000"  # type: ignore[index]


@pytest.mark.smoke
def test_save_figure_supports_pdf_svg(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    out_pdf = save_figure(fig, tmp_path / "fig.pdf", provenance={"git_sha": "abc"})
    assert out_pdf.suffix == ".pdf"
    sidecar_pdf = (tmp_path / "fig.pdf").with_suffix(".meta.json")
    assert sidecar_pdf.exists()

    out_svg = save_figure(fig, tmp_path / "fig.svg", provenance={"git_sha": "abc"})
    assert out_svg.suffix == ".svg"


@pytest.mark.smoke
def test_save_figure_rejects_unknown_suffix(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match=r"\.png|\.pdf|\.svg|sorted"):
        save_figure(fig, tmp_path / "fig.jpg")
