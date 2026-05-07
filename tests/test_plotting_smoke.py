"""Smoke tests for plotting helpers — verify shapes, return types, no crashes.

Adapted from prompt_injection_detector/tests/test_plotting.py.
Headless matplotlib backend is set in conftest.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from eval_toolkit.plotting import (
    DEFAULT_FIGSIZE,
    PALETTE,
    plot_confusion_matrix_grid,
    plot_lift_ci,
    plot_metric_bars,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_score_histograms,
    save_figure,
    set_plot_style,
)


@dataclass(frozen=True, slots=True)
class _StubCI:
    """Minimal duck-typed stand-in for BootstrapCI for plot_lift_ci tests."""

    point_estimate: float
    ci_low: float
    ci_high: float


@pytest.fixture(autouse=True)
def _close_figures_after_each_test() -> None:
    yield
    plt.close("all")


@pytest.fixture
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.int64)
    s = (y + rng.normal(0, 0.3, size=200)).astype(np.float64)
    return y, s


@pytest.mark.smoke
def test_set_plot_style_applies_palette() -> None:
    set_plot_style()
    # Cycle should include the palette colors
    cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in cycle]
    assert PALETTE["negative"] in colors


@pytest.mark.smoke
def test_plot_pr_curve_returns_figure(synthetic: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = synthetic
    fig = plot_pr_curve(y, s, label="model", threshold=0.5, prevalence=0.5)
    assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_plot_reliability_diagram(synthetic: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = synthetic
    s_clipped = np.clip(s, 0, 1)
    fig = plot_reliability_diagram(y, s_clipped, n_bins=10)
    assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_plot_confusion_matrix_grid_class_names() -> None:
    matrices = {"model_a": np.array([[40, 5], [3, 12]]), "model_b": np.array([[42, 3], [4, 11]])}
    fig = plot_confusion_matrix_grid(matrices, class_names=("benign", "malicious"))
    assert isinstance(fig, Figure)
    # Tick labels should reflect class_names
    axes = fig.axes[0]
    labels = [t.get_text() for t in axes.get_xticklabels()]
    assert labels == ["benign", "malicious"]


@pytest.mark.smoke
def test_plot_metric_bars_with_default_formatter() -> None:
    values = {"slice_a": 0.85, "slice_b": 0.72, "slice_c": 0.91}
    fig = plot_metric_bars(values, ylabel="PR-AUC")
    assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_plot_metric_bars_with_custom_formatter() -> None:
    """Injectable label_formatter is honored."""
    values = {"a": 0.5, "b": 0.7}
    fig = plot_metric_bars(values, label_formatter=str.upper)
    axes = fig.axes[0]
    labels = [t.get_text() for t in axes.get_xticklabels()]
    assert "A" in labels and "B" in labels


@pytest.mark.smoke
def test_plot_score_histograms() -> None:
    rng = np.random.default_rng(0)
    scores = {"slice_x": rng.uniform(0, 1, 100), "slice_y": rng.uniform(0, 1, 80)}
    fig = plot_score_histograms(scores, scorer_name="model")
    assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_plot_lift_ci_with_stub_ci() -> None:
    estimates = {
        "A_vs_B": _StubCI(point_estimate=0.05, ci_low=0.01, ci_high=0.09),
        "C_vs_B": _StubCI(point_estimate=-0.02, ci_low=-0.05, ci_high=0.01),
    }
    fig = plot_lift_ci(estimates, zero_line=True)
    assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_save_figure_writes_png_and_sidecar(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "test.png"
    provenance = {"git_sha": "abc123", "run_id": "test-run"}
    saved = save_figure(fig, target, dpi=150, provenance=provenance)
    assert saved == target.resolve()
    assert target.exists()
    sidecar = target.with_suffix(".meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["git_sha"] == "abc123"
    assert meta["figure_dpi"] == "150"
    assert "timestamp_utc" in meta


@pytest.mark.smoke
def test_save_figure_validates_suffix(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="\\.png"):
        save_figure(fig, tmp_path / "bad.jpg")


@pytest.mark.smoke
def test_plot_pr_curve_validates_input() -> None:
    with pytest.raises(TypeError):
        plot_pr_curve([0, 1, 0, 1], np.array([0.1, 0.9, 0.2, 0.8]))  # type: ignore[arg-type]
