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
    make_palette,
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
def test_save_figure_permitted_suffixes_restricts_to_caller_set(tmp_path: Path) -> None:
    """``permitted_suffixes={'.png'}`` rejects ``.pdf`` even though it's a default-allowed suffix.

    Downstream projects (e.g. PID) can lock to PNG-only via this kwarg.
    """
    fig, _ = plt.subplots()
    # Default permits .pdf; restricted set should reject it.
    with pytest.raises(ValueError, match="\\.png"):
        save_figure(fig, tmp_path / "out.pdf", permitted_suffixes={".png"})
    # PNG still works with the restricted set.
    out = save_figure(fig, tmp_path / "out.png", permitted_suffixes={".png"})
    assert out.exists()


@pytest.mark.smoke
def test_save_figure_skip_env_var_honors_caller_supplied_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom ``skip_env_var`` controls write-suppression independently of default.

    Setting ``EVAL_TOOLKIT_SKIP_SAVEFIG`` should NOT suppress when the caller
    asked for a different env-var name; setting the caller-supplied name
    should suppress.
    """
    fig, _ = plt.subplots()
    target = tmp_path / "skipped.png"

    # Default env not set; explicit env var name not set → write happens.
    monkeypatch.delenv("EVAL_TOOLKIT_SKIP_SAVEFIG", raising=False)
    monkeypatch.delenv("PID_SKIP_SAVEFIG", raising=False)
    save_figure(fig, target, skip_env_var="PID_SKIP_SAVEFIG")
    assert target.exists()

    # Toolkit's default env-var name should NOT trigger the project-specific opt-out.
    target.unlink()
    monkeypatch.setenv("EVAL_TOOLKIT_SKIP_SAVEFIG", "1")
    save_figure(fig, target, skip_env_var="PID_SKIP_SAVEFIG")
    assert target.exists()  # written despite the toolkit-default env-var being set

    # Caller-supplied env-var name DOES trigger suppression.
    target.unlink()
    monkeypatch.setenv("PID_SKIP_SAVEFIG", "1")
    save_figure(fig, target, skip_env_var="PID_SKIP_SAVEFIG")
    assert not target.exists()  # suppressed


@pytest.mark.unit
def test_make_palette_default_returns_standard_roles() -> None:
    """Default ``make_palette()`` matches the toolkit's :data:`PALETTE` constant."""
    p = make_palette()
    assert p["negative"] == "#004488"
    assert p["positive"] == "#BB5566"
    assert p["accent"] == "#DDAA33"
    assert p["baseline"] == "#999999"
    # Must match the module-level PALETTE on the four core roles.
    for role in ("negative", "positive", "accent", "baseline"):
        assert p[role] == PALETTE[role]


@pytest.mark.unit
def test_make_palette_extras_extend_standard_roles() -> None:
    """Extras are added alongside the four standard roles (PID semantics).

    PID's project-specific roles ('benign', 'injection', 'emphasis') extend
    the palette without removing the standard roles.
    """
    p = make_palette(benign="#004488", injection="#BB5566", emphasis="#DDAA33")
    # Project-specific keys present.
    assert p["benign"] == "#004488"
    assert p["injection"] == "#BB5566"
    assert p["emphasis"] == "#DDAA33"
    # Standard roles still present (additive, not replacement).
    assert p["negative"] == "#004488"
    assert p["accent"] == "#DDAA33"


@pytest.mark.unit
def test_make_palette_returns_frozen_mapping() -> None:
    """The returned mapping rejects mutation (MappingProxyType)."""
    p = make_palette()
    with pytest.raises(TypeError):
        p["new_key"] = "#000000"  # type: ignore[index]


@pytest.mark.smoke
def test_plot_pr_curve_validates_input() -> None:
    with pytest.raises(TypeError):
        plot_pr_curve([0, 1, 0, 1], np.array([0.1, 0.9, 0.2, 0.8]))  # type: ignore[arg-type]
