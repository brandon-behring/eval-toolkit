"""Visualization helpers for binary classification evaluation.

Six high-level plot helpers (PR curve, reliability diagram, confusion-matrix
grid, metric bars, score histograms, lift CI) plus style and provenance-aware
``save_figure``. matplotlib-only.

All public functions accept ``np.ndarray`` inputs (callers extract from
DataFrames via ``.values``). Each helper accepts ``title`` and ``figsize``;
single-panel helpers also accept ``ax``.

The palette uses semantic role names: ``negative``, ``positive``, ``baseline``,
``accent``. Domain-specific labeling is parameterized via ``class_names``
(confusion matrix) or ``slice_formatter`` (bar/histogram helpers).

``save_figure`` writes provenance to both PNG iTXt chunks and a sidecar JSON
so saved figures stay file-traceable. Set ``EVAL_TOOLKIT_SKIP_SAVEFIG=1`` in
the environment to skip writes (useful for dry-run notebook iteration).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Container, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

if TYPE_CHECKING:
    from eval_toolkit.bootstrap import BootstrapCI

__all__ = [
    "DEFAULT_FIGSIZE",
    "PALETTE",
    "PLOT_STYLE",
    "make_palette",
    "plot_bootstrap_distribution",
    "plot_confusion_matrix_grid",
    "plot_lift_ci",
    "plot_metric_bars",
    "plot_pr_curve",
    "plot_reliability_diagram",
    "plot_score_histograms",
    "save_figure",
    "set_plot_style",
]

# Semantic palette roles per Decision (palette role rename from PID's
# benign/injection to generic negative/positive/baseline/accent).
# Wrapped in MappingProxyType to prevent inadvertent caller mutation;
# downstream code can still read all entries normally.
PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "negative": "#004488",  # navy   — negative / "good outcome" / TN/TP diagonal
        "positive": "#BB5566",  # rose   — positive / alert / FP/FN off-diagonal
        "accent": "#DDAA33",  # gold   — emphasis / threshold marker
        "baseline": "#999999",  # gray   — reference line / calibration diagonal
    }
)

PLOT_STYLE: dict[str, Any] = {
    "axes.prop_cycle": cycler(
        "color", [PALETTE["negative"], PALETTE["accent"], PALETTE["positive"]]
    ),
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
}

DEFAULT_FIGSIZE: tuple[float, float] = (6.0, 4.0)


def make_palette(
    *,
    negative: str = "#004488",
    positive: str = "#BB5566",
    accent: str = "#DDAA33",
    baseline: str = "#999999",
    **extras: str,
) -> Mapping[str, str]:
    """Construct a semantic-role color palette for downstream projects.

    Returns a frozen mapping (via :class:`types.MappingProxyType`) keyed by
    semantic role names. Standard roles are ``negative`` (good outcome,
    diagonal of confusion matrix), ``positive`` (alert, off-diagonal),
    ``accent`` (threshold marker / highlight), and ``baseline`` (reference
    line, calibration diagonal). Pass any number of additional named keyword
    arguments to extend the palette with project-specific roles
    (e.g., ``benign="#004488"``, ``injection="#BB5566"`` for prompt-injection
    framing).

    Parameters
    ----------
    negative, positive, accent, baseline : str, optional
        Hex color strings for the four standard semantic roles. Defaults
        match the toolkit's :data:`PALETTE` constant.
    **extras : str
        Project-specific role keys (e.g., ``benign``, ``injection``,
        ``emphasis``). All values must be valid hex color strings.

    Returns
    -------
    Mapping[str, str]
        Frozen palette mapping role → hex color. Mutation attempts raise
        ``TypeError``.

    Examples
    --------
    Default palette:

    >>> p = make_palette()
    >>> p["negative"]
    '#004488'

    Project-specific extension (PID semantics):

    >>> p = make_palette(benign="#004488", injection="#BB5566", emphasis="#DDAA33")
    >>> p["benign"]
    '#004488'
    >>> p["injection"]
    '#BB5566'
    >>> p["negative"]  # standard roles still present
    '#004488'

    Mutation prevented:

    >>> try:
    ...     p["new_key"] = "#000000"
    ... except TypeError:
    ...     print("frozen")
    frozen
    """
    base: dict[str, str] = {
        "negative": negative,
        "positive": positive,
        "accent": accent,
        "baseline": baseline,
    }
    base.update(extras)
    return MappingProxyType(base)


def set_plot_style() -> None:
    """Apply ``PLOT_STYLE`` rcParams + ``PALETTE`` color cycle.

    Idempotent. Call once per notebook or script before any plotting code.
    """
    plt.rcParams.update(PLOT_STYLE)


_DEFAULT_PERMITTED_SUFFIXES: frozenset[str] = frozenset({".png", ".pdf", ".svg"})


def save_figure(
    fig: Figure,
    path: Path,
    *,
    dpi: int = 300,
    provenance: dict[str, str] | None = None,
    permitted_suffixes: Container[str] = _DEFAULT_PERMITTED_SUFFIXES,
    skip_env_var: str = "EVAL_TOOLKIT_SKIP_SAVEFIG",
) -> Path:
    """Save figure to disk with optional provenance metadata.

    Honors a configurable skip-env-var (default ``EVAL_TOOLKIT_SKIP_SAVEFIG``)
    — when set to ``"1"``, returns the resolved path without writing.

    When ``provenance`` is provided, persists the metadata in two places:

    1. PNG iTXt chunks via ``fig.savefig(metadata=...)``. Travels with the
       PNG when copied/shared. ``.pdf`` and ``.svg`` skip the iTXt step.
    2. Sidecar ``{path.stem}.meta.json`` next to the figure. Human-readable
       and works for all permitted suffixes.

    Both sidecar and iTXt contain the caller-supplied keys plus auto-fields
    ``timestamp_utc`` (ISO-8601), ``matplotlib_version``, ``figure_dpi``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : pathlib.Path
        Output path. Suffix must be in ``permitted_suffixes``.
    dpi : int, optional
        Resolution. Default 300.
    provenance : dict[str, str] or None, optional
        Caller-supplied provenance keys (e.g., ``git_sha``, ``run_id``).
    permitted_suffixes : Container[str], optional
        Allowed file-extension suffixes (with leading dot). Default
        ``{".png", ".pdf", ".svg"}``. Downstream projects can restrict to a
        single format (e.g. ``permitted_suffixes={".png"}``) for stable
        artifact pipelines, or extend to additional matplotlib-supported
        suffixes (``.eps``, ``.ps``, etc.).
    skip_env_var : str, optional
        Environment-variable name that, when set to ``"1"``, suppresses
        writes and returns the resolved path. Default
        ``"EVAL_TOOLKIT_SKIP_SAVEFIG"``. Downstream projects can pass their
        own (e.g. ``skip_env_var="PID_SKIP_SAVEFIG"``) for project-specific
        opt-out controls.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Raises
    ------
    ValueError
        If ``path.suffix`` is not in ``permitted_suffixes``, or if ``dpi``
        is non-positive.

    Notes
    -----
    PNG sidecar JSON is always written when ``provenance`` is supplied;
    iTXt embedded metadata is added for ``.png`` only (matplotlib supports
    iTXt natively via ``metadata=`` kwarg). ``.pdf`` and ``.svg`` ship the
    same sidecar JSON without embedded metadata.
    """
    if path.suffix not in permitted_suffixes:
        # Container is the broadest acceptable type, but most realistic
        # implementations (set, frozenset, list, tuple) are also iterable.
        if isinstance(permitted_suffixes, Iterable):
            permitted_repr: object = sorted(permitted_suffixes)
        else:
            permitted_repr = permitted_suffixes
        raise ValueError(f"path must end in {permitted_repr}, got {path.suffix!r}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    resolved = path.resolve()
    if os.environ.get(skip_env_var) == "1":
        return resolved

    path.parent.mkdir(parents=True, exist_ok=True)

    if provenance is not None:
        # Use the canonical figure_metadata helper from provenance.py for the
        # sidecar payload — keeps the provenance schema consistent across
        # save_figure / make_run_dir / file_sha256 callers.
        from eval_toolkit.provenance import figure_metadata  # noqa: PLC0415

        combined = figure_metadata(dict(provenance), dpi=dpi)
        # iTXt embedded metadata only supported for PNG.
        if path.suffix == ".png":
            fig.savefig(path, dpi=dpi, metadata=combined)
        else:
            fig.savefig(path, dpi=dpi)
        sidecar = path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(combined, indent=2, sort_keys=True))
    else:
        fig.savefig(path, dpi=dpi)
    return resolved


def _ensure_ndarray(name: str, value: object) -> np.ndarray:
    """Strict ndarray contract."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(value).__name__}")
    return value


def _validate_pair(y_true: np.ndarray, y_other: np.ndarray, *, other_name: str) -> None:
    """Shared shape + domain validation for PR / reliability inputs."""
    if y_true.shape != y_other.shape:
        raise ValueError(
            f"y_true and {other_name} must have the same shape, "
            f"got {y_true.shape} vs {y_other.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if not np.isfinite(y_other).all():
        raise ValueError(f"{other_name} contains NaN or inf")
    classes = np.unique(y_true)
    if classes.size < 2:
        raise ValueError(
            "y_true must contain at least one positive and one negative example; "
            "PR / reliability metrics are undefined for a single class"
        )


def _resolve_axes(
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes]:
    """Reuse caller's Axes (and parent Figure) or create fresh."""
    if ax is not None:
        return cast(Figure, ax.figure), ax
    fig, axes = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE)
    return fig, axes


def _maybe_add_legend(axes: Axes) -> None:
    """Add a legend only when at least one labeled artist exists."""
    handles, _ = axes.get_legend_handles_labels()
    if handles:
        axes.legend(loc="best")


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    label: str | None = None,
    threshold: float | None = None,
    prevalence: float | None = None,
    baseline_curve: tuple[np.ndarray, np.ndarray] | None = None,
    baseline_label: str = "baseline",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Precision-recall curve.

    Parameters
    ----------
    y_true, y_score : np.ndarray
        Labels and scores.
    label : str or None, optional
        Legend label for the main curve.
    threshold : float or None, optional
        Draw a star marker at the (recall, precision) point closest to this
        threshold.
    prevalence : float or None, optional
        Draw a horizontal reference line at this y-value (e.g., positive
        class prevalence).
    baseline_curve : tuple of (recall, precision) np.ndarrays, optional
        Optional baseline curve to overlay (e.g., a simpler reference model).
    baseline_label : str, optional
        Legend label for the baseline overlay (default ``"baseline"``).
    title : str or None, optional
    figsize : tuple of float or None, optional
    ax : matplotlib Axes or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = _ensure_ndarray("y_true", y_true)
    y_score = _ensure_ndarray("y_score", y_score)
    _validate_pair(y_true, y_score, other_name="y_score")

    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if prevalence is not None and not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    if baseline_curve is not None:
        if not (isinstance(baseline_curve, tuple) and len(baseline_curve) == 2):
            raise ValueError("baseline_curve must be a (recall, precision) tuple")
        bl_recall = _ensure_ndarray("baseline_curve[0]", baseline_curve[0])
        bl_precision = _ensure_ndarray("baseline_curve[1]", baseline_curve[1])
        if bl_recall.shape != bl_precision.shape:
            raise ValueError(
                f"baseline_curve recall and precision must have same shape, "
                f"got {bl_recall.shape} vs {bl_precision.shape}"
            )

    fig, axes = _resolve_axes(ax, figsize)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    axes.plot(recall, precision, color=PALETTE["negative"], label=label, linewidth=1.5)

    if baseline_curve is not None:
        bl_recall = np.asarray(baseline_curve[0])
        bl_precision = np.asarray(baseline_curve[1])
        axes.plot(
            bl_recall,
            bl_precision,
            color=PALETTE["baseline"],
            linewidth=1.0,
            linestyle="--",
            label=baseline_label,
            zorder=1,
        )
    if prevalence is not None:
        axes.axhline(
            prevalence,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label=f"prevalence={prevalence:.3f}",
        )
    if threshold is not None:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        axes.scatter(
            recall[idx],
            precision[idx],
            color=PALETTE["accent"],
            marker="*",
            s=120,
            zorder=5,
            label=f"τ={threshold:.3f}",
            edgecolor="black",
            linewidth=0.5,
        )

    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    if title is not None:
        axes.set_title(title)
    _maybe_add_legend(axes)
    fig.tight_layout()
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    bin_counts: np.ndarray | None = None,
    xlabel: str = "Mean Predicted Probability",
    ylabel: str = "Observed Fraction of Positives",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Equal-width-binned reliability diagram with diagonal reference.

    When ``bin_counts`` is provided, scales point sizes by counts (bubble
    encoding) so sparsely-populated bins look smaller, signaling lower trust.

    Parameters
    ----------
    y_true, y_prob : np.ndarray
        Labels and calibrated probabilities.
    n_bins : int, optional
        Number of equal-width bins. Default 10.
    bin_counts : np.ndarray or None, optional
        Per-bin counts (shape ``(n_bins,)``); enables bubble encoding.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str or None, optional
    figsize : tuple of float or None, optional
    ax : matplotlib Axes or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = _ensure_ndarray("y_true", y_true)
    y_prob = _ensure_ndarray("y_prob", y_prob)
    _validate_pair(y_true, y_prob, other_name="y_prob")

    if bin_counts is not None:
        bin_counts = _ensure_ndarray("bin_counts", bin_counts)
        if bin_counts.shape != (n_bins,):
            raise ValueError(f"bin_counts must have shape ({n_bins},), got {bin_counts.shape}")
        if (bin_counts < 0).any():
            raise ValueError("bin_counts must be non-negative")

    fig, axes = _resolve_axes(ax, figsize)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    axes.plot(
        [0, 1],
        [0, 1],
        color=PALETTE["baseline"],
        linestyle="--",
        linewidth=0.8,
        label="Perfect calibration",
        zorder=1,
    )

    if bin_counts is not None and bin_counts.size > 0:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(mean_predicted_value, bin_edges) - 1, 0, n_bins - 1)
        sizes = 30.0 + 4.0 * bin_counts[bin_idx]
    else:
        sizes = np.full_like(mean_predicted_value, 50.0)

    axes.scatter(
        mean_predicted_value,
        fraction_of_positives,
        s=sizes,
        color=PALETTE["negative"],
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    axes.plot(
        mean_predicted_value,
        fraction_of_positives,
        color=PALETTE["negative"],
        linewidth=1.0,
        zorder=2,
    )

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    if title is not None:
        axes.set_title(title)
    _maybe_add_legend(axes)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_grid(
    matrices: dict[str, np.ndarray],
    *,
    class_names: tuple[str, str] = ("negative", "positive"),
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """N-panel confusion-matrix grid (one per scorer).

    Two-tone semantic encoding: diagonal cells (TP, TN — "correct") use a
    white→PALETTE['negative'] gradient; off-diagonal cells (FP, FN — "errors")
    use a white→PALETTE['positive'] gradient. Each cell shows
    ``'{count}\\n({percent:.0%})'``.

    Parameters
    ----------
    matrices : dict[str, np.ndarray]
        ``{scorer_name: 2x2 confusion matrix}``.
    class_names : tuple of (str, str), optional
        Tick labels for the two classes (default ``("negative", "positive")``).
    title : str or None, optional
    figsize : tuple of float or None, optional

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``matrices`` is empty or any matrix is not 2x2.
    """
    if not matrices:
        raise ValueError("matrices must contain at least one panel")
    for key, matrix in matrices.items():
        matrix = _ensure_ndarray(f"matrices[{key!r}]", matrix)
        if matrix.shape != (2, 2):
            raise ValueError(f"matrices must be 2x2, got shape {matrix.shape} for key {key!r}")

    n = len(matrices)
    actual_figsize = figsize or (3.5 * n, 3.5)
    fig, axes_list = plt.subplots(1, n, figsize=actual_figsize, squeeze=False)
    axes_arr = axes_list[0]

    negative_cmap = LinearSegmentedColormap.from_list(
        "white_to_negative", ["#ffffff", PALETTE["negative"]]
    )
    positive_cmap = LinearSegmentedColormap.from_list(
        "white_to_positive", ["#ffffff", PALETTE["positive"]]
    )

    diag_mask = np.eye(2, dtype=bool)
    for axes, (name, matrix) in zip(axes_arr, matrices.items(), strict=True):
        m = np.asarray(matrix, dtype=float)
        total = m.sum()
        diag_max = max(float(m[diag_mask].max()), 1.0)
        offdiag_max = max(float(m[~diag_mask].max()), 1.0)

        for (i, j), value in np.ndenumerate(m):
            on_diag = i == j
            norm = value / (diag_max if on_diag else offdiag_max)
            cmap = negative_cmap if on_diag else positive_cmap
            axes.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm),
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            cell_total = total if total > 0 else 1.0
            percent = value / cell_total
            text_color = "white" if norm > 0.55 else "#222222"
            axes.text(
                j,
                i,
                f"{int(value)}\n({percent:.0%})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

        axes.set_xticks([0, 1])
        axes.set_yticks([0, 1])
        axes.set_xticklabels(list(class_names))
        axes.set_yticklabels(list(class_names))
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")
        axes.set_title(name)
        axes.set_xlim(-0.5, 1.5)
        axes.set_ylim(1.5, -0.5)
        axes.set_aspect("equal")
        axes.grid(False)
        for spine in axes.spines.values():
            spine.set_visible(False)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metric_bars(
    values: dict[str, float],
    *,
    color: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    label_formatter: Callable[[str], str] | None = None,
    sort_key: Callable[[str], Any] | None = None,
) -> Figure:
    """Bar chart for a ``{label: metric}`` mapping.

    Parameters
    ----------
    values : dict[str, float]
    color : str or None, optional
        Bar color. Default is ``PALETTE["negative"]``.
    ylabel, title : str or None, optional
    figsize : tuple of float or None, optional
    label_formatter : callable or None, optional
        Maps raw key → display label. Default is identity.
    sort_key : callable or None, optional
        Maps raw key → sort key. Default is alphabetical.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not values:
        raise ValueError("values must contain at least one entry")

    fmt = label_formatter or (lambda k: k)
    skey = sort_key or (lambda k: k)
    sorted_items = sorted(values.items(), key=lambda kv: skey(kv[0]))
    labels = [fmt(k) for k, _ in sorted_items]
    bar_values = [v for _, v in sorted_items]

    fig, axes = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE)
    bar_color = color or PALETTE["negative"]
    axes.bar(labels, bar_values, color=bar_color, edgecolor="black", linewidth=0.5)
    upper = max(bar_values)
    axes.set_ylim(0.0, max(1.0, upper * 1.05))
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    axes.tick_params(axis="x", rotation=30)
    for tick in axes.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def plot_score_histograms(
    scores_by_slice: dict[str, np.ndarray],
    *,
    scorer_name: str | None = None,
    bins: int = 30,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    label_formatter: Callable[[str], str] | None = None,
    sort_key: Callable[[str], Any] | None = None,
) -> Figure:
    """Overlaid score-distribution histograms, one per slice.

    Parameters
    ----------
    scores_by_slice : dict[str, np.ndarray]
        ``{slice_name: 1-D score array}``.
    scorer_name : str or None, optional
        If set and ``title`` is None, used as part of an auto-generated title.
    bins : int, optional
        Histogram bins. Default 30.
    title, figsize : optional
    label_formatter, sort_key : callable or None, optional
        See :func:`plot_metric_bars`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not scores_by_slice:
        raise ValueError("scores_by_slice must contain at least one slice")

    fmt = label_formatter or (lambda k: k)
    skey = sort_key or (lambda k: k)

    sorted_items: list[tuple[str, np.ndarray]] = []
    for key, arr in scores_by_slice.items():
        validated = _ensure_ndarray(f"scores_by_slice[{key!r}]", arr)
        if validated.ndim != 1:
            raise ValueError(f"scores_by_slice[{key!r}] must be 1D, got shape {validated.shape}")
        if validated.size == 0:
            raise ValueError(f"scores_by_slice[{key!r}] is empty")
        if not np.isfinite(validated).all():
            raise ValueError(f"scores_by_slice[{key!r}] contains NaN or inf")
        sorted_items.append((key, validated))
    sorted_items.sort(key=lambda kv: skey(kv[0]))

    palette_cycle = [
        PALETTE["negative"],
        PALETTE["positive"],
        PALETTE["accent"],
        PALETTE["baseline"],
    ]

    fig, axes = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE)
    for i, (key, arr) in enumerate(sorted_items):
        color = palette_cycle[i % len(palette_cycle)]
        axes.hist(
            arr,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            histtype="stepfilled",
            alpha=0.4,
            edgecolor=color,
            facecolor=color,
            linewidth=1.0,
            label=fmt(key),
        )

    axes.set_xlabel("Score")
    axes.set_ylabel("Density")
    axes.set_xlim(0.0, 1.0)
    full_title = title
    if full_title is None and scorer_name is not None:
        full_title = f"Score distribution: {scorer_name}"
    if full_title is not None:
        axes.set_title(full_title)
    _maybe_add_legend(axes)
    fig.tight_layout()
    return fig


def plot_lift_ci(
    estimates: dict[str, BootstrapCI],
    *,
    zero_line: bool = True,
    xlabel: str = "Δ (lift over reference)",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Forest-plot-style CIs over comparisons.

    Each row = one comparison key × its 95% CI. When ``zero_line=True``,
    draws a vertical reference at x=0: a CI overlapping zero means the lift
    is not statistically significant.

    Parameters
    ----------
    estimates : dict[str, BootstrapCI]
        Anything with ``point_estimate``, ``ci_low``, ``ci_high`` attributes
        is accepted (duck-typed; not strictly the toolkit's
        :class:`~eval_toolkit.bootstrap.BootstrapCI`).
    zero_line : bool, optional
        Draw a vertical reference at x=0. Default True.
    xlabel : str, optional
        X-axis label (default ``"Δ (lift over reference)"``).
    title, figsize, ax : optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not estimates:
        raise ValueError("estimates must contain at least one comparison")
    for key, est in estimates.items():
        for attr in ("point_estimate", "ci_low", "ci_high"):
            if not hasattr(est, attr):
                raise TypeError(
                    f"estimates[{key!r}] must be a BootstrapCI-like object with "
                    f"point_estimate/ci_low/ci_high attributes"
                )

    keys = list(estimates.keys())
    points = np.array([estimates[k].point_estimate for k in keys])
    ci_low = np.array([estimates[k].ci_low for k in keys])
    ci_high = np.array([estimates[k].ci_high for k in keys])
    yerr = np.array([points - ci_low, ci_high - points])

    resolved_figsize = figsize or (DEFAULT_FIGSIZE[0], max(2.5, 0.5 * len(keys) + 1.5))
    fig, axes = _resolve_axes(ax, resolved_figsize)
    y_positions = np.arange(len(keys))[::-1]

    if zero_line:
        axes.axvline(
            0.0,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    axes.errorbar(
        points,
        y_positions,
        xerr=np.abs(yerr),
        fmt="o",
        color=PALETTE["negative"],
        ecolor=PALETTE["negative"],
        elinewidth=1.2,
        capsize=4,
        markersize=6,
        zorder=3,
    )

    axes.set_yticks(y_positions)
    axes.set_yticklabels(keys)
    axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)
    fig.tight_layout()
    return fig


def plot_bootstrap_distribution(
    deltas: np.ndarray,
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    bins: int = 30,
    title: str | None = None,
    xlabel: str = "Δ (bootstrap-sampled)",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Histogram of bootstrap-sampled deltas with optional CI overlay.

    Useful for diagnosing CI shape (skew, multimodality, normality
    assumption violations) that the scalar ``(ci_low, ci_high)`` summary
    hides. If ``ci_low`` and ``ci_high`` are supplied, vertical lines
    are drawn at the CI bounds plus a reference at zero.

    Parameters
    ----------
    deltas : np.ndarray, shape (n_resamples,)
        1-D array of bootstrap-sampled deltas.
    ci_low, ci_high : float or None, optional
        CI bounds to overlay as vertical lines. Both must be supplied
        together or left as ``None``.
    bins : int, optional
        Histogram bin count. Default 30.
    title, xlabel : optional
    figsize : tuple, optional
    ax : matplotlib.axes.Axes, optional
        Reuse caller's axes.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``deltas`` is empty / non-1-D / contains NaN/Inf, or if exactly
        one of ``ci_low`` / ``ci_high`` is supplied.

    Notes
    -----
    Skewed distributions visible in this plot indicate that the
    half-width-derived ``σ̂`` in :func:`eval_toolkit.bootstrap.mde_from_ci`
    is biased; in that case prefer :func:`eval_toolkit.bootstrap.paired_mde`
    which computes ``σ`` from the deltas directly.

    See Also
    --------
    eval_toolkit.plotting.plot_lift_ci :
        Forest plot of multiple CI summaries (when shape diagnostics are
        not needed).
    """
    deltas_arr = _ensure_ndarray("deltas", deltas)
    if deltas_arr.ndim != 1:
        raise ValueError(f"deltas must be 1-D, got shape {deltas_arr.shape}")
    if deltas_arr.size == 0:
        raise ValueError("deltas is empty")
    if not np.isfinite(deltas_arr).all():
        raise ValueError("deltas contains NaN or inf")
    if (ci_low is None) != (ci_high is None):
        raise ValueError("ci_low and ci_high must both be supplied or both None")

    fig, axes = _resolve_axes(ax, figsize)
    axes.hist(
        deltas_arr,
        bins=bins,
        color=PALETTE["negative"],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    axes.axvline(0.0, color=PALETTE["baseline"], linestyle="--", linewidth=0.8, alpha=0.7)
    if ci_low is not None and ci_high is not None:
        for x in (ci_low, ci_high):
            axes.axvline(
                x,
                color=PALETTE["accent"],
                linestyle="-",
                linewidth=1.2,
                label=f"CI bound: {x:+.4f}",
            )
        _maybe_add_legend(axes)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("count")
    if title is not None:
        axes.set_title(title)
    fig.tight_layout()
    return fig
