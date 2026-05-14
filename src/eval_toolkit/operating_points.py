"""Cross-slice operating-point transfer for binary classification.

The core workflow is:

1. Fit one or more threshold selectors on a mixed-class slice.
2. Apply the fitted thresholds to any target slice.

Target slices may be mixed-class, all-positive, or all-negative. Mixed-class
targets report the usual threshold metrics. Single-class targets report the
quantity that remains meaningful at an externally selected threshold:
recall for all-positive slices, FPR/specificity for all-negative slices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from eval_toolkit.metrics import (
    metrics_at_threshold,
    single_class_threshold_metrics,
)
from eval_toolkit.thresholds import ThresholdSelector

__all__ = [
    "FittedOperatingPoint",
    "OperatingPointSpec",
    "apply_operating_points",
    "fit_operating_points",
]


@dataclass(frozen=True, slots=True)
class FittedOperatingPoint:
    """A threshold selected on one scorer/slice pair.

    Parameters
    ----------
    criterion : str
        Selector label, usually ``selector.criterion``.
    threshold : float
        Decision boundary selected on the fit slice.
    fitted_on_slice : str
        Name of the slice used to fit the threshold.
    scorer_name : str
        Scorer whose scores were used to fit the threshold.
    n_fit, n_positive_fit, n_negative_fit : int
        Fit-slice counts recorded for provenance.
    fit_metrics : dict
        ``metrics_at_threshold`` on the fit slice at the selected threshold.
    """

    criterion: str
    threshold: float
    fitted_on_slice: str
    scorer_name: str
    n_fit: int
    n_positive_fit: int
    n_negative_fit: int
    fit_metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperatingPointSpec:
    """Harness instruction for cross-slice operating-point transfer.

    ``evaluate(..., operating_point_specs=[...])`` fits each selector on
    ``fit_slice`` and applies the resulting thresholds to every named target
    in ``apply_slices``. If ``scorer_names`` is empty, all evaluated scorers
    participate.
    """

    fit_slice: str
    apply_slices: tuple[str, ...]
    selectors: tuple[ThresholdSelector, ...]
    name: str = "transferred"
    scorer_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the spec while preserving tuple fields for immutability."""
        object.__setattr__(self, "apply_slices", tuple(self.apply_slices))
        object.__setattr__(self, "selectors", tuple(self.selectors))
        object.__setattr__(self, "scorer_names", tuple(self.scorer_names))
        if not self.fit_slice:
            raise ValueError("fit_slice must be non-empty")
        if not self.apply_slices:
            raise ValueError("apply_slices must be non-empty")
        if not self.selectors:
            raise ValueError("selectors must be non-empty")
        if not self.name:
            raise ValueError("name must be non-empty")
        if any(not s for s in self.apply_slices):
            raise ValueError("apply_slices must not contain empty names")
        if any(not s for s in self.scorer_names):
            raise ValueError("scorer_names must not contain empty names")


def fit_operating_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
    selectors: Sequence[ThresholdSelector],
    *,
    fitted_on_slice: str = "",
    scorer_name: str = "",
) -> dict[str, FittedOperatingPoint]:
    """Fit selectors on a mixed-class slice and record threshold provenance.

    Raises from the underlying selector if the data cannot support threshold
    selection, for example single-class labels or no feasible target
    precision/recall/FPR threshold.

    Raises
    ------
    ValueError
        If the fit slice is single-class (the function requires both
        positive and negative labels for threshold fitting). Selector
        raises (e.g. :exc:`RuntimeError` on infeasible target rates) are
        propagated unchanged.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    labels = set(np.unique(y_true_arr).tolist())
    if len(labels) != 2:
        raise ValueError(
            "fit_operating_points requires a mixed-class fit slice; " f"got labels {sorted(labels)}"
        )

    out: dict[str, FittedOperatingPoint] = {}
    n_positive = int(np.sum(y_true_arr))
    n_fit = int(len(y_true_arr))
    for selector in selectors:
        selected = selector.select(y_true_arr, y_score_arr)
        fit_metrics = metrics_at_threshold(y_true_arr, y_score_arr, selected.threshold)
        out[selected.criterion] = FittedOperatingPoint(
            criterion=selected.criterion,
            threshold=float(selected.threshold),
            fitted_on_slice=fitted_on_slice,
            scorer_name=scorer_name,
            n_fit=n_fit,
            n_positive_fit=n_positive,
            n_negative_fit=n_fit - n_positive,
            fit_metrics=dict(fit_metrics),
        )
    return out


def apply_operating_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fitted: Mapping[str, FittedOperatingPoint],
    *,
    applied_to_slice: str = "",
    scorer_name: str = "",
) -> dict[str, dict[str, object]]:
    """Apply fitted thresholds to a mixed-class or single-class target slice.

    Raises
    ------
    ValueError
        If ``y_true`` contains labels outside ``{0, 1}``.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    labels = set(np.unique(y_true_arr).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"labels must be binary in {{0, 1}}, got {sorted(labels)}")

    out: dict[str, dict[str, object]] = {}
    for criterion, op in fitted.items():
        metrics: dict[str, object]
        if len(labels) == 1:
            metrics = dict(single_class_threshold_metrics(y_true_arr, y_score_arr, op.threshold))
        else:
            metrics = dict(metrics_at_threshold(y_true_arr, y_score_arr, op.threshold))
        metrics["threshold_provenance"] = {
            "criterion": op.criterion,
            "fitted_on_slice": op.fitted_on_slice,
            "applied_to_slice": applied_to_slice,
            "scorer_name": scorer_name or op.scorer_name,
            "threshold": op.threshold,
            "n_fit": op.n_fit,
            "n_positive_fit": op.n_positive_fit,
            "n_negative_fit": op.n_negative_fit,
        }
        out[str(criterion)] = metrics
    return out
