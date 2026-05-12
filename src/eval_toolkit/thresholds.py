"""Threshold selection: pluggable :class:`ThresholdSelector` Protocol + reference impls.

A :class:`ThresholdSelector` is anything implementing
``select(y_true, y_score) -> ThresholdResult``. The toolkit ships six reference
impls covering the rules seen in real binary-classification eval workflows:

- :class:`MaxF1Selector` — argmax F1 over the PR curve
- :class:`TargetRecallSelector(recall)` — *highest* threshold meeting recall ≥ target
- :class:`TargetPrecisionSelector(precision)` — *smallest* threshold meeting precision ≥ target
- :class:`TargetFPRSelector(fpr)` — *smallest* threshold meeting FPR ≤ target on the ROC curve
- :class:`YoudenJSelector` — argmax (TPR − FPR) on the ROC curve
- :class:`CostSensitiveSelector(cost_matrix)` — Bayes-optimal threshold from
  prior + cost matrix (assumes ``y_score`` is a calibrated probability)

Replaces the v0.6 string-criterion API in :func:`metrics.select_threshold`.
See ``CHANGELOG.md`` for the migration mapping.

References
----------
.. [1] Lipton, Z., Elkan, C., & Naryanaswamy, B. "Optimal thresholding of
       classifiers to maximize F1 measure." ECML PKDD 2014. arXiv:1402.1892.
.. [2] Youden, W. J. "Index for rating diagnostic tests." Cancer 3(1), 1950.
.. [3] Elkan, C. "The foundations of cost-sensitive learning." IJCAI 2001.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm as _scipy_norm
from sklearn.metrics import precision_recall_curve, roc_curve

from eval_toolkit.calibration import CostMatrix
from eval_toolkit.metrics import ThresholdResult, _validate_inputs, metrics_at_threshold

__all__ = [
    "CISafeThresholdSelector",
    "CostSensitiveSelector",
    "MaxF1Selector",
    "TargetFPRSelector",
    "TargetPrecisionSelector",
    "TargetRecallSelector",
    "ThresholdPolicyMetadata",
    "ThresholdSelector",
    "WilsonInterval",
    "YoudenJSelector",
    "select_threshold",
    "wilson_interval",
]


@runtime_checkable
class ThresholdSelector(Protocol):
    """Selects a decision threshold from ``(y_true, y_score)``.

    Implementations are typically frozen dataclasses with config in the
    constructor (e.g., ``TargetRecallSelector(recall=0.90)``) and a
    :meth:`select` method that performs the sweep. The returned
    :class:`~eval_toolkit.metrics.ThresholdResult` is the interchange type so
    every selector composes with :func:`metrics.metrics_at_threshold` and the
    paired-bootstrap operating-point machinery.

    Attributes
    ----------
    criterion : str
        Stable label for this selector instance, written into the
        :class:`ThresholdResult` and used as a dict key in headline output.
        Implementations may expose it as a class-level default
        (``MaxF1Selector``) or as a derived property (``TargetRecallSelector``).

    Notes
    -----
    Runtime-checkable: ``isinstance(obj, ThresholdSelector)`` returns ``True``
    for any object exposing both :meth:`select` and :attr:`criterion`. Mirrors
    the :class:`eval_toolkit.harness.Scorer` Protocol pattern.
    """

    @property
    def criterion(self) -> str:  # pragma: no cover
        """Stable label written into the resulting :class:`ThresholdResult`."""
        ...

    def select(  # pragma: no cover
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> ThresholdResult:
        """Return the chosen threshold and its precision / recall / F1."""
        ...


def _f1(precision: float, recall: float) -> float:
    """F1 with safe denominator. Matches the existing ``select_threshold`` body."""
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def _pr_curve_trim(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PR curve with the trailing ``recall=0`` row dropped.

    sklearn returns ``N+1`` precision / recall but ``N`` thresholds; the
    ``[:-1]`` trim aligns indices into a single-axis selection.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        raise RuntimeError("PR curve has no thresholds — y_score may be constant")
    return precisions[:-1], recalls[:-1], thresholds


def _roc_curve_trim(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve with the leading "predict all negative" sentinel row dropped.

    sklearn pads ``thresholds[0]`` with ``max(y_score) + 1`` (or ``+inf`` in
    older versions) representing the trivial "predict everything negative"
    point. It is never an interesting operating point for selection — drop it
    so threshold values returned by ROC-based selectors stay finite and
    bounded by the score range.
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    if len(thresholds) == 0:
        raise RuntimeError("ROC curve has no thresholds — y_score may be constant")
    score_max = float(np.asarray(y_score).max())
    if len(thresholds) >= 2 and (np.isinf(thresholds[0]) or thresholds[0] > score_max):
        return fprs[1:], tprs[1:], thresholds[1:]
    return fprs, tprs, thresholds


def _result_at(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, criterion: str
) -> ThresholdResult:
    """Build a :class:`ThresholdResult` by evaluating metrics at ``threshold``."""
    m = metrics_at_threshold(y_true, y_score, threshold)
    return ThresholdResult(
        threshold=float(threshold),
        f1=float(m["f1"]),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        criterion=criterion,
    )


def _record_float(row: Mapping[str, object], key: str) -> float:
    """Read a numeric candidate-record field as float."""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"candidate record field {key!r} must be numeric")
    return float(value)


def _record_float_or_inf(row: Mapping[str, object], key: str) -> float:
    """Read an optional numeric candidate-record field, defaulting to +inf."""
    value = row[key]
    if value is None:
        return float("inf")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"candidate record field {key!r} must be numeric or null")
    return float(value)


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """Wilson score interval for a binomial rate."""

    low: float | None
    high: float | None
    confidence: float
    successes: int
    n: int

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "successes": self.successes,
            "n": self.n,
        }


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for a binomial rate."""
    if successes < 0:
        raise ValueError("successes must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes > n:
        raise ValueError("successes cannot exceed n")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return WilsonInterval(
            low=None,
            high=None,
            confidence=confidence,
            successes=successes,
            n=n,
        )
    z = float(_scipy_norm.ppf(0.5 + confidence / 2.0))
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))) / denom
    return WilsonInterval(
        low=float(max(0.0, centre - margin)),
        high=float(min(1.0, centre + margin)),
        confidence=confidence,
        successes=successes,
        n=n,
    )


@dataclass(frozen=True, slots=True)
class ThresholdPolicyMetadata:
    """Pre-registration metadata for thresholded operating claims."""

    calibration_slice: str
    score_column: str
    selector: str
    constraints: dict[str, float]
    claim_enabled: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the policy metadata."""
        if not self.calibration_slice:
            raise ValueError("calibration_slice must be non-empty")
        if not self.score_column:
            raise ValueError("score_column must be non-empty")
        if not self.selector:
            raise ValueError("selector must be non-empty")
        if not self.constraints:
            raise ValueError("constraints must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "calibration_slice": self.calibration_slice,
            "score_column": self.score_column,
            "selector": self.selector,
            "constraints": dict(self.constraints),
            "claim_enabled": self.claim_enabled,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CISafeThresholdSelector:
    """Select a threshold whose rate confidence bounds satisfy constraints.

    The selector does not ship target defaults. Callers must provide one or
    more explicit constraints, for example ``max_fpr`` and
    ``max_fpr_ci_upper`` for a low-FPR claim.
    """

    max_fpr: float | None = None
    max_fpr_ci_upper: float | None = None
    min_recall: float | None = None
    min_recall_ci_lower: float | None = None
    confidence: float = 0.95
    criterion: str = "ci_safe"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate constraint shape."""
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        constraints = (
            self.max_fpr,
            self.max_fpr_ci_upper,
            self.min_recall,
            self.min_recall_ci_lower,
        )
        if all(value is None for value in constraints):
            raise ValueError("at least one CI-safe threshold constraint is required")
        for name, value in (
            ("max_fpr", self.max_fpr),
            ("max_fpr_ci_upper", self.max_fpr_ci_upper),
            ("min_recall", self.min_recall),
            ("min_recall_ci_lower", self.min_recall_ci_lower),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the best accepted threshold."""
        accepted = [row for row in self.candidate_records(y_true, y_score) if row["accepted"]]
        if not accepted:
            raise RuntimeError("no threshold satisfies the configured CI-safe constraints")
        best = min(
            accepted,
            key=lambda row: (
                -_record_float(row, "recall"),
                _record_float_or_inf(row, "fpr_ci_high"),
                _record_float(row, "fpr"),
                -_record_float(row, "threshold"),
            ),
        )
        return _result_at(y_true, y_score, _record_float(best, "threshold"), self.criterion)

    def candidate_records(self, y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, object]]:
        """Return every candidate threshold with Wilson rate intervals."""
        _validate_inputs(y_true, y_score)
        y_true_arr = np.asarray(y_true, dtype=int).ravel()
        y_score_arr = np.asarray(y_score, dtype=float).ravel()
        thresholds = np.unique(y_score_arr[np.isfinite(y_score_arr)])
        if thresholds.size == 0:
            return []
        neg_scores = np.sort(y_score_arr[y_true_arr == 0])
        pos_scores = np.sort(y_score_arr[y_true_arr == 1])
        n_neg = int(neg_scores.size)
        n_pos = int(pos_scores.size)
        rows: list[dict[str, object]] = []
        for threshold in thresholds:
            fp = int(n_neg - np.searchsorted(neg_scores, threshold, side="left"))
            tp = int(n_pos - np.searchsorted(pos_scores, threshold, side="left"))
            tn = n_neg - fp
            fn = n_pos - tp
            rows.append(self._candidate_record(float(threshold), tp=tp, fp=fp, tn=tn, fn=fn))
        return rows

    def selected_operating_point(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        *,
        bootstrap_selected: bool = False,
        n_resamples: int = 1000,
        seed: int = 42,
    ) -> dict[str, object]:
        """Return selected threshold metadata and optional bootstrap CIs."""
        selected = self.select(y_true, y_score)
        records = self.candidate_records(y_true, y_score)
        selected_record = next(
            row for row in records if _record_float(row, "threshold") == float(selected.threshold)
        )
        out: dict[str, object] = {
            "threshold_result": asdict(selected),
            "selected_record": selected_record,
            "n_candidates": len(records),
            "n_accepted": sum(1 for row in records if row["accepted"]),
            "constraints": self.constraints,
            "metadata": dict(self.metadata),
        }
        if bootstrap_selected:
            out["bootstrap_selected"] = _bootstrap_threshold_metric_cis(
                y_true,
                y_score,
                selected.threshold,
                n_resamples=n_resamples,
                seed=seed,
            )
        return out

    @property
    def constraints(self) -> dict[str, float]:
        """Configured threshold constraints."""
        out: dict[str, float] = {}
        for key in ("max_fpr", "max_fpr_ci_upper", "min_recall", "min_recall_ci_lower"):
            value = getattr(self, key)
            if value is not None:
                out[key] = float(value)
        return out

    def _candidate_record(
        self,
        threshold: float,
        *,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> dict[str, object]:
        n_pos = tp + fn
        n_neg = fp + tn
        fpr = float(fp / n_neg) if n_neg else 0.0
        recall = float(tp / n_pos) if n_pos else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = _f1(precision, recall)
        fpr_ci = wilson_interval(fp, n_neg, confidence=self.confidence)
        recall_ci = wilson_interval(tp, n_pos, confidence=self.confidence)
        rejection_reasons: list[str] = []
        if self.max_fpr is not None and fpr > self.max_fpr:
            rejection_reasons.append("fpr_exceeds_max")
        if self.max_fpr_ci_upper is not None and (
            fpr_ci.high is None or fpr_ci.high > self.max_fpr_ci_upper
        ):
            rejection_reasons.append("fpr_ci_upper_exceeds_max")
        if self.min_recall is not None and recall < self.min_recall:
            rejection_reasons.append("recall_below_min")
        if self.min_recall_ci_lower is not None and (
            recall_ci.low is None or recall_ci.low < self.min_recall_ci_lower
        ):
            rejection_reasons.append("recall_ci_lower_below_min")
        return {
            "threshold": threshold,
            "fpr": fpr,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fpr_ci_low": fpr_ci.low,
            "fpr_ci_high": fpr_ci.high,
            "recall_ci_low": recall_ci.low,
            "recall_ci_high": recall_ci.high,
            "confidence": self.confidence,
            "accepted": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }


def _bootstrap_threshold_metric_cis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Percentile bootstrap CIs for selected-threshold operating metrics."""
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_score_arr = np.asarray(y_score, dtype=float).ravel()
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have identical shape")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"fpr": [], "recall": [], "precision": [], "f1": []}
    n = len(y_true_arr)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        metrics = metrics_at_threshold(y_true_arr[idx], y_score_arr[idx], threshold)
        for key in samples:
            samples[key].append(float(metrics[key]))
    return {
        key: {
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "n_resamples": n_resamples,
            "method": "percentile",
        }
        for key, values in samples.items()
    }


@dataclass(frozen=True, slots=True)
class MaxF1Selector:
    """Argmax F1 over the PR curve. Replaces the v0.6 ``criterion="max_f1"``.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.thresholds import MaxF1Selector
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    >>> tr = MaxF1Selector().select(y, s)
    >>> tr.f1
    1.0
    """

    criterion: str = "max_f1"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the threshold maximizing F1."""
        _validate_inputs(y_true, y_score)
        precisions, recalls, thresholds = _pr_curve_trim(y_true, y_score)
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
        idx = int(np.argmax(f1s))
        return ThresholdResult(
            threshold=float(thresholds[idx]),
            f1=_f1(float(precisions[idx]), float(recalls[idx])),
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
            criterion=self.criterion,
        )


@dataclass(frozen=True, slots=True)
class TargetRecallSelector:
    """Highest threshold meeting recall ≥ target — most-precise feasible point.

    Replaces ``criterion="recall_0.90"`` / ``"recall_0.95"`` (and
    ``"recall@0.90"`` from prompt-injection-sdd's local fork).

    Parameters
    ----------
    recall : float
        Target recall floor in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``recall`` is outside ``(0, 1]``.
    RuntimeError
        At ``select`` time, if no threshold achieves recall ≥ target.
    """

    recall: float

    def __post_init__(self) -> None:
        """Validate the recall target."""
        if not 0.0 < self.recall <= 1.0:
            raise ValueError(f"recall must be in (0, 1], got {self.recall}")

    @property
    def criterion(self) -> str:
        """Stable label embedding the target recall."""
        return f"recall_{self.recall:.2f}"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the highest threshold meeting recall ≥ ``self.recall``."""
        _validate_inputs(y_true, y_score)
        precisions, recalls, thresholds = _pr_curve_trim(y_true, y_score)
        # Recall is monotonically non-increasing in threshold; eligible is
        # contiguous starting at index 0; eligible[-1] is the highest
        # threshold (most-precise) still satisfying the recall floor.
        eligible = np.where(recalls >= self.recall)[0]
        if len(eligible) == 0:
            raise RuntimeError(
                f"No threshold achieves recall ≥ {self.recall}; "
                f"max recall = {recalls.max():.3f}"
            )
        idx = int(eligible[-1])
        return ThresholdResult(
            threshold=float(thresholds[idx]),
            f1=_f1(float(precisions[idx]), float(recalls[idx])),
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
            criterion=self.criterion,
        )


@dataclass(frozen=True, slots=True)
class TargetPrecisionSelector:
    """Smallest threshold meeting precision ≥ target — highest-recall feasible point.

    NEW in v0.7.0: required by ``prompt-injection-sdd``'s ``"precision@0.90"``
    workflow, which previously forced a local re-implementation of
    ``select_threshold``.

    Parameters
    ----------
    precision : float
        Target precision floor in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``precision`` is outside ``(0, 1]``.
    RuntimeError
        At ``select`` time, if no threshold achieves precision ≥ target.

    Notes
    -----
    Precision is *not* monotonic in threshold (it can oscillate as positives
    vanish), so we cannot exit early on the first non-eligible point — the
    full PR curve is scanned.
    """

    precision: float

    def __post_init__(self) -> None:
        """Validate the precision target."""
        if not 0.0 < self.precision <= 1.0:
            raise ValueError(f"precision must be in (0, 1], got {self.precision}")

    @property
    def criterion(self) -> str:
        """Stable label embedding the target precision."""
        return f"precision_{self.precision:.2f}"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the smallest threshold meeting precision ≥ ``self.precision``."""
        _validate_inputs(y_true, y_score)
        precisions, recalls, thresholds = _pr_curve_trim(y_true, y_score)
        eligible = np.where(precisions >= self.precision)[0]
        if len(eligible) == 0:
            raise RuntimeError(
                f"No threshold achieves precision ≥ {self.precision}; "
                f"max precision = {precisions.max():.3f}"
            )
        # Smallest-threshold-meeting-target = first eligible index.
        # PR-curve thresholds are sorted ascending.
        idx = int(eligible[0])
        return ThresholdResult(
            threshold=float(thresholds[idx]),
            f1=_f1(float(precisions[idx]), float(recalls[idx])),
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
            criterion=self.criterion,
        )


@dataclass(frozen=True, slots=True)
class TargetFPRSelector:
    """Smallest threshold meeting FPR ≤ target — highest-TPR feasible ROC point.

    Useful for "screening" workflows where downstream cost is dominated by
    false positives (e.g., classifier flagging a queue for human review).

    Parameters
    ----------
    fpr : float
        Target false-positive-rate ceiling in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``fpr`` is outside ``[0, 1]``.
    RuntimeError
        At ``select`` time, if no threshold achieves FPR ≤ target.
    """

    fpr: float

    def __post_init__(self) -> None:
        """Validate the FPR ceiling."""
        if not 0.0 <= self.fpr <= 1.0:
            raise ValueError(f"fpr must be in [0, 1], got {self.fpr}")

    @property
    def criterion(self) -> str:
        """Stable label embedding the target FPR."""
        return f"fpr_{self.fpr:.2f}"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the smallest threshold (highest TPR) meeting FPR ≤ target."""
        _validate_inputs(y_true, y_score)
        fprs, _tprs, thresholds = _roc_curve_trim(y_true, y_score)
        eligible = np.where(fprs <= self.fpr)[0]
        if len(eligible) == 0:
            raise RuntimeError(
                f"No threshold achieves FPR ≤ {self.fpr}; " f"min fpr = {fprs.min():.3f}"
            )
        # Trimmed ROC: thresholds are sorted descending, FPR is non-decreasing
        # as threshold decreases. Eligible is contiguous from index 0;
        # eligible[-1] is the largest FPR still meeting the ceiling = highest
        # TPR (most-recall) feasible point.
        idx = int(eligible[-1])
        return _result_at(y_true, y_score, float(thresholds[idx]), self.criterion)


@dataclass(frozen=True, slots=True)
class YoudenJSelector:
    """Argmax of Youden's J statistic ``J = TPR − FPR`` over the ROC curve.

    Threshold-free, balanced criterion that does not require choosing a
    target rate. Often the right default when costs are unknown or roughly
    symmetric and the consumer wants a single "best" operating point.

    References
    ----------
    .. [1] Youden, W. J. "Index for rating diagnostic tests." Cancer 3(1),
           1950.
    """

    criterion: str = "youden_j"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the threshold maximizing Youden's J."""
        _validate_inputs(y_true, y_score)
        fprs, tprs, thresholds = _roc_curve_trim(y_true, y_score)
        j = tprs - fprs
        idx = int(np.argmax(j))
        return _result_at(y_true, y_score, float(thresholds[idx]), self.criterion)


@dataclass(frozen=True, slots=True)
class CostSensitiveSelector:
    """Bayes-optimal threshold from a :class:`~eval_toolkit.calibration.CostMatrix`.

    Closed-form Elkan 2001: ``t* = c_fp · (1 − π) / (c_fp · (1 − π) + c_fn · π)``.
    Assumes ``y_score`` is a calibrated probability in ``[0, 1]``; for raw
    scores, apply Platt / temperature scaling first
    (:func:`eval_toolkit.calibration.fit_platt_calibrator` etc.).

    Parameters
    ----------
    cost_matrix : CostMatrix
        Carries ``prior``, ``fp_cost``, ``fn_cost``. Threshold derives from
        :attr:`CostMatrix.bayes_threshold`.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.calibration import CostMatrix
    >>> from eval_toolkit.thresholds import CostSensitiveSelector
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    >>> cm = CostMatrix(prior=0.5, fp_cost=1.0, fn_cost=1.0)
    >>> tr = CostSensitiveSelector(cm).select(y, s)
    >>> tr.threshold
    0.5
    """

    cost_matrix: CostMatrix

    @property
    def criterion(self) -> str:
        """Stable label embedding the cost-matrix triple."""
        cm = self.cost_matrix
        return f"cost_sensitive_prior={cm.prior:.3f}" f"_fp={cm.fp_cost:.2f}_fn={cm.fn_cost:.2f}"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """Return the Bayes-optimal threshold for this cost matrix."""
        _validate_inputs(y_true, y_score)
        threshold = self.cost_matrix.bayes_threshold
        return _result_at(y_true, y_score, threshold, self.criterion)


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: ThresholdSelector,
) -> ThresholdResult:
    """Dispatch to a :class:`ThresholdSelector` instance.

    Convenience wrapper that's equivalent to ``criterion.select(y_true, y_score)``;
    its only job is to enforce the v0.7.0 contract and produce a helpful
    migration error if a v0.6 string is passed.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels in ``{0, 1}``.
    y_score : np.ndarray
        Real-valued scores.
    criterion : ThresholdSelector
        Any object implementing ``select(y_true, y_score) -> ThresholdResult``.

    Returns
    -------
    ThresholdResult

    Raises
    ------
    TypeError
        If ``criterion`` is not a :class:`ThresholdSelector` (e.g., a v0.6
        string is passed). The error message includes the migration mapping.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.thresholds import MaxF1Selector, select_threshold
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    >>> tr = select_threshold(y, s, criterion=MaxF1Selector())
    >>> tr.f1
    1.0

    .. versionchanged:: 0.7.0
        BREAKING — ``criterion`` now requires a :class:`ThresholdSelector`
        instance. The v0.6 string form (``"max_f1"`` /  ``"recall_0.90"`` /
        ``"recall_0.95"``) is removed. Migration:

        ====================  ===========================================
        v0.6                  v0.7
        ====================  ===========================================
        ``"max_f1"``          :class:`MaxF1Selector()`
        ``"recall_0.90"``     :class:`TargetRecallSelector(0.90)`
        ``"recall_0.95"``     :class:`TargetRecallSelector(0.95)`
        ``"precision@p"``     :class:`TargetPrecisionSelector(p)`  *(new)*
        ``"recall@p"``        :class:`TargetRecallSelector(p)`
        ====================  ===========================================

        Passing a string raises :class:`TypeError` whose message includes
        this mapping.
    """
    if not isinstance(criterion, ThresholdSelector):
        raise TypeError(
            "select_threshold requires a ThresholdSelector instance "
            f"(v0.7.0+); got {type(criterion).__name__}={criterion!r}.\n"
            "Migration:\n"
            "  'max_f1'      -> MaxF1Selector()\n"
            "  'recall_0.90' -> TargetRecallSelector(0.90)\n"
            "  'recall_0.95' -> TargetRecallSelector(0.95)\n"
            "  'precision@p' -> TargetPrecisionSelector(p)\n"
            "  'recall@p'    -> TargetRecallSelector(p)\n"
            "See CHANGELOG v0.7.0 for the full guide."
        )
    return criterion.select(y_true, y_score)
