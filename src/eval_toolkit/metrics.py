"""Binary classification metrics: PR-AUC, ROC-AUC, F1 at threshold, ECE, prior-shift projection.

Wraps sklearn references with input validation, single-class slice handling,
and threshold-free score-distribution summaries. All functions accept raw
numpy arrays (binary y_true in {0, 1}, real-valued y_score) and return
JSON-serializable values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from eval_toolkit.thresholds import ThresholdSelector
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from eval_toolkit.artifacts import skipped_metric

EmptyStrategy = Literal["raise", "return_none", "skipped_metric"]


class _SentinelOk:
    """Marker returned by :func:`_empty_strategy_guard` for non-degenerate input.

    Distinct type so mypy can narrow the union via ``isinstance``; using
    ``None`` would conflict with ``empty_strategy="return_none"`` returning
    ``None`` as a real value.
    """


_SENTINEL_OK: Final[_SentinelOk] = _SentinelOk()


def _empty_strategy_guard(
    y_true: np.ndarray,
    strategy: EmptyStrategy,
    metric_name: str,
    *,
    check_single_class: bool = True,
) -> float | None | dict[str, object] | _SentinelOk:
    """Short-circuit return value for degenerate inputs under empty_strategy.

    Returns ``_SENTINEL_OK`` if the input is non-degenerate (proceed to
    metric computation). Otherwise returns ``None`` or a structured
    :func:`~eval_toolkit.artifacts.skipped_metric` dict per the strategy.

    Parameters
    ----------
    y_true : np.ndarray
        Labels to inspect.
    strategy : EmptyStrategy
        How to handle empty / single-class input. ``"raise"`` is rejected here
        — callers should fall through to the validator and sklearn call.
    metric_name : str
        Used in the structured reason string for ``"skipped_metric"``.
    check_single_class : bool, optional
        If ``True`` (default), single-class ``y_true`` is also treated as
        degenerate. Required for AUC metrics; ``False`` for ``brier_score``
        which is valid on single-class.
    """
    if strategy not in {"return_none", "skipped_metric"}:
        raise ValueError(
            f"empty_strategy must be one of 'raise', 'return_none', "
            f"'skipped_metric'; got {strategy!r}"
        )
    y_arr = np.asarray(y_true)
    n = int(y_arr.size)
    if n == 0:
        if strategy == "return_none":
            return None
        return skipped_metric(f"{metric_name}: empty input (n=0)", n=0)
    if check_single_class:
        unique = np.unique(y_arr)
        if unique.size < 2:
            if strategy == "return_none":
                return None
            return skipped_metric(
                f"{metric_name}: single-class y_true (only {unique.tolist()})",
                n=n,
                unique=unique.tolist(),
            )
    return _SENTINEL_OK


__all__ = [
    "DEFAULT_ASSUMED_PRIORS",
    "ThresholdResult",
    "brier_decomposition",
    "brier_score",
    "expected_calibration_error",
    "expected_calibration_error_debiased",
    "expected_calibration_error_equal_mass",
    "expected_calibration_error_l2",
    "expected_calibration_error_l2_debiased",
    "headline_metrics",
    "metrics_at_threshold",
    "pr_auc",
    "precision_at_prior",
    "quantile_stratified_pr_auc",
    "quantile_stratified_report",
    "roc_auc",
    "score_distribution_summary",
    "single_class_threshold_metrics",
    "stratified_recall",
]


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Outcome of operating-point selection at a given criterion.

    Parameters
    ----------
    threshold : float
        Decision boundary on the score; predictions are positive when
        ``y_score >= threshold``.
    f1 : float
        F1 score at this threshold ∈ [0, 1].
    precision : float
        Precision at this threshold ∈ [0, 1].
    recall : float
        Recall (true-positive rate) at this threshold ∈ [0, 1].
    criterion : str
        Selection-rule label (e.g. ``"max_f1"``, ``"recall_0.90"``).

    Examples
    --------
    >>> tr = ThresholdResult(
    ...     threshold=0.5, f1=0.8, precision=0.9, recall=0.72, criterion="max_f1"
    ... )
    >>> tr.threshold, tr.criterion
    (0.5, 'max_f1')

    Notes
    -----
    Frozen value-type; equality is field-wise. The selection rule is recorded
    in ``criterion`` so downstream consumers (plots, reports) can label
    operating points without re-deriving the rule.

    References
    ----------
    .. [1] Davis, J. & Goadrich, M. "The relationship between precision-recall
           and ROC curves." ICML 2006.
    """

    threshold: float
    f1: float
    precision: float
    recall: float
    criterion: str


@overload
def pr_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["raise"] = ...
) -> float: ...
@overload
def pr_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["return_none"]
) -> float | None: ...
@overload
def pr_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["skipped_metric"]
) -> float | dict[str, object]: ...
def pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    empty_strategy: EmptyStrategy = "raise",
) -> float | None | dict[str, object]:
    """Average precision (PR-AUC).

    Primary metric for rare-positive binary classification because it is
    sensitive to changes in the positive class, unlike ROC-AUC which can be
    inflated by abundant true negatives.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Real-valued scores; higher means more positive.
    empty_strategy : {"raise", "return_none", "skipped_metric"}, optional
        How to handle degenerate input (empty array or single-class ``y_true``).
        Default ``"raise"`` preserves pre-v0.13 behavior — the underlying
        sklearn call raises. ``"return_none"`` short-circuits with ``None``
        (useful for callers that emit ``None`` columns for degenerate slices).
        ``"skipped_metric"`` returns a structured
        :func:`~eval_toolkit.artifacts.skipped_metric` dict so callers can
        thread the reason through JSON artifacts. Closes F1.2 from the V4
        consumer feedback log.

    Returns
    -------
    float | None | dict[str, object]
        PR-AUC ∈ [prevalence, 1]. Wraps ``sklearn.metrics.average_precision_score``.
        With non-default ``empty_strategy``, the return type widens — see
        overload variants.

    Raises
    ------
    ValueError
        If shapes mismatch, dimensions are wrong, labels are not binary, or
        the input is empty / single-class AND ``empty_strategy="raise"``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> s = np.array([0.1, 0.9, 0.2, 0.8])
    >>> pr_auc(y, s)
    1.0

    Notes
    -----
    PR-AUC is the area under the precision-recall curve, computed by sklearn
    as the weighted mean of precisions at each threshold:

    .. math:: \\mathrm{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` are precision and recall at the :math:`n`-th threshold.

    See Also
    --------
    eval_toolkit.metrics.roc_auc : ROC-AUC; less informative under imbalance.
    eval_toolkit.metrics.headline_metrics : Bundle including PR-AUC.
    sklearn.metrics.average_precision_score : Underlying sklearn implementation.

    References
    ----------
    .. [1] Davis, J. & Goadrich, M. "The relationship between precision-recall
           and ROC curves." ICML 2006.
    .. [2] Saito, T. & Rehmsmeier, M. "The precision-recall plot is more
           informative than the ROC plot when evaluating binary classifiers
           on imbalanced datasets." PLOS ONE 10(3), 2015.
    """
    if empty_strategy != "raise":
        guarded = _empty_strategy_guard(y_true, empty_strategy, "pr_auc")
        if not isinstance(guarded, _SentinelOk):
            return guarded
    _validate_inputs(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


@overload
def roc_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["raise"] = ...
) -> float: ...
@overload
def roc_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["return_none"]
) -> float | None: ...
@overload
def roc_auc(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["skipped_metric"]
) -> float | dict[str, object]: ...
def roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    empty_strategy: EmptyStrategy = "raise",
) -> float | None | dict[str, object]:
    """ROC-AUC.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Real-valued scores; higher means more positive.
    empty_strategy : {"raise", "return_none", "skipped_metric"}, optional
        Degenerate-input handling — see :func:`pr_auc` for full semantics.
        Default ``"raise"`` preserves pre-v0.13 behavior.

    Returns
    -------
    float | None | dict[str, object]
        ROC-AUC ∈ [0, 1]. 0.5 = random; 1.0 = perfect ranking. With non-default
        ``empty_strategy``, the return type widens — see overload variants.

    Raises
    ------
    ValueError
        If shapes mismatch, dimensions are wrong, labels are not binary, or
        input is empty / single-class AND ``empty_strategy="raise"``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> s = np.array([0.1, 0.9, 0.2, 0.8])
    >>> roc_auc(y, s)
    1.0

    Notes
    -----
    ROC-AUC is invariant to monotone transforms of ``y_score``: for any
    strictly monotone :math:`f`, :math:`\\mathrm{ROC-AUC}(y, s) = \\mathrm{ROC-AUC}(y, f(s))`.
    Inversion: :math:`\\mathrm{ROC-AUC}(y, -s) = 1 - \\mathrm{ROC-AUC}(y, s)`.

    See Also
    --------
    eval_toolkit.metrics.pr_auc : Prefer PR-AUC under class imbalance.
    sklearn.metrics.roc_auc_score : Underlying sklearn implementation.

    References
    ----------
    .. [1] Hanley, J. A. & McNeil, B. J. "The meaning and use of the area under
           a receiver operating characteristic (ROC) curve." Radiology 143(1),
           1982.
    """
    if empty_strategy != "raise":
        guarded = _empty_strategy_guard(y_true, empty_strategy, "roc_auc")
        if not isinstance(guarded, _SentinelOk):
            return guarded
    _validate_inputs(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Precision / recall / F1 / accuracy / TN/FP/FN/TP at a fixed threshold.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Scores.
    threshold : float
        Decision boundary; ``y_score >= threshold`` is the positive prediction.

    Returns
    -------
    dict
        Keys: ``threshold``, ``f1``, ``precision``, ``recall``, ``accuracy``,
        ``fpr`` (false-positive rate), ``fnr`` (false-negative rate),
        ``tn``, ``fp``, ``fn``, ``tp``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> s = np.array([0.1, 0.9, 0.2, 0.8])
    >>> result = metrics_at_threshold(y, s, threshold=0.5)
    >>> result["f1"], result["tp"], result["fp"]
    (1.0, 2, 0)
    >>> result["fpr"], result["fnr"]
    (0.0, 0.0)
    """
    _validate_inputs(y_true, y_score)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    n_pos = tp + fn
    n_neg = tn + fp
    fpr = float(fp) / max(int(n_neg), 1)
    fnr = float(fn) / max(int(n_pos), 1)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(n, 1)),
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def single_class_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int | str]:
    """Operating metrics for all-positive or all-negative slices.

    PR-AUC, ROC-AUC, and max-F1 are not meaningful when one class is absent.
    This helper reports the operating-point quantity that still answers a
    deployment question:

    - all-positive slices: recall at an externally selected threshold
    - all-negative slices: false-positive rate and specificity

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels (must all be 0 or all be 1).
    y_score : np.ndarray, shape (n,)
        Scores.
    threshold : float
        Decision boundary supplied externally (e.g., from a mixed-class slice).

    Returns
    -------
    dict
        Includes ``slice_class`` ∈ {``all_positive``, ``all_negative``} and
        the corresponding operating-point metric (``recall@threshold`` or
        ``fpr@threshold``).

    Raises
    ------
    ValueError
        If the slice is not single-class.
    """
    _validate_inputs(y_true, y_score)
    unique = set(np.unique(np.asarray(y_true)).tolist())
    if len(unique) != 1:
        raise ValueError(f"expected a single-class slice, got labels {sorted(unique)}")

    threshold_metrics = metrics_at_threshold(y_true, y_score, threshold)
    n = int(len(y_true))
    n_positive = int(np.sum(y_true))
    n_negative = n - n_positive
    out: dict[str, float | int | str] = {
        "threshold": float(threshold),
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "tp": int(threshold_metrics["tp"]),
        "fp": int(threshold_metrics["fp"]),
        "tn": int(threshold_metrics["tn"]),
        "fn": int(threshold_metrics["fn"]),
    }
    if n_positive == n:
        out["slice_class"] = "all_positive"
        out["recall@threshold"] = float(threshold_metrics["recall"])
    else:
        out["slice_class"] = "all_negative"
        fpr = float(threshold_metrics["fp"] / max(n_negative, 1))
        out["fpr@threshold"] = fpr
        out["specificity@threshold"] = 1.0 - fpr
    return out


def stratified_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    strata: np.ndarray,
    *,
    with_ci: bool = False,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Recall (TPR) per categorical stratum.

    Generalization of "per-family recall" — useful any time you want to check
    whether a detector's recall holds up uniformly across subgroups.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Scores.
    threshold : float
        Decision boundary.
    strata : np.ndarray, shape (n,)
        Categorical labels (any hashable type; coerced to string for grouping).
        ``None`` / NaN values are coerced to the string ``"unlabeled"``.
    with_ci : bool, optional
        If ``True``, attach a Wilson scoring confidence interval per stratum
        on the recall (binomial proportion). Default ``False``.
    confidence : float, optional
        Two-sided confidence level (default 0.95). Ignored unless
        ``with_ci=True``.

    Returns
    -------
    dict
        ``{stratum_label: {"recall", "n", "tp", "fn"}}``. Strata with zero
        positives are reported with ``recall=NaN`` and ``n=0``. When
        ``with_ci=True`` each stratum dict also contains ``"ci_low"`` and
        ``"ci_high"`` (Wilson scoring CI).

    Raises
    ------
    ValueError
        If ``strata`` length differs from ``y_true`` length.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 1, 0, 0])
    >>> s = np.array([0.9, 0.4, 0.7, 0.2, 0.1])
    >>> strata = np.array(["A", "B", "A", "B", "A"])
    >>> result = stratified_recall(y, s, threshold=0.5, strata=strata)
    >>> sorted(result.keys())
    ['A', 'B']
    >>> result["A"]["recall"], result["A"]["n"]
    (1.0, 2)

    See Also
    --------
    eval_toolkit.metrics.quantile_stratified_pr_auc :
        PR-AUC on a numeric stratifier window (vs categorical strata here).

    References
    ----------
    .. [1] Hardt, M., Price, E., & Srebro, N. "Equality of opportunity in
           supervised learning." NIPS 2016. (Per-group recall is the
           "equal opportunity" fairness criterion.)
    """
    _validate_inputs(y_true, y_score)
    strata_arr = np.asarray(strata)
    if strata_arr.shape != np.asarray(y_true).shape:
        raise ValueError(
            f"strata shape {strata_arr.shape} != y_true shape {np.asarray(y_true).shape}"
        )

    # Coerce to string for groupby; treat None/NaN as "unlabeled". The
    # isinstance(v, float) guard ensures np.isnan only sees floats (incl.
    # numpy.float64), which never raises TypeError — no try/except needed.
    def _coerce(v: object) -> str:
        if v is None:
            return "unlabeled"
        if isinstance(v, float) and np.isnan(v):
            return "unlabeled"
        return str(v)

    strata_str = np.array([_coerce(v) for v in strata_arr])
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    if with_ci and not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    out: dict[str, dict[str, float | int]] = {}
    for stratum_np in np.unique(strata_str):
        stratum = str(stratum_np)  # coerce numpy str to plain Python str
        mask = strata_str == stratum_np
        positives_mask = mask & (y_true_arr == 1)
        n_pos = int(positives_mask.sum())
        if n_pos == 0:
            entry: dict[str, float | int] = {
                "recall": float("nan"),
                "n": 0,
                "tp": 0,
                "fn": 0,
            }
            if with_ci:
                entry["ci_low"] = float("nan")
                entry["ci_high"] = float("nan")
            out[stratum] = entry
            continue
        tp = int((y_pred[positives_mask] == 1).sum())
        entry = {
            "recall": tp / n_pos,
            "n": n_pos,
            "tp": tp,
            "fn": n_pos - tp,
        }
        if with_ci:
            ci_low, ci_high = _wilson_ci(tp, n_pos, confidence=confidence)
            entry["ci_low"] = ci_low
            entry["ci_high"] = ci_high
        out[stratum] = entry
    return out


def _wilson_ci(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson scoring confidence interval for a binomial proportion.

    Closed-form 2-sided CI for ``p̂ = k/n`` at the given confidence level.
    Wilson is preferred over the normal-approximation Wald interval for
    small ``n`` or extreme ``p̂`` (closer to 0 or 1) — the toolkit choice
    matches sklearn / statsmodels defaults for ``method='wilson'``.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    confidence : float, optional
        Two-sided confidence level (default 0.95).

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` Wilson CI bounds.

    References
    ----------
    .. [1] Wilson, E. B. "Probable inference, the law of succession, and
           statistical inference." JASA 22(158), 1927.
    """
    from scipy.stats import norm  # noqa: PLC0415  (scoped optional import)

    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))) / denom
    return float(centre - margin), float(centre + margin)


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error on equal-width probability bins.

    At low base rates, several equal-width bins are mostly empty and the
    estimate becomes dominated by sparse-bin variance.
    :func:`expected_calibration_error_equal_mass` is the audit-recommended
    companion for imbalanced data.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities ∈ [0, 1].
    n_bins : int, optional
        Number of equal-width bins (default 10). Must be ≥ 2.

    Returns
    -------
    float
        ECE ∈ [0, 1]. 0 = perfectly calibrated.

    Raises
    ------
    ValueError
        If ``n_bins < 2`` or input shapes are wrong.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> s = np.array([0.1, 0.2, 0.8, 0.9])
    >>> round(expected_calibration_error(y, s, n_bins=2), 3)
    0.15

    Notes
    -----
    .. math::

        \\mathrm{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{n} |\\mathrm{acc}(B_m) - \\mathrm{conf}(B_m)|

    where :math:`B_m` is the :math:`m`-th bin, :math:`\\mathrm{acc}` is the
    empirical positive rate in the bin, and :math:`\\mathrm{conf}` is the
    mean predicted score.

    References
    ----------
    .. [1] DeGroot, M. H. & Fienberg, S. E. "The comparison and evaluation of
           forecasters." The Statistician 32(1-2), 1983.
    .. [2] Naeini, M. P., Cooper, G., & Hauskrecht, M. "Obtaining well
           calibrated probabilities using Bayesian binning." AAAI 2015.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bin_edges, right=False) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        confidence = float(np.mean(y_score[mask]))
        empirical = float(np.mean(y_true[mask]))
        ece += (mask.sum() / n) * abs(confidence - empirical)
    return float(ece)


def expected_calibration_error_equal_mass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE on equal-mass (quantile) bins.

    With class imbalance, equal-width bins concentrate most data in 1-2 bins.
    Equal-mass binning gives every bin the same number of examples, so each
    bin contributes a comparable amount of evidence to the ECE estimate.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities.
    n_bins : int, optional
        Number of quantile bins (default 10). Must be ≥ 2 and ≤ ``n``.

    Returns
    -------
    float
        ECE ∈ [0, 1].

    Raises
    ------
    ValueError
        If ``n_bins < 2`` or ``n < n_bins``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = (y + rng.normal(0, 0.5, size=200)).clip(0, 1)
    >>> 0.0 <= expected_calibration_error_equal_mass(y, s) <= 1.0
    True

    Notes
    -----
    Quantile-binned ECE replaces equal-width bins with bin edges placed at
    score quantiles, so each bin holds roughly :math:`n / M` examples. This
    eliminates the sparse-bin variance that dominates equal-width ECE under
    class imbalance.

    References
    ----------
    .. [1] Nixon, J., et al. "Measuring calibration in deep learning."
           CVPR Workshops 2019. (Discussion of equal-mass binning rationale.)
    .. [2] DeGroot, M. H. & Fienberg, S. E. "The comparison and evaluation of
           forecasters." The Statistician 32(1-2), 1983.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"n={n} smaller than n_bins={n_bins}; cannot form quantile bins")
    order = np.argsort(np.asarray(y_score), kind="stable")
    sorted_score = np.asarray(y_score)[order]
    sorted_true = np.asarray(y_true)[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        confidence = float(np.mean(sorted_score[lo:hi]))
        empirical = float(np.mean(sorted_true[lo:hi]))
        ece += ((hi - lo) / n) * abs(confidence - empirical)
    return float(ece)


def expected_calibration_error_l2(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    r"""Equal-mass L2 ECE — root mean squared bin-level miscalibration.

    .. math::

        \mathrm{ECE}_2 = \sqrt{\sum_k \frac{n_k}{n} (\bar p_k - \bar y_k)^2}

    Companion to :func:`expected_calibration_error_equal_mass` (which uses L1
    on the absolute difference). The L2 form is what Kumar 2019 [#kumar]_
    proves admits a clean closed-form bias correction —
    see :func:`expected_calibration_error_l2_debiased`.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities ∈ [0, 1].
    n_bins : int, optional
        Number of equal-mass bins (default 10).

    Returns
    -------
    float
        L2 ECE in [0, 1]. 0 = perfectly calibrated.

    Raises
    ------
    ValueError
        If ``y_true``/``y_score`` shape, dtype, or value-range checks fail
        (re-raised from input validators); if ``n_bins < 2``; or if
        ``n < n_bins`` (cannot form quantile bins).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, size=500)
    >>> s = (y + rng.normal(0, 0.3, 500)).clip(0, 1)
    >>> 0.0 <= expected_calibration_error_l2(y, s) <= 1.0
    True

    References
    ----------
    .. [#kumar] Kumar, A., Liang, P. S., & Ma, T. "Verified uncertainty
       calibration." NeurIPS 2019. arXiv:1909.10155.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"n={n} smaller than n_bins={n_bins}; cannot form quantile bins")
    order = np.argsort(np.asarray(y_score), kind="stable")
    sorted_score = np.asarray(y_score)[order]
    sorted_true = np.asarray(y_true)[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    sq_err = 0.0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        n_k = hi - lo
        p_bar_k = float(np.mean(sorted_score[lo:hi]))
        y_bar_k = float(np.mean(sorted_true[lo:hi]))
        sq_err += (n_k / n) * (p_bar_k - y_bar_k) ** 2
    return float(np.sqrt(sq_err))


def expected_calibration_error_l2_debiased(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    r"""Bias-corrected L2 ECE per Kumar 2019 [#kumar]_ §3.3.

    The plug-in L2 ECE estimator is positively biased: random
    miscalibration "noise" inflates the squared error term by
    :math:`\bar p_k (1 - \bar p_k) / n_k` per bin even when the true
    calibration error is zero. Kumar 2019 proves the bias is
    closed-form-removable for the L2 (squared) ECE:

    .. math::

        \widehat{\mathrm{ECE}}_2^{\,2,\, \text{deb}} =
            \widehat{\mathrm{ECE}}_2^{\,2} -
            \sum_k \frac{n_k}{n} \cdot \frac{\bar p_k (1 - \bar p_k)}{n_k}

    The result is then square-rooted (clipped at 0 if the bias correction
    drives the estimate negative on well-calibrated data).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities ∈ [0, 1].
    n_bins : int, optional
        Number of equal-mass bins (default 10).

    Returns
    -------
    float
        Debiased L2 ECE in [0, 1]. The bias correction can drive the
        estimate to exactly 0 on well-calibrated data; this is correct
        Kumar 2019 behavior, not a bug.

    Raises
    ------
    ValueError
        If ``y_true``/``y_score`` shape, dtype, or value-range checks fail
        (re-raised from input validators); if ``n_bins < 2``; or if
        ``n < n_bins`` (cannot form quantile bins).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Perfectly calibrated synthetic data: y ~ Bernoulli(s)
    >>> n = 5000
    >>> s = rng.uniform(0, 1, size=n)
    >>> y = (rng.uniform(0, 1, size=n) < s).astype(int)
    >>> # Plug-in ECE has positive bias; debiased should be ~0.
    >>> plug_in = expected_calibration_error_l2(y, s)
    >>> debiased = expected_calibration_error_l2_debiased(y, s)
    >>> debiased <= plug_in + 1e-9
    True

    See Also
    --------
    eval_toolkit.metrics.expected_calibration_error_l2 :
        Plug-in L2 ECE without bias correction.
    eval_toolkit.metrics.expected_calibration_error_equal_mass :
        L1 (absolute-deviation) ECE — no closed-form bias correction.

    References
    ----------
    .. [#kumar] Kumar, A., Liang, P. S., & Ma, T. "Verified uncertainty
       calibration." NeurIPS 2019. arXiv:1909.10155. (§3.3 closed-form
       debiased L2 ECE.)
    .. [1] Roelofs, R., Cain, N., Shlens, J., & Mozer, M. "Mitigating bias
       in calibration error estimation." AISTATS 2022.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"n={n} smaller than n_bins={n_bins}; cannot form quantile bins")
    order = np.argsort(np.asarray(y_score), kind="stable")
    sorted_score = np.asarray(y_score)[order]
    sorted_true = np.asarray(y_true)[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    sq_err = 0.0
    bias = 0.0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        n_k = hi - lo
        p_bar_k = float(np.mean(sorted_score[lo:hi]))
        y_bar_k = float(np.mean(sorted_true[lo:hi]))
        sq_err += (n_k / n) * (p_bar_k - y_bar_k) ** 2
        # Kumar 2019 bias correction: per-bin variance of a Bernoulli with
        # mean p_bar_k (≈ true bin mean), divided by n_k samples.
        bias += (n_k / n) * (p_bar_k * (1.0 - p_bar_k) / n_k)
    debiased_sq = max(sq_err - bias, 0.0)
    return float(np.sqrt(debiased_sq))


def expected_calibration_error_debiased(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    *,
    n_sweep: int = 200,
    seed: int = 42,
) -> float:
    r"""Bias-corrected L1 ECE via simulated-H0 Monte-Carlo (Roelofs 2022 spirit).

    The plug-in L1 ECE has positive O(M/n) bias — random miscalibration
    "noise" inflates the absolute-deviation term per bin even on
    perfectly-calibrated data. Unlike the L2 form (where Kumar 2019 gives
    a closed-form bias correction in
    :func:`expected_calibration_error_l2_debiased`), L1 has no closed-form
    correction and must be estimated.

    This estimator uses the **simulated-H0 bootstrap**: resample synthetic
    labels :math:`y_i^* \\sim \\mathrm{Bernoulli}(s_i)` (assuming the
    scores ARE the true probabilities), compute the plug-in L1 ECE on each
    resample, and average to estimate the bias under the null hypothesis
    of perfect calibration. The debiased estimate is then

    .. math::

        \widehat{\mathrm{ECE}}^{\,\text{deb}}_1 = \max\!\big(0,\,
        \widehat{\mathrm{ECE}}_1 - \widehat{\mathrm{bias}}_1\big).

    Same conceptual move as Roelofs 2022's ECE_SWEEP estimator (which
    uses cross-validation rather than simulated-H0), trading a bit of
    fidelity to the literal SWEEP construction for a substantially
    simpler implementation.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities ∈ [0, 1].
    n_bins : int, optional
        Number of equal-mass bins (default 10).
    n_sweep : int, optional
        Number of simulated-H0 resamples for the bias estimate. Default
        ``200``. Larger → tighter bias estimate, more compute.
    seed : int, optional
        RNG seed for the simulated H0 resamples.

    Returns
    -------
    float
        Debiased L1 ECE in [0, 1]. The bias correction can drive the
        estimate to exactly 0 on well-calibrated data.

    Raises
    ------
    ValueError
        Same conditions as :func:`expected_calibration_error_equal_mass`,
        plus ``n_sweep < 10``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Perfectly calibrated synthetic data
    >>> n = 2000
    >>> s = rng.uniform(0, 1, size=n)
    >>> y = (rng.uniform(0, 1, size=n) < s).astype(int)
    >>> plug_in = expected_calibration_error_equal_mass(y, s)
    >>> debiased = expected_calibration_error_debiased(y, s, n_sweep=100, seed=0)
    >>> debiased <= plug_in + 1e-9
    True

    See Also
    --------
    eval_toolkit.metrics.expected_calibration_error_l2_debiased :
        Closed-form (Kumar 2019) debiased L2 ECE — no Monte-Carlo, faster.
    eval_toolkit.metrics.expected_calibration_error_equal_mass :
        Plug-in L1 ECE without bias correction.

    Notes
    -----
    Compute cost is :math:`O(n_{\\text{sweep}} \\cdot n)` for the
    Monte-Carlo bias estimate. With ``n_sweep=200`` and ``n=1000``, this
    is ~200K Bernoulli draws + ECE computations — fast (~0.5s).

    References
    ----------
    .. [1] Roelofs, R., Cain, N., Shlens, J., & Mozer, M. "Mitigating bias
           in calibration error estimation." AISTATS 2022. (ECE_SWEEP via
           cross-validation; this implementation uses simulated H0 instead.)
    .. [2] Kumar, A., Liang, P. S., & Ma, T. "Verified uncertainty
           calibration." NeurIPS 2019. arXiv:1909.10155.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    if n_sweep < 10:
        raise ValueError(f"n_sweep must be ≥ 10, got {n_sweep}")
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"n={n} smaller than n_bins={n_bins}; cannot form quantile bins")

    plug_in = expected_calibration_error_equal_mass(y_true, y_score, n_bins=n_bins)

    # Simulated-H0 bias estimate: under H0 (scores ARE true probabilities),
    # the expected plug-in ECE on Bernoulli(s) labels gives the bias floor.
    s_arr = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    bias_estimates = np.empty(n_sweep, dtype=np.float64)
    for i in range(n_sweep):
        y_synth = (rng.uniform(0, 1, size=n) < s_arr).astype(int)
        bias_estimates[i] = expected_calibration_error_equal_mass(y_synth, s_arr, n_bins=n_bins)
    bias = float(bias_estimates.mean())
    return float(max(0.0, plug_in - bias))


@overload
def brier_score(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["raise"] = ...
) -> float: ...
@overload
def brier_score(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["return_none"]
) -> float | None: ...
@overload
def brier_score(
    y_true: np.ndarray, y_score: np.ndarray, *, empty_strategy: Literal["skipped_metric"]
) -> float | dict[str, object]: ...
def brier_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    empty_strategy: EmptyStrategy = "raise",
) -> float | None | dict[str, object]:
    r"""Brier score (mean squared error between predicted probabilities and labels).

    Strictly proper scoring rule: minimized in expectation iff the forecast
    equals the true conditional probability. Captures both calibration and
    resolution that ECE alone misses; see :func:`brier_decomposition` for
    the Murphy 1973 partition.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities in ``[0, 1]``.
    empty_strategy : {"raise", "return_none", "skipped_metric"}, optional
        Empty-input handling — see :func:`pr_auc`. Note: unlike PR/ROC-AUC,
        ``brier_score`` is well-defined on single-class slices, so the guard
        only short-circuits on ``n=0``. Default ``"raise"`` preserves pre-v0.13
        behavior.

    Returns
    -------
    float | None | dict[str, object]
        Brier score in ``[0, 1]``. ``0`` is perfect; ``0.25`` is the
        constant-prevalence forecast; ``1`` is maximally wrong. With
        non-default ``empty_strategy``, the return type widens — see overload
        variants.

    Raises
    ------
    ValueError
        On shape mismatch, NaN/Inf scores, non-binary labels, scores outside
        ``[0, 1]``, or empty input AND ``empty_strategy="raise"``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> s = np.array([0.1, 0.9, 0.2, 0.8])
    >>> round(brier_score(y, s), 4)
    0.025

    Notes
    -----
    .. math:: \mathrm{BS} = \frac{1}{n} \sum_i (p_i - y_i)^2

    See Also
    --------
    eval_toolkit.metrics.brier_decomposition :
        Murphy 1973 reliability/resolution/uncertainty partition.
    eval_toolkit.metrics.expected_calibration_error :
        Calibration-only metric; ECE = 0 does not imply BS = 0.

    References
    ----------
    .. [1] Brier, G. W. "Verification of forecasts expressed in terms of
           probability." Monthly Weather Review 78(1), 1950.
    """
    if empty_strategy != "raise":
        guarded = _empty_strategy_guard(
            y_true, empty_strategy, "brier_score", check_single_class=False
        )
        if not isinstance(guarded, _SentinelOk):
            return guarded
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    r"""Murphy 1973 [#murphy]_ decomposition of the Brier score.

    Partitions the Brier score into three additive components:

    .. math:: \mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}

    where:

    - **REL** (reliability, lower is better) is the expected squared
      distance between predicted probability and the empirical positive
      rate within each bin: :math:`\sum_k \frac{n_k}{n} (\bar p_k - \bar y_k)^2`.
    - **RES** (resolution, higher is better) is the variance of bin
      empirical rates around the marginal: :math:`\sum_k \frac{n_k}{n} (\bar y_k - \bar y)^2`.
    - **UNC** (uncertainty, irreducible) is the marginal variance:
      :math:`\bar y (1 - \bar y)`.

    The decomposition is exact in expectation when bin assignments are
    independent of the labels; for finite samples the partition holds up to
    binning approximation.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Calibrated probabilities ``∈ [0, 1]``.
    n_bins : int, optional
        Number of equal-mass bins (default 10). Must be ≥ 2 and ≤ ``n``.

    Returns
    -------
    dict
        ``{"brier", "reliability", "resolution", "uncertainty"}``. The
        identity ``brier ≈ reliability - resolution + uncertainty`` holds
        to ``≈ 1e-9`` modulo binning approximation.

    Raises
    ------
    ValueError
        Same conditions as :func:`brier_score`, plus invalid ``n_bins``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, size=500)
    >>> s = (y + rng.normal(0, 0.3, size=500)).clip(0, 1)
    >>> result = brier_decomposition(y, s, n_bins=10)
    >>> set(result.keys()) == {"brier", "reliability", "resolution", "uncertainty"}
    True
    >>> # The decomposition identity holds to within binning approximation:
    >>> approx = result["reliability"] - result["resolution"] + result["uncertainty"]
    >>> abs(result["brier"] - approx) < 0.05
    True

    Notes
    -----
    Equal-mass binning is used (vs equal-width) following Nixon et al.
    2019's recommendation; under class imbalance the equal-width variant
    concentrates most mass in 1-2 bins and the resolution term collapses.

    References
    ----------
    .. [#murphy] Murphy, A. H. "A new vector partition of the probability
       score." Journal of Applied Meteorology 12, 1973.
    """
    _validate_inputs(y_true, y_score)
    _validate_calibrated_score(y_score)
    if n_bins < 2:
        raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"n={n} smaller than n_bins={n_bins}; cannot form quantile bins")

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    bs = float(np.mean((p - y) ** 2))
    y_bar = float(np.mean(y))
    uncertainty = y_bar * (1.0 - y_bar)

    # Equal-mass binning per Nixon 2019.
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)

    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        n_k = hi - lo
        p_bar_k = float(np.mean(sorted_p[lo:hi]))
        y_bar_k = float(np.mean(sorted_y[lo:hi]))
        weight = n_k / n
        reliability += weight * (p_bar_k - y_bar_k) ** 2
        resolution += weight * (y_bar_k - y_bar) ** 2
    return {
        "brier": bs,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def quantile_stratified_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    stratifier: np.ndarray,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> dict[str, float]:
    """PR-AUC on the central [q_low, q_high] range of any 1-D stratifier.

    Useful when you suspect a metric has accidentally learned a confounder
    correlated with the stratifier (e.g., text length, time-of-day, document
    size). Filter the tails (where one class typically dominates) and recompute
    PR-AUC on the central window where both classes are represented.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Scores.
    stratifier : np.ndarray, shape (n,)
        Numeric values used to define the quantile window.
    q_low, q_high : float, optional
        Quantile bounds for the kept window (default 0.25 to 0.75).

    Returns
    -------
    dict
        ``{"pr_auc", "n", "n_positive", "n_negative", "stratifier_low",
        "stratifier_high", "q_low", "q_high"}``.

    Raises
    ------
    ValueError
        If shapes mismatch, quantile bounds invalid, or the kept window has
        too few positives/negatives (< 10) for a reliable PR-AUC.

    Notes
    -----
    When ``stratifier`` is the score itself, this method becomes a partial
    AUC over a score-quantile window — the same construct as McClish 1989's
    "partial area under the ROC curve" (in PR space rather than ROC).

    See Also
    --------
    eval_toolkit.metrics.stratified_recall :
        Categorical-stratum recall.

    References
    ----------
    .. [1] McClish, D. K. "Analyzing a portion of the ROC curve."
           Medical Decision Making 9(3), 1989. (Partial-AUC framework.)
    """
    _validate_inputs(y_true, y_score)
    strat_arr = np.asarray(stratifier)
    if strat_arr.shape != np.asarray(y_true).shape:
        raise ValueError(
            f"stratifier shape {strat_arr.shape} != y_true shape {np.asarray(y_true).shape}"
        )
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError(f"need 0 ≤ q_low < q_high ≤ 1, got {q_low}, {q_high}")
    lo, hi = np.quantile(strat_arr, [q_low, q_high])
    mask = (strat_arr >= lo) & (strat_arr <= hi)
    if not mask.any():
        raise ValueError("no rows in stratifier window")
    sub_y = np.asarray(y_true)[mask]
    sub_s = np.asarray(y_score)[mask]
    n_pos = int((sub_y == 1).sum())
    n_neg = int((sub_y == 0).sum())
    if n_pos < 10 or n_neg < 10:
        raise ValueError(f"stratified subset too imbalanced for PR-AUC: pos={n_pos}, neg={n_neg}")
    return {
        "pr_auc": pr_auc(sub_y, sub_s),
        "n": int(mask.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "stratifier_low": float(lo),
        "stratifier_high": float(hi),
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


def quantile_stratified_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    stratifier: np.ndarray,
    *,
    q_low: float = 0.25,
    q_high: float = 0.75,
    gap_threshold: float = 0.05,
) -> dict[str, float | bool]:
    """Full vs trimmed PR-AUC report with a gap-flag (SDD reporting convention).

    Wraps :func:`pr_auc` (full) + :func:`quantile_stratified_pr_auc` (trimmed)
    into the four-field report shape used by ``prompt-injection-sdd`` style
    reports: a *full* PR-AUC over all rows, a *trimmed* PR-AUC over the
    central ``[q_low, q_high]`` quantile window of ``stratifier``, the
    arithmetic ``gap = full - trimmed``, and a boolean ``gap_flag`` set when
    ``gap > gap_threshold``.

    A positive gap means the model exploits a confounder correlated with
    ``stratifier`` (e.g., text length, time-of-day, document source); the
    `gap_flag` surfaces this as a single auditable bit alongside the
    headline metric — useful for at-a-glance reviewer-friendly tables.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score : np.ndarray, shape (n,)
        Scores.
    stratifier : np.ndarray, shape (n,)
        Numeric values defining the quantile window.
    q_low, q_high : float, optional
        Quantile bounds for the trimmed window. Default ``(0.25, 0.75)``.
    gap_threshold : float, optional
        Threshold above which ``gap_flag`` is True. Default ``0.05``
        (SDD convention).

    Returns
    -------
    dict
        Keys: ``full``, ``trimmed``, ``gap``, ``gap_flag``.

    Raises
    ------
    ValueError
        Forwarded from :func:`pr_auc` and :func:`quantile_stratified_pr_auc`.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit import quantile_stratified_report
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> y = rng.binomial(1, 0.3, size=n)
    >>> s = np.clip(y * 0.6 + rng.normal(0, 0.3, size=n), 0, 1)
    >>> lengths = rng.integers(10, 200, size=n)
    >>> report = quantile_stratified_report(y, s, lengths)
    >>> bool(report["full"] >= report["trimmed"] - 0.5)  # sanity
    True
    >>> isinstance(report["gap_flag"], bool | np.bool_)
    True

    See Also
    --------
    eval_toolkit.metrics.quantile_stratified_pr_auc :
        The trimmed-PR-AUC primitive this report wraps.
    eval_toolkit.metrics.pr_auc :
        The full-PR-AUC primitive.

    Notes
    -----
    See ``docs/methodology/length_stratification.md`` for the methodology
    motivation (McClish 1989 partial-AUC framing; when stratified vs.
    global PR-AUC).
    """
    full = float(pr_auc(y_true, y_score))
    trimmed_block = quantile_stratified_pr_auc(
        y_true, y_score, stratifier, q_low=q_low, q_high=q_high
    )
    trimmed = float(trimmed_block["pr_auc"])
    gap = full - trimmed
    return {
        "full": full,
        "trimmed": trimmed,
        "gap": gap,
        "gap_flag": gap > gap_threshold,
    }


def precision_at_prior(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    assumed_prior: float,
) -> dict[str, float]:
    r"""Project precision under a different positive-class prior.

    Operating points tuned at one prevalence are systematically too permissive
    or too restrictive at deployment-time prevalence. This applies Bayes' rule
    to extrapolate precision under a different positive prior, holding TPR
    and FPR fixed.

    Parameters
    ----------
    y_true, y_score : np.ndarray
        Eval-set labels and scores.
    threshold : float
        Decision boundary on ``y_score``.
    assumed_prior : float
        Hypothesized deployment prior π ∈ (0, 1).

    Returns
    -------
    dict
        ``{tpr, fpr, eval_prior, assumed_prior, precision_at_eval_prior,
        precision_at_assumed_prior, threshold}``.

    Raises
    ------
    ValueError
        If ``assumed_prior`` is outside (0, 1) or the eval set lacks both
        classes (cannot estimate both TPR and FPR).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = y + rng.normal(0, 0.3, size=200)
    >>> result = precision_at_prior(y, s, threshold=0.5, assumed_prior=0.01)
    >>> 0.0 <= result["precision_at_assumed_prior"] <= 1.0
    True

    Notes
    -----
    Bayes' rule with TPR and FPR fixed:

    .. math::

        P(y=1 | \hat{y}=1) = \frac{\pi \cdot \mathrm{TPR}}{\pi \cdot \mathrm{TPR} + (1-\pi) \cdot \mathrm{FPR}}

    Caveat: assumes the *class-conditional* score distributions on the eval
    set match deployment. If the input distribution shifts, TPR/FPR move and
    this projection no longer holds.

    References
    ----------
    .. [1] Saerens, M., Latinne, P., & Decaestecker, C. "Adjusting the outputs
           of a classifier to new a priori probabilities: A simple procedure."
           Neural Computation 14(1), 2002.
    """
    _validate_inputs(y_true, y_score)
    if not 0.0 < assumed_prior < 1.0:
        raise ValueError(f"assumed_prior must be in (0, 1), got {assumed_prior}")
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    n_pos = int(y_true_arr.sum())
    n_neg = int(len(y_true_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"need both classes for precision_at_prior; got n_pos={n_pos}, n_neg={n_neg}"
        )
    tpr = float(((y_pred == 1) & (y_true_arr == 1)).sum() / n_pos)
    fpr = float(((y_pred == 1) & (y_true_arr == 0)).sum() / n_neg)
    eval_prior = n_pos / (n_pos + n_neg)
    eval_precision = (eval_prior * tpr) / max(eval_prior * tpr + (1.0 - eval_prior) * fpr, 1e-12)
    assumed_precision = (assumed_prior * tpr) / max(
        assumed_prior * tpr + (1.0 - assumed_prior) * fpr, 1e-12
    )
    return {
        "tpr": tpr,
        "fpr": fpr,
        "eval_prior": float(eval_prior),
        "assumed_prior": float(assumed_prior),
        "precision_at_eval_prior": float(eval_precision),
        "precision_at_assumed_prior": float(assumed_precision),
        "threshold": float(threshold),
    }


DEFAULT_ASSUMED_PRIORS: Final[tuple[float, ...]] = (0.001, 0.01, 0.05)


def score_distribution_summary(scores: np.ndarray) -> dict[str, float | int]:
    """Threshold-free score-distribution summary.

    Reports ``mean / median / std / q25 / q75 / n`` so seed-stability of model
    confidence on a slice is auditable without committing to a deployment
    threshold. Especially useful for single-class slices where PR-AUC and
    threshold-dependent metrics are degenerate.

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
        1-D array of predicted probabilities or scores.

    Returns
    -------
    dict
        ``{"n", "mean", "median", "std", "q25", "q75"}``.

    Raises
    ------
    ValueError
        If ``scores`` is empty or contains NaN/inf.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> scores = rng.uniform(0, 1, size=100)
    >>> result = score_distribution_summary(scores)
    >>> result["n"]
    100
    """
    arr = np.asarray(scores).astype(float).ravel()
    if arr.size == 0:
        raise ValueError("scores is empty")
    if not np.isfinite(arr).all():
        raise ValueError("scores contains NaN or inf")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def headline_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strata: np.ndarray | None = None,
    assumed_priors: tuple[float, ...] = DEFAULT_ASSUMED_PRIORS,
) -> dict[str, object]:
    """Bundle PR-AUC + ROC-AUC + 3 operating-point F1s + per-stratum recall (if provided).

    Also reports both equal-width and equal-mass ECE and
    ``precision_at_prior`` at the max-F1 threshold across ``assumed_priors``
    for prior-shift sensitivity.

    Parameters
    ----------
    y_true, y_score : np.ndarray
        Labels and scores.
    strata : np.ndarray or None, optional
        Categorical groupings for stratified recall. If None, that section is
        omitted.
    assumed_priors : tuple of float, optional
        Priors at which to project precision (default ``(0.001, 0.01, 0.05)``).

    Returns
    -------
    dict
        Headline bundle with keys: ``n``, ``n_positive``, ``pr_auc``,
        ``roc_auc``, ``operating_points``, ``ece_equal_width``,
        ``ece_equal_mass``, ``ece`` (alias for equal-width), and optionally
        ``per_stratum_recall_at_max_f1``, ``precision_at_prior``.

    Notes
    -----
    For single-class slices, PR-AUC / ROC-AUC / threshold-selected metrics are
    set to ``None`` and a ``metric_note`` field explains why; use
    :func:`single_class_threshold_metrics` for those slices instead.
    """
    y_true_arr = np.asarray(y_true)
    unique_labels = set(np.unique(y_true_arr).tolist())
    is_single_class = len(unique_labels) == 1
    out: dict[str, object] = {
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }
    if is_single_class:
        only_label = int(next(iter(unique_labels)))
        out["pr_auc"] = None
        out["roc_auc"] = None
        out["single_class_label"] = only_label
        out["metric_note"] = (
            "single-class slice; PR-AUC/ROC-AUC/threshold-selected F1 are not meaningful. "
            "Use externally selected threshold operating metrics instead."
        )
    else:
        out["pr_auc"] = pr_auc(y_true, y_score)
        out["roc_auc"] = roc_auc(y_true, y_score)
    # Late import: thresholds.py imports ThresholdResult / metrics_at_threshold
    # from this module, so module-level imports would be circular.
    from eval_toolkit.thresholds import MaxF1Selector, TargetRecallSelector

    operating_points: dict[str, Mapping[str, object]] = {}
    selectors_for_headline: list[ThresholdSelector] = [
        MaxF1Selector(),
        TargetRecallSelector(0.90),
        TargetRecallSelector(0.95),
    ]
    for selector in selectors_for_headline:
        label = selector.criterion
        if is_single_class:
            operating_points[label] = {
                "skipped": "single-class slice; threshold must be selected on a mixed-class slice"
            }
            continue
        try:
            tr = selector.select(y_true, y_score)
            operating_points[label] = metrics_at_threshold(y_true, y_score, tr.threshold)
        except RuntimeError as exc:
            operating_points[label] = {"error": str(exc)}
    out["operating_points"] = operating_points

    # Stratified recall reported at max-F1 threshold (when strata provided)
    if strata is not None:
        if is_single_class:
            out["per_stratum_recall_at_max_f1"] = {
                "skipped": "single-class slice; threshold selected on mixed-class slice required"
            }
        else:
            try:
                mf1_thresh = MaxF1Selector().select(y_true, y_score).threshold
                out["per_stratum_recall_at_max_f1"] = stratified_recall(
                    y_true, y_score, mf1_thresh, strata
                )
            except RuntimeError as exc:
                out["per_stratum_recall_at_max_f1"] = {"error": str(exc)}

    out["ece_equal_width"] = expected_calibration_error(y_true, y_score)
    try:
        out["ece_equal_mass"] = expected_calibration_error_equal_mass(y_true, y_score)
    except ValueError as exc:
        out["ece_equal_mass"] = float("nan")
        out["ece_equal_mass_error"] = str(exc)
    out["ece"] = out["ece_equal_width"]

    if is_single_class:
        out["precision_at_prior"] = {
            "skipped": "single-class slice cannot estimate both TPR and FPR"
        }
        return out

    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    if n_pos > 0 and n_neg > 0:
        try:
            mf1_thresh = MaxF1Selector().select(y_true, y_score).threshold
            out["precision_at_prior"] = {
                f"{p:g}": precision_at_prior(y_true, y_score, mf1_thresh, p) for p in assumed_priors
            }
        except RuntimeError as exc:
            out["precision_at_prior"] = {"error": str(exc)}
    return out


def _validate_inputs(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Common input validation. Fail-fast diagnostic errors."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(f"y_true shape {y_true_arr.shape} != y_score shape {y_score_arr.shape}")
    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true_arr.shape}")
    if len(y_true_arr) == 0:
        raise ValueError("y_true is empty")
    unique = set(np.unique(y_true_arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must be binary (0/1), got values {unique}")
    # NaN/Inf guard — silent ranking distortions if scores carry non-finite
    # values; harmonizes with score_distribution_summary's own guard.
    if not np.isfinite(y_score_arr).all():
        raise ValueError("y_score contains NaN or inf")


def _validate_calibrated_score(y_score: np.ndarray, name: str = "y_score") -> None:
    """Probability-range validation for calibration-aware metrics.

    Calibration metrics (ECE variants) are only meaningful when ``y_score``
    is in ``[0, 1]``. Raw logits silently produce a meaningless ECE; this
    guard fails loudly with a diagnostic.
    """
    arr = np.asarray(y_score)
    if arr.size == 0:
        return  # _validate_inputs catches empty arrays
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"{name} must be in [0, 1] for calibration metrics; got "
            f"range [{float(arr.min()):.4g}, {float(arr.max()):.4g}]. "
            "If you have logits, apply softmax/sigmoid first."
        )
