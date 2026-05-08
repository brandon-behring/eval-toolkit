"""Binary classification metrics: PR-AUC, ROC-AUC, F1 at threshold, ECE, prior-shift projection.

Wraps sklearn references with input validation, single-class slice handling,
and threshold-free score-distribution summaries. All functions accept raw
numpy arrays (binary y_true in {0, 1}, real-valued y_score) and return
JSON-serializable values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "DEFAULT_ASSUMED_PRIORS",
    "OperatingPoint",
    "ThresholdResult",
    "expected_calibration_error",
    "expected_calibration_error_equal_mass",
    "headline_metrics",
    "metrics_at_threshold",
    "pr_auc",
    "precision_at_prior",
    "quantile_stratified_pr_auc",
    "roc_auc",
    "score_distribution_summary",
    "select_threshold",
    "single_class_threshold_metrics",
    "stratified_recall",
]

OperatingPoint = Literal["max_f1", "recall_0.90", "recall_0.95"]


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


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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

    Returns
    -------
    float
        PR-AUC ∈ [prevalence, 1]. Wraps ``sklearn.metrics.average_precision_score``.

    Raises
    ------
    ValueError
        If shapes mismatch, dimensions are wrong, the input is empty, or
        labels are not binary.

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

    References
    ----------
    .. [1] Davis, J. & Goadrich, M. "The relationship between precision-recall
           and ROC curves." ICML 2006.
    .. [2] Saito, T. & Rehmsmeier, M. "The precision-recall plot is more
           informative than the ROC plot when evaluating binary classifiers
           on imbalanced datasets." PLOS ONE 10(3), 2015.
    """
    _validate_inputs(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Real-valued scores; higher means more positive.

    Returns
    -------
    float
        ROC-AUC ∈ [0, 1]. 0.5 = random; 1.0 = perfect ranking.

    Raises
    ------
    ValueError
        If shapes mismatch, dimensions are wrong, or labels are not binary.

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

    References
    ----------
    .. [1] Hanley, J. A. & McNeil, B. J. "The meaning and use of the area under
           a receiver operating characteristic (ROC) curve." Radiology 143(1),
           1982.
    """
    _validate_inputs(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: OperatingPoint = "max_f1",
) -> ThresholdResult:
    """Pick a threshold per the named operating-point rule.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Real-valued scores.
    criterion : {"max_f1", "recall_0.90", "recall_0.95"}, optional
        - ``max_f1``: argmax F1 over the PR-curve thresholds.
        - ``recall_0.90``: smallest threshold achieving recall ≥ 0.90 (best precision under that).
        - ``recall_0.95``: smallest threshold achieving recall ≥ 0.95.

    Returns
    -------
    ThresholdResult
        Frozen dataclass with ``threshold``, ``f1``, ``precision``, ``recall``, ``criterion``.

    Raises
    ------
    ValueError
        If ``criterion`` is not recognized.
    RuntimeError
        If ``y_score`` is constant (no PR-curve thresholds) or no threshold
        achieves the recall target.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    >>> tr = select_threshold(y, s, criterion="max_f1")
    >>> tr.f1
    1.0
    """
    _validate_inputs(y_true, y_score)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # `precision_recall_curve` returns N+1 precision/recall but N thresholds.
    precisions, recalls = precisions[:-1], recalls[:-1]
    if len(thresholds) == 0:
        raise RuntimeError("PR curve has no thresholds — y_score may be constant")

    if criterion == "max_f1":
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
        idx = int(np.argmax(f1s))
    elif criterion in ("recall_0.90", "recall_0.95"):
        target = float(criterion.split("_")[1])
        # smallest threshold (most permissive on positives) achieving recall ≥ target;
        # PR-curve thresholds are in increasing order, recall decreases with threshold.
        eligible = np.where(recalls >= target)[0]
        if len(eligible) == 0:
            raise RuntimeError(
                f"No threshold achieves recall ≥ {target}; max recall = {recalls.max():.3f}"
            )
        idx = int(eligible[-1])
    else:
        raise ValueError(f"unknown criterion: {criterion!r}")

    return ThresholdResult(
        threshold=float(thresholds[idx]),
        f1=float(2 * precisions[idx] * recalls[idx] / max(precisions[idx] + recalls[idx], 1e-12)),
        precision=float(precisions[idx]),
        recall=float(recalls[idx]),
        criterion=criterion,
    )


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
        ``tn``, ``fp``, ``fn``, ``tp``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> s = np.array([0.1, 0.9, 0.2, 0.8])
    >>> result = metrics_at_threshold(y, s, threshold=0.5)
    >>> result["f1"], result["tp"], result["fp"]
    (1.0, 2, 0)
    """
    _validate_inputs(y_true, y_score)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(n, 1)),
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

    Returns
    -------
    dict
        ``{stratum_label: {"recall", "n", "tp", "fn"}}``. Strata with zero
        positives are reported with ``recall=NaN`` and ``n=0``.

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
    """
    _validate_inputs(y_true, y_score)
    strata_arr = np.asarray(strata)
    if strata_arr.shape != np.asarray(y_true).shape:
        raise ValueError(
            f"strata shape {strata_arr.shape} != y_true shape {np.asarray(y_true).shape}"
        )

    # Coerce to string for groupby; treat None/NaN as "unlabeled".
    def _coerce(v: object) -> str:
        if v is None:
            return "unlabeled"
        try:
            if isinstance(v, float) and np.isnan(v):
                return "unlabeled"
        except TypeError:
            pass
        return str(v)

    strata_str = np.array([_coerce(v) for v in strata_arr])
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    out: dict[str, dict[str, float | int]] = {}
    for stratum_np in np.unique(strata_str):
        stratum = str(stratum_np)  # coerce numpy str to plain Python str
        mask = strata_str == stratum_np
        positives_mask = mask & (y_true_arr == 1)
        n_pos = int(positives_mask.sum())
        if n_pos == 0:
            out[stratum] = {"recall": float("nan"), "n": 0, "tp": 0, "fn": 0}
            continue
        tp = int((y_pred[positives_mask] == 1).sum())
        out[stratum] = {"recall": tp / n_pos, "n": n_pos, "tp": tp, "fn": n_pos - tp}
    return out


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


DEFAULT_ASSUMED_PRIORS: tuple[float, ...] = (0.001, 0.01, 0.05)


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
    operating_points: dict[str, Mapping[str, object]] = {}
    criteria: tuple[OperatingPoint, ...] = ("max_f1", "recall_0.90", "recall_0.95")
    for crit in criteria:
        if is_single_class:
            operating_points[crit] = {
                "skipped": "single-class slice; threshold must be selected on a mixed-class slice"
            }
            continue
        try:
            tr = select_threshold(y_true, y_score, criterion=crit)
            operating_points[crit] = metrics_at_threshold(y_true, y_score, tr.threshold)
        except RuntimeError as exc:
            operating_points[crit] = {"error": str(exc)}
    out["operating_points"] = operating_points

    # Stratified recall reported at max-F1 threshold (when strata provided)
    if strata is not None:
        if is_single_class:
            out["per_stratum_recall_at_max_f1"] = {
                "skipped": "single-class slice; threshold selected on mixed-class slice required"
            }
        else:
            try:
                mf1_thresh = select_threshold(y_true, y_score, criterion="max_f1").threshold
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
            mf1_thresh = select_threshold(y_true, y_score, criterion="max_f1").threshold
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
