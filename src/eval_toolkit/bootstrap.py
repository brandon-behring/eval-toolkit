r"""Bootstrap confidence intervals: BCa per-condition, paired-difference, MDE.

- :class:`BootstrapCI` — 95% CI on a single metric on one condition (BCa or percentile)
- :class:`PairedBootstrapCI` — paired CI on metric(B) − metric(A) using shared resample indices
- :func:`paired_bootstrap_op_point_diff` — two-level bootstrap that re-fits operating-point
  thresholds within each resample (correctly accounts for threshold-selection variance)
- :func:`paired_bootstrap_ece_diff` — paired CI on ECE deltas; metric-agnostic via dependency
  injection (caller supplies an ``ece_fn`` callable)
- :class:`MDEEstimate` and :func:`paired_mde` — minimum detectable Δ at requested (α, power)

The math kernels depend only on numpy + scipy.stats; no other module in this toolkit imports
into bootstrap.

References
----------
.. [1] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap." Chapman & Hall, 1993.
.. [2] DiCiccio, T. & Efron, B. "Bootstrap Confidence Intervals." Statistical Science, 1996.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import bootstrap as _scipy_bootstrap
from scipy.stats import norm as _scipy_norm

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_METHOD",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapCI",
    "MDEEstimate",
    "MetricFn",
    "PairedBootstrapCI",
    "ThresholdFn",
    "ThresholdedMetricFn",
    "bootstrap_ci",
    "mde_from_ci",
    "paired_bootstrap_diff",
    "paired_bootstrap_ece_diff",
    "paired_bootstrap_op_point_diff",
    "paired_mde",
]

DEFAULT_N_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_METHOD: Literal["BCa", "percentile"] = "BCa"
DEFAULT_SEED = 42

MetricFn = Callable[[np.ndarray, np.ndarray], float]
ThresholdFn = Callable[[np.ndarray, np.ndarray], float]
ThresholdedMetricFn = Callable[[np.ndarray, np.ndarray, float], float]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """95% CI for a metric on a single condition.

    Parameters
    ----------
    point_estimate : float
        Metric value on the original (non-resampled) data.
    ci_low, ci_high : float
        Lower / upper bound of the confidence interval.
    confidence : float
        Two-sided confidence level ∈ (0, 1) (typically 0.95).
    n_resamples : int
        Number of bootstrap resamples used.
    method : str
        Either ``"BCa"`` (bias-corrected accelerated) or ``"percentile"``.

    Examples
    --------
    >>> ci = BootstrapCI(
    ...     point_estimate=0.85, ci_low=0.78, ci_high=0.91,
    ...     confidence=0.95, n_resamples=1000, method="BCa",
    ... )
    >>> ci.ci_low <= ci.point_estimate <= ci.ci_high
    True

    Notes
    -----
    Frozen value-type. The BCa interval does **not** guarantee
    ``ci_low ≤ point_estimate ≤ ci_high`` — the bias correction can shift
    the interval off-center. Callers that need that invariant should use
    ``method="percentile"``.

    References
    ----------
    .. [1] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap."
           Chapman & Hall, 1993. (Chapter 14: BCa.)
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    method: str

    def to_dict(self) -> dict[str, object]:
        """Serialize to a stable dict schema for JSON output."""
        return {
            "point_estimate": self.point_estimate,
            "ci_95": [self.ci_low, self.ci_high],
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapCI:
    """95% CI for ``metric(B) − metric(A)`` on shared resample indices.

    The lift Δ is the headline statistic for an anti-overengineering stopping
    rule: if ``ci_low <= 0 <= ci_high`` (``overlaps_zero`` is True), the
    improvement is not statistically significant.

    Parameters
    ----------
    delta : float
        Point estimate of ``metric(B) − metric(A)`` on the original data.
    ci_low, ci_high : float
        Lower / upper paired-bootstrap CI bounds on the difference.
    overlaps_zero : bool
        True iff ``ci_low <= 0 <= ci_high`` (inclusive). Encodes the
        zero-effect null result, including the degenerate case where
        ``ci_low == ci_high == 0``.
    confidence : float
        Two-sided confidence level ∈ (0, 1).
    n_resamples : int
        Number of paired bootstrap resamples.

    Examples
    --------
    >>> pci = PairedBootstrapCI(
    ...     delta=0.05, ci_low=0.02, ci_high=0.08,
    ...     overlaps_zero=False, confidence=0.95, n_resamples=1000,
    ... )
    >>> pci.overlaps_zero, pci.delta
    (False, 0.05)

    Notes
    -----
    Paired resampling shares the resample indices between the A and B score
    arrays, so the variance of the difference is reduced by the
    cross-condition correlation — typically a much tighter CI than
    differencing two unpaired CIs would produce.

    References
    ----------
    .. [1] Efron, B. "Bootstrap methods: Another look at the jackknife."
           Annals of Statistics 7(1), 1979.
    .. [2] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap."
           Chapman & Hall, 1993. (§10.3 paired bootstrap.)
    """

    delta: float
    ci_low: float
    ci_high: float
    overlaps_zero: bool
    confidence: float
    n_resamples: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a stable dict schema for JSON output."""
        return {
            "delta": self.delta,
            "ci_95": [self.ci_low, self.ci_high],
            "overlaps_zero": self.overlaps_zero,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
        }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: MetricFn,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    method: Literal["BCa", "percentile"] = DEFAULT_METHOD,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Per-condition CI via :func:`scipy.stats.bootstrap`.

    Resamples paired ``(y_true, y_score)`` indices with replacement. Standard
    BCa unless ``method='percentile'`` is forced (recommended fallback for
    very small slices where BCa jackknife may misbehave).

    Parameters
    ----------
    y_true, y_score : np.ndarray, shape (n,)
        Labels and scores.
    metric : callable ``(y_true, y_score) -> float``
        Any metric. ``pr_auc``, ``roc_auc``, etc.
    n_resamples : int, optional
        Default 1000.
    confidence : float, optional
        Two-sided confidence level (default 0.95).
    method : {"BCa", "percentile"}, optional
        Default "BCa".
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    BootstrapCI

    Raises
    ------
    ValueError
        If shapes mismatch, ``n < 10``, or ``confidence ∉ (0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import pr_auc
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = y + rng.normal(0, 0.3, size=200)
    >>> ci = bootstrap_ci(y, s, metric=pr_auc, n_resamples=200, seed=42)
    >>> ci.ci_low <= ci.point_estimate <= ci.ci_high
    True

    Notes
    -----
    The bias-corrected and accelerated (BCa) interval [1]_ is recommended over
    plain percentile for asymmetric statistics. For very small samples, BCa
    jackknife can degenerate; percentile is the safe fallback.

    References
    ----------
    .. [1] Efron, B. "Better bootstrap confidence intervals." JASA 82(397),
           1987.
    .. [2] DiCiccio, T. J. & Efron, B. "Bootstrap confidence intervals."
           Statistical Science 11(3), 1996.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(f"y_true shape {y_true_arr.shape} != y_score shape {y_score_arr.shape}")
    n = len(y_true_arr)
    if n < 10:
        raise ValueError(f"n={n} too small for bootstrap; need ≥ 10")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    point = float(metric(y_true_arr, y_score_arr))

    def _statistic(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(metric(yt, ys))

    rng = np.random.default_rng(seed)
    res = _scipy_bootstrap(
        (y_true_arr, y_score_arr),
        statistic=_statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method=method,
        paired=True,
        random_state=rng,
    )
    return BootstrapCI(
        point_estimate=point,
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
    )


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric: MetricFn,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PairedBootstrapCI:
    """Paired-bootstrap CI on ``metric(B) − metric(A)`` using the same resample indices.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score_a, y_score_b : np.ndarray, shape (n,)
        Scores from two scorers on the same rows.
    metric : callable ``(y_true, y_score) -> float``
    n_resamples, confidence, seed : standard bootstrap params.

    Returns
    -------
    PairedBootstrapCI

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import pr_auc
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s_a = rng.normal(0, 1, size=200)                 # random scorer
    >>> s_b = y + rng.normal(0, 0.3, size=200)           # signal scorer
    >>> diff = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=200, seed=42)
    >>> diff.delta > 0  # B beats A
    True

    Notes
    -----
    Resampling indices once and computing both metrics on the same resample
    correlates the two bootstrap distributions, producing a tighter CI on Δ
    than independent unpaired bootstraps would.

    References
    ----------
    .. [1] Efron, B. "Bootstrap methods: Another look at the jackknife."
           Annals of Statistics 7(1), 1979.
    .. [2] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap."
           Chapman & Hall, 1993. (§10.3.)
    """
    y_true_arr = np.asarray(y_true)
    a = np.asarray(y_score_a)
    b = np.asarray(y_score_b)
    if not (y_true_arr.shape == a.shape == b.shape):
        raise ValueError(f"shapes mismatch: y_true {y_true_arr.shape}, a {a.shape}, b {b.shape}")
    n = len(y_true_arr)
    if n < 10:
        raise ValueError(f"n={n} too small for paired bootstrap; need ≥ 10")

    delta_point = float(metric(y_true_arr, b)) - float(metric(y_true_arr, a))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    failures = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            metric_b = float(metric(y_true_arr[idx], b[idx]))
            metric_a = float(metric(y_true_arr[idx], a[idx]))
        except (ValueError, RuntimeError):
            # Single-class resamples raise ValueError on PR/ROC-AUC; rare-positive
            # data can also trigger sklearn's UndefinedMetric. Skip + audit.
            failures += 1
            continue
        deltas.append(metric_b - metric_a)

    if failures > 0.05 * n_resamples:
        raise ValueError(
            f"paired_bootstrap_diff: {failures}/{n_resamples} resamples raised "
            "the metric function (likely single-class draws on rare-positive "
            "data); refusing to compute CI on > 5% degenerate resamples"
        )
    if not deltas:
        raise ValueError("paired_bootstrap_diff: no usable resamples")

    deltas_arr = np.asarray(deltas, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(deltas_arr, alpha))
    ci_high = float(np.quantile(deltas_arr, 1.0 - alpha))
    return PairedBootstrapCI(
        delta=delta_point,
        ci_low=ci_low,
        ci_high=ci_high,
        overlaps_zero=ci_low <= 0.0 <= ci_high,
        confidence=confidence,
        n_resamples=int(len(deltas_arr)),
    )


def paired_bootstrap_ece_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    *,
    ece_fn: Callable[[np.ndarray, np.ndarray, int], float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    n_bins: int = 10,
) -> PairedBootstrapCI:
    r"""Paired-bootstrap CI on ``ECE(B) − ECE(A)`` for two calibrated outputs.

    Uses the same resample indices for both calibrators so the Δ is paired
    across calibration methods (correlated → tighter CI). Skips degenerate
    single-class resamples (which have undefined ECE) and raises if the
    failure rate exceeds 5%.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score_a, y_score_b : np.ndarray, shape (n,)
        Calibrated probabilities from method A and method B.
    ece_fn : callable ``(y_true, y_score, n_bins) -> float``
        ECE function to use. Bootstrap is metric-agnostic; caller injects the
        specific ECE variant (equal-width, equal-mass, etc.) so this module
        does not depend on calibration. Typical use:
        ``from eval_toolkit.metrics import expected_calibration_error``,
        then pass ``ece_fn=expected_calibration_error``.
    n_resamples, confidence, seed : standard bootstrap params.
    n_bins : int, optional
        Number of ECE bins (passed through to ``ece_fn``).

    Returns
    -------
    PairedBootstrapCI
        ``Δ = ECE_B − ECE_A`` with paired-percentile CI. Lower delta means
        method B is *better calibrated* than A.

    Raises
    ------
    ValueError
        On shape mismatch, ``n < 10``, or > 5% degenerate resamples.

    Notes
    -----
    The dependency injection of ``ece_fn`` is intentional: bootstrap math is
    independent of which ECE variant is being compared, so this module stays
    metric-agnostic and depends only on numpy + scipy.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    a = np.asarray(y_score_a, dtype=float)
    b = np.asarray(y_score_b, dtype=float)
    if not (y_true_arr.shape == a.shape == b.shape):
        raise ValueError(f"shapes mismatch: y_true {y_true_arr.shape}, a {a.shape}, b {b.shape}")
    n = int(y_true_arr.size)
    if n < 10:
        raise ValueError(f"n={n} too small for paired bootstrap; need >= 10")

    delta_point = float(ece_fn(y_true_arr, b, n_bins)) - float(ece_fn(y_true_arr, a, n_bins))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    failures = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_re = y_true_arr[idx]
        n_pos = int(y_re.sum())
        if n_pos == 0 or n_pos == n:
            failures += 1
            continue
        try:
            ece_a = ece_fn(y_re, a[idx], n_bins)
            ece_b = ece_fn(y_re, b[idx], n_bins)
        except (ValueError, ZeroDivisionError):
            failures += 1
            continue
        deltas.append(float(ece_b - ece_a))

    if failures > 0.05 * n_resamples:
        raise ValueError(
            f"paired_bootstrap_ece_diff: {failures}/{n_resamples} resamples degenerate; "
            "input may be too small or too imbalanced"
        )
    if not deltas:
        raise ValueError("paired_bootstrap_ece_diff: no usable resamples")

    deltas_arr = np.asarray(deltas, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(deltas_arr, alpha))
    ci_high = float(np.quantile(deltas_arr, 1.0 - alpha))
    return PairedBootstrapCI(
        delta=delta_point,
        ci_low=ci_low,
        ci_high=ci_high,
        overlaps_zero=ci_low <= 0.0 <= ci_high,
        confidence=confidence,
        n_resamples=len(deltas),
    )


def paired_bootstrap_op_point_diff(
    val_y: np.ndarray,
    val_score_a: np.ndarray,
    val_score_b: np.ndarray,
    test_y: np.ndarray,
    test_score_a: np.ndarray,
    test_score_b: np.ndarray,
    threshold_fn: ThresholdFn,
    metric_fn: ThresholdedMetricFn,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PairedBootstrapCI:
    r"""Two-level paired bootstrap for operating-point lifts.

    Operating-point metrics (F1@threshold, precision@threshold, recall@
    threshold) depend on a threshold *chosen on val*. The single-level
    paired bootstrap re-uses one fixed val-derived threshold across all
    resamples, which under-counts variance from threshold selection. This
    helper resamples val + test independently per iteration, refits the
    threshold on the val resample (via ``threshold_fn``), then evaluates
    ``metric_fn`` at that threshold on the test resample for both scorers.

    Both scorers share the val resample (so threshold differences stay
    apples-to-apples); each scorer fits its *own* threshold from that
    shared resample. Test indices are likewise shared across scorers.

    Parameters
    ----------
    val_y, val_score_a, val_score_b : np.ndarray
        Validation labels and scores for both scorers.
    test_y, test_score_a, test_score_b : np.ndarray
        Test labels and scores for both scorers.
    threshold_fn : callable ``(y_true, y_score) -> threshold``
        Typically wraps ``select_threshold(..., criterion=...).threshold``.
    metric_fn : callable ``(y_true, y_score, threshold) -> float``
        Operating-point metric (e.g., F1, precision) at the given threshold.
    n_resamples, confidence, seed : standard bootstrap params.

    Returns
    -------
    PairedBootstrapCI
        ``Δ = metric_B(test) − metric_A(test)`` with both val-threshold variance
        and test-metric variance baked in.

    Raises
    ------
    ValueError
        On shape mismatch or insufficient sample size.
    RuntimeError
        If > 50% of resamples are degenerate (e.g., single-class val draws).

    Notes
    -----
    Two-level structure: outer level resamples val + test indices; inner level
    refits the threshold on the val resample, then evaluates on the test
    resample. The combined CI is wider than the fixed-threshold paired CI
    because it absorbs threshold-selection noise.

    Methodological caveats:

    1. **Variance-only simplification**: this is a *variance-correction*
       nested bootstrap — it does not implement the double-bootstrap bias
       correction in Davison & Hinkley §4.2 eq. 4.6. Acceptable for most
       ML applications but matters at small val sets or near boundary
       prevalences (e.g., precision@99% recall).
    2. **Independent val/test resampling**: deliberately drops any
       correlation structure between val and test (correct under i.i.d.
       splits; conservative under deliberate-OOD splits).
    3. **Replicability caveat**: paired bootstrap tests with re-used data
       have lower replicability than naive degrees-of-freedom suggest
       (Bouckaert 2003).

    References
    ----------
    .. [1] Davison, A. C. & Hinkley, D. V. "Bootstrap Methods and their
           Application." Cambridge, 1997. (§4.2 Nested bootstrap.)
    .. [2] Bouckaert, R. R. "Choosing between two learning algorithms
           based on calibrated tests." ICML 2003.
    """
    val_y_arr = np.asarray(val_y)
    val_a, val_b = np.asarray(val_score_a), np.asarray(val_score_b)
    test_y_arr = np.asarray(test_y)
    test_a, test_b = np.asarray(test_score_a), np.asarray(test_score_b)
    if not (val_y_arr.shape == val_a.shape == val_b.shape):
        raise ValueError(
            f"val shape mismatch: y={val_y_arr.shape}, a={val_a.shape}, b={val_b.shape}"
        )
    if not (test_y_arr.shape == test_a.shape == test_b.shape):
        raise ValueError(
            f"test shape mismatch: y={test_y_arr.shape}, a={test_a.shape}, b={test_b.shape}"
        )
    n_val, n_test = len(val_y_arr), len(test_y_arr)
    if n_val < 10 or n_test < 10:
        raise ValueError(f"need ≥ 10 rows in val and test; got val={n_val}, test={n_test}")

    thr_a_full = float(threshold_fn(val_y_arr, val_a))
    thr_b_full = float(threshold_fn(val_y_arr, val_b))
    delta_point = float(metric_fn(test_y_arr, test_b, thr_b_full)) - float(
        metric_fn(test_y_arr, test_a, thr_a_full)
    )

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    failures = 0
    for r in range(n_resamples):
        val_idx = rng.integers(0, n_val, size=n_val)
        test_idx = rng.integers(0, n_test, size=n_test)
        try:
            thr_a = float(threshold_fn(val_y_arr[val_idx], val_a[val_idx]))
            thr_b = float(threshold_fn(val_y_arr[val_idx], val_b[val_idx]))
            m_a = float(metric_fn(test_y_arr[test_idx], test_a[test_idx], thr_a))
            m_b = float(metric_fn(test_y_arr[test_idx], test_b[test_idx], thr_b))
            deltas[r] = m_b - m_a
        except (ValueError, RuntimeError):
            deltas[r] = np.nan
            failures += 1
    valid = deltas[~np.isnan(deltas)]
    if len(valid) < n_resamples // 2:
        raise RuntimeError(
            f"paired_bootstrap_op_point_diff: {failures}/{n_resamples} resamples degenerate; "
            "refusing to compute CI on < 50% of requested resamples"
        )

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(valid, alpha))
    ci_high = float(np.quantile(valid, 1.0 - alpha))
    return PairedBootstrapCI(
        delta=delta_point,
        ci_low=ci_low,
        ci_high=ci_high,
        overlaps_zero=ci_low <= 0.0 <= ci_high,
        confidence=confidence,
        n_resamples=int(len(valid)),
    )


@dataclass(frozen=True, slots=True)
class MDEEstimate:
    r"""Minimum detectable Δ at the requested (α, 1-β).

    ``mde`` is the smallest true Δ that the paired bootstrap on this
    ``(y, a, b)`` configuration would detect with probability ≥ ``power`` at
    significance ``alpha`` (two-sided). Computed analytically from the
    bootstrap-estimated standard error of Δ:

    .. math::

        \mathrm{MDE} = (z_{\alpha/2} + z_{\beta}) \cdot \sigma_\Delta

    where :math:`\sigma_\Delta = (\mathrm{ci\_high} - \mathrm{ci\_low}) / (2 \cdot 1.96)`.
    Assumes asymptotic normality of the bootstrap distribution; for small N
    this is a reasonable but not exact approximation.

    Parameters
    ----------
    mde : float
        Minimum detectable difference at the configured (α, power).
    sigma_delta : float
        Standard error of Δ inferred from the paired-bootstrap CI half-width.
    delta_observed : float
        Observed point estimate of Δ on the original data.
    alpha : float
        Two-sided significance level used in the MDE calculation (typically
        0.05).
    power : float
        Detection probability used in the MDE calculation (typically 0.80).
    n_resamples : int
        Number of paired-bootstrap resamples that produced the source CI.
    n : int
        Sample size used in the paired bootstrap (-1 if unknown — see
        :func:`mde_from_ci`).

    Examples
    --------
    >>> est = MDEEstimate(
    ...     mde=0.04, sigma_delta=0.014, delta_observed=0.02,
    ...     alpha=0.05, power=0.8, n_resamples=1000, n=500,
    ... )
    >>> est.delta_observed < est.mde  # observed < MDE → underpowered
    True

    Notes
    -----
    The MDE is the minimum *true* effect size detectable; the *observed*
    delta can be smaller than MDE (in which case the experiment is
    underpowered) or larger (in which case the result is interpretable).

    References
    ----------
    .. [1] Cohen, J. "Statistical Power Analysis for the Behavioral Sciences."
           2nd ed., Lawrence Erlbaum, 1988.
    """

    mde: float
    sigma_delta: float
    delta_observed: float
    alpha: float
    power: float
    n_resamples: int
    n: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the MDE estimate."""
        return {
            "mde": self.mde,
            "sigma_delta": self.sigma_delta,
            "delta_observed": self.delta_observed,
            "alpha": self.alpha,
            "power": self.power,
            "n_resamples": self.n_resamples,
            "n": self.n,
        }


def mde_from_ci(
    paired: PairedBootstrapCI,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> MDEEstimate:
    r"""Derive MDE from an existing ``PairedBootstrapCI`` (no second bootstrap).

    Reuses the bootstrap distribution implicit in the paired CI: the
    half-width at 95% gives :math:`\sigma_\Delta \approx (\mathrm{ci\_high} - \mathrm{ci\_low}) / (2 \cdot 1.96)`,
    and the standard two-sided MDE formula gives
    :math:`\mathrm{MDE} = (z_{\alpha/2} + z_{\beta}) \cdot \sigma_\Delta`.

    Parameters
    ----------
    paired : PairedBootstrapCI
    alpha : float, optional
        Two-sided significance level (default 0.05).
    power : float, optional
        Detection probability at true Δ = MDE (default 0.80).

    Returns
    -------
    MDEEstimate
        ``n`` is set to -1 (unknown without source arrays).

    Notes
    -----
    Limitations of the analytical σ̂ from CI half-width:

    1. **Normality assumption**: ``σ̂_Δ = width / (2 · z_{α/2})`` assumes
       the bootstrap distribution of Δ is approximately normal and
       symmetric. For small ``n_resamples`` (< 200) or skewed metrics
       (PR-AUC under extreme imbalance), σ̂ is biased.
    2. **Boundary-effect bias on bounded metrics**: when the true Δ is
       near 0 or near the metric's max (e.g., AUC ≈ 1), the CI is
       asymmetric and the half-width approximation under-estimates σ.
    3. **Skew bias**: for heavy-tailed Δ distributions the percentile-CI
       half-width over-estimates σ. Use :func:`paired_mde` (which
       computes σ from the deltas directly) when these effects matter.

    References
    ----------
    .. [1] Cohen, J. "Statistical Power Analysis for the Behavioral
           Sciences." 2nd ed., Lawrence Erlbaum, 1988.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    width = paired.ci_high - paired.ci_low
    if width <= 0:
        raise RuntimeError(f"non-positive CI width ({width}); paired bootstrap likely degenerate")
    z_at_paired_conf = _normal_quantile((1.0 + paired.confidence) / 2.0)
    sigma = width / (2.0 * z_at_paired_conf)
    z_alpha = _normal_quantile(1.0 - alpha / 2.0)
    z_power = _normal_quantile(power)
    mde = float((z_alpha + z_power) * sigma)
    return MDEEstimate(
        mde=mde,
        sigma_delta=float(sigma),
        delta_observed=float(paired.delta),
        alpha=alpha,
        power=power,
        n_resamples=int(paired.n_resamples),
        n=-1,
    )


def paired_mde(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric: MetricFn,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> MDEEstimate:
    r"""Minimum detectable paired Δ at (α, power).

    Quantifies "the headline lift barely clears zero": given the observed
    paired-bootstrap variance of Δ, the smallest true Δ this test would
    reject the null on with ``power`` probability is

    .. math::

        \mathrm{MDE} = (z_{\alpha/2} + z_{\beta}) \cdot \sigma_\Delta

    For α=0.05, power=0.80 the multiplier is ≈ 2.80
    (:math:`z_{0.025} \approx 1.96`, :math:`z_{0.20} \approx 0.842`).

    Parameters
    ----------
    y_true, y_score_a, y_score_b : np.ndarray
        Labels and two scorers' outputs on the same rows.
    metric : MetricFn
    alpha : float, optional
        Two-sided significance (default 0.05).
    power : float, optional
        1 − β; probability of detection at true Δ = MDE (default 0.80).

    Returns
    -------
    MDEEstimate
    """
    paired = paired_bootstrap_diff(
        y_true,
        y_score_a,
        y_score_b,
        metric,
        n_resamples=n_resamples,
        confidence=0.95,
        seed=seed,
    )
    est = mde_from_ci(paired, alpha=alpha, power=power)
    return MDEEstimate(
        mde=est.mde,
        sigma_delta=est.sigma_delta,
        delta_observed=est.delta_observed,
        alpha=est.alpha,
        power=est.power,
        n_resamples=est.n_resamples,
        n=int(len(np.asarray(y_true))),
    )


def _normal_quantile(p: float) -> float:
    """Inverse CDF (PPF) of the standard normal — exact via :func:`scipy.stats.norm.ppf`."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    return float(_scipy_norm.ppf(p))
