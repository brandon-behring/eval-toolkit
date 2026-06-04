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

import functools
import logging
import warnings
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.stats import bootstrap as _scipy_bootstrap
from scipy.stats import norm as _scipy_norm
from scipy.stats import rankdata as _scipy_rankdata

from eval_toolkit._parallel import parallel_map
from eval_toolkit._rng import RNGLike, SeedLike, spawn_seed_sequences

_logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_METHOD",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapCI",
    "CorrectionMethod",
    "DeLongResult",
    "MDEEstimate",
    "MetricFn",
    "PairedBootstrapCI",
    "ThresholdFn",
    "ThresholdedMetricFn",
    "block_bootstrap_on_folds",
    "bonferroni_correct",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "correct_p_values",
    "cross_validate_metric",
    "cv_clt_ci",
    "delong_roc_variance",
    "fdr_bh_correct",
    "mde_from_ci",
    "paired_bootstrap_diff",
    "paired_bootstrap_ece_diff",
    "paired_bootstrap_op_point_diff",
    "paired_mde",
    "stratified_cluster_bootstrap_ci",
]

DEFAULT_N_RESAMPLES: Final[int] = 1000
DEFAULT_CONFIDENCE: Final[float] = 0.95
DEFAULT_METHOD: Final[Literal["BCa", "percentile"]] = "BCa"
DEFAULT_SEED: Final[int] = 42

MetricFn = Callable[[np.ndarray, np.ndarray], float]
ThresholdFn = Callable[[np.ndarray, np.ndarray], float]
ThresholdedMetricFn = Callable[[np.ndarray, np.ndarray, float], float]
CorrectionMethod = Literal["bh", "bonferroni", "none"]


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
        """Serialize to a stable, self-describing dict schema for JSON output.

        v0.48 BREAKING (§5B): schema rewritten to drop the hard-coded
        ``"ci_95"`` key that lied when ``confidence != 0.95``. The new
        schema names the bounds neutrally and carries the actual
        confidence level in a dedicated field; consumers can read
        ``confidence`` to interpret the bound semantics.

        Before v0.48:
            {"point_estimate": p, "ci_95": [l, h], "confidence": 0.95,
             "n_resamples": N, "method": "BCa"}

        v0.48+:
            {"point": p, "low": l, "high": h, "confidence": 0.95,
             "n_resamples": N, "method": "BCa"}

        Migration: rename ``point_estimate`` → ``point``; replace the
        ``ci_95`` list-of-two with separate ``low`` + ``high`` keys.
        """
        return {
            "point": self.point_estimate,
            "low": self.ci_low,
            "high": self.ci_high,
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
        """Serialize to a stable, self-describing dict schema for JSON output.

        v0.48 BREAKING (§5B): same rewrite as :meth:`BootstrapCI.to_dict`.
        ``"ci_95"`` is replaced by ``"low"`` + ``"high"``; ``"confidence"``
        carries the actual level.

        Before v0.48:
            {"delta": d, "ci_95": [l, h], "overlaps_zero": b,
             "confidence": 0.95, "n_resamples": N}

        v0.48+:
            {"delta": d, "low": l, "high": h, "overlaps_zero": b,
             "confidence": 0.95, "n_resamples": N}
        """
        return {
            "delta": self.delta,
            "low": self.ci_low,
            "high": self.ci_high,
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
    method: Literal["BCa", "percentile", "studentized"] = DEFAULT_METHOD,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
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
    method : {"BCa", "percentile", "studentized"}, optional
        Default "BCa".
    rng : RNGLike | SeedLike | None, optional
        RNG argument per `Scientific Python SPEC 7 <https://scientific-python.org/specs/spec-0007/>`_.
        Int seed (default ``DEFAULT_SEED=42``), ``Generator``, or ``None`` (entropy).
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). Only effective when
        ``method='studentized'`` (which has the only Python-level outer loop
        in this function — the BCa/percentile paths delegate to
        :func:`scipy.stats.bootstrap` which has no ``n_jobs`` knob).
        Setting ``n_jobs != 1`` with a non-studentized method raises
        ``ValueError``. See :ref:`methodology/parallelism`.

    Returns
    -------
    BootstrapCI

    Raises
    ------
    ValueError
        If shapes mismatch, ``n < 10``, ``confidence ∉ (0, 1)``, or
        ``n_jobs != 1`` is supplied with a non-studentized ``method``.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import pr_auc
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = y + rng.normal(0, 0.3, size=200)
    >>> ci = bootstrap_ci(y, s, metric=pr_auc, n_resamples=200, rng=42)
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
    if n_jobs != 1 and method != "studentized":
        raise ValueError(
            f"n_jobs={n_jobs} requires method='studentized'; got method={method!r}. "
            "The BCa and percentile paths delegate to scipy.stats.bootstrap "
            "which does not expose parallelism. Use n_jobs=1 with BCa/percentile, "
            "or method='studentized' with n_jobs > 1."
        )

    _logger.debug(
        "bootstrap_ci: metric=%s n=%d n_resamples=%d method=%s confidence=%.3f rng=%r n_jobs=%d",
        getattr(metric, "__name__", repr(metric)),
        n,
        n_resamples,
        method,
        confidence,
        rng,
        n_jobs,
    )

    point = float(metric(y_true_arr, y_score_arr))

    def _statistic(yt: np.ndarray, ys: np.ndarray) -> float:
        return float(metric(yt, ys))

    if method == "studentized":
        ci_low, ci_high = _bootstrap_t_ci(
            y_true_arr,
            y_score_arr,
            metric,
            point,
            n_resamples=n_resamples,
            confidence=confidence,
            rng=rng,
            n_jobs=n_jobs,
        )
    else:
        rng = np.random.default_rng(rng)
        res = _scipy_bootstrap(
            (y_true_arr, y_score_arr),
            statistic=_statistic,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method=method,
            paired=True,
            random_state=rng,
        )
        ci_low = float(res.confidence_interval.low)
        ci_high = float(res.confidence_interval.high)
        # R9 follow-on (F-bootstrap-1): scipy.stats.bootstrap with BCa can
        # degenerate at small n with ceiling/floor metrics or skewed
        # distributions — the jackknife acceleration constant blows up or
        # produces NaN bounds. scipy emits DegenerateDataWarning but does
        # NOT raise; the returned CI has ci_low == ci_high == point OR
        # NaN bounds. Pre-v0.51 the R8-C4(b) RNG bug spuriously varied
        # bootstrap streams and could mask degeneracy; post-v0.51 with
        # correct RNG the brittleness is exposed. Surface a UserWarning
        # so callers know to switch to method='percentile' OR use a
        # larger n. (R9 third-audit finding F-bootstrap-1; my hunt of
        # modules neither auditor cited; verified at
        # audit-verification-round-9-v0.51.0.md Part 2.)
        if method == "BCa" and (
            not np.isfinite(ci_low) or not np.isfinite(ci_high) or (ci_low == ci_high == point)
        ):
            warnings.warn(
                f"bootstrap_ci: BCa degenerated on n={n} "
                f"(metric={getattr(metric, '__name__', repr(metric))!r}; "
                f"ci_low={ci_low}, ci_high={ci_high}, point={point}). "
                "Common causes: small n with ceiling/floor metric, "
                "near-constant scores, or skewed distributions where "
                "the jackknife acceleration is undefined. Use "
                "method='percentile' for a safer fallback at small n.",
                UserWarning,
                stacklevel=2,
            )
    return BootstrapCI(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
    )


def _bootstrap_t_step(
    seed_seq: np.random.SeedSequence,
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: MetricFn,
) -> tuple[tuple[float, float] | None, str | None]:
    """One outer-resample step for studentized bootstrap-t.

    Returns ``(result, first_failure)`` where:
    - ``result`` is ``(theta_b, se_b)`` from one outer-resample +
      inner-jackknife pair, or ``None`` for any degenerate case
    - ``first_failure`` is a short string describing the first underlying
      exception encountered (outer metric call or inner-jackknife LOO),
      or ``None`` if no exception was raised. Caller surfaces the first
      cross-resample failure-string in the >5% degenerate error.

    The inner jackknife stays sequential per resample (O(n) per call); only
    the outer n_resamples loop benefits from parallelism.
    """
    rng = np.random.default_rng(seed_seq)
    n = int(len(y_true))
    idx = rng.integers(0, n, size=n)
    y_b = y_true[idx]
    s_b = y_score[idx]
    first_failure: str | None = None
    try:
        theta_b = float(metric(y_b, s_b))
    except (ValueError, RuntimeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    loo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        try:
            loo[i] = float(metric(np.delete(y_b, i), np.delete(s_b, i)))
        except (ValueError, RuntimeError) as exc:
            if first_failure is None:
                first_failure = f"{type(exc).__name__}: {exc}"
    valid = ~np.isnan(loo)
    if int(valid.sum()) < 2:
        return None, first_failure
    loo_mean = float(np.nanmean(loo))
    jack_var = (n - 1.0) / n * float(np.nansum((loo[valid] - loo_mean) ** 2))
    if jack_var <= 0.0:
        return None, first_failure
    return (theta_b, float(np.sqrt(jack_var))), first_failure


def _bootstrap_t_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: MetricFn,
    point: float,
    *,
    n_resamples: int,
    confidence: float,
    rng: RNGLike | SeedLike | None,
    n_jobs: int = 1,
) -> tuple[float, float]:
    r"""Studentized bootstrap-t CI per Algeshiemer 2024 / Davison & Hinkley §5.2.

    Outer loop: B bootstrap resamples → ``θ̂_b`` per resample.
    Inner loop: jackknife within each resample → ``SE_b`` per resample.
    Pivot: ``T_b = (θ̂_b - θ̂) / SE_b``.
    CI: ``[θ̂ - q_{1-α/2}(T) · SE, θ̂ - q_{α/2}(T) · SE]`` where ``SE`` is
    the bootstrap standard error of ``θ̂``.

    Best CI coverage of any non-nested method per Algeshiemer 2024
    simulations, at the cost of an extra factor ~n compute (per-resample
    jackknife). Use for high-stakes inference where coverage matters.

    Skips degenerate resamples (single-class draws causing the metric to
    raise); raises if > 5% of resamples are degenerate.
    """
    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(_bootstrap_t_step, y_true=y_true, y_score=y_score, metric=metric)
    raw_results = parallel_map(step, seed_seqs, n_jobs=n_jobs, description="bootstrap_t")
    valid_pairs = [r for r, _ in raw_results if r is not None]
    n_valid = len(valid_pairs)
    if n_valid < n_resamples * 0.95:
        # Surface the first underlying failure for diagnostic context.
        first_failure: str | None = next(
            (failure for _, failure in raw_results if failure is not None), None
        )
        first_msg = (
            f"; first underlying failure: {first_failure}" if first_failure is not None else ""
        )
        raise ValueError(
            f"_bootstrap_t_ci: {n_resamples - n_valid}/{n_resamples} resamples "
            f"degenerate (single-class draws or zero jackknife variance); "
            f"refusing to compute studentized CI on > 5% degenerate resamples"
            f"{first_msg}"
        )

    theta_v = np.array([t for t, _ in valid_pairs], dtype=np.float64)
    se_v = np.array([s for _, s in valid_pairs], dtype=np.float64)
    pivots = (theta_v - point) / se_v
    se_overall = float(np.std(theta_v, ddof=1))
    alpha = (1.0 - confidence) / 2.0
    q_lo = float(np.quantile(pivots, alpha))
    q_hi = float(np.quantile(pivots, 1.0 - alpha))
    # CI is asymmetric — pivot quantiles are subtracted in reverse order.
    return point - q_hi * se_overall, point - q_lo * se_overall


def _paired_bootstrap_diff_step(
    seed_seq: np.random.SeedSequence,
    *,
    y_true_arr: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    metric: MetricFn,
) -> float | None:
    """One paired-bootstrap resample step (module-level for parallel_map picklability).

    Returns the delta ``metric(B) - metric(A)`` on the resampled indices, or
    ``None`` if the resample is degenerate (single-class draw → metric raises).
    Caller filters None values + tracks failure rate.
    """
    rng = np.random.default_rng(seed_seq)
    n = len(y_true_arr)
    idx = rng.integers(0, n, size=n)
    try:
        metric_b = float(metric(y_true_arr[idx], b[idx]))
        metric_a = float(metric(y_true_arr[idx], a[idx]))
    except (ValueError, RuntimeError):
        return None
    return metric_b - metric_a


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric: MetricFn,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
) -> PairedBootstrapCI:
    """Paired-bootstrap CI on ``metric(B) − metric(A)`` using the same resample indices.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels.
    y_score_a, y_score_b : np.ndarray, shape (n,)
        Scores from two scorers on the same rows.
    metric : callable ``(y_true, y_score) -> float``
        Must be picklable when ``n_jobs != 1`` (lambdas not supported).
    n_resamples, confidence, rng : standard bootstrap params (``rng`` per SPEC 7).
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). ``n_jobs > 1`` uses
        joblib loky; ``n_jobs=-1`` uses all cores; ``n_jobs=0`` is rejected.
        See :ref:`methodology/parallelism` for the contract.

    Returns
    -------
    PairedBootstrapCI

    Raises
    ------
    ValueError
        If ``y_true``, ``y_score_a``, ``y_score_b`` do not share the same
        shape; if ``n < 10`` (too small for paired bootstrap); if more
        than 5% of resamples raised in ``metric`` (rare-positive
        degeneracy); or if no resamples produced a usable Δ; or if
        ``n_jobs == 0``.
    TypeError
        If ``n_jobs != 1`` and ``metric`` is not picklable.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import pr_auc
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s_a = rng.normal(0, 1, size=200)                 # random scorer
    >>> s_b = y + rng.normal(0, 0.3, size=200)           # signal scorer
    >>> diff = paired_bootstrap_diff(y, s_a, s_b, pr_auc, n_resamples=200, rng=42)
    >>> diff.delta > 0  # B beats A
    True

    Notes
    -----
    Resampling indices once and computing both metrics on the same resample
    correlates the two bootstrap distributions, producing a tighter CI on Δ
    than independent unpaired bootstraps would.

    Per-resample seeding uses :class:`numpy.random.SeedSequence.spawn` so
    ``n_jobs > 1`` produces bit-for-bit-identical CI to ``n_jobs = 1`` for
    the same caller-supplied ``seed``. This is the v0.34.0 reproducibility
    contract; the RNG-stream did change from the v0.33.x single-Generator
    pattern, so exact CI bounds differ slightly from prior versions on
    identical seeds (statistically equivalent bootstraps either way).

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
    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(
        _paired_bootstrap_diff_step,
        y_true_arr=y_true_arr,
        a=a,
        b=b,
        metric=metric,
    )
    raw_results = parallel_map(step, seed_seqs, n_jobs=n_jobs, description="paired_bootstrap_diff")
    failures = sum(1 for r in raw_results if r is None)
    deltas = [r for r in raw_results if r is not None]

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


def _paired_bootstrap_ece_diff_step(
    seed_seq: np.random.SeedSequence,
    *,
    y_true_arr: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ece_fn: Callable[[np.ndarray, np.ndarray, int], float],
    n_bins: int,
) -> float | None:
    """One paired-ECE resample step (module-level for parallel_map picklability).

    Returns ``ECE(B) - ECE(A)`` on the resampled indices, or ``None`` if the
    resample is single-class (ECE undefined). Caller filters None values +
    tracks failure rate.
    """
    rng = np.random.default_rng(seed_seq)
    n = len(y_true_arr)
    idx = rng.integers(0, n, size=n)
    y_re = y_true_arr[idx]
    n_pos = int(y_re.sum())
    if n_pos == 0 or n_pos == n:
        return None
    try:
        ece_a = ece_fn(y_re, a[idx], n_bins)
        ece_b = ece_fn(y_re, b[idx], n_bins)
    except (ValueError, ZeroDivisionError):
        return None
    return float(ece_b - ece_a)


def paired_bootstrap_ece_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    *,
    ece_fn: Callable[[np.ndarray, np.ndarray, int], float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_bins: int = 10,
    n_jobs: int = 1,
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
    n_resamples, confidence, rng : standard bootstrap params (``rng`` per SPEC 7).
    n_bins : int, optional
        Number of ECE bins (passed through to ``ece_fn``).
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). See
        :ref:`methodology/parallelism`. ``ece_fn`` must be picklable when
        ``n_jobs != 1``.

    Returns
    -------
    PairedBootstrapCI
        ``Δ = ECE_B − ECE_A`` with paired-percentile CI. Lower delta means
        method B is *better calibrated* than A.

    Raises
    ------
    ValueError
        On shape mismatch, ``n < 10``, > 5% degenerate resamples, or
        ``n_jobs == 0``.
    TypeError
        If ``n_jobs != 1`` and ``ece_fn`` is not picklable.

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
    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(
        _paired_bootstrap_ece_diff_step,
        y_true_arr=y_true_arr,
        a=a,
        b=b,
        ece_fn=ece_fn,
        n_bins=n_bins,
    )
    raw_results = parallel_map(
        step, seed_seqs, n_jobs=n_jobs, description="paired_bootstrap_ece_diff"
    )
    failures = sum(1 for r in raw_results if r is None)
    deltas = [r for r in raw_results if r is not None]

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


def _paired_bootstrap_op_point_diff_step(
    seed_seq: np.random.SeedSequence,
    *,
    val_y_arr: np.ndarray,
    val_a: np.ndarray,
    val_b: np.ndarray,
    test_y_arr: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    threshold_fn: ThresholdFn,
    metric_fn: ThresholdedMetricFn,
) -> float | None:
    """One two-level op-point resample step (module-level for parallel_map picklability).

    Resamples val + test indices independently; refits threshold on val
    resample (per scorer); evaluates metric_fn at that threshold on the
    test resample for both scorers. Returns ``m_B - m_A`` or ``None`` on
    any threshold-fit / metric-eval failure (e.g., single-class val draw).
    """
    rng = np.random.default_rng(seed_seq)
    n_val = len(val_y_arr)
    n_test = len(test_y_arr)
    val_idx = rng.integers(0, n_val, size=n_val)
    test_idx = rng.integers(0, n_test, size=n_test)
    try:
        thr_a = float(threshold_fn(val_y_arr[val_idx], val_a[val_idx]))
        thr_b = float(threshold_fn(val_y_arr[val_idx], val_b[val_idx]))
        m_a = float(metric_fn(test_y_arr[test_idx], test_a[test_idx], thr_a))
        m_b = float(metric_fn(test_y_arr[test_idx], test_b[test_idx], thr_b))
    except (ValueError, RuntimeError):
        return None
    return m_b - m_a


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
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
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
        Typically wraps ``ThresholdSelector.select(...).threshold`` (e.g.
        ``lambda y, s: MaxF1Selector().select(y, s).threshold``).
    metric_fn : callable ``(y_true, y_score, threshold) -> float``
        Operating-point metric (e.g., F1, precision) at the given threshold.
    n_resamples, confidence, rng : standard bootstrap params (``rng`` per SPEC 7).
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). See
        :ref:`methodology/parallelism`. Both ``threshold_fn`` and
        ``metric_fn`` must be picklable when ``n_jobs != 1``.

    Returns
    -------
    PairedBootstrapCI
        ``Δ = metric_B(test) − metric_A(test)`` with both val-threshold variance
        and test-metric variance baked in.

    Raises
    ------
    ValueError
        On shape mismatch, insufficient sample size, or ``n_jobs == 0``.
    RuntimeError
        If > 50% of resamples are degenerate (e.g., single-class val draws).
    TypeError
        If ``n_jobs != 1`` and ``threshold_fn`` / ``metric_fn`` are not
        picklable.

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
    # Defensive identity-guard: the two-level bootstrap resamples val + test
    # indices INDEPENDENTLY (see _paired_bootstrap_op_point_diff_step). Passing
    # the same Python object for val and test causes ~63.2% overlap on each
    # resample, violating the val/test independence assumption that lets the
    # CI absorb threshold-selection variance honestly. Partition the data
    # before calling — see docs/source/methodology/thresholds.md.
    if val_y is test_y:
        raise ValueError(
            "paired_bootstrap_op_point_diff: val_y and test_y are the same array. "
            "The two-level bootstrap requires DISJOINT val + test slices; the "
            "resampler draws val_idx and test_idx independently, so identical "
            "arrays cause ~63.2% overlap and violate the independence assumption. "
            "Partition your data first (e.g., val = arr[:n//2], test = arr[n//2:])."
        )

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

    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(
        _paired_bootstrap_op_point_diff_step,
        val_y_arr=val_y_arr,
        val_a=val_a,
        val_b=val_b,
        test_y_arr=test_y_arr,
        test_a=test_a,
        test_b=test_b,
        threshold_fn=threshold_fn,
        metric_fn=metric_fn,
    )
    raw_results = parallel_map(
        step, seed_seqs, n_jobs=n_jobs, description="paired_bootstrap_op_point_diff"
    )
    failures = sum(1 for r in raw_results if r is None)
    valid_list = [r for r in raw_results if r is not None]
    valid = np.asarray(valid_list, dtype=np.float64)
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
    ci: BootstrapCI | PairedBootstrapCI,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> MDEEstimate:
    r"""Derive MDE from an existing ``BootstrapCI`` or ``PairedBootstrapCI``.

    Reuses the bootstrap distribution implicit in the supplied CI: the
    half-width at confidence :math:`c` gives :math:`\sigma \approx
    (\mathrm{ci\_high} - \mathrm{ci\_low}) / (2 \cdot z_{(1+c)/2})`, and the
    standard two-sided MDE formula gives
    :math:`\mathrm{MDE} = (z_{\alpha/2} + z_{\beta}) \cdot \sigma`.

    **v0.34.0 BREAKING**: the first parameter is now ``ci`` (was ``paired``)
    and accepts both ``BootstrapCI`` (single-condition CI) and
    ``PairedBootstrapCI`` (paired-difference CI). Positional callers are
    unaffected; keyword callers must migrate
    ``mde_from_ci(paired=x)`` → ``mde_from_ci(ci=x)`` or
    ``mde_from_ci(x)``. See ``docs/source/DEPRECATION.md`` for the one-time-
    exception justification.

    Parameters
    ----------
    ci : BootstrapCI or PairedBootstrapCI
        Either kind of bootstrap CI dataclass. The function extracts
        ``ci_low``, ``ci_high``, ``confidence``, and ``n_resamples`` from
        either; for ``PairedBootstrapCI`` the observed effect is
        ``ci.delta``; for ``BootstrapCI`` it is ``ci.point_estimate``.
    alpha : float, optional
        Two-sided significance level (default 0.05).
    power : float, optional
        Detection probability at true effect = MDE (default 0.80).

    Returns
    -------
    MDEEstimate
        ``n`` is set to -1 (unknown without source arrays).

    Raises
    ------
    ValueError
        If ``alpha`` or ``power`` is not in (0, 1).
    RuntimeError
        If the supplied CI has non-positive or non-finite width (bootstrap
        likely degenerate; no usable variance signal). The non-finite branch
        catches the case where ``scipy.stats.bootstrap`` returns NaN bounds
        from a degenerate-jackknife BCa run (small n + ceiling/floor metric
        + skewed distribution); without this guard the NaN width would
        silently propagate to ``MDEEstimate.mde = NaN``. Added in v0.51
        per R10 micro-audit follow-on F3.

    Notes
    -----
    Limitations of the analytical σ̂ from CI half-width:

    1. **Normality assumption**: ``σ̂ = width / (2 · z_{α/2})`` assumes
       the bootstrap distribution is approximately normal and symmetric.
       For small ``n_resamples`` (< 200) or skewed metrics (PR-AUC under
       extreme imbalance), σ̂ is biased.
    2. **Boundary-effect bias on bounded metrics**: when the true effect
       is near 0 or near the metric's max (e.g., AUC ≈ 1), the CI is
       asymmetric and the half-width approximation under-estimates σ.
    3. **Skew bias**: for heavy-tailed distributions the percentile-CI
       half-width over-estimates σ. Use :func:`paired_mde` (which
       computes σ from the deltas directly) when these effects matter
       and a paired-difference CI is the use case.

    References
    ----------
    .. [1] Cohen, J. "Statistical Power Analysis for the Behavioral
           Sciences." 2nd ed., Lawrence Erlbaum, 1988.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    width = ci.ci_high - ci.ci_low
    # R9 follow-on (F-bootstrap-2): NaN width bypasses `<= 0` check in IEEE
    # float comparison (NaN <= 0 is False). Scipy's BCa can return NaN bounds
    # on degenerate jackknife; the resulting NaN width would silently
    # propagate to MDEEstimate.mde = NaN. Explicit `not np.isfinite`
    # branch surfaces the degeneracy as RuntimeError.
    if width <= 0 or not np.isfinite(width):
        raise RuntimeError(
            f"non-positive or non-finite CI width ({width}); bootstrap likely degenerate"
        )
    z_at_ci_conf = _normal_quantile((1.0 + ci.confidence) / 2.0)
    sigma = width / (2.0 * z_at_ci_conf)
    z_alpha = _normal_quantile(1.0 - alpha / 2.0)
    z_power = _normal_quantile(power)
    mde = float((z_alpha + z_power) * sigma)
    # Effect-size attribute differs by CI type: paired carries `.delta` as the
    # observed difference; marginal carries `.point_estimate` as the observed
    # single-condition metric. Duck-typed extraction keeps the body branch-free.
    delta_observed = float(getattr(ci, "delta", getattr(ci, "point_estimate", 0.0)))
    return MDEEstimate(
        mde=mde,
        sigma_delta=float(sigma),
        delta_observed=delta_observed,
        alpha=alpha,
        power=power,
        n_resamples=int(ci.n_resamples),
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
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
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
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). Forwarded to the
        internal :func:`paired_bootstrap_diff` call which carries the
        compute. ``mde_from_ci`` is O(1) and ignored. See
        :ref:`methodology/parallelism`.

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
        rng=rng,
        n_jobs=n_jobs,
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


def cv_clt_ci(
    fold_metrics: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapCI:
    r"""CV-corrected confidence interval per Bayle et al. 2020 [#bayle]_ Theorem 3.1.

    Computes a confidence interval on the cross-validation mean metric
    that correctly accounts for fold-level dependence. The standard
    "naive" CI (compute std-of-folds then divide by sqrt(K)) had long
    been suspected to be anti-conservative because the folds share
    training data. Bayle et al. 2020 prove that the naive sample-variance
    estimator (with ``ddof=1``) gives valid asymptotic coverage under
    stability conditions, resolving the historical concern that fold
    correlation makes it anti-conservative. No additional correction
    factor is applied.

    The variance estimator (Bayle 2020 Theorem 3.1) is just the standard
    sample variance over per-fold metrics:

    .. math::

        \widehat{\sigma}^2_{\mathrm{CV-CLT}} = \frac{1}{K - 1}
        \sum_{f=1}^{K} (\widehat{\theta}_f - \bar{\theta})^2

    where :math:`\widehat{\theta}_f` is the metric on fold :math:`f` and
    :math:`\bar{\theta}` is the mean over folds. The CI is then
    :math:`\bar{\theta} \pm z_{\alpha/2} \cdot \widehat{\sigma}_{\mathrm{CV-CLT}}
    / \sqrt{K}`.

    This helper does **not** run the CV — callers supply the already-fit
    per-fold metric estimates. eval-toolkit does not currently ship a
    cross-validation orchestrator (gated on a separate design conversation
    about fold strategy + reproducibility); this function is the standalone
    inference primitive for callers using their own CV runner.

    Parameters
    ----------
    fold_metrics : np.ndarray, shape (K,)
        Per-fold metric estimates. Need ≥ 2 folds.
    confidence : float, optional
        Two-sided confidence level (default 0.95).

    Returns
    -------
    BootstrapCI
        With ``method="cv_clt"`` and ``n_resamples=K``. ``point_estimate``
        is the mean of ``fold_metrics``.

    Raises
    ------
    ValueError
        If ``fold_metrics`` has fewer than 2 entries, contains non-finite
        values, or ``confidence`` is outside (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> # 5-fold CV PR-AUC estimates (already computed externally):
    >>> folds = np.array([0.83, 0.81, 0.85, 0.79, 0.84])
    >>> ci = cv_clt_ci(folds, confidence=0.95)
    >>> ci.method
    'cv_clt'
    >>> bool(ci.ci_low <= ci.point_estimate <= ci.ci_high)
    True

    See Also
    --------
    eval_toolkit.bootstrap.bootstrap_ci :
        Single-test-set CI (no CV); use that for typical eval workflows.

    References
    ----------
    .. [#bayle] Bayle, P., Bayle, A., Janson, L., & Mackey, L.
       "Cross-validation confidence intervals for test error." Annals
       of Statistics 48(6), 2020.
    """
    arr = np.asarray(fold_metrics, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"fold_metrics must be 1-D, got shape {arr.shape}")
    K = int(arr.size)
    if K < 2:
        raise ValueError(f"fold_metrics must have ≥ 2 entries, got K={K}")
    if not np.isfinite(arr).all():
        raise ValueError("fold_metrics contains NaN or inf")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    point = float(arr.mean())
    # Bayle 2020 Theorem 3.1: the naive sample-variance estimator (ddof=1)
    # gives valid asymptotic coverage under stability conditions — no extra
    # correction factor is applied for fold correlation.
    sigma_hat = float(np.std(arr, ddof=1))
    z = _normal_quantile(0.5 + confidence / 2.0)
    margin = z * sigma_hat / np.sqrt(K)
    return BootstrapCI(
        point_estimate=point,
        ci_low=point - margin,
        ci_high=point + margin,
        confidence=confidence,
        n_resamples=K,
        method="cv_clt",
    )


def block_bootstrap_on_folds(
    fold_metrics: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
) -> BootstrapCI:
    r"""Block bootstrap on folds: resample K folds with replacement; percentile CI on mean.

    Sibling primitive to :func:`cv_clt_ci`. Where :func:`cv_clt_ci` relies on
    Bayle et al. 2020's CV-CLT — the naive sample-variance estimator gives
    valid asymptotic coverage under stability + fold exchangeability — the
    block bootstrap is more *conservative* under
    fold-level **non-exchangeability** — situations where the K folds are
    not interchangeable (e.g., source-disjoint LODO folds where one source
    is intrinsically harder than the others). The sensitivity-check
    pattern: if
    ``block_bootstrap_CI_halfwidth / cv_clt_CI_halfwidth > 1.5`` then
    LODO non-exchangeability dominates within-fold variance and the
    block-bootstrap CI is the safer headline.

    The algorithm: draw ``n_resamples`` K-element samples with replacement
    from ``fold_metrics``; for each sample compute the mean; return the
    ``[alpha/2, 1 - alpha/2]`` percentile CI on the empirical distribution
    of resample means.

    This helper does **not** run the CV; callers supply per-fold metric
    estimates as in :func:`cv_clt_ci`. The body is one vectorized numpy
    call (no per-resample Python loop) — does NOT accept ``n_jobs`` since
    parallel-spawn overhead would dominate the O(seconds) total.

    Parameters
    ----------
    fold_metrics : np.ndarray, shape (K,)
        Per-fold metric estimates. Need ≥ 2 folds.
    n_resamples : int, optional
        Number of bootstrap resamples (default 1000). 10000 is typical for
        the cross-fold sensitivity-check use case (runs in O(seconds)).
    confidence : float, optional
        Two-sided confidence level (default 0.95).
    rng : RNGLike | SeedLike | None, optional
        RNG argument per `Scientific Python SPEC 7 <https://scientific-python.org/specs/spec-0007/>`_.
        Int seed (default ``DEFAULT_SEED=42``), ``Generator``, or ``None`` (entropy).

    Returns
    -------
    BootstrapCI
        With ``method="block_bootstrap"``, ``n_resamples`` as supplied,
        ``point_estimate=mean(fold_metrics)``.

    Raises
    ------
    ValueError
        If ``fold_metrics`` is not 1-D, has < 2 entries, contains non-finite
        values, or ``confidence`` is outside (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> folds = np.array([0.83, 0.81, 0.85, 0.79, 0.84])
    >>> ci = block_bootstrap_on_folds(folds, n_resamples=2000, rng=42)
    >>> ci.method
    'block_bootstrap'
    >>> bool(ci.ci_low <= ci.point_estimate <= ci.ci_high)
    True

    See Also
    --------
    eval_toolkit.bootstrap.cv_clt_ci :
        The CV-CLT sibling primitive; use it as the headline CI when folds
        are exchangeable + use ``block_bootstrap_on_folds`` as the spoke
        for the sensitivity-check pattern.

    References
    ----------
    .. [1] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap."
           Chapman & Hall, 1993. (§8 Bootstrap of stratified data.)
    """
    arr = np.asarray(fold_metrics, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"fold_metrics must be 1-D; got shape {arr.shape}")
    K = arr.shape[0]
    if K < 2:
        raise ValueError(f"fold_metrics must have K ≥ 2 folds; got K={K}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("fold_metrics must be all-finite; got NaN or Inf")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")

    rng = np.random.default_rng(rng)
    # Vectorized: (n_resamples, K) index draws, gather, mean along axis 1.
    idx = rng.integers(0, K, size=(n_resamples, K))
    resample_means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(resample_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        point_estimate=float(arr.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=int(n_resamples),
        method="block_bootstrap",
    )


def _label_cluster_units(y_true: np.ndarray, groups: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Index rows by ``(label, group)``: per label, a list of per-group row-index arrays.

    A group that appears under both labels contributes a **separate** index array to each label's
    list — the resample unit is ``(label, group)``, so a mixed-label group (e.g. a document with
    both a poisoned and a benign variant sharing one id) splits into one positive unit and one
    negative unit, resampled independently. Helper for :func:`cluster_bootstrap_ci`.
    """
    units: dict[int, list[np.ndarray]] = {}
    for lab in np.unique(y_true):
        lab_rows = np.flatnonzero(y_true == lab)
        order = np.argsort(groups[lab_rows], kind="stable")
        sorted_rows = lab_rows[order]
        sorted_groups = groups[lab_rows][order]
        # Split at the boundaries between consecutive distinct group ids (post-sort).
        cut = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
        units[int(lab)] = np.split(sorted_rows, cut)
    return units


def _cluster_bootstrap_step(
    seed_seq: np.random.SeedSequence,
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    units: dict[int, list[np.ndarray]],
    statistic: MetricFn,
    resample_labels: tuple[int, ...],
) -> float | None:
    """One cluster-resampled draw of ``statistic`` (module-level for parallel_map picklability).

    For each label, its ``(label, group)`` units are resampled with replacement when the label is
    in ``resample_labels``, else all its rows are held fixed. Returns the statistic on the gathered
    rows, or ``None`` if the draw is degenerate (statistic raises — e.g. a single-class draw).
    """
    rng = np.random.default_rng(seed_seq)
    parts: list[np.ndarray] = []
    for lab, group_rows in units.items():
        if lab in resample_labels:
            chosen = rng.integers(0, len(group_rows), size=len(group_rows))
            parts.extend(group_rows[c] for c in chosen)
        else:
            parts.extend(group_rows)
    idx = np.concatenate(parts)
    try:
        return float(statistic(y_true[idx], y_score[idx]))
    except (ValueError, RuntimeError):
        return None


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    statistic: MetricFn,
    *,
    resample_labels: tuple[int, ...] = (0, 1),
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
) -> BootstrapCI:
    r"""Label-stratified **cluster** (group) bootstrap percentile CI for a single-condition metric.

    Resamples whole ``groups`` (clusters) with replacement rather than individual rows, so the CI
    is honest under intra-cluster correlation (multiple prompts sharing one attack payload; a
    document contributing both a poisoned and a benign row). The resample unit is ``(label,
    group)``: by default (``resample_labels=(0, 1)``) positive-clusters and negative-clusters are
    resampled **separately**, preserving the per-class cluster split so a draw is never
    single-class. Pass ``resample_labels=(1,)`` to resample only the positive clusters while holding
    all negatives fixed (the payload-cluster convention).

    Where :func:`bootstrap_ci` resamples **rows** (i.i.d. assumption) and
    :func:`block_bootstrap_on_folds` resamples **per-fold scalars**, this resamples **clusters of
    rows** — the missing middle for grouped eval data. The analytic row-level AUROC-difference CI
    (:func:`delong_roc_variance`) assumes row independence and so under-covers on clustered data,
    which is the motivation for this estimator.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in ``{0, 1}``.
    y_score : np.ndarray, shape (n,)
        Scores aligned with ``y_true``.
    groups : np.ndarray, shape (n,)
        Cluster id per row (any sortable dtype — ints or strings).
    statistic : callable ``(y_true, y_score) -> float``
        Metric to bootstrap (e.g. ``roc_auc``). Must be **picklable** when ``n_jobs != 1`` (a named
        top-level function — lambdas / closures are rejected).
    resample_labels : tuple[int, ...], optional
        Which label strata are cluster-resampled (default ``(0, 1)`` — both). Labels not listed are
        held fixed (all their rows always included). Must be non-empty.
    n_resamples, confidence, rng : standard bootstrap params (``rng`` per SPEC 7).
    n_jobs : int, optional
        Parallel workers (default 1 — sequential). ``n_jobs=-1`` uses all cores; ``n_jobs=0`` is
        rejected. Per-resample seeding via :func:`spawn_seed_sequences` makes the CI **bit-for-bit
        identical across ``n_jobs``** for a fixed ``rng``. See :ref:`methodology/parallelism`.

    Returns
    -------
    BootstrapCI
        ``method="cluster_percentile"``; ``point_estimate = statistic(y_true, y_score)`` on the full
        data; ``[alpha/2, 1 - alpha/2]`` percentile CI over the cluster-resampled distribution.

    Raises
    ------
    ValueError
        On shape mismatch, non-1-D input, ``n < 10``, ``confidence`` outside (0, 1), empty
        ``resample_labels``, a ``resample_labels`` entry absent from ``y_true``, ``n_jobs == 0``, or
        > 5% degenerate resamples.
    TypeError
        If ``n_jobs != 1`` and ``statistic`` is not picklable.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import roc_auc
    >>> rng = np.random.default_rng(0)
    >>> groups = np.repeat(np.arange(40), 5)        # 40 clusters of 5 rows
    >>> y = (groups % 2).astype(int)                # cluster-pure labels
    >>> s = y + rng.normal(0, 0.3, size=y.size)
    >>> ci = cluster_bootstrap_ci(y, s, groups, roc_auc, n_resamples=200, rng=0)
    >>> ci.method
    'cluster_percentile'
    >>> bool(0.0 <= ci.ci_low <= ci.ci_high <= 1.0)
    True

    Notes
    -----
    For a *gap* statistic with a fixed offset (e.g. ``Gx = val_auc − test_auc`` with ``val_auc``
    held fixed), bootstrap the variable term and shift the bounds: ``Gx_low = val_auc − ci.ci_high``,
    ``Gx_high = val_auc − ci.ci_low``. For a one-sided 95% bound, pass ``confidence=0.90`` and read
    the relevant bound.

    References
    ----------
    .. [1] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap." Chapman & Hall, 1993.
           (§8 — bootstrapping stratified / clustered data.)
    .. [2] Field, C. A. & Welsh, A. H. "Bootstrapping clustered data." JRSS-B 69(3), 2007.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    groups_arr = np.asarray(groups)
    if not (y_true_arr.shape == y_score_arr.shape == groups_arr.shape):
        raise ValueError(
            f"shapes mismatch: y_true {y_true_arr.shape}, y_score {y_score_arr.shape}, "
            f"groups {groups_arr.shape}"
        )
    if y_true_arr.ndim != 1:
        raise ValueError(f"inputs must be 1-D; got shape {y_true_arr.shape}")
    n = int(y_true_arr.size)
    if n < 10:
        raise ValueError(f"n={n} too small for cluster bootstrap; need ≥ 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    resample_labels = tuple(int(x) for x in resample_labels)
    if not resample_labels:
        raise ValueError("resample_labels must be non-empty (nothing would be resampled)")
    present = {int(v) for v in np.unique(y_true_arr).tolist()}
    missing = set(resample_labels) - present
    if missing:
        raise ValueError(
            f"resample_labels {sorted(missing)} absent from y_true (present: {sorted(present)})"
        )

    point = float(statistic(y_true_arr, y_score_arr))
    units = _label_cluster_units(y_true_arr, groups_arr)
    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(
        _cluster_bootstrap_step,
        y_true=y_true_arr,
        y_score=y_score_arr,
        units=units,
        statistic=statistic,
        resample_labels=resample_labels,
    )
    raw = parallel_map(step, seed_seqs, n_jobs=n_jobs, description="cluster_bootstrap_ci")
    failures = sum(1 for r in raw if r is None)
    vals = [r for r in raw if r is not None]
    if failures > 0.05 * n_resamples:
        raise ValueError(
            f"cluster_bootstrap_ci: {failures}/{n_resamples} resamples degenerate "
            "(statistic raised — e.g. single-class draws); refusing to compute CI on > 5% degenerate"
        )
    if not vals:
        raise ValueError("cluster_bootstrap_ci: no usable resamples")
    arr = np.asarray(vals, dtype=np.float64)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        point_estimate=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=int(len(vals)),
        method="cluster_percentile",
    )


def _stratified_cluster_step(
    seed_seq: np.random.SeedSequence,
    *,
    strata_data: dict[Hashable, tuple[np.ndarray, np.ndarray]],
    strata_units: dict[Hashable, dict[int, list[np.ndarray]]],
    per_stratum_metric: MetricFn,
    combine: Callable[[Mapping[Hashable, float]], float],
    resample_labels: tuple[int, ...],
) -> float | None:
    """One stratified-cluster draw of ``combine({key: metric})`` (module-level for picklability).

    Each stratum's ``(label, group)`` units are resampled independently (per ``resample_labels``;
    labels not listed are held fixed) with a single per-resample RNG, the per-stratum metric is
    computed on the gathered rows, and ``combine`` reduces the ``{key: metric}`` map to one scalar.
    Returns ``None`` if any stratum metric or ``combine`` raises (a degenerate draw).
    """
    rng = np.random.default_rng(seed_seq)
    by_key: dict[Hashable, float] = {}
    for key, units in strata_units.items():
        y_k, s_k = strata_data[key]
        parts: list[np.ndarray] = []
        for lab, group_rows in units.items():
            if lab in resample_labels:
                chosen = rng.integers(0, len(group_rows), size=len(group_rows))
                parts.extend(group_rows[c] for c in chosen)
            else:
                parts.extend(group_rows)
        idx = np.concatenate(parts)
        try:
            by_key[key] = float(per_stratum_metric(y_k[idx], s_k[idx]))
        except (ValueError, RuntimeError):
            return None
    try:
        return float(combine(by_key))
    except (ValueError, RuntimeError, KeyError, ZeroDivisionError):
        return None


def stratified_cluster_bootstrap_ci(
    strata: Mapping[Hashable, tuple[np.ndarray, np.ndarray, np.ndarray]],
    per_stratum_metric: MetricFn,
    combine: Callable[[Mapping[Hashable, float]], float],
    *,
    resample_labels: tuple[int, ...] = (0, 1),
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
    n_jobs: int = 1,
) -> BootstrapCI:
    r"""Cluster bootstrap of a **composite** statistic over several independent strata.

    Generalises :func:`cluster_bootstrap_ci` (one condition, one metric) to a statistic that is a
    user-supplied **reduction over several independently-resampled cluster strata** — the shape that
    leave-one-group-out transfer gaps take in practice: a per-seed (and per-group) cluster resample,
    averaged/combined into one scalar (a seed-averaged ROC-AUC gap, a mean-over-carriers gap, a
    top-minus-bottom per-type AUPRC contrast, …). Each bootstrap iteration resamples every stratum's
    ``(label, group)`` clusters (label-stratified, per ``resample_labels``; labels not listed are
    held fixed), computes ``per_stratum_metric`` on each, and reduces the ``{key: metric}`` map with
    ``combine``; the percentile CI is over those reduced values.

    :func:`cluster_bootstrap_ci` is the single-stratum, identity-reduce special case
    (``strata={0: (y, score, groups)}``, ``combine=lambda m: m[0]``).

    Parameters
    ----------
    strata : Mapping[key, (y_true, y_score, groups)]
        Independent resample-units keyed by any hashable (e.g. ``seed`` / ``(carrier, seed)`` /
        ``(attack_type, seed)``). Each value is three aligned 1-D arrays. Iteration order is the
        mapping's order (stable ⇒ deterministic).
    per_stratum_metric : callable ``(y_true, y_score) -> float``
        Metric computed on each stratum's resampled rows (e.g. ``roc_auc``, ``pr_auc``). Must be
        **picklable** when ``n_jobs != 1``.
    combine : callable ``Mapping[key, float] -> float``
        Reduces the per-stratum metrics to the composite statistic (e.g.
        ``val − mean_seed(m)``; ``mean_carrier(val[c] − mean_seed(m[c, ·]))``; a top−bottom contrast).
        Closes over any fixed quantities (val ROC, the type partition) — pass a **picklable**
        top-level function or ``functools.partial`` when ``n_jobs != 1`` (lambdas are fine at
        ``n_jobs == 1``).
    resample_labels : tuple[int, ...], optional
        Which label strata are cluster-resampled within each stratum (default ``(0, 1)``);
        ``(1,)`` resamples only positive clusters, holding negatives fixed (the payload-cluster
        convention). Labels not present in a given stratum are simply skipped there.
    n_resamples, confidence, rng : standard bootstrap params (``rng`` per SPEC 7).
    n_jobs : int, optional
        Parallel workers (default 1). Per-resample seeding via :func:`spawn_seed_sequences` makes the
        CI **bit-for-bit identical across ``n_jobs``**; ``n_jobs=-1`` uses all cores. See
        :ref:`methodology/parallelism`.

    Returns
    -------
    BootstrapCI
        ``method="stratified_cluster_percentile"``; ``point_estimate = combine({key:
        per_stratum_metric(y, score)})`` on the full data; ``[alpha/2, 1 - alpha/2]`` percentile CI.

    Raises
    ------
    ValueError
        On empty ``strata``, a stratum whose arrays mismatch shape / are not 1-D, empty
        ``resample_labels``, ``confidence`` outside (0, 1), ``n_jobs == 0``, or > 5% degenerate
        resamples.
    TypeError
        If ``n_jobs != 1`` and ``per_stratum_metric`` / ``combine`` are not picklable.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import roc_auc
    >>> def _block(seed):
    ...     g = np.repeat(np.arange(20), 4)                 # 20 clusters of 4
    ...     y = (g % 2).astype(int)
    ...     s = y + np.random.default_rng(seed).normal(0, 0.3, size=y.size)
    ...     return y, s, g
    >>> strata = {0: _block(0), 1: _block(1)}               # two seeds
    >>> ci = stratified_cluster_bootstrap_ci(
    ...     strata, roc_auc, lambda m: float(np.mean(list(m.values()))),
    ...     n_resamples=200, rng=0)
    >>> ci.method
    'stratified_cluster_percentile'
    >>> bool(0.0 <= ci.ci_low <= ci.ci_high <= 1.0)
    True

    Notes
    -----
    For a *gap* with a fixed offset (``Gx = val − stat``, ``val`` fixed), fold the offset into
    ``combine`` (``combine`` returns ``val − mean(m.values())``) so the CI is on ``Gx`` directly.

    References
    ----------
    .. [1] Efron, B. & Tibshirani, R. "An Introduction to the Bootstrap." Chapman & Hall, 1993.
           (§8 — stratified / clustered resampling.)
    .. [2] Field, C. A. & Welsh, A. H. "Bootstrapping clustered data." JRSS-B 69(3), 2007.
    """
    if not strata:
        raise ValueError("strata must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    resample_labels = tuple(int(x) for x in resample_labels)
    if not resample_labels:
        raise ValueError("resample_labels must be non-empty (nothing would be resampled)")

    strata_data: dict[Hashable, tuple[np.ndarray, np.ndarray]] = {}
    strata_units: dict[Hashable, dict[int, list[np.ndarray]]] = {}
    for key, triple in strata.items():
        y_a, s_a, g_a = (np.asarray(triple[0]), np.asarray(triple[1]), np.asarray(triple[2]))
        if not (y_a.shape == s_a.shape == g_a.shape) or y_a.ndim != 1:
            raise ValueError(
                f"stratum {key!r}: y/score/groups must be aligned 1-D arrays; got "
                f"{y_a.shape}, {s_a.shape}, {g_a.shape}"
            )
        strata_data[key] = (y_a, s_a)
        strata_units[key] = _label_cluster_units(y_a, g_a)

    point = float(combine({k: float(per_stratum_metric(*strata_data[k])) for k in strata_data}))
    seed_seqs = spawn_seed_sequences(rng, n_resamples)
    step = functools.partial(
        _stratified_cluster_step,
        strata_data=strata_data,
        strata_units=strata_units,
        per_stratum_metric=per_stratum_metric,
        combine=combine,
        resample_labels=resample_labels,
    )
    raw = parallel_map(step, seed_seqs, n_jobs=n_jobs, description="stratified_cluster_bootstrap_ci")
    failures = sum(1 for r in raw if r is None)
    vals = [r for r in raw if r is not None]
    if failures > 0.05 * n_resamples:
        raise ValueError(
            f"stratified_cluster_bootstrap_ci: {failures}/{n_resamples} resamples degenerate "
            "(a per-stratum metric or combine raised); refusing CI on > 5% degenerate"
        )
    if not vals:
        raise ValueError("stratified_cluster_bootstrap_ci: no usable resamples")
    arr = np.asarray(vals, dtype=np.float64)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        point_estimate=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=int(len(vals)),
        method="stratified_cluster_percentile",
    )


def cross_validate_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: MetricFn,
    k: int = 5,
    stratified: bool = True,
    rng: RNGLike | SeedLike | None = DEFAULT_SEED,
) -> np.ndarray:
    r"""K-fold cross-validation of a metric on caller-supplied scores.

    Eval-only flavor: caller has ``(y_true, y_score)`` for the whole
    dataset (typically from a model that has already been trained); this
    helper just slices into K folds, computes ``metric`` on each, and
    returns the per-fold values. Pairs with :func:`cv_clt_ci` for valid
    Bayle 2020 confidence intervals on the CV mean.

    .. note::

        This does NOT re-train a model per fold. The toolkit is a pure
        eval-methodology library; for train+eval cross-validation use
        :func:`sklearn.model_selection.cross_validate` directly and feed
        the per-fold metric values to :func:`cv_clt_ci`.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Scores aligned with ``y_true``.
    metric : callable ``(y_true, y_score) -> float``
        Any metric. Single-class folds are skipped (NaN in result) — the
        caller filters NaNs before passing to ``cv_clt_ci`` if needed.
    k : int, optional
        Number of folds. Default ``5``. Must be ≥ 2.
    stratified : bool, optional
        If ``True`` (default), use ``StratifiedKFold`` so each fold
        preserves the class balance. Recommended for binary
        classification under class imbalance.
    rng : RNGLike | SeedLike | None, optional
        RNG per SPEC 7 — derived to int at the sklearn ``KFold/StratifiedKFold`` boundary.

    Returns
    -------
    np.ndarray, shape (k,)
        Per-fold metric values. NaN entries indicate folds where the
        metric raised (e.g., single-class draw on rare-positive data).

    Raises
    ------
    ValueError
        On shape mismatch, ``k < 2``, ``k > n``, or > 50% NaN folds
        (which would make the CI uninterpretable).

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit.metrics import pr_auc
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> y = rng.binomial(1, 0.3, size=n).astype(int)
    >>> s = np.clip(y * 0.6 + rng.normal(0, 0.3, n), 0, 1)
    >>> folds = cross_validate_metric(y, s, metric=pr_auc, k=5, rng=42)
    >>> folds.shape
    (5,)
    >>> bool(np.all(0.0 <= folds[~np.isnan(folds)]))
    True

    See Also
    --------
    eval_toolkit.bootstrap.cv_clt_ci :
        Compute a CV-corrected confidence interval from the per-fold
        values returned here.
    """
    from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415

    y_arr = np.asarray(y_true).astype(int)
    s_arr = np.asarray(y_score, dtype=float)
    if y_arr.shape != s_arr.shape:
        raise ValueError(f"y_true shape {y_arr.shape} != y_score shape {s_arr.shape}")
    n = int(y_arr.size)
    if k < 2:
        raise ValueError(f"k must be ≥ 2, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")

    # Derive an int seed for sklearn — sklearn KFold's random_state accepts
    # int | None | RandomState (not Generator) across versions <1.4; safer to
    # derive at the boundary than pin a higher sklearn minimum.
    rng = np.random.default_rng(rng)
    sklearn_seed = int(rng.integers(0, 2**31 - 1))

    splitter: KFold | StratifiedKFold
    if stratified:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=sklearn_seed)
        fold_iter = splitter.split(np.zeros(n), y_arr)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=sklearn_seed)
        fold_iter = splitter.split(np.zeros(n))

    fold_metrics = np.full(k, np.nan, dtype=np.float64)
    # Capture first underlying exception so the n_failed raise can quote it
    # (was silent contextlib.suppress; "likely single-class" guess is unhelpful
    # when the actual cause is a different upstream error).
    first_failure: str | None = None
    for f, (_train_idx, test_idx) in enumerate(fold_iter):
        try:
            fold_metrics[f] = float(metric(y_arr[test_idx], s_arr[test_idx]))
        except (ValueError, RuntimeError) as exc:
            if first_failure is None:
                first_failure = f"fold {f}: {type(exc).__name__}: {exc}"

    n_failed = int(np.isnan(fold_metrics).sum())
    if n_failed > k // 2:
        first_msg = (
            f"; first underlying failure: {first_failure}" if first_failure is not None else ""
        )
        raise ValueError(
            f"cross_validate_metric: {n_failed}/{k} folds raised the metric "
            f"(likely single-class folds on rare-positive data); refusing to "
            f"return CV result with > 50% degenerate folds"
            f"{first_msg}"
        )
    return fold_metrics


# ---------------------------------------------------------------------------
# DeLong correlated-ROC ΔAUC z-test (v0.20.0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """Result of a DeLong paired ROC-AUC comparison.

    Returned by :func:`delong_roc_variance`. Carries point AUCs for both
    conditions, the variance of their difference, a two-sided ``z`` and
    ``p_value`` against the null ``AUC_a == AUC_b``, and a 95% CI on the
    delta.

    Parameters
    ----------
    auc_a, auc_b : float
        Per-condition ROC-AUC point estimates.
    delta_auc : float
        ``auc_a - auc_b``.
    var : float
        DeLong variance estimate of ``delta_auc``.
    z : float
        ``delta_auc / sqrt(var)`` (NaN if var is zero).
    p_value : float
        Two-sided p-value against ``H0: delta_auc == 0``.
    ci_low, ci_high : float
        95% normal-approx CI on ``delta_auc``
        (``delta_auc ± 1.96 * sqrt(var)``).
    """

    auc_a: float
    auc_b: float
    delta_auc: float
    var: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        """JSON-serializable dict; NaN/Inf become :func:`float`."""
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "delta_auc": self.delta_auc,
            "var": self.var,
            "z": self.z,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def _delong_structural(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Sun & Xu 2014 structural components for one condition.

    Returns ``(V10, V01, auc)`` where ``V10`` is length ``m = len(pos)``,
    ``V01`` is length ``n = len(neg)``, and ``auc = mean(V10) =
    1 - mean(V01)``. Uses midranks (``scipy.stats.rankdata`` average
    method) to handle ties.
    """
    m = len(pos_scores)
    n = len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("delong_roc_variance requires at least one positive and one negative")
    combined = np.concatenate([pos_scores, neg_scores])
    combined_ranks = _scipy_rankdata(combined, method="average")
    tx10 = combined_ranks[:m]
    tx01 = combined_ranks[m:]
    tx11 = _scipy_rankdata(pos_scores, method="average")
    tx00 = _scipy_rankdata(neg_scores, method="average")
    v10 = (tx10 - tx11) / n
    v01 = 1.0 - (tx01 - tx00) / m
    auc = float(np.mean(v10))
    return v10, v01, auc


def delong_roc_variance(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> DeLongResult:
    """DeLong's variance of the paired ROC-AUC difference.

    Implements the Sun & Xu 2014 fast variant of DeLong's correlated-AUC
    test. Returns a :class:`DeLongResult` with point AUCs, ``delta_auc``,
    DeLong variance, z, two-sided p-value, and a 95% normal-approx CI on
    the delta.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels in ``{0, 1}``. Must contain at least one of each.
    y_score_a, y_score_b : np.ndarray
        Scores for the two conditions on the SAME rows (paired).

    Returns
    -------
    DeLongResult

    Raises
    ------
    ValueError
        If shapes mismatch, labels are not binary, or fewer than one
        positive or negative is present.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = np.array([0]*50 + [1]*50)
    >>> sa = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.0, 1, 50)])
    >>> sb = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.2, 1, 50)])
    >>> result = delong_roc_variance(y, sa, sb)
    >>> bool(result.delta_auc <= 0)  # B is stronger
    True
    """
    y_true_arr = np.asarray(y_true)
    y_a = np.asarray(y_score_a, dtype=float)
    y_b = np.asarray(y_score_b, dtype=float)
    if y_true_arr.shape != y_a.shape or y_a.shape != y_b.shape:
        raise ValueError(
            "delong_roc_variance: y_true, y_score_a, y_score_b must share shape "
            f"(got {y_true_arr.shape}, {y_a.shape}, {y_b.shape})"
        )
    unique = {int(v) for v in np.unique(y_true_arr).tolist()}
    if not unique.issubset({0, 1}):
        raise ValueError(f"delong_roc_variance: y_true must be binary {{0, 1}}, got {unique}")
    pos_mask = y_true_arr == 1
    neg_mask = y_true_arr == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError(
            "delong_roc_variance: need at least one positive and one negative "
            f"row (got m={m}, n={n})"
        )
    v10_a, v01_a, auc_a = _delong_structural(y_a[pos_mask], y_a[neg_mask])
    v10_b, v01_b, auc_b = _delong_structural(y_b[pos_mask], y_b[neg_mask])
    delta = auc_a - auc_b

    # 2x2 covariance matrices for V10 and V01 (between A and B).
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    # Var(AUC_A - AUC_B) = (s10[0,0] - 2*s10[0,1] + s10[1,1])/m
    #                   + (s01[0,0] - 2*s01[0,1] + s01[1,1])/n
    var_delta = (s10[0, 0] - 2.0 * s10[0, 1] + s10[1, 1]) / m + (
        s01[0, 0] - 2.0 * s01[0, 1] + s01[1, 1]
    ) / n
    var_delta = float(max(var_delta, 0.0))  # clamp tiny negative FP noise

    if var_delta == 0.0:
        z = float("nan")
        p_value = float("nan")
        half_ci = 0.0
    else:
        se = float(np.sqrt(var_delta))
        z = delta / se
        p_value = 2.0 * float(1.0 - _scipy_norm.cdf(abs(z)))
        half_ci = 1.959963984540054 * se  # 1.96 to higher precision

    return DeLongResult(
        auc_a=float(auc_a),
        auc_b=float(auc_b),
        delta_auc=float(delta),
        var=var_delta,
        z=float(z),
        p_value=float(p_value),
        ci_low=float(delta - half_ci),
        ci_high=float(delta + half_ci),
    )


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------


def fdr_bh_correct(p_values: np.ndarray) -> np.ndarray:
    r"""Benjamini-Hochberg false-discovery-rate correction.

    Standard step-up FDR-control procedure for the multiple-comparisons
    problem (e.g., correcting many DeLong / paired-bootstrap p-values
    across slices, scorers, folds). Less conservative than Bonferroni;
    controls expected proportion of false discoveries rather than
    family-wise error rate.

    Parameters
    ----------
    p_values : np.ndarray, shape (m,)
        1-D array of raw p-values; values outside ``[0, 1]`` raise.

    Returns
    -------
    np.ndarray, shape (m,)
        q-values (BH-adjusted p-values) in the same position order as
        the input. The i-th q-value is
        :math:`\min_{k \ge i} (m / k) \, p_{(k)}` clipped to
        ``[0, 1]``, where :math:`p_{(k)}` is the k-th smallest input.

    Raises
    ------
    ValueError
        If input is empty or contains values outside ``[0, 1]``.

    References
    ----------
    .. [1] Benjamini, Y. & Hochberg, Y. "Controlling the False Discovery
           Rate: A Practical and Powerful Approach to Multiple Testing."
           Journal of the Royal Statistical Society B 57(1), 1995.

    See Also
    --------
    bonferroni_correct : Family-wise error rate alternative (strict).
    correct_p_values : Dispatch helper covering both methods.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p_values must be non-empty")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        bad = p[~((p >= 0.0) & (p <= 1.0))]
        raise ValueError(f"p_values outside [0, 1]: {bad[:5]}")
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    # BH step-up: q_(k) = min over i >= k of (n / i) * p_(i)
    ranks = np.arange(1, n + 1, dtype=float)
    raw_q = sorted_p * n / ranks
    # Monotone enforcement: q_(k) ≤ q_(k+1) ≤ ... ≤ q_(n)
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def bonferroni_correct(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni family-wise error rate correction.

    Strict FWER control; multiplies each p-value by the number of
    tests and clips at 1. Conservative but simple; pick this when the
    strongest multiple-comparisons guarantee is required.

    Parameters
    ----------
    p_values : np.ndarray, shape (m,)
        1-D array of raw p-values; values outside ``[0, 1]`` raise.

    Returns
    -------
    np.ndarray, shape (m,)
        Corrected p-values ``min(p * m, 1)`` in input position order.

    Raises
    ------
    ValueError
        If input is empty or contains values outside ``[0, 1]``.

    See Also
    --------
    fdr_bh_correct : Less conservative FDR-control alternative.
    correct_p_values : Dispatch helper covering both methods.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p_values must be non-empty")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        bad = p[~((p >= 0.0) & (p <= 1.0))]
        raise ValueError(f"p_values outside [0, 1]: {bad[:5]}")
    clipped: np.ndarray = np.clip(p * p.size, 0.0, 1.0)
    return clipped


def correct_p_values(
    p_values: np.ndarray,
    *,
    method: CorrectionMethod = "bh",
) -> np.ndarray:
    """Dispatch helper for multiple-comparisons correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values; passed through to the selected correction.
    method : {"bh", "bonferroni", "none"}, optional
        ``"bh"`` (default): Benjamini-Hochberg FDR control.
        ``"bonferroni"``: Bonferroni FWER control (conservative).
        ``"none"``: pass-through; returns ``np.asarray(p_values).ravel()``.

    Returns
    -------
    np.ndarray
        Corrected p-values (a.k.a. q-values when ``method="bh"``).

    Notes
    -----
    This function is numpy-only by module convention (the ``bootstrap``
    sub-module depends only on numpy + scipy.stats). The canonical pandas
    idiom for DataFrame-shaped p-values is::

        df["q_value"] = correct_p_values(df["p_value"].to_numpy(), method="bh")

    Pass ``df[col].to_numpy()`` (1-D) on the way in; assign the returned
    array back to a new column (the DataFrame index is preserved by the
    column assignment, not by this function).

    Raises
    ------
    ValueError
        If ``method`` is not one of the documented options, or if
        the underlying correction raises on invalid p-values.
    """
    if method == "bh":
        return fdr_bh_correct(p_values)
    if method == "bonferroni":
        return bonferroni_correct(p_values)
    if method == "none":
        return np.asarray(p_values, dtype=float).ravel()
    raise ValueError(f"method must be 'bh' | 'bonferroni' | 'none'; got {method!r}")
