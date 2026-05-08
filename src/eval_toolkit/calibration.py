r"""Calibration: reliability curves, Bayes-optimal thresholds, isotonic/Platt/temperature scaling.

Public surface:

- :func:`reliability_curve` — bin-level calibration data
  (DeGroot & Fienberg 1983 [#degroot]_; Niculescu-Mizil & Caruana 2005 [#nm05]_)
- :func:`bayes_optimal_threshold` — closed-form cost-sensitive decision boundary
  (Elkan 2001 [#elkan]_); :class:`CostMatrix` packages prior + costs + abstain cost.
- :func:`fit_isotonic_calibrator` — Niculescu-Mizil & Caruana 2005 [#nm05]_
- :func:`fit_platt_calibrator` — Platt 1999 [#platt]_ sigmoid scaling
- :func:`fit_temperature` — Guo et al. 2017 [#guo]_ — fits T on val *logits* (literature standard)
- :func:`fit_temperature_oracle` — Guo et al. 2017 [#guo]_ — fits T on *probabilities*; diagnostic
  upper-bound only (T is fit on the data it then scores).

References
----------
.. [#degroot] DeGroot, M. H. & Fienberg, S. E. "The Comparison and Evaluation of Forecasters."
   *The Statistician* 32 (1/2): 12-22, 1983.
.. [#elkan] Elkan, C. "The Foundations of Cost-Sensitive Learning." IJCAI 2001.
.. [#guo] Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. "On Calibration of Modern Neural Networks."
   ICML 2017. arXiv:1706.04599.
.. [#nm05] Niculescu-Mizil, A. & Caruana, R. "Predicting Good Probabilities With Supervised
   Learning." ICML 2005.
.. [#platt] Platt, J. "Probabilistic Outputs for Support Vector Machines and Comparisons to
   Regularized Likelihood Methods." *Advances in Large Margin Classifiers*, 1999.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "DEFAULT_FN_COST",
    "DEFAULT_FP_COST",
    "DEFAULT_N_BINS",
    "DEFAULT_PRIOR",
    "DEFAULT_STRATEGY",
    "CostMatrix",
    "bayes_optimal_threshold",
    "fit_isotonic_calibrator",
    "fit_platt_calibrator",
    "fit_temperature",
    "fit_temperature_oracle",
    "reliability_curve",
]

DEFAULT_N_BINS = 10
DEFAULT_STRATEGY: Literal["uniform", "quantile"] = "quantile"

# Example cost-matrix defaults (rare-positive deployment surface). These are
# illustrative scaffolding; a real cost matrix should come from stakeholder
# elicitation, not library defaults.
DEFAULT_PRIOR = 0.01
DEFAULT_FP_COST = 1.0
DEFAULT_FN_COST = 10.0


def reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: Literal["uniform", "quantile"] = DEFAULT_STRATEGY,
) -> dict[str, object]:
    """Bin-level calibration data wrapping :func:`sklearn.calibration.calibration_curve`.

    Returns a JSON-friendly dict with bin centers, observed positive rates,
    per-bin counts, and both equal-width and equal-mass ECE summaries.
    Single-class slices are skipped with an explicit marker.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Predicted probabilities in [0, 1].
    n_bins : int, optional
        Number of bins (default 10).
    strategy : {"uniform", "quantile"}, optional
        Equal-width vs equal-mass binning. Default "quantile".

    Returns
    -------
    dict
        Either the calibration record with keys ``prob_true``, ``prob_pred``,
        ``bin_edges``, ``n_per_bin``, ``ece_equal_mass``, ``ece_equal_width``,
        ``n_bins``, ``strategy``, ``n``, ``n_positive``,
        or ``{"skipped": "...", "n", "n_positive"}`` for a single-class slice.

    Raises
    ------
    ValueError
        On shape mismatch, empty input, ``n_bins <= 1``, or unknown strategy.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = (y + rng.normal(0, 0.5, size=200)).clip(0, 1)
    >>> result = reliability_curve(y, s, n_bins=5, strategy="uniform")
    >>> sorted(result.keys())[:5]
    ['bin_edges', 'ece_equal_mass', 'ece_equal_width', 'n', 'n_bins']
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(f"shape mismatch: y_true {y_true_arr.shape}, y_score {y_score_arr.shape}")
    if y_true_arr.size == 0:
        raise ValueError("y_true is empty")
    if n_bins <= 1:
        raise ValueError(f"n_bins must be > 1, got {n_bins}")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    n = int(y_true_arr.size)
    n_positive = int(y_true_arr.sum())
    if n_positive == 0 or n_positive == n:
        return {
            "skipped": (
                "single-class slice; calibration is degenerate (per-bin observed "
                "rates are constant 0 or 1)."
            ),
            "n": n,
            "n_positive": n_positive,
        }

    prob_true, prob_pred = calibration_curve(
        y_true_arr, y_score_arr, n_bins=n_bins, strategy=strategy
    )

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.quantile(y_score_arr, np.linspace(0.0, 1.0, n_bins + 1))
    n_per_bin, _ = np.histogram(y_score_arr, bins=bin_edges)

    ece_equal_mass = _ece_via_calibration_curve(y_true_arr, y_score_arr, n_bins, "quantile")
    ece_equal_width = _ece_via_calibration_curve(y_true_arr, y_score_arr, n_bins, "uniform")

    return {
        "n": n,
        "n_positive": n_positive,
        "n_bins": int(n_bins),
        "strategy": strategy,
        "prob_true": [float(x) for x in prob_true],
        "prob_pred": [float(x) for x in prob_pred],
        "bin_edges": [float(x) for x in bin_edges],
        "n_per_bin": [int(x) for x in n_per_bin],
        "ece_equal_mass": float(ece_equal_mass),
        "ece_equal_width": float(ece_equal_width),
    }


def _ece_via_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
    strategy: Literal["uniform", "quantile"],
) -> float:
    """ECE computed via sklearn's ``calibration_curve`` (handles empty bins).

    Used internally by :func:`reliability_curve`. For metric-only ECE in
    bootstrap contexts, use ``eval_toolkit.metrics.expected_calibration_error``.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy=strategy)
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.quantile(y_score, np.linspace(0.0, 1.0, n_bins + 1))
    n_per_bin, _ = np.histogram(y_score, bins=bin_edges)
    non_empty_mask = n_per_bin > 0
    n_per_bin_nonempty = n_per_bin[non_empty_mask]
    if len(n_per_bin_nonempty) != len(prob_true):
        n_per_bin_nonempty = np.full(len(prob_true), len(y_score) / max(len(prob_true), 1))
    weights = n_per_bin_nonempty / max(int(n_per_bin_nonempty.sum()), 1)
    return float((weights * np.abs(prob_true - prob_pred)).sum())


def bayes_optimal_threshold(π: float, c_fp: float, c_fn: float) -> float:
    r"""Bayes-optimal threshold per Elkan 2001 [#elkan]_ cost-sensitive derivation.

    For a calibrated probabilistic classifier P(y=1 | x), the cost-minimizing
    decision rule is "predict 1 iff score ≥ t*" with:

    .. math:: t^* = \frac{c_{FP} \cdot (1 - π)}{c_{FP} \cdot (1 - π) + c_{FN} \cdot π}

    Parameters
    ----------
    π : float
        Deployment positive-class prior P(y=1) ∈ [0, 1].
        (π = empirical positive prior; English alias on first appearance per the
        Unicode-identifier convention in STYLE.md.)
    c_fp : float
        Cost of a false positive. Must be > 0.
    c_fn : float
        Cost of a false negative. Must be > 0.

    Returns
    -------
    float
        Optimal threshold ∈ [0, 1].

    Raises
    ------
    ValueError
        If π is outside [0, 1] or costs are non-positive.

    Examples
    --------
    Symmetric costs at prior=0.5: threshold should equal the prior.

    >>> bayes_optimal_threshold(0.5, c_fp=1.0, c_fn=1.0)
    0.5

    Rare-positive case with FN 10× more expensive than FP:

    >>> round(bayes_optimal_threshold(0.01, c_fp=1.0, c_fn=10.0), 4)
    0.9083

    Edge cases:

    >>> bayes_optimal_threshold(0.0, c_fp=1.0, c_fn=1.0)
    1.0
    >>> bayes_optimal_threshold(1.0, c_fp=1.0, c_fn=1.0)
    0.0

    Notes
    -----
    Symmetric costs (c_fp == c_fn) collapse the formula to t* = 1 - π.
    Equivalently, when costs are equal the optimal threshold is the *negative*
    prior — predicting 1 whenever P(y=1 | x) > P(y=0).

    Attribution caveat: Elkan 2001 §4 derives the **prior-independent**
    posterior-formula ``t* = c_fp / (c_fp + c_fn)`` for thresholding a
    *Bayes-calibrated* posterior P(y=1 | x). The formula implemented here
    is the **prior-corrected** form for thresholding raw scores at a known
    deployment prior π, which agrees with Elkan only under symmetric costs.
    For our intended use (deployment prior + asymmetric costs) the
    prior-corrected form is what the user wants — but the citation should
    be read as "Elkan 2001 cost-sensitive framework", not literal §4.

    References
    ----------
    .. [#elkan] Elkan, C. "The foundations of cost-sensitive learning." IJCAI
       2001.
    """
    if not 0.0 <= π <= 1.0:
        raise ValueError(f"π (prior) must be in [0, 1], got {π}")
    if c_fp <= 0:
        raise ValueError(f"c_fp must be > 0, got {c_fp}")
    if c_fn <= 0:
        raise ValueError(f"c_fn must be > 0, got {c_fn}")

    if π == 0.0:
        return 1.0
    if π == 1.0:
        return 0.0
    numerator = c_fp * (1.0 - π)
    denominator = numerator + c_fn * π
    return float(numerator / denominator)


@dataclass(frozen=True, slots=True)
class CostMatrix:
    r"""Frozen scaffolding for FP/FN/abstain costs at an assumed prior.

    Pairs a deployment prior with FP/FN costs (and optionally an abstain cost
    for selective classification). The :attr:`bayes_threshold` property
    composes :func:`bayes_optimal_threshold`.

    Parameters
    ----------
    prior : float, optional
        Assumed deployment prevalence P(y=1). Default 0.01.
    fp_cost : float, optional
        Cost of a false positive. Default 1.0.
    fn_cost : float, optional
        Cost of a false negative. Default 10.0.
    abstain_cost : float or None, optional
        Optional cost of abstaining/escalating. ``None`` means abstention is
        not allowed in this policy.
    notes : str, optional
        Free-form annotation.

    Examples
    --------
    >>> cm = CostMatrix(prior=0.5, fp_cost=1.0, fn_cost=1.0)
    >>> cm.bayes_threshold
    0.5
    """

    prior: float = DEFAULT_PRIOR
    fp_cost: float = DEFAULT_FP_COST
    fn_cost: float = DEFAULT_FN_COST
    abstain_cost: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the cost-matrix triple."""
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError(f"prior must be in [0, 1], got {self.prior}")
        if self.fp_cost <= 0:
            raise ValueError(f"fp_cost must be > 0, got {self.fp_cost}")
        if self.fn_cost <= 0:
            raise ValueError(f"fn_cost must be > 0, got {self.fn_cost}")
        if self.abstain_cost is not None and self.abstain_cost < 0:
            raise ValueError(f"abstain_cost must be >= 0 if set, got {self.abstain_cost}")

    @property
    def bayes_threshold(self) -> float:
        """Compose :func:`bayes_optimal_threshold` using this matrix's fields."""
        return bayes_optimal_threshold(self.prior, self.fp_cost, self.fn_cost)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form with the derived threshold."""
        return {
            "prior": self.prior,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "abstain_cost": self.abstain_cost,
            "notes": self.notes,
            "bayes_threshold": self.bayes_threshold,
        }


_SCORE_CLIP_LO = 1e-7
_SCORE_CLIP_HI = 1.0 - 1e-7


def _validate_calibrator_inputs(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shared input validation for the three calibrator fitters."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(f"shape mismatch: y_true {y_true_arr.shape}, y_score {y_score_arr.shape}")
    if y_true_arr.size == 0:
        raise ValueError("y_true is empty")
    if not np.isfinite(y_score_arr).all():
        raise ValueError("y_score contains NaN or inf")
    n_pos = int(y_true_arr.sum())
    if n_pos == 0 or n_pos == y_true_arr.size:
        raise ValueError(
            f"y_true must contain both classes; got n={y_true_arr.size}, n_positive={n_pos}"
        )
    return y_true_arr, y_score_arr


def fit_isotonic_calibrator(
    y_true: np.ndarray, y_score: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Niculescu-Mizil & Caruana 2005 [#nm05]_ isotonic regression.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Predicted probabilities in [0, 1].

    Returns
    -------
    callable
        Maps raw scores to monotonically calibrated probabilities,
        clipped to [0, 1] via ``out_of_bounds="clip"``.

    Raises
    ------
    ValueError
        On shape mismatch, empty input, non-finite scores, or single-class
        ``y_true`` (calibration is degenerate).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = (y + rng.normal(0, 0.5, size=200)).clip(0, 1)
    >>> g = fit_isotonic_calibrator(y, s)
    >>> calibrated = g(s)
    >>> bool(calibrated.min() >= 0.0 and calibrated.max() <= 1.0)
    True

    Notes
    -----
    Isotonic regression fits a monotonic step function from raw scores to
    calibrated probabilities. The fit is non-parametric; on small fitting
    sets it can overfit. Niculescu-Mizil & Caruana 2005 §5 finds isotonic
    competitive with Platt only at **n ≳ 1000**; below ~1000 Platt scaling
    (or :class:`fit_beta_calibrator`) typically generalizes better. Prefer
    Platt / Beta for small calibration sets.

    References
    ----------
    .. [1] Niculescu-Mizil, A. & Caruana, R. "Predicting good probabilities
           with supervised learning." ICML 2005.
    .. [2] Zadrozny, B. & Elkan, C. "Transforming classifier scores into
           accurate multiclass probability estimates." KDD 2002.
    """
    y_true_arr, y_score_arr = _validate_calibrator_inputs(y_true, y_score)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_score_arr, y_true_arr)

    def apply(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).ravel()
        if not np.isfinite(arr).all():
            raise ValueError("scores contains NaN or inf")
        out: np.ndarray = np.asarray(iso.predict(arr), dtype=float)
        return out

    return apply


def _platt_loss_grad(
    ab: np.ndarray, scores: np.ndarray, smoothed_targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """Binomial NLL + gradient under Laplace-smoothed targets (Lin 2007 §2).

    Parameters
    ----------
    ab : np.ndarray, shape (2,)
        Sigmoid parameters ``(a, b)``; the calibrated score is
        :math:`\\sigma(a \\cdot s + b)`.
    scores : np.ndarray, shape (n,)
        Raw scores ``F`` (Platt's notation).
    smoothed_targets : np.ndarray, shape (n,)
        Laplace-smoothed targets ``T`` per Lin 2007 (avoids
        log-of-zero singularities under MLE).

    Returns
    -------
    loss : float
        Total NLL.
    grad : np.ndarray, shape (2,)
        Gradient w.r.t. ``(a, b)``.
    """
    a, b = ab
    z = a * scores + b
    # Stable: NLL_i = T·log(1+exp(-z)) + (1-T)·log(1+exp(z))
    pos_part = smoothed_targets * np.logaddexp(0.0, -z)
    neg_part = (1.0 - smoothed_targets) * np.logaddexp(0.0, z)
    loss = float((pos_part + neg_part).sum())
    # dNLL/dz_i = σ(z_i) - T_i
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))
    err = sigmoid_z - smoothed_targets
    grad = np.array([float(np.dot(err, scores)), float(err.sum())])
    return loss, grad


def fit_platt_calibrator(
    y_true: np.ndarray, y_score: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    r"""Platt 1999 [#platt]_ sigmoid scaling with Lin 2007 [#lin]_ Laplace-smoothed targets.

    Canonical Platt scaling: fits :math:`\sigma(a \cdot s + b)` to maximize
    the binomial likelihood under Laplace-smoothed targets

    .. math::

        T_i = \frac{n_+ + 1}{n_+ + 2} \quad \text{if } y_i = 1, \qquad
        T_i = \frac{1}{n_- + 2} \quad \text{if } y_i = 0,

    where :math:`n_+` and :math:`n_-` are the positive and negative counts.
    The smoothing avoids the MLE singularity at zero/one counts and matches
    :class:`sklearn.calibration._SigmoidCalibration` to within optimizer
    tolerance.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Predicted probabilities or scores.

    Returns
    -------
    callable
        Maps raw scores to calibrated probabilities via the fitted sigmoid
        :math:`\sigma(a \cdot s + b)`.

    Raises
    ------
    ValueError
        On shape mismatch, empty input, non-finite scores, or single-class
        ``y_true``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = (y + rng.normal(0, 0.5, size=200))
    >>> g = fit_platt_calibrator(y, s)
    >>> out = g(s)
    >>> bool(out.min() > 0.0 and out.max() < 1.0)
    True

    Notes
    -----
    Platt scaling fits the two-parameter sigmoid

    .. math:: P(y=1 \mid s) = \sigma(a \cdot s + b) = \frac{1}{1 + \exp(-(a s + b))}

    by maximum-likelihood under Lin 2007's Laplace-smoothed targets. Unlike
    isotonic, the parametric form regularizes small samples but cannot
    correct strongly non-monotone miscalibration.

    Initialization follows sklearn / Lin 2007: ``a₀ = 0``, ``b₀ = log((n_- + 1) / (n_+ + 1))``;
    the optimizer is L-BFGS-B with analytic gradient.

    Behavior change vs eval-toolkit ≤ 0.2.0: previous versions wrapped
    :class:`sklearn.linear_model.LogisticRegression` with default L2
    regularization. v0.3.0 implements canonical Platt directly to match
    :class:`sklearn.calibration._SigmoidCalibration` (Lin 2007). Empirical
    delta on imbalanced data is ~1–3% ECE.

    References
    ----------
    .. [#platt] Platt, J. "Probabilistic outputs for support vector machines
       and comparisons to regularized likelihood methods." Advances in Large
       Margin Classifiers, 1999.
    .. [#lin] Lin, H. T., Lin, C. J., & Weng, R. C. "A note on Platt's
       probabilistic outputs for support vector machines." Machine Learning
       68(3), 2007.
    """
    y_true_arr, y_score_arr = _validate_calibrator_inputs(y_true, y_score)

    n_pos = float(np.sum(y_true_arr > 0))
    n_neg = float(y_true_arr.size - n_pos)
    smoothed = np.empty_like(y_score_arr)
    smoothed[y_true_arr > 0] = (n_pos + 1.0) / (n_pos + 2.0)
    smoothed[y_true_arr <= 0] = 1.0 / (n_neg + 2.0)

    ab_init = np.array([0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))])
    result = minimize(
        _platt_loss_grad,
        ab_init,
        args=(y_score_arr, smoothed),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(f"Platt calibration optimization failed: {result.message}")
    a_fit, b_fit = float(result.x[0]), float(result.x[1])

    def apply(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).ravel()
        if not np.isfinite(arr).all():
            raise ValueError("scores contains NaN or inf")
        z = a_fit * arr + b_fit
        out: np.ndarray = (1.0 / (1.0 + np.exp(-z))).astype(float)
        return out

    return apply


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> dict[str, float]:
    r"""Single-parameter temperature scaling per Guo et al. 2017 [#guo]_.

    Fits a scalar T > 0 on validation logits to minimize negative log-likelihood:

    .. math:: T^* = \arg\min_T - \frac{1}{n}\sum_i \log p_{y_i}(x_i / T)

    where :math:`p_y(x / T) = \mathrm{softmax}(x/T)_y`. T scales the entire
    logit vector before softmax, so accuracy (argmax) is preserved exactly
    while the confidence distribution flattens (T > 1) or sharpens (T < 1).

    Parameters
    ----------
    val_logits : np.ndarray, shape (n, 2)
        Validation logits for the binary classifier (column 0 = negative class,
        column 1 = positive class).
    val_labels : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    bounds : tuple of float, optional
        ``(lo, hi)`` bracket for ``T``. Default ``(0.05, 20.0)``.

    Returns
    -------
    dict
        Keys: ``temperature`` (T*), ``nll_pre`` (NLL at T=1), ``nll_post``
        (NLL at T=T*), ``improvement`` (nll_pre - nll_post; ≥ 0 always).

    Raises
    ------
    ValueError
        If ``val_logits`` shape is not (n, 2), shapes mismatch, or labels are
        not binary.
    RuntimeError
        If the bounded scalar optimizer fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # synthesize uncalibrated logits with known T_true = 3.0
    >>> base = rng.normal(size=(500, 2))
    >>> labels = (base[:, 1] > base[:, 0]).astype(int)
    >>> logits = base * 3.0  # makes scores overconfident
    >>> result = fit_temperature(logits, labels)
    >>> 0.05 <= result['temperature'] <= 20.0
    True
    >>> result['nll_post'] <= result['nll_pre']  # always non-increasing
    True

    Notes
    -----
    Temperature scaling preserves accuracy exactly because dividing all
    logits by the same scalar does not change the argmax. It only rescales
    the *confidence* (max softmax probability), which is what miscalibration
    in modern overconfident networks measures.

    References
    ----------
    .. [#guo] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On
       calibration of modern neural networks." ICML 2017. arXiv:1706.04599.
    """
    if val_logits.ndim != 2 or val_logits.shape[1] != 2:
        raise ValueError(f"val_logits must be (n, 2), got shape {val_logits.shape}")
    if val_logits.shape[0] != val_labels.shape[0]:
        raise ValueError(
            f"length mismatch: logits {val_logits.shape[0]} vs labels {val_labels.shape[0]}"
        )
    if val_logits.shape[0] == 0:
        raise ValueError("val_logits is empty")
    if not np.isfinite(val_logits).all():
        raise ValueError("val_logits contains NaN or inf")
    if not set(np.unique(val_labels).tolist()).issubset({0, 1}):
        raise ValueError("val_labels must be binary (0/1)")
    n_pos = int(np.sum(val_labels))
    if n_pos == 0 or n_pos == val_labels.shape[0]:
        raise ValueError(
            f"val_labels must contain both classes; got n={val_labels.shape[0]}, "
            f"n_positive={n_pos}"
        )

    nll_pre = _negative_log_likelihood(1.0, val_logits, val_labels)
    res = minimize_scalar(
        _negative_log_likelihood,
        bounds=bounds,
        method="bounded",
        args=(val_logits, val_labels),
    )
    if not res.success:
        raise RuntimeError(f"temperature optimization failed: {res.message}")
    t_opt = float(res.x)
    nll_post = _negative_log_likelihood(t_opt, val_logits, val_labels)
    return {
        "temperature": t_opt,
        "nll_pre": nll_pre,
        "nll_post": nll_post,
        "improvement": nll_pre - nll_post,
    }


def _negative_log_likelihood(t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """NLL of softmax(logits / T) against true labels."""
    if t <= 0:
        return float("inf")
    log_probs = log_softmax(logits / t, axis=-1)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature_oracle(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, Callable[[np.ndarray], np.ndarray]]:
    r"""**DIAGNOSTIC ONLY** — fit-on-test oracle T-scaling per Guo et al. 2017 [#guo]_.

    .. warning::

        **Do not use this function as a deployment policy.** It fits ``T``
        on the same data the returned callable scores — the canonical
        "fit-on-test" methodological pitfall. ECE measured on the fitted
        scores is systematically **under**-estimated, sometimes by 50% or
        more (Vaicenavicius 2019 §3, Kumar 2019 §5, Roelofs 2022). Use
        :func:`fit_temperature` (fit on a separate validation set) for
        deployment; use this function only to compute a diagnostic
        upper bound on what *any* single-T recalibration could achieve
        if T were chosen optimally per slice.

    Internally inverts probabilities to logits via :math:`\log(p / (1 - p))`,
    fits T to minimize NLL on the T-scaled logits, then exposes a callable
    that applies :math:`\sigma(\mathrm{logit} / T)`.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary labels in {0, 1}.
    y_score : np.ndarray, shape (n,)
        Predicted probabilities in (0, 1). Scores at the extremes {0, 1} are
        clipped to [1e-7, 1-1e-7] so the logit inversion is finite.

    Returns
    -------
    tuple
        ``(T_optimal, apply)`` where ``apply`` maps any input probability array
        through :math:`\sigma(\mathrm{logit}(p) / T_{optimal})`.

    Raises
    ------
    ValueError
        On shape mismatch, empty input, non-finite scores, or single-class
        ``y_true``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.integers(0, 2, size=200)
    >>> s = (y + rng.normal(0, 0.5, size=200)).clip(0.01, 0.99)
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", UserWarning)
    ...     T_opt, apply = fit_temperature_oracle(y, s)
    >>> T_opt > 0
    True
    """
    import warnings as _warnings  # noqa: PLC0415  (deferred to keep top of file lean)

    _warnings.warn(
        "fit_temperature_oracle is fit-on-test and produces an under-estimated "
        "ECE; use fit_temperature with a held-out validation set for deployment. "
        "This warning may be suppressed in test contexts: "
        "`warnings.simplefilter('ignore', UserWarning)`.",
        UserWarning,
        stacklevel=2,
    )
    y_true_arr, y_score_arr = _validate_calibrator_inputs(y_true, y_score)

    def _logits_from_probs(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, _SCORE_CLIP_LO, _SCORE_CLIP_HI)
        out: np.ndarray = np.log(clipped / (1.0 - clipped))
        return out

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out: np.ndarray = 1.0 / (1.0 + np.exp(-z))
        return out

    logits = _logits_from_probs(y_score_arr)

    def nll_at_t(t: float) -> float:
        if t <= 0:
            return float("inf")
        scaled = logits / t
        # Stable log-sigmoid via softplus identity.
        log_p1 = -np.logaddexp(0.0, -scaled)
        log_p0 = -np.logaddexp(0.0, scaled)
        return float(-(y_true_arr * log_p1 + (1 - y_true_arr) * log_p0).sum())

    result = minimize_scalar(
        nll_at_t,
        bounds=(0.05, 20.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    t_optimal = float(result.x)

    def apply(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).ravel()
        if not np.isfinite(arr).all():
            raise ValueError("scores contains NaN or inf")
        scaled = _logits_from_probs(arr) / t_optimal
        out: np.ndarray = _sigmoid(scaled).astype(float)
        return out

    return t_optimal, apply
