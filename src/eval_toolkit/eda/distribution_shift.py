"""Covariate-shift quantification: proxy-A-distance, MMD, kNN purity.

This module is **Job-3** of the EDA-first research program — the
distribution-shift companion to the Job-2 lexical diagnostics
(:mod:`eval_toolkit.eda.lexical_association`). It quantifies *how far apart* two
text populations sit in a feature space, per the methods the prompt-injection
post-mortem identified as the ones that would have flagged the OOD wall *before*
training:

- **Proxy-A-distance (PAD)** — Ben-David et al. (2006, 2010). A linear domain
  classifier's cross-validated error becomes ``PAD = 2(1 − 2ε)``: ``0`` =
  indistinguishable, ``2`` = perfectly separable. Uses a **linear** classifier
  with **fixed strong regularization** + **k-fold CV** held-out error — *not* the
  popular high-``C`` RBF-SVM-on-``predict_proba`` recipe, which overfits to
  ``PAD ≈ 2`` at small ``n``.
- **Maximum Mean Discrepancy (MMD)** — Gretton et al. (2012). The **unbiased**
  RBF-kernel U-statistic with a **median-heuristic bandwidth** and a
  **permutation test** p-value (Phipson & Smyth 2010: ``p = (1 + count) / (B +
  1)``, never zero).
- **kNN purity** — the mean fraction of each point's ``k`` nearest neighbours
  sharing its domain label (``0.5`` ≈ mixed/no-shift for a balanced pair, ``1.0``
  = cleanly separated).

Critical caveats (interpret accordingly — these are *pre-registered* in
``portfolio:experiments/eda/OOD_WALL_PREDICTION/criteria.md``):

- PAD / MMD measure **covariate** shift only. Distance is **necessary but not
  sufficient** for OOD collapse (Kpotufe & Martinet 2018; Zhao et al. 2019) — the
  wall is shortcut-mediated, so these must be *fused* with the Job-2
  shortcut-exposure signals, never used alone.
- A **non-significant** MMD p-value is **not** evidence of "no shift" — kernel
  two-sample power decays with dimension and small ``n`` (Ramdas et al. 2015).
- **Cross-dataset** distances conflate covariate shift with label-semantics
  mismatch; treat them **ordinally**, with confound controls, never as a
  calibrated covariate distance.

Stability tier
--------------
``eval_toolkit.eda.*`` — **Tier-2** per ADR 0003. The public functions take
**feature matrices** (:class:`numpy.ndarray`), so the module is base-install-safe
(NumPy + SciPy + scikit-learn). To embed text first, use
:func:`eval_toolkit.embeddings.make_minilm_embedder` (the optional
``[embeddings]`` extra) or any vectorizer, then pass the arrays here. Import
explicitly::

    from eval_toolkit.eda import distribution_shift, proxy_a_distance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

__all__ = [
    "DEFAULT_KNN_K",
    "DEFAULT_MMD_PERMUTATIONS",
    "DEFAULT_PAD_C",
    "DEFAULT_PAD_FOLDS",
    "DistributionShiftResult",
    "KnnPurityResult",
    "MmdResult",
    "PadResult",
    "distribution_shift",
    "knn_purity",
    "maximum_mean_discrepancy",
    "median_bandwidth",
    "proxy_a_distance",
]

# Fixed strong regularization for the PAD domain classifier. A small C (strong
# regularization) is deliberate: it prevents the linear classifier from
# memorizing small samples into a spurious PAD ≈ 2 (the documented PAD pitfall).
DEFAULT_PAD_C: Final[float] = 0.1

# Cross-validation folds for the PAD domain-classifier error estimate.
DEFAULT_PAD_FOLDS: Final[int] = 5

# Permutation count for the MMD null. ≥ 1000 for a usable p-value resolution
# (100 floors p at ~1/101); the portfolio pre-registers ≥ 1000.
DEFAULT_MMD_PERMUTATIONS: Final[int] = 1000

# Neighbours for kNN purity (excludes self).
DEFAULT_KNN_K: Final[int] = 5

# Cap on the number of points used to estimate the median-heuristic bandwidth —
# the full pairwise-distance matrix is O(n²); a capped subsample is enough for a
# stable median.
_BANDWIDTH_MAX_SAMPLES: Final[int] = 1000


def _validate_pair(x_a: np.ndarray, x_b: np.ndarray) -> tuple[int, int]:
    """Validate two feature matrices and return ``(n_a, n_b)``.

    Raises
    ------
    ValueError
        If either array is not 2-D, is empty, or the feature dimensions differ.
    """
    if x_a.ndim != 2 or x_b.ndim != 2:
        raise ValueError(f"x_a and x_b must be 2-D, got ndim {x_a.ndim} and {x_b.ndim}")
    if x_a.shape[0] == 0 or x_b.shape[0] == 0:
        raise ValueError("x_a and x_b must both be non-empty")
    if x_a.shape[1] != x_b.shape[1]:
        raise ValueError(
            f"feature-dimension mismatch: x_a has {x_a.shape[1]}, x_b has {x_b.shape[1]}"
        )
    return x_a.shape[0], x_b.shape[0]


@dataclass(frozen=True, slots=True)
class PadResult:
    """Proxy-A-distance between two feature populations.

    Parameters
    ----------
    pad : float
        ``2 * (1 - 2 * error)``, clamped to ``[0.0, 2.0]``. ``0`` =
        indistinguishable domains; ``2`` = perfectly separable.
    domain_classifier_error : float
        The cross-validated 0/1 error of the linear domain classifier.
    n_folds : int
        Effective number of CV folds used.
    c : float
        The (fixed) inverse-regularization strength of the linear classifier.
    n_a, n_b : int
        Sample counts.
    ci_low, ci_high : float or None
        Percentile bootstrap CI bounds on ``pad`` (``None`` if not bootstrapped).
    """

    pad: float
    domain_classifier_error: float
    n_folds: int
    c: float
    n_a: int
    n_b: int
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the result."""
        return {
            "pad": self.pad,
            "domain_classifier_error": self.domain_classifier_error,
            "n_folds": self.n_folds,
            "c": self.c,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True, slots=True)
class MmdResult:
    """Maximum Mean Discrepancy between two feature populations.

    Parameters
    ----------
    mmd_squared : float
        The unbiased RBF-kernel MMD² U-statistic (may be slightly negative under
        the null — retained, not clamped).
    bandwidth : float
        The RBF bandwidth σ used (``gamma = 1 / (2 σ²)``).
    p_value : float
        Permutation-test p-value, ``(1 + count) / (n_permutations + 1)``.
    n_permutations : int
        Number of label permutations.
    n_a, n_b : int
        Sample counts.
    ci_low, ci_high : float or None
        Percentile bootstrap CI bounds on ``mmd_squared`` (``None`` if not
        bootstrapped).
    """

    mmd_squared: float
    bandwidth: float
    p_value: float
    n_permutations: int
    n_a: int
    n_b: int
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the result."""
        return {
            "mmd_squared": self.mmd_squared,
            "bandwidth": self.bandwidth,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True, slots=True)
class KnnPurityResult:
    """kNN domain purity between two feature populations.

    Parameters
    ----------
    mean_purity : float
        Mean fraction of each point's ``k`` nearest neighbours that share its
        domain label. ``~0.5`` (balanced) = mixed / no shift; ``1.0`` = cleanly
        separated.
    k : int
        Neighbours considered (excludes self).
    n_a, n_b : int
        Sample counts.
    """

    mean_purity: float
    k: int
    n_a: int
    n_b: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the result."""
        return {
            "mean_purity": self.mean_purity,
            "k": self.k,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


@dataclass(frozen=True, slots=True)
class DistributionShiftResult:
    """Combined PAD + MMD + kNN-purity shift report for one pair of populations."""

    pad: PadResult
    mmd: MmdResult
    knn: KnnPurityResult

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of all three sub-results."""
        return {
            "pad": self.pad.to_dict(),
            "mmd": self.mmd.to_dict(),
            "knn": self.knn.to_dict(),
        }


def median_bandwidth(
    x: np.ndarray, *, max_samples: int = _BANDWIDTH_MAX_SAMPLES, random_state: int = 0
) -> float:
    """Median-heuristic RBF bandwidth σ = median pairwise Euclidean distance.

    For ``n > max_samples`` rows a deterministic random subsample is used (the
    full pairwise matrix is O(n²)). The returned σ is meant to be computed once
    on a **frozen pooled reference** and held fixed across comparisons.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n, d)`` feature matrix (typically the pooled A∪B reference).
    max_samples : int, optional
        Subsample cap for the pairwise-distance computation.
    random_state : int, optional
        Seed for the subsample.

    Returns
    -------
    float
        The median pairwise distance.

    Raises
    ------
    ValueError
        If ``x`` has fewer than 2 rows, or all points are identical (σ = 0).
    """
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("median_bandwidth needs a 2-D array with >= 2 rows")
    if x.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    sigma = float(np.median(pdist(x)))
    if sigma <= 0.0:
        raise ValueError("degenerate bandwidth: all pooled points are identical (σ = 0)")
    return sigma


def proxy_a_distance(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    c: float = DEFAULT_PAD_C,
    n_folds: int = DEFAULT_PAD_FOLDS,
    n_bootstrap: int = 0,
    random_state: int = 0,
) -> PadResult:
    """Proxy-A-distance via a CV'd linear domain classifier (Ben-David 2010).

    Trains a linear SVM to separate corpus A (label 0) from corpus B (label 1)
    and converts its **cross-validated** 0/1 error ε into ``PAD = 2(1 − 2ε)``.
    A high error (≈ 0.5, domains indistinguishable) → PAD ≈ 0; a low error
    (domains separable) → PAD → 2. The classifier is **linear** with **fixed
    strong regularization** (small ``c``) so it cannot trivially memorize small
    samples into a spurious PAD ≈ 2.

    To get the within-population sanity floor (PAD ≈ 0), call this with two random
    halves of a *single* population.

    Parameters
    ----------
    x_a, x_b : numpy.ndarray
        ``(n, d)`` feature matrices, same ``d``.
    c : float, optional
        Inverse-regularization strength of :class:`~sklearn.svm.LinearSVC`.
        Default :data:`DEFAULT_PAD_C` (small = strong regularization).
    n_folds : int, optional
        Requested stratified CV folds; clamped to ``min(n_folds, n_a, n_b)`` and
        must end ``>= 2``. Default :data:`DEFAULT_PAD_FOLDS`.
    n_bootstrap : int, optional
        If ``> 0``, resample within each corpus that many times to attach a
        percentile 95% CI to ``pad``. Default ``0`` (skip).
    random_state : int, optional
        Seed for CV shuffling, the classifier, and the bootstrap. Default ``0``.

    Returns
    -------
    PadResult

    Raises
    ------
    ValueError
        On invalid feature matrices, or if fewer than 2 folds are possible
        (each corpus needs >= 2 rows).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(0.0, 1.0, size=(40, 5))
    >>> b = rng.normal(6.0, 1.0, size=(40, 5))  # far-separated
    >>> r = proxy_a_distance(a, b, n_folds=4)
    >>> r.pad > 1.5  # clearly separable
    True
    >>> # within-population split → PAD floor near 0
    >>> floor = proxy_a_distance(a[:20], a[20:], n_folds=4)
    >>> floor.pad < 0.6
    True
    """
    n_a, n_b = _validate_pair(x_a, x_b)
    folds = min(n_folds, n_a, n_b)
    if folds < 2:
        raise ValueError(
            f"need >= 2 samples per corpus for >= 2 CV folds, got n_a={n_a}, n_b={n_b}"
        )

    pad, error = _pad_once(x_a, x_b, c=c, folds=folds, random_state=random_state)

    ci_low: float | None = None
    ci_high: float | None = None
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        boots = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            ia = rng.integers(0, n_a, size=n_a)
            ib = rng.integers(0, n_b, size=n_b)
            boots[i], _ = _pad_once(x_a[ia], x_b[ib], c=c, folds=folds, random_state=random_state)
        ci_low = float(np.percentile(boots, 2.5))
        ci_high = float(np.percentile(boots, 97.5))

    return PadResult(
        pad=pad,
        domain_classifier_error=error,
        n_folds=folds,
        c=c,
        n_a=n_a,
        n_b=n_b,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _pad_once(
    x_a: np.ndarray, x_b: np.ndarray, *, c: float, folds: int, random_state: int
) -> tuple[float, float]:
    """Single PAD estimate; returns ``(pad, cv_error)``."""
    x = np.vstack([x_a, x_b])
    y = np.concatenate([np.zeros(x_a.shape[0], dtype=int), np.ones(x_b.shape[0], dtype=int)])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    clf = LinearSVC(C=c, random_state=random_state)
    preds = cross_val_predict(clf, x, y, cv=cv)
    error = float(np.mean(preds != y))
    pad = 2.0 * (1.0 - 2.0 * error)
    # Clamp: a below-chance classifier (error > 0.5) means "indistinguishable",
    # i.e. PAD 0, not a negative distance.
    pad = float(min(max(pad, 0.0), 2.0))
    return pad, error


def _mmd2_from_kernel(k: np.ndarray, m: int) -> float:
    """Unbiased MMD² U-statistic from a pooled kernel matrix.

    ``k`` is the ``(m + n, m + n)`` Gram matrix with the first ``m`` rows/cols =
    corpus A. Diagonal terms are dropped (the unbiased estimator).
    """
    n = k.shape[0] - m
    k_xx = k[:m, :m]
    k_yy = k[m:, m:]
    k_xy = k[:m, m:]
    term_a = (float(k_xx.sum()) - float(np.trace(k_xx))) / (m * (m - 1))
    term_b = (float(k_yy.sum()) - float(np.trace(k_yy))) / (n * (n - 1))
    term_c = 2.0 * float(k_xy.sum()) / (m * n)
    return float(term_a + term_b - term_c)


def maximum_mean_discrepancy(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    bandwidth: float | None = None,
    n_permutations: int = DEFAULT_MMD_PERMUTATIONS,
    n_bootstrap: int = 0,
    random_state: int = 0,
) -> MmdResult:
    """Unbiased RBF-kernel MMD² with a permutation-test p-value (Gretton 2012).

    Computes the unbiased MMD² U-statistic with an RBF kernel, then a
    permutation test: pool the samples, repeatedly re-split into sizes
    ``(n_a, n_b)``, and count how often the permuted MMD² meets or exceeds the
    observed value. The p-value is ``(1 + count) / (n_permutations + 1)``
    (Phipson & Smyth 2010 — never zero).

    Parameters
    ----------
    x_a, x_b : numpy.ndarray
        ``(n, d)`` feature matrices, same ``d``. Each needs ``>= 2`` rows
        (the unbiased estimator divides by ``n(n-1)``).
    bandwidth : float, optional
        RBF bandwidth σ (``gamma = 1/(2σ²)``). If ``None``, the median heuristic
        on the pooled A∪B sample is used. Pass a frozen reference σ to keep MMD
        comparable across folds.
    n_permutations : int, optional
        Label permutations for the null. Default :data:`DEFAULT_MMD_PERMUTATIONS`.
    n_bootstrap : int, optional
        If ``> 0``, resample within each corpus to attach a percentile 95% CI to
        ``mmd_squared``. Default ``0`` (skip).
    random_state : int, optional
        Seed for the permutations and bootstrap. Default ``0``.

    Returns
    -------
    MmdResult

    Raises
    ------
    ValueError
        On invalid feature matrices, ``n_permutations < 1``, fewer than 2 rows in
        either corpus, or a degenerate (σ = 0) bandwidth.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(0.0, 1.0, size=(30, 4))
    >>> b = rng.normal(5.0, 1.0, size=(30, 4))
    >>> r = maximum_mean_discrepancy(a, b, n_permutations=200)
    >>> r.mmd_squared > 0.1 and r.p_value < 0.05  # clear, significant shift
    True
    >>> same = maximum_mean_discrepancy(a, a.copy(), n_permutations=200)
    >>> same.p_value > 0.05  # identical → not significant
    True
    """
    n_a, n_b = _validate_pair(x_a, x_b)
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    if n_a < 2 or n_b < 2:
        raise ValueError(f"each corpus needs >= 2 rows for unbiased MMD, got {n_a}, {n_b}")

    pooled = np.vstack([x_a, x_b])
    sigma = (
        bandwidth if bandwidth is not None else median_bandwidth(pooled, random_state=random_state)
    )
    if sigma <= 0.0:
        raise ValueError("bandwidth must be > 0")
    gamma = 1.0 / (2.0 * sigma * sigma)
    k = rbf_kernel(pooled, gamma=gamma)

    observed = _mmd2_from_kernel(k, n_a)

    rng = np.random.default_rng(random_state)
    total = n_a + n_b
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(total)
        kp = k[np.ix_(perm, perm)]
        if _mmd2_from_kernel(kp, n_a) >= observed:
            count += 1
    p_value = (1.0 + count) / (n_permutations + 1.0)

    ci_low: float | None = None
    ci_high: float | None = None
    if n_bootstrap > 0:
        boots = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            ia = rng.integers(0, n_a, size=n_a)
            ib = rng.integers(0, n_b, size=n_b)
            kb = rbf_kernel(np.vstack([x_a[ia], x_b[ib]]), gamma=gamma)
            boots[i] = _mmd2_from_kernel(kb, n_a)
        ci_low = float(np.percentile(boots, 2.5))
        ci_high = float(np.percentile(boots, 97.5))

    return MmdResult(
        mmd_squared=float(observed),
        bandwidth=float(sigma),
        p_value=float(p_value),
        n_permutations=n_permutations,
        n_a=n_a,
        n_b=n_b,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def knn_purity(x_a: np.ndarray, x_b: np.ndarray, *, k: int = DEFAULT_KNN_K) -> KnnPurityResult:
    """Mean fraction of each point's ``k`` nearest neighbours sharing its domain.

    Pools A (label 0) and B (label 1), finds each point's ``k`` nearest
    neighbours (excluding itself), and averages the fraction that share the
    point's label. For a balanced pair, ``~0.5`` means the domains are mixed (no
    shift) and ``1.0`` means they are cleanly separated.

    Parameters
    ----------
    x_a, x_b : numpy.ndarray
        ``(n, d)`` feature matrices, same ``d``.
    k : int, optional
        Neighbours per point (excludes self). Clamped to ``n_a + n_b - 1``.
        Default :data:`DEFAULT_KNN_K`.

    Returns
    -------
    KnnPurityResult

    Raises
    ------
    ValueError
        On invalid feature matrices or ``k < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(0.0, 1.0, size=(30, 4))
    >>> b = rng.normal(8.0, 1.0, size=(30, 4))  # far apart
    >>> knn_purity(a, b, k=5).mean_purity > 0.95
    True
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_a, n_b = _validate_pair(x_a, x_b)
    total = n_a + n_b
    eff_k = min(k, total - 1)
    pooled = np.vstack([x_a, x_b])
    labels = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    nn = NearestNeighbors(n_neighbors=eff_k + 1).fit(pooled)
    neighbours = nn.kneighbors(pooled, return_distance=False)[:, 1:]  # drop self
    same = labels[neighbours] == labels[:, None]
    return KnnPurityResult(
        mean_purity=float(same.mean()),
        k=eff_k,
        n_a=n_a,
        n_b=n_b,
    )


def distribution_shift(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    pad_c: float = DEFAULT_PAD_C,
    pad_folds: int = DEFAULT_PAD_FOLDS,
    bandwidth: float | None = None,
    n_permutations: int = DEFAULT_MMD_PERMUTATIONS,
    knn_k: int = DEFAULT_KNN_K,
    n_bootstrap: int = 0,
    random_state: int = 0,
) -> DistributionShiftResult:
    """Run PAD + MMD + kNN-purity on one pair of feature populations.

    Convenience orchestrator returning all three shift measures in a single
    :class:`DistributionShiftResult`. See the per-metric functions for the
    parameter semantics and caveats (distance is necessary-not-sufficient for
    OOD collapse; fuse with shortcut-exposure).

    Parameters
    ----------
    x_a, x_b : numpy.ndarray
        ``(n, d)`` feature matrices, same ``d``.
    pad_c, pad_folds, bandwidth, n_permutations, knn_k, n_bootstrap, random_state
        Forwarded to :func:`proxy_a_distance`, :func:`maximum_mean_discrepancy`,
        and :func:`knn_purity`.

    Returns
    -------
    DistributionShiftResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(0.0, 1.0, size=(30, 4))
    >>> b = rng.normal(5.0, 1.0, size=(30, 4))
    >>> res = distribution_shift(a, b, pad_folds=4, n_permutations=200)
    >>> res.pad.pad > 1.0 and res.mmd.p_value < 0.05 and res.knn.mean_purity > 0.9
    True
    """
    pad = proxy_a_distance(
        x_a, x_b, c=pad_c, n_folds=pad_folds, n_bootstrap=n_bootstrap, random_state=random_state
    )
    mmd = maximum_mean_discrepancy(
        x_a,
        x_b,
        bandwidth=bandwidth,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    knn = knn_purity(x_a, x_b, k=knn_k)
    return DistributionShiftResult(pad=pad, mmd=mmd, knn=knn)
