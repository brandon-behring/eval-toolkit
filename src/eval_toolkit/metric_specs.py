"""First-party `MetricSpec` instances for v0.46 :func:`~eval_toolkit.scorecard`.

The namespace ships **threshold-free** specs only (Decision R). Each
unparameterized metric is a module-level singleton; parameterized metrics
(currently just the ECE family) are factory callables with LRU-cached
identity (so ``ece(n_bins=15) is ece(n_bins=15)``).

Threshold-dependent metrics (F1, accuracy, precision, recall) are
intentionally **NOT** in this namespace — they need a threshold and live in
:func:`~eval_toolkit.metrics.metrics_at_threshold` +
:class:`~eval_toolkit.thresholds.ThresholdSelector`. Adding them to
`scorecard()` would require an operating-point spec family + CI-policy
contract (fix-vs-refit threshold per resample); deferred to v1.x if user
demand surfaces.

Examples
--------
>>> from eval_toolkit import scorecard, metric_specs as ms
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> y = rng.integers(0, 2, 200)
>>> s = rng.random(200)
>>> r = scorecard(y, s, metrics=[
...     ms.pr_auc, ms.roc_auc, ms.brier, ms.ece(n_bins=15),
... ], bootstrap=False)
>>> sorted(r.keys())
['brier', 'ece_n_bins_15_strategy_uniform', 'pr_auc', 'roc_auc']

Identity stability for singletons:

>>> ms.pr_auc is ms.pr_auc
True

Identity stability for parameterized factories (LRU-cached):

>>> ms.ece(n_bins=15) is ms.ece(n_bins=15)
True
>>> ms.ece(n_bins=15) is ms.ece(n_bins=10)
False
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from eval_toolkit._scorecard import MetricSpec
from eval_toolkit.metrics import brier_score as _brier_score
from eval_toolkit.metrics import expected_calibration_error as _ece_uniform
from eval_toolkit.metrics import (
    expected_calibration_error_equal_mass as _ece_equal_mass,
)
from eval_toolkit.metrics import pr_auc as _pr_auc
from eval_toolkit.metrics import roc_auc as _roc_auc

__all__ = [
    "brier",
    "ece",
    "make_spec_name",
    "pr_auc",
    "roc_auc",
]


# Strategy choices for ECE; both shipped as part of v0.46.
ECEStrategy = Literal["uniform", "quantile"]


# ─────────────────────────────────────────────────────────────────────────────
# Unparameterized singleton specs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _PrAucSpec:
    """Internal :class:`MetricSpec` for area under the PR curve."""

    name: str = "pr_auc"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(_pr_auc(y_true, y_score))


@dataclass(frozen=True, slots=True)
class _RocAucSpec:
    """Internal :class:`MetricSpec` for area under the ROC curve."""

    name: str = "roc_auc"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(_roc_auc(y_true, y_score))


@dataclass(frozen=True, slots=True)
class _BrierSpec:
    """Internal :class:`MetricSpec` for the Brier score."""

    name: str = "brier"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(_brier_score(y_true, y_score))


#: PR-AUC singleton spec (area under the precision-recall curve).
pr_auc: MetricSpec = _PrAucSpec()

#: ROC-AUC singleton spec (area under the receiver-operating-characteristic curve).
roc_auc: MetricSpec = _RocAucSpec()

#: Brier-score singleton spec (mean squared error of probabilistic predictions).
brier: MetricSpec = _BrierSpec()


# ─────────────────────────────────────────────────────────────────────────────
# Parameterized factories (LRU-cached for identity stability)
# ─────────────────────────────────────────────────────────────────────────────


# Valid strategy values for ECE specs. Locked at v0.46.1 to prevent the
# Round 6 R6-F1 footgun where `ece(strategy="typo")` silently dispatched to
# quantile ECE and returned a scorecard cell with status="ok" under an
# invalid key. See `docs/source/audit_findings.md` Round 6.
_ECE_VALID_STRATEGIES: frozenset[str] = frozenset({"uniform", "quantile"})


def _validate_ece_strategy(strategy: str) -> None:
    """Validate ECE strategy value; raise ValueError with context if invalid.

    Shared between the factory (eager validation) and ``_EceSpec.compute`` (defence in
    depth for direct construction paths that bypass the factory).
    """
    if strategy not in _ECE_VALID_STRATEGIES:
        raise ValueError(f"ECE strategy must be 'uniform' or 'quantile'; got {strategy!r}")


@dataclass(frozen=True, slots=True)
class _EceSpec:
    """Internal :class:`MetricSpec` for expected calibration error.

    Carries the bin count + strategy in the spec; the name encodes both so
    multiple ECE specs in the same scorecard get distinct keys.
    """

    n_bins: int
    strategy: ECEStrategy

    @property
    def name(self) -> str:
        """Stable key: ``"ece_n_bins_{n}_strategy_{s}"``, alphabetized kwargs."""
        return f"ece_n_bins_{self.n_bins}_strategy_{self.strategy}"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        # Defence-in-depth strategy validation — the factory validates first,
        # but a caller bypassing the factory and constructing `_EceSpec` directly
        # would otherwise produce a wrong-metric scorecard cell silently.
        _validate_ece_strategy(self.strategy)
        if self.strategy == "uniform":
            return float(_ece_uniform(y_true, y_score, n_bins=self.n_bins))
        return float(_ece_equal_mass(y_true, y_score, n_bins=self.n_bins))


@lru_cache(maxsize=128)
def ece(*, n_bins: int = 15, strategy: ECEStrategy = "uniform") -> MetricSpec:
    """Factory for an ECE :class:`MetricSpec` with configurable bins + strategy.

    LRU-cached so identity is stable for repeated calls with the same kwargs:
    ``ece(n_bins=15) is ece(n_bins=15)``. The cache is bounded at 128 entries
    (Decision J memory cap); after that, the least-recently-used entry is
    evicted and the next call with that signature constructs a fresh instance
    (identity no longer holds, but value-equality through ``__eq__`` still does
    since the underlying dataclass is frozen + slots).

    Parameters
    ----------
    n_bins : int, optional
        Number of bins. Default ``15`` (matches Hines et al. / the existing
        ``expected_calibration_error`` default-after-discussion). Must be
        ``>= 2``.
    strategy : {"uniform", "quantile"}, optional
        Bin layout:

        - ``"uniform"`` (default) — equal-width bins on ``[0, 1]``. Dispatches
          to :func:`~eval_toolkit.metrics.expected_calibration_error`.
        - ``"quantile"`` — equal-mass bins on the score quantiles. Dispatches
          to :func:`~eval_toolkit.calibration.expected_calibration_error_equal_mass`.
          Audit-recommended for imbalanced score distributions where uniform
          bins leave low-density regions noisy.

    Returns
    -------
    MetricSpec
        Spec instance with stable ``name`` and ``compute()``.

    Examples
    --------
    >>> ece(n_bins=15).name
    'ece_n_bins_15_strategy_uniform'
    >>> ece(n_bins=10, strategy="quantile").name
    'ece_n_bins_10_strategy_quantile'

    Invalid strategies raise ``ValueError`` eagerly (v0.46.1+; Round 6 R6-F1
    fix — prior to v0.46.1 this silently dispatched to quantile ECE):

    >>> ece(strategy="typo")
    Traceback (most recent call last):
        ...
    ValueError: ECE strategy must be 'uniform' or 'quantile'; got 'typo'

    Raises
    ------
    ValueError
        If ``strategy`` is not in ``{"uniform", "quantile"}``.
    """
    _validate_ece_strategy(strategy)
    return _EceSpec(n_bins=n_bins, strategy=strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Spec-name canonicalization (Decision R6-H — Round 6 Gemini R6-F4)
# ─────────────────────────────────────────────────────────────────────────────


def make_spec_name(prefix: str, **kwargs: object) -> str:
    """Canonicalize a parameterized :class:`MetricSpec` ``name``.

    Convention: alphabetized kwargs joined by underscore, snake-cased. Mirrors
    the v0.46 ECE encoding rule (``ece(n_bins=15, strategy="uniform")`` →
    ``"ece_n_bins_15_strategy_uniform"``).

    Use in custom :class:`MetricSpec` implementations to avoid silent key drift
    when constructor argument order changes — without canonicalization, two
    callers passing the same kwargs in different orders can spawn distinct dict
    keys in :class:`~eval_toolkit.Scorecard`.

    Placement (Decision R6-H, locked at Round 6): exposed via
    :mod:`eval_toolkit.metric_specs.__all__` only; not in the top-level package
    ``__all__``. The Tier-2 additive-only contract (ADR 0003) permits the
    helper to gain optional kwargs (e.g. custom value formatters) without a
    SemVer-major bump.

    Parameters
    ----------
    prefix : str
        Base name — typically the metric family (``"ece"``, ``"pr_auc"``, etc.).
    **kwargs : object
        Spec parameters. Keys are alphabetized; values rendered via
        ``str(value)``. Only finite, repr-stable values are supported.

    Returns
    -------
    str
        Canonicalized spec name. With zero kwargs, returns ``prefix`` unchanged.

    Examples
    --------
    >>> make_spec_name("ece", n_bins=15, strategy="uniform")
    'ece_n_bins_15_strategy_uniform'
    >>> make_spec_name("ece", strategy="uniform", n_bins=15)
    'ece_n_bins_15_strategy_uniform'
    >>> make_spec_name("custom_metric", alpha=0.1, beta=2)
    'custom_metric_alpha_0.1_beta_2'
    >>> make_spec_name("pr_auc")
    'pr_auc'
    """
    if not kwargs:
        return prefix
    parts: list[str] = [prefix]
    for key in sorted(kwargs):
        parts.append(key)
        parts.append(str(kwargs[key]))
    return "_".join(parts)
