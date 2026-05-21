"""Primary metric surface for v1.0 — `scorecard()` plus `MetricSpec` / `MetricResult` / `Scorecard`.

This module is the v1.0 evolution of "give me PR-AUC, ROC-AUC, Brier, ECE on
one slice, with bootstrap CIs, in one call." It supersedes the v0.x pattern of
importing scalar metrics from the top-level package (those are soft-deprecated
at v0.46 and removed at v0.47 per the v1.0 plan).

Key design points (each is a locked decision in the v1.0 plan):

- **Threshold-free first-party specs only** (Decision R). The default
  `metric_specs` namespace ships only metrics whose point estimate is defined
  from ``(y_true, y_score)`` alone: PR-AUC, ROC-AUC, Brier, ECE. F1, accuracy,
  precision, recall remain reachable via `metrics_at_threshold` +
  `ThresholdSelector` — they need a threshold and that's a separate concern
  with its own provenance machinery.

- **Status-aware cells** (Decision S). Each result cell is a
  :class:`MetricResult` carrying ``value | None``, ``status`` ∈
  ``{ok, skipped, error}``, a human-readable ``reason``, and an optional
  :class:`~eval_toolkit.bootstrap.BootstrapCI`. Single-class slices (where
  PR-AUC / ROC-AUC are undefined) produce ``status="skipped"`` rather than
  raising; the rest of the scorecard still computes. This mirrors the
  existing ``MetricState`` vocabulary at ``artifacts.py:30-61``.

- **Type-safe dict-subscript access** (Decision I).
  :class:`Scorecard` is a :class:`~collections.abc.Mapping` ``[str, MetricResult]``.
  Callers access results by string key (``r["pr_auc"].value``), which catches
  typos at runtime (``KeyError``) rather than silently returning ``Any`` like
  attribute-based access would under ``mypy --strict``.

- **Per-cell error isolation**. One metric's failure (whether undefined,
  unexpected exception, or bootstrap-CI inability) does not abort the whole
  scorecard. The failing cell records its state; the rest compute normally.

- **Skipped detection** uses the existing :func:`~eval_toolkit.metrics.is_metric_defined_for_slice`
  primitive (v0.39, closes #39; extended at v0.46 prep to recognize `pr_auc` /
  `roc_auc` aliases). No new public exception, no new Protocol method
  (Decision X.2).

Stability:

- `MetricSpec` is a **Tier-2 Protocol** per Decision M — frozen at v1.0
  modulo additive subprotocols.
- `MetricResult` and `Scorecard` are public stable types. `to_dict()` schema
  is stable; new keys may be added (additive only) in minor releases.

See Also
--------
eval_toolkit.metric_specs : The first-party spec namespace (`pr_auc`,
    `roc_auc`, `brier`, `ece(...)`).
eval_toolkit.artifacts.MetricState : The existing status vocabulary
    (`ok`/`skipped`/`error`) reused here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from eval_toolkit.bootstrap import BootstrapCI, bootstrap_ci
from eval_toolkit.metrics import is_metric_defined_for_slice

_logger = logging.getLogger(__name__)

__all__ = [
    "MetricResult",
    "MetricSpec",
    "Scorecard",
    "scorecard",
]


# Status vocabulary mirrors eval_toolkit.artifacts.MetricState (artifacts.py:30).
MetricStatus = Literal["ok", "skipped", "error"]


@runtime_checkable
class MetricSpec(Protocol):
    """Public Protocol for metric specifications consumed by :func:`scorecard`.

    A spec is identified by a stable string name and knows how to compute its
    point estimate from ``(y_true, y_score)``. Specs are typically created
    via :mod:`eval_toolkit.metric_specs` (e.g. ``metric_specs.pr_auc``,
    ``metric_specs.ece(n_bins=15)``). Custom user specs satisfy the Protocol
    by exposing both members.

    Attributes
    ----------
    name : str
        Stable identifier used as the :class:`Scorecard` key. Parameterized
        specs encode their parameters in the name (alphabetized, snake-cased)
        so ``ece(n_bins=15)`` → ``"ece_n_bins_15"`` and ``ece(n_bins=15)`` and
        ``ece(n_bins=10)`` produce distinct cells.

    Notes
    -----
    Tier-2 Protocol (frozen at v1.0 per Decision M). Method signature
    changes require v2.0; additive subprotocols are permitted in minor
    releases.

    For consistent skipped-status detection, prefer registering the spec's
    name with :data:`~eval_toolkit.metrics.SINGLE_CLASS_INCOMPATIBLE_METRICS`
    if the metric is undefined on single-class slices (ranking metrics like
    PR-AUC / ROC-AUC). The default :class:`Scorecard` flow uses
    :func:`~eval_toolkit.metrics.is_metric_defined_for_slice` for this check;
    specs whose names aren't in that registry default to "defined" and the
    spec's ``compute()`` exception (if any) yields ``status="error"``.

    Examples
    --------
    Built-in spec:

    >>> import numpy as np
    >>> from eval_toolkit import metric_specs as ms
    >>> isinstance(ms.pr_auc, object) and ms.pr_auc.name == "pr_auc"
    True

    Custom user spec (structural Protocol satisfaction):

    >>> class MyCustomSpec:
    ...     name = "always_half"
    ...     def compute(self, y_true, y_score):
    ...         return 0.5
    >>> from eval_toolkit import MetricSpec
    >>> isinstance(MyCustomSpec(), MetricSpec)
    True
    """

    @property
    def name(self) -> str:  # pragma: no cover
        """Stable identifier for this spec; used as the :class:`Scorecard` key.

        Implementations may expose ``name`` as a frozen-dataclass field (which
        behaves as a read-only attribute under Protocol structural checks) or
        as a real ``@property`` — both satisfy the contract. Mirrors the
        :class:`~eval_toolkit.thresholds.ThresholdSelector` Protocol's
        ``criterion`` pattern.
        """
        ...

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:  # pragma: no cover
        """Return the point estimate as a float (no CI).

        :func:`scorecard` adds bootstrap CI separately via
        :func:`~eval_toolkit.bootstrap.bootstrap_ci` when ``bootstrap=True``.

        Implementations should raise ``ValueError`` (or any specific exception)
        on internal validation failure; :func:`scorecard` catches and converts
        to ``MetricResult(status="error", reason=str(exc))``.
        """
        ...


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Per-metric scorecard cell with status + reason + optional CI.

    Reuses the ``ok``/``skipped``/``error`` vocabulary from
    :class:`~eval_toolkit.artifacts.MetricState` (Decision S).

    Parameters
    ----------
    value : float or None
        Point estimate. ``None`` exactly when ``status != "ok"``.
    status : Literal["ok", "skipped", "error"]
        Outcome state. ``"ok"`` = computed successfully; ``"skipped"`` =
        metric is mathematically undefined on this slice (e.g., PR-AUC on
        single-class); ``"error"`` = unexpected exception from ``compute()``.
    reason : str
        Human-readable explanation. Empty when ``status == "ok"``; populated
        for ``skipped`` (cites the undefined-on condition) and ``error``
        (the exception ``str(exc)``).
    ci : BootstrapCI or None
        Bootstrap confidence interval when both ``bootstrap=True`` was passed
        and the point estimate could be computed without exception. ``None``
        when ``bootstrap=False``, when ``status != "ok"``, or when the
        bootstrap itself couldn't run (e.g., ``n < 10`` floor in
        :func:`bootstrap_ci`). The reason for a missing CI on an
        otherwise-``ok`` cell is recorded in ``reason``.
    """

    value: float | None
    status: MetricStatus
    reason: str = ""
    ci: BootstrapCI | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization. Stable schema (additive only)."""
        d: dict[str, Any] = {
            "value": self.value,
            "status": self.status,
            "reason": self.reason,
        }
        if self.ci is not None:
            d["ci"] = self.ci.to_dict()
        return d


class Scorecard(Mapping[str, MetricResult]):
    """Read-only :class:`~collections.abc.Mapping` ``[str, MetricResult]`` over a slice's metric battery.

    Returned by :func:`scorecard`. Subscript access by metric spec name:

    >>> # r = scorecard(y_true, y_score, metrics=[ms.pr_auc, ms.brier])
    >>> # r["pr_auc"].value, r["pr_auc"].ci

    Type-safe under ``mypy --strict`` (no ``__getattr__`` magic; typos raise
    ``KeyError`` at runtime).

    Notes
    -----
    The container is read-only — there is no ``__setitem__``. To compose
    multiple scorecards (e.g., per-slice grids), aggregate at the caller
    side (concat the ``to_dict()`` outputs or build a pandas DataFrame from
    the rows).

    See Also
    --------
    Scorecard.to_dict : JSON-friendly serialization.
    Scorecard.to_pandas : One-row DataFrame view.
    """

    __slots__ = ("_results",)

    def __init__(self, results: Mapping[str, MetricResult]) -> None:
        # Defensive copy: callers cannot mutate the container through their
        # original mapping reference.
        self._results: dict[str, MetricResult] = dict(results)

    def __getitem__(self, key: str) -> MetricResult:
        return self._results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __contains__(self, key: object) -> bool:
        return key in self._results

    def __repr__(self) -> str:
        return f"Scorecard({list(self._results.keys())!r})"

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """JSON-friendly serialization. Stable schema (additive only).

        Returns a flat ``{spec_name: result_dict}`` mapping. Each result_dict
        carries ``value``, ``status``, ``reason``, and optionally ``ci``.
        """
        return {name: result.to_dict() for name, result in self._results.items()}

    def to_pandas(self) -> Any:
        """One-row DataFrame with metric names as the (multi)column index.

        Lazy pandas import — only available if pandas is installed. Raises
        ``ImportError`` with an install hint when pandas is missing.

        The DataFrame has 1 row (one slice) and a 2-level column index:
        outer = metric name, inner = field name in ``{"value", "status",
        "reason", "ci_low", "ci_high", "confidence", "n_resamples",
        "method"}``. CI-related columns (``ci_low``, ``ci_high``,
        ``confidence``, ``n_resamples``, ``method``) are sentinel-valued
        (``NaN`` for numeric, ``""`` for string) when no CI is present
        (status="skipped" / "error", or bootstrap=False).

        Decision R6-C (Round 6 audit, Gemini F3): the v0.47 expansion adds
        ``n_resamples`` + ``method`` so the schema is lossless against
        :meth:`BootstrapCI.to_dict` — trace provenance no longer drops in
        the DataFrame view.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Scorecard.to_pandas() requires pandas. "
                "Install with: pip install eval-toolkit[dataframe]"
            ) from exc

        cols: list[tuple[str, str]] = []
        values: list[Any] = []
        for name, result in self._results.items():
            cols.extend(
                [
                    (name, "value"),
                    (name, "status"),
                    (name, "reason"),
                    (name, "ci_low"),
                    (name, "ci_high"),
                    (name, "confidence"),
                    (name, "n_resamples"),
                    (name, "method"),
                ]
            )
            values.extend(
                [
                    result.value if result.value is not None else float("nan"),
                    result.status,
                    result.reason,
                    result.ci.ci_low if result.ci is not None else float("nan"),
                    result.ci.ci_high if result.ci is not None else float("nan"),
                    result.ci.confidence if result.ci is not None else float("nan"),
                    result.ci.n_resamples if result.ci is not None else float("nan"),
                    result.ci.method if result.ci is not None else "",
                ]
            )

        index = pd.MultiIndex.from_tuples(cols, names=["metric", "field"])
        return pd.DataFrame([values], columns=index)


def scorecard(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metrics: Sequence[MetricSpec],
    bootstrap: bool = True,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> Scorecard:
    """Compute multiple metrics on a single slice; return a :class:`Scorecard`.

    The v1.0 primary metric surface (Decision A/I/S/X). For each spec in
    ``metrics``, computes the point estimate and (when ``bootstrap=True``)
    a :class:`BootstrapCI`. Per-cell status reflects whether the metric is
    defined on the slice, whether ``compute()`` succeeded, and whether the
    bootstrap could run.

    Parameters
    ----------
    y_true : numpy.ndarray, shape (n,)
        Binary labels in ``{0, 1}``.
    y_score : numpy.ndarray, shape (n,)
        Predicted scores (typically probabilities in ``[0, 1]``; the contract
        does not enforce range — individual specs validate as appropriate).
    metrics : Sequence[MetricSpec]
        List of spec instances. Typically from :mod:`eval_toolkit.metric_specs`:
        ``[ms.pr_auc, ms.roc_auc, ms.brier, ms.ece(n_bins=15)]``. Custom user
        specs satisfying the :class:`MetricSpec` Protocol structurally are
        accepted.
    bootstrap : bool, optional
        Whether to compute bootstrap CIs alongside point estimates. Default
        ``True`` (Decision B). Set ``False`` for cheap point-only batches.
    n_resamples : int, optional
        Bootstrap resamples per metric. Default ``1000``. Passed through to
        :func:`~eval_toolkit.bootstrap.bootstrap_ci`.
    confidence : float, optional
        Two-sided CI level ∈ ``(0, 1)``. Default ``0.95``.
    seed : int or None, optional
        Bootstrap RNG seed. Default ``None``, which is treated as ``seed=0``
        for reproducibility — eval-toolkit's evaluation pipelines are
        deterministic by default. Pass an explicit integer to control the
        bootstrap RNG; pass a value derived from
        ``np.random.SeedSequence().entropy`` for non-deterministic sampling.
        Decision R6-A (Round 6 audit) locked the deterministic-by-default
        contract; the prior docstring framing was incorrect.

    Returns
    -------
    Scorecard
        Read-only :class:`Mapping`-typed container; subscript-access by spec
        name.

    Raises
    ------
    ValueError
        On absolute input validation failures: length mismatch, empty arrays,
        bootstrap config out of bounds. Per-metric failures are isolated into
        :class:`MetricResult` cells, not raised.

    Examples
    --------
    >>> import numpy as np
    >>> from eval_toolkit import scorecard, metric_specs as ms
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 500)
    >>> s = rng.random(500)
    >>> r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True,
    ...               n_resamples=200, seed=0)
    >>> r["pr_auc"].status
    'ok'
    >>> r["pr_auc"].ci is not None
    True
    >>> r["brier"].status
    'ok'

    Single-class slice yields skipped (not raised):

    >>> r_sc = scorecard(np.zeros(100, dtype=int), rng.random(100),
    ...                  metrics=[ms.pr_auc, ms.brier], bootstrap=False)
    >>> r_sc["pr_auc"].status
    'skipped'
    >>> r_sc["pr_auc"].value is None
    True
    >>> r_sc["brier"].status
    'ok'

    Notes
    -----
    **Per-cell error isolation.** When one metric raises inside ``compute()``,
    other metrics still compute. The failing cell carries
    ``status="error"`` + ``reason=str(exc)``; the surrounding scorecard
    does not raise.

    **Skipped detection.** Each spec's ``name`` is checked against
    :func:`~eval_toolkit.metrics.is_metric_defined_for_slice` (v0.39). Names
    recognized as ranking metrics undefined on single-class slices yield
    ``status="skipped"`` without invoking ``compute()``. Names the registry
    doesn't recognize fall through to the attempt-compute path.

    **Bootstrap unavailable.** When ``bootstrap=True`` but
    :func:`bootstrap_ci` cannot run (e.g., the slice has ``n < 10`` samples),
    the cell still records ``status="ok"`` with a populated ``reason``
    explaining the missing CI; ``ci`` stays ``None``.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    _validate_scorecard_inputs(
        y_true_arr,
        y_score_arr,
        n_resamples=n_resamples,
        confidence=confidence,
        bootstrap=bootstrap,
    )
    _validate_unique_spec_names(metrics)

    is_single_class = bool(np.unique(y_true_arr).size < 2)

    results: dict[str, MetricResult] = {}
    for spec in metrics:
        results[spec.name] = _evaluate_spec(
            spec,
            y_true_arr,
            y_score_arr,
            is_single_class=is_single_class,
            bootstrap=bootstrap,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )

    return Scorecard(results)


def _evaluate_spec(
    spec: MetricSpec,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    is_single_class: bool,
    bootstrap: bool,
    n_resamples: int,
    confidence: float,
    seed: int | None,
) -> MetricResult:
    """Compute one spec's MetricResult, isolating per-spec failures."""
    if not is_metric_defined_for_slice(spec.name, is_single_class=is_single_class):
        return MetricResult(
            value=None,
            status="skipped",
            reason=f"{spec.name} not defined on single-class slice",
        )

    try:
        point = float(spec.compute(y_true, y_score))
    except (MemoryError, RecursionError, KeyboardInterrupt, SystemExit):
        # Process-exhaustion / user-interrupt signals must propagate;
        # per-cell isolation is for application-level errors only.
        # Decision R6-F5 (Round 6 audit, Gemini).
        raise
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional (per-cell isolation)
        return MetricResult(
            value=None,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
        )

    if not bootstrap:
        return MetricResult(value=point, status="ok")

    try:
        ci = bootstrap_ci(
            y_true,
            y_score,
            lambda y_t, y_s: spec.compute(y_t, y_s),
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed if seed is not None else 0,
        )
    except (MemoryError, RecursionError, KeyboardInterrupt, SystemExit):
        # Same R6-F5 invariant for the bootstrap path.
        raise
    except Exception as exc:  # noqa: BLE001
        # Point estimate succeeded; the bootstrap couldn't (e.g., n < 10
        # floor from bootstrap.py:198, BCa degeneracy, etc.). Record the
        # reason but keep status="ok" since the point estimate is valid.
        return MetricResult(
            value=point,
            status="ok",
            reason=f"bootstrap unavailable: {type(exc).__name__}: {exc}",
        )

    return MetricResult(value=point, status="ok", ci=ci)


def _validate_scorecard_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_resamples: int,
    confidence: float,
    bootstrap: bool,
) -> None:
    """Absolute input validation (raises) — per-cell concerns are caught later."""
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D (n,); got ndim={y_true.ndim}")
    if y_score.ndim != 1:
        raise ValueError(f"y_score must be 1-D (n,); got ndim={y_score.ndim}")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            "y_true and y_score must have matching length; "
            f"got y_true.shape[0]={y_true.shape[0]}, y_score.shape[0]={y_score.shape[0]}"
        )
    if y_true.size == 0:
        raise ValueError("y_true is empty; provide at least one sample")
    if bootstrap:
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1 when bootstrap=True; got {n_resamples}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}")


def _validate_unique_spec_names(metrics: Sequence[MetricSpec]) -> None:
    """Reject duplicate MetricSpec.name values in a single scorecard() call.

    Locked by Decision R6-B (Round 6 audit, Codex R6-F3): silent last-wins
    overwrite is not a documented Mapping[str, MetricResult] contract. Force
    the caller to disambiguate so we never lose data on user error.
    """
    seen: dict[str, int] = {}
    for i, spec in enumerate(metrics):
        if spec.name in seen:
            raise ValueError(
                f"Duplicate MetricSpec name {spec.name!r} at index {i} "
                f"(previously at index {seen[spec.name]}); each spec must have a "
                f"unique name for the Scorecard Mapping[str, MetricResult] contract."
            )
        seen[spec.name] = i
