"""Slice-aware evaluation harness for binary scorers.

Public surface:

- :class:`Scorer` Protocol — anything with ``predict_proba(X) -> np.ndarray``
- :class:`SliceAwareScorer` Protocol — optional ``should_score_slice(name)`` hook
- :class:`EvalSlice` — DataFrame wrapper with configurable column names
- :class:`RunResult` — JSON-serializable run container (versioned schema)
- :func:`evaluate_scorer_on_slice` — score one model on one slice
- :func:`evaluate` — pure orchestrator: scores × slices → RunResult (no IO)
- :func:`evaluate_folded` — fold aggregator: Splitter × seeds → RunResult
  with ``by_fold`` and auto-CV-CI ``fold_summary``
- :func:`with_claim_report` — attach generic claim-gate evidence to a
  frozen ``RunResult``
- :func:`write_run_result` — IO wrapper: write RunResult to ``run_dir/results.json``

The pure/IO split lets callers test :func:`evaluate` deterministically without
touching the filesystem; :func:`write_run_result` is the only IO sink.

v0.7.0 additions:

- ``leakage_checks`` / ``on_leakage`` params on :func:`evaluate`: run a
  sequence of :class:`~eval_toolkit.leakage.LeakageCheck` over the input
  slices before evaluation; raise on error-severity findings by default.
- ``on_scorer_error`` param: when ``"record"``, captures any
  ``Scorer.predict_proba`` exception per (slice, scorer) instead of failing
  the whole run.
- ``RunResult.by_fold`` / ``fold_summary`` / ``schema_version="v1"`` fields
  (additive; defaults preserve backward compat).
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import pandas as pd

from eval_toolkit.artifacts import (
    error_metric,
    sanitize_for_json,
    skipped_metric,
    write_json_strict,
)
from eval_toolkit.bootstrap import (
    bootstrap_ci,
    cv_clt_ci,
    mde_from_ci,
    paired_bootstrap_diff,
)
from eval_toolkit.calibration import PlattFit, maximum_calibration_error
from eval_toolkit.metrics import brier_score, headline_metrics, pr_auc, roc_auc
from eval_toolkit.operating_points import (
    FittedOperatingPoint,
    OperatingPointSpec,
    apply_operating_points,
    fit_operating_points,
)
from eval_toolkit.protocols import Scorer, SliceAwareScorer
from eval_toolkit.thresholds import TargetFPRSelector

if TYPE_CHECKING:
    from eval_toolkit.leakage import LeakageCheck
    from eval_toolkit.splits import Splitter

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "RUN_RESULT_SCHEMA_VERSION",
    "EvalSlice",
    "RunResult",
    "Scorer",
    "SliceAwareScorer",
    "evaluate",
    "evaluate_folded",
    "evaluate_scorer_on_slice",
    "with_claim_report",
    "write_run_result",
]

DEFAULT_BOOTSTRAP_RESAMPLES: Final[int] = 1000
RUN_RESULT_SCHEMA_VERSION: Final[str] = "v1"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalSlice:
    """A single eval slice (dev test, OOD slice, ablation slice, etc.).

    Parameters
    ----------
    name : str
        Slice identifier.
    df : pandas.DataFrame
        Must contain ``feature_col`` and ``label_col``; ``strata_col`` if set.
    description : str, optional
        Human-readable slice description.
    feature_col : str, optional
        Column holding the feature passed to ``Scorer.predict_proba``.
        Default ``"text"``.
    label_col : str, optional
        Column holding binary labels in {0, 1}. Default ``"label"``.
    strata_col : str or None, optional
        Optional categorical column for stratified recall reporting.
        Default ``None``.
    """

    name: str
    df: pd.DataFrame
    description: str = ""
    feature_col: str = "text"
    label_col: str = "label"
    strata_col: str | None = None

    def __post_init__(self) -> None:
        """Validate the minimum column and label contract."""
        for col in (self.feature_col, self.label_col):
            if col not in self.df.columns:
                raise KeyError(f"slice {self.name!r}: missing column {col!r}")
        if self.strata_col is not None and self.strata_col not in self.df.columns:
            raise KeyError(f"slice {self.name!r}: missing strata column {self.strata_col!r}")
        if (~self.df[self.label_col].isin({0, 1})).any():
            raise ValueError(f"slice {self.name!r}: labels must be in {{0, 1}}")

    @property
    def y_true(self) -> np.ndarray:
        """Binary labels as a 1-D NumPy array."""
        arr: np.ndarray = self.df[self.label_col].astype(int).to_numpy()
        return arr

    @property
    def features(self) -> list[str]:
        """Feature column as a plain list for scorer compatibility."""
        out: list[str] = self.df[self.feature_col].tolist()
        return out

    @property
    def strata(self) -> np.ndarray | None:
        """Stratifier column as np.ndarray, or None if unset."""
        if self.strata_col is None:
            return None
        out: np.ndarray = self.df[self.strata_col].to_numpy()
        return out


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a full evaluation run.

    Frozen dataclass: result fields must be fully populated before construction.
    Callers building results incrementally should accumulate into local dicts
    and pass them to the constructor.

    Parameters
    ----------
    run_id : str
        Caller-supplied run identifier (timestamp / UUID).
    git_sha : str or None
        Repo HEAD commit SHA at run time, or ``None``.
    config : dict[str, object]
        Eval-time configuration (n_resamples, seed, scorer / slice names,
        paired_diffs). Distinct from :class:`~eval_toolkit.manifest.RunManifest`
        which captures *environment* fingerprint.
    by_slice : dict[str, dict[str, object]]
        Per-slice results. Empty for fold-aggregated runs (see ``by_fold``).
    by_fold : dict[str, "RunResult"], optional
        Per-fold raw :class:`RunResult` keyed by composite ID
        (``"seed=42/fold=0"``). Populated by :func:`evaluate_folded`;
        empty for non-folded runs (backward compat).
    fold_summary : dict[str, dict[str, object]], optional
        Auto-computed CV-CI summary per (slice, scorer, metric), keyed
        ``[slice_name][scorer_name][metric] = {"mean", "ci_low", "ci_high",
        "n_folds"}``. Populated by :func:`evaluate_folded`; empty otherwise.
    claim_report : dict[str, object], optional
        Optional generic :class:`eval_toolkit.claims.ClaimReport` payload.
        Empty means no claim gates were evaluated for this run.
    schema_version : str
        JSON schema version. ``"v1"`` for v0.7.0+; downstream parsers gate
        on this.

    .. versionchanged:: 0.7.0
        Added ``by_fold``, ``fold_summary``, ``schema_version`` (additive,
        defaults empty / ``"v1"`` — backward compatible).
    """

    run_id: str
    git_sha: str | None
    config: dict[str, object]
    by_slice: dict[str, dict[str, object]] = field(default_factory=dict)
    by_fold: dict[str, RunResult] = field(default_factory=dict)
    fold_summary: dict[str, dict[str, object]] = field(default_factory=dict)
    claim_report: dict[str, object] = field(default_factory=dict)
    prediction_artifacts: list[dict[str, object]] = field(default_factory=list)
    evidence_axes: list[dict[str, object]] = field(default_factory=list)
    pairing_metadata: dict[str, object] = field(default_factory=dict)
    aggregate_evidence: dict[str, object] = field(default_factory=dict)
    threshold_policy: dict[str, object] = field(default_factory=dict)
    schema_version: str = RUN_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Serialize using the stable JSON schema (v1 — see ``schema_version``)."""
        out = sanitize_for_json(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "git_sha": self.git_sha,
                "config": self.config,
                "by_slice": self.by_slice,
                "by_fold": {k: v.to_dict() for k, v in self.by_fold.items()},
                "fold_summary": self.fold_summary,
                "claim_report": self.claim_report,
                "prediction_artifacts": self.prediction_artifacts,
                "evidence_axes": self.evidence_axes,
                "pairing_metadata": self.pairing_metadata,
                "aggregate_evidence": self.aggregate_evidence,
                "threshold_policy": self.threshold_policy,
            }
        )
        if not isinstance(out, dict):
            raise TypeError("RunResult.to_dict expected a mapping payload")
        return out


def with_claim_report(result: RunResult, report: object) -> RunResult:
    """Return a copy of ``result`` with a serialized claim report attached.

    ``RunResult`` is frozen, so claim evidence is attached by value rather than
    mutation. ``report`` may be a mapping or any object exposing ``to_dict()``,
    including :class:`eval_toolkit.claims.ClaimReport`.
    """
    claim_report = _object_to_dict(report, what="claim report")
    return RunResult(
        run_id=result.run_id,
        git_sha=result.git_sha,
        config=result.config,
        by_slice=result.by_slice,
        by_fold=result.by_fold,
        fold_summary=result.fold_summary,
        claim_report=claim_report,
        prediction_artifacts=result.prediction_artifacts,
        evidence_axes=result.evidence_axes,
        pairing_metadata=result.pairing_metadata,
        aggregate_evidence=result.aggregate_evidence,
        threshold_policy=result.threshold_policy,
        schema_version=result.schema_version,
    )


def _object_to_dict(obj: object, *, what: str) -> dict[str, object]:
    """Normalize a mapping or ``to_dict`` object to a plain dict."""
    if isinstance(obj, Mapping):
        return dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, Mapping):
            return dict(out)
    raise TypeError(f"expected {what} mapping or object with to_dict(), got {type(obj).__name__}")


def _should_score_slice(scorer: Scorer, slice_name: str) -> bool:
    """Honor optional slice-aware scorer hooks without widening the base Protocol."""
    should_score = getattr(scorer, "should_score_slice", None)
    if should_score is None:
        return True
    result = should_score(slice_name)
    if not isinstance(result, bool):
        raise TypeError(
            f"{type(scorer).__name__}.should_score_slice() must return bool, "
            f"got {type(result).__name__}"
        )
    return result


def _skipped_scorer_result(slice_: EvalSlice, reason: str) -> dict[str, object]:
    """Schema-compatible placeholder for a scorer intentionally skipped on a slice."""
    return {
        "skipped": reason,
        "n": int(len(slice_.df)),
        "n_positive": int(slice_.y_true.sum()),
        "scores": [],
    }


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: object,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap (low, high) CI on an AUC metric; return BootstrapCI.to_dict() or sentinel.

    Mirrors :func:`evaluate_scorer_on_slice`'s existing PR-AUC bootstrap
    logic so :func:`_evaluate_scores` can compute ROC-AUC CI on the same
    code path (closes V4's bootstrap-roc-auc need for C11).
    """
    if len({int(v) for v in y_true}) < 2:
        return skipped_metric("single-class slice; AUC CI is not meaningful")
    if len(y_true) < 30:
        return skipped_metric(f"n={len(y_true)} < 30")
    try:
        ci = bootstrap_ci(
            y_true,
            y_score,
            metric_fn,  # type: ignore[arg-type]
            n_resamples=n_resamples,
            method="BCa",
            seed=seed,
        )
        return ci.to_dict()
    except (ValueError, RuntimeError) as exc:
        return error_metric(str(exc))


def _evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    strata: np.ndarray | None,
    n_resamples: int,
    seed: int,
    fpr_ladder: list[float] | None,
    compute_mce: bool,
    compute_brier: bool,
    bootstrap_roc_auc: bool,
) -> dict[str, object]:
    """Compute the harness metric block for a (y_true, y_score) pair.

    v0.22.0 private helper used by :func:`evaluate_scorer_on_slice` to
    produce a single metric block. Called once with the raw scores and
    optionally again with calibrated scores; the calibrated-side dict is
    merged under ``*_calibrated`` keys by the public function.

    Always includes the v0.7.0 baseline (headline_metrics + pr_auc_ci +
    scores + is_single_class). Conditionally adds ``roc_auc_ci``,
    ``tpr_at_fpr``, ``mce``, ``brier_score`` keys per kwargs.
    """
    metrics = headline_metrics(y_true, y_score, strata=strata)
    is_single_class = len({int(v) for v in y_true}) == 1
    metrics["is_single_class"] = is_single_class
    metrics["pr_auc_ci"] = _bootstrap_auc_ci(
        y_true, y_score, pr_auc, n_resamples=n_resamples, seed=seed
    )
    if bootstrap_roc_auc:
        metrics["roc_auc_ci"] = _bootstrap_auc_ci(
            y_true, y_score, roc_auc, n_resamples=n_resamples, seed=seed
        )
    if fpr_ladder is not None:
        tpr_at_fpr: dict[str, object] = {}
        if is_single_class:
            for target in fpr_ladder:
                tpr_at_fpr[f"{target}"] = None
        else:
            for target in fpr_ladder:
                try:
                    result = TargetFPRSelector(fpr=target).select(y_true, y_score)
                    tpr_at_fpr[f"{target}"] = float(result.recall)
                except RuntimeError:
                    tpr_at_fpr[f"{target}"] = None
        metrics["tpr_at_fpr"] = tpr_at_fpr
    if compute_brier:
        try:
            metrics["brier_score"] = brier_score(y_true, y_score, empty_strategy="return_none")
        except (ValueError, RuntimeError) as exc:
            metrics["brier_score"] = error_metric(str(exc))
    if compute_mce:
        try:
            metrics["mce"] = maximum_calibration_error(y_true, y_score)
        except (ValueError, RuntimeError) as exc:
            metrics["mce"] = error_metric(str(exc))
    metrics["scores"] = y_score.tolist()
    return dict(metrics)


def evaluate_scorer_on_slice(
    scorer: Scorer,
    slice_: EvalSlice,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    on_scorer_error: Literal["raise", "record"] = "raise",
    precomputed_scores: np.ndarray | None = None,
    attack_style: str | None = None,
    fpr_ladder: list[float] | None = None,
    compute_mce: bool = False,
    compute_brier: bool = False,
    calibrator: PlattFit | None = None,
    bootstrap_roc_auc: bool = False,
) -> dict[str, object]:
    """Score one scorer on one slice; return headline + bootstrap CI on PR-AUC.

    Single-class slices (all-positive or all-negative): PR-AUC, ROC-AUC, and
    threshold-selected F1 are not meaningful; the result includes a
    ``"skipped"`` field for those metrics.

    Parameters
    ----------
    scorer : Scorer
    slice_ : EvalSlice
    n_resamples : int, optional
        Bootstrap resamples for PR-AUC CI. Default 1000.
    seed : int, optional
        RNG seed. Default 42.
    on_scorer_error : {"raise", "record"}, optional
        v0.7.0 — when ``"record"``, catch any ``Scorer.predict_proba``
        exception and return a ``{"error", "exc_type", "traceback"}`` dict
        instead of failing. Default ``"raise"`` (loud during dev/CI).
    precomputed_scores : np.ndarray or None, optional
        v0.22.0 — if provided, skip ``scorer.predict_proba`` and use this
        array as ``y_score``. Shape must match ``len(slice_.df)``. Used by
        callers that cache scores across per-slice variants (e.g. V4's
        per-attack-style decomposition).
    attack_style : str or None, optional
        v0.22.0 — pass-through label that lands in the result dict's
        ``attack_style`` key. No metric effect.
    fpr_ladder : list[float] or None, optional
        v0.22.0 — when set, also compute TPR at each FPR via
        :class:`TargetFPRSelector`; emitted under ``tpr_at_fpr`` as
        ``{str(fpr): tpr_value_or_None}``.
    compute_mce : bool, optional
        v0.22.0 — when True, also compute
        :func:`maximum_calibration_error`; emitted under ``mce``.
    compute_brier : bool, optional
        v0.22.0 — when True, also compute :func:`brier_score`; emitted
        under ``brier_score``.
    calibrator : PlattFit or None, optional
        v0.22.0 — when provided, apply to ``y_score`` to produce
        ``y_score_calibrated``, then recompute every requested metric on
        the calibrated scores; merged into the result under
        ``*_calibrated`` keys (``pr_auc_calibrated``,
        ``roc_auc_calibrated``, ``brier_score_calibrated``,
        ``ece_calibrated``, ``mce_calibrated``, ``tpr_at_fpr_calibrated``,
        ``scores_calibrated``, plus the ``pr_auc_ci`` /
        ``roc_auc_ci`` companions).
    bootstrap_roc_auc : bool, optional
        v0.22.0 — when True (and ``n_resamples > 0`` and the slice is
        mixed-class), also bootstrap ROC-AUC CI; emitted under
        ``roc_auc_ci``. Default ``False`` preserves the v0.7-v0.21
        contract (PR-AUC CI only).

    Returns
    -------
    dict
        Headline metrics + ``pr_auc_ci`` + raw scores. On caught error
        (``on_scorer_error="record"``), the dict carries
        ``{"error", "exc_type", "traceback", "n", "n_positive", "scores": []}``
        — same shape downstream consumers expect, plus the error fields.
    """
    y_true = slice_.y_true
    if precomputed_scores is not None:
        if precomputed_scores.shape != (len(slice_.df),):
            raise ValueError(
                f"precomputed_scores shape {precomputed_scores.shape} does not "
                f"match slice length {len(slice_.df)}"
            )
        y_score = np.asarray(precomputed_scores)
    else:
        try:
            y_score = scorer.predict_proba(slice_.features)
        except Exception as exc:
            if on_scorer_error == "raise":
                raise
            err = {
                "error": str(exc),
                "error_state": error_metric(str(exc), exc_type=type(exc).__name__),
                "exc_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "n": int(len(slice_.df)),
                "n_positive": int(y_true.sum()),
                "scores": [],
            }
            if attack_style is not None:
                err["attack_style"] = attack_style
            return err

    metrics = _evaluate_scores(
        y_true,
        y_score,
        strata=slice_.strata,
        n_resamples=n_resamples,
        seed=seed,
        fpr_ladder=fpr_ladder,
        compute_mce=compute_mce,
        compute_brier=compute_brier,
        bootstrap_roc_auc=bootstrap_roc_auc,
    )

    if calibrator is not None:
        y_score_calibrated = np.asarray(calibrator(y_score))
        calibrated = _evaluate_scores(
            y_true,
            y_score_calibrated,
            strata=slice_.strata,
            n_resamples=n_resamples,
            seed=seed,
            fpr_ladder=fpr_ladder,
            compute_mce=compute_mce,
            compute_brier=compute_brier,
            bootstrap_roc_auc=bootstrap_roc_auc,
        )
        # Merge calibrated block under *_calibrated keys; preserve raw keys.
        for k, v in calibrated.items():
            if k in ("n", "n_positive", "is_single_class", "metric_note"):
                continue  # invariant across raw / calibrated; skip duplicate
            metrics[f"{k}_calibrated"] = v

    if attack_style is not None:
        metrics["attack_style"] = attack_style
    return metrics


def evaluate(
    scorers: dict[str, Scorer],
    slices: Sequence[EvalSlice],
    *,
    run_id: str,
    git_sha: str | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    paired_diffs: list[tuple[str, str]] | None = None,
    seed: int = 42,
    extra_config: Mapping[str, object] | None = None,
    leakage_checks: Sequence[LeakageCheck] = (),
    on_leakage: Literal["raise", "record", "skip"] = "raise",
    on_scorer_error: Literal["raise", "record"] = "raise",
    operating_point_specs: Sequence[OperatingPointSpec] = (),
) -> RunResult:
    """Run every scorer on every slice; return a pure :class:`RunResult` (no IO).

    Parameters
    ----------
    scorers : dict[str, Scorer]
        Named scorers to evaluate.
    slices : sequence of EvalSlice
    run_id : str
        Caller-supplied run identifier (e.g., a timestamp). Pure functions don't
        capture the wall-clock; pass an ID built once outside.
    git_sha : str or None, optional
        Optional git SHA for provenance. Caller computes this if needed; pass
        ``None`` to omit. Pure functions don't shell out to git.
    n_resamples : int, optional
        Bootstrap resamples per CI. Default 1000.
    paired_diffs : list of (str, str) tuples, optional
        Pairs ``(a, b)`` for which to compute paired bootstrap on
        ``pr_auc(b) - pr_auc(a)`` per slice.
    seed : int, optional
        RNG seed. Default 42.
    extra_config : Mapping or None, optional
        Additional config keys to record in the result.
    leakage_checks : sequence of LeakageCheck, optional
        v0.7.0 — Sequence of pluggable leakage validators run over the
        slices before evaluation. Slices are exposed to the checks as a
        ``{slice.name: slice}`` mapping. Default empty (skip).
    on_leakage : {"raise", "record", "skip"}, optional
        v0.7.0 — Behavior when ``leakage_checks`` produces error-severity
        findings. ``"raise"`` (default) raises ``RuntimeError`` listing the
        findings; ``"record"`` records the report in
        ``RunResult.config["leakage_report"]`` and continues; ``"skip"``
        records nothing and continues.
    on_scorer_error : {"raise", "record"}, optional
        v0.7.0 — Threaded into every :func:`evaluate_scorer_on_slice` call.
        ``"record"`` captures Scorer exceptions per (slice, scorer) instead
        of failing the whole run.
    operating_point_specs : sequence of OperatingPointSpec, optional
        Fit thresholds on one mixed-class slice and apply them to named target
        slices. Results are attached under each scorer's
        ``"transferred_operating_points"`` block. Default empty (skip).

    Returns
    -------
    RunResult
        Pure result; no filesystem touched. Pass to :func:`write_run_result`
        to persist.

    Raises
    ------
    ValueError
        If ``scorers`` or ``slices`` is empty.
    RuntimeError
        If ``on_leakage="raise"`` and any leakage check produced an
        error-severity finding.
    """
    if not scorers:
        raise ValueError("at least one scorer required")
    if not slices:
        raise ValueError("at least one slice required")

    config: dict[str, object] = {
        "n_resamples": n_resamples,
        "seed": seed,
        "scorers": list(scorers.keys()),
        "slices": [s.name for s in slices],
        "paired_diffs": paired_diffs or [],
        "on_scorer_error": on_scorer_error,
    }
    if extra_config:
        config.update(dict(extra_config))

    # Run leakage checks before any scoring (per Q2 decision).
    if leakage_checks:
        # Late import to avoid circular dependency: leakage.py imports EvalSlice.
        from eval_toolkit.leakage import run_leakage_checks

        slices_dict = {s.name: s for s in slices}
        report = run_leakage_checks(list(leakage_checks), slices_dict)
        config["on_leakage"] = on_leakage
        if on_leakage != "skip":
            config["leakage_report"] = report.to_dict()
        if on_leakage == "raise" and report.has_errors():
            errors_summary = "; ".join(f"{f.check_name}: {f.message}" for f in report.errors())
            raise RuntimeError(
                f"Leakage checks produced {len(report.errors())} error finding(s): "
                f"{errors_summary}. Pass on_leakage='record' to continue with the "
                "report captured in RunResult.config, or 'skip' to drop the report."
            )

    by_slice: dict[str, dict[str, object]] = {}
    score_cache: dict[tuple[str, str], np.ndarray] = {}
    slices_by_name = {s.name: s for s in slices}

    for slice_ in slices:
        _logger.info(
            "[slice %s] n=%d, positives=%d",
            slice_.name,
            len(slice_.df),
            int(slice_.y_true.sum()),
        )
        slice_data: dict[str, dict[str, object]] = {}
        scores_by_scorer: dict[str, np.ndarray] = {}
        for sname, scorer in scorers.items():
            if not _should_score_slice(scorer, slice_.name):
                reason = f"slice {slice_.name!r} not in scorer allow-list"
                slice_data[sname] = _skipped_scorer_result(slice_, reason)
                _logger.info("    skipped %s: %s", sname, reason)
                continue
            t0 = time.time()
            slice_data[sname] = evaluate_scorer_on_slice(
                scorer,
                slice_,
                n_resamples=n_resamples,
                seed=seed,
                on_scorer_error=on_scorer_error,
            )
            # If the scorer raised under on_scorer_error="record", scores is [].
            # Subsequent paired-diff machinery sees the empty array and will
            # short-circuit on the same len-check it already does for skipped
            # scorers; no special-case needed.
            scores_by_scorer[sname] = np.asarray(slice_data[sname]["scores"], dtype=np.float64)
            score_cache[(slice_.name, sname)] = scores_by_scorer[sname]
            elapsed = time.time() - t0
            pr = slice_data[sname].get("pr_auc")
            pr_display = f"{pr:.4f}" if isinstance(pr, float) else "N/A"
            _logger.info("    %s: PR-AUC=%s (%.1fs)", sname, pr_display, elapsed)

        diffs: dict[str, dict[str, object]] = {}
        is_single_class = len({int(v) for v in slice_.y_true}) == 1
        if paired_diffs:
            for a, b in paired_diffs:
                if a not in scorers or b not in scorers:
                    continue
                if a not in scores_by_scorer or b not in scores_by_scorer:
                    diffs[f"{b}_minus_{a}"] = {"skipped": "one or both scorers skipped this slice"}
                    continue
                if is_single_class:
                    diffs[f"{b}_minus_{a}"] = {"skipped": "single-class slice; PR-AUC Δ degenerate"}
                    continue
                if len(slice_.y_true) < 30:
                    diffs[f"{b}_minus_{a}"] = {"skipped": f"n={len(slice_.y_true)} < 30"}
                    continue
                pdiff = paired_bootstrap_diff(
                    slice_.y_true,
                    scores_by_scorer[a],
                    scores_by_scorer[b],
                    pr_auc,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                pdiff_dict = pdiff.to_dict()
                try:
                    pdiff_dict["mde_at_80_power"] = mde_from_ci(
                        pdiff, alpha=0.05, power=0.80
                    ).to_dict()
                except (ValueError, RuntimeError) as exc:
                    pdiff_dict["mde_at_80_power"] = {"error": str(exc)}
                diffs[f"{b}_minus_{a}"] = pdiff_dict

        by_slice[slice_.name] = {
            "n": int(len(slice_.df)),
            "n_positive": int(slice_.y_true.sum()),
            "by_scorer": slice_data,
            "paired_diffs": diffs,
        }

    if operating_point_specs:
        _attach_transferred_operating_points(
            by_slice=by_slice,
            slices_by_name=slices_by_name,
            score_cache=score_cache,
            scorer_names=list(scorers.keys()),
            specs=operating_point_specs,
        )

    return RunResult(run_id=run_id, git_sha=git_sha, config=config, by_slice=by_slice)


def _attach_transferred_operating_points(
    *,
    by_slice: dict[str, dict[str, object]],
    slices_by_name: Mapping[str, EvalSlice],
    score_cache: Mapping[tuple[str, str], np.ndarray],
    scorer_names: Sequence[str],
    specs: Sequence[OperatingPointSpec],
) -> None:
    """Mutate ``by_slice`` to attach opt-in cross-slice operating-point metrics."""
    for spec in specs:
        names = list(spec.scorer_names) if spec.scorer_names else list(scorer_names)
        if spec.fit_slice not in slices_by_name:
            _record_spec_error(by_slice, spec, names, f"fit slice {spec.fit_slice!r} not found")
            continue

        fit_slice = slices_by_name[spec.fit_slice]
        fitted_by_scorer: dict[str, object] = {}
        for scorer_name in names:
            fit_scores = score_cache.get((spec.fit_slice, scorer_name))
            if fit_scores is None or len(fit_scores) != len(fit_slice.y_true):
                fitted_by_scorer[scorer_name] = {
                    "error": "fit scorer skipped, errored, or produced no scores"
                }
                continue
            try:
                fitted_by_scorer[scorer_name] = fit_operating_points(
                    fit_slice.y_true,
                    fit_scores,
                    spec.selectors,
                    fitted_on_slice=spec.fit_slice,
                    scorer_name=scorer_name,
                )
            except (ValueError, RuntimeError) as exc:
                fitted_by_scorer[scorer_name] = {"error": str(exc)}

        for target_name in spec.apply_slices:
            if target_name not in slices_by_name:
                _record_spec_error(
                    by_slice,
                    spec,
                    names,
                    f"apply slice {target_name!r} not found",
                    target_slice=target_name,
                )
                continue
            target_slice = slices_by_name[target_name]
            for scorer_name in names:
                scorer_block = _scorer_result_block(by_slice, target_name, scorer_name)
                transfer_block = _transfer_result_block(scorer_block)
                spec_block: dict[str, object] = {}
                transfer_block[spec.name] = spec_block

                fitted = fitted_by_scorer.get(scorer_name)
                if not isinstance(fitted, dict) or "error" in fitted:
                    spec_block["error"] = (
                        str(fitted.get("error", "threshold fitting failed"))
                        if isinstance(fitted, dict)
                        else "threshold fitting failed"
                    )
                    continue

                target_scores = score_cache.get((target_name, scorer_name))
                if target_scores is None or len(target_scores) != len(target_slice.y_true):
                    spec_block["skipped"] = "target scorer skipped, errored, or produced no scores"
                    continue
                try:
                    spec_block.update(
                        apply_operating_points(
                            target_slice.y_true,
                            target_scores,
                            cast(Mapping[str, FittedOperatingPoint], fitted),
                            applied_to_slice=target_name,
                            scorer_name=scorer_name,
                        )
                    )
                except (ValueError, RuntimeError) as exc:
                    spec_block["error"] = str(exc)


def _scorer_result_block(
    by_slice: dict[str, dict[str, object]],
    slice_name: str,
    scorer_name: str,
) -> dict[str, object]:
    """Return the mutable scorer result block, creating a minimal one if absent."""
    slice_block = by_slice.get(slice_name)
    if not isinstance(slice_block, dict):
        slice_block = {}
        by_slice[slice_name] = slice_block

    raw_by_scorer = slice_block.get("by_scorer")
    if not isinstance(raw_by_scorer, dict):
        slice_block["by_scorer"] = {}
        raw_by_scorer = slice_block["by_scorer"]
    by_scorer = cast(dict[str, object], raw_by_scorer)

    raw_scorer_block = by_scorer.get(scorer_name)
    if not isinstance(raw_scorer_block, dict):
        raw_scorer_block = {}
        by_scorer[scorer_name] = raw_scorer_block
    scorer_block = cast(dict[str, object], raw_scorer_block)
    return scorer_block


def _transfer_result_block(scorer_block: dict[str, object]) -> dict[str, object]:
    """Return/create the mutable transferred-operating-points block."""
    raw_transfer = scorer_block.get("transferred_operating_points")
    if not isinstance(raw_transfer, dict):
        raw_transfer = {}
        scorer_block["transferred_operating_points"] = raw_transfer
    transfer_block: dict[str, object] = raw_transfer
    return transfer_block


def _record_spec_error(
    by_slice: dict[str, dict[str, object]],
    spec: OperatingPointSpec,
    scorer_names: Sequence[str],
    message: str,
    *,
    target_slice: str | None = None,
) -> None:
    """Attach a spec-level error under target scorer blocks."""
    targets = [target_slice] if target_slice is not None else list(spec.apply_slices)
    for slice_name in targets:
        by_slice.setdefault(
            slice_name,
            {"n": 0, "n_positive": 0, "by_scorer": {}, "paired_diffs": {}},
        )
        for scorer_name in scorer_names:
            scorer_block = _scorer_result_block(by_slice, slice_name, scorer_name)
            transfer_block = _transfer_result_block(scorer_block)
            transfer_block[spec.name] = {"error": message}


def _extract_metric_value(slice_dict: object, metric: str) -> float | None:
    """Pull a numeric metric from one ``by_slice[scorer]`` dict, or ``None``."""
    if not isinstance(slice_dict, dict):
        return None
    val = slice_dict.get(metric)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _build_fold_summary(
    by_fold: Mapping[str, RunResult],
    slice_names: Sequence[str],
    scorer_names: Sequence[str],
    summary_metrics: Sequence[str] = ("pr_auc", "roc_auc"),
) -> dict[str, dict[str, object]]:
    """Aggregate per-fold metrics into ``[slice][scorer][metric] = CV-CI dict``.

    For each (slice, scorer, metric) triple, collect the per-fold values and
    pass them to :func:`eval_toolkit.bootstrap.cv_clt_ci`. Folds where the
    scorer was skipped or errored contribute ``np.nan`` (graceful per-fold
    degradation) and ``cv_clt_ci`` failures (fewer than 2 numeric folds, etc.)
    degrade to ``{"skipped": "<reason>"}``.
    """
    summary: dict[str, dict[str, object]] = {}
    for slice_name in slice_names:
        per_scorer: dict[str, object] = {}
        for scorer_name in scorer_names:
            per_metric: dict[str, object] = {}
            for metric in summary_metrics:
                fold_values: list[float] = []
                for fold_result in by_fold.values():
                    slice_block = fold_result.by_slice.get(slice_name, {})
                    if not isinstance(slice_block, dict):
                        continue
                    by_scorer = slice_block.get("by_scorer", {})
                    if not isinstance(by_scorer, dict):
                        continue
                    value = _extract_metric_value(by_scorer.get(scorer_name), metric)
                    fold_values.append(value if value is not None else float("nan"))
                arr = np.asarray(fold_values, dtype=np.float64)
                numeric = arr[~np.isnan(arr)]
                if len(numeric) < 2:
                    per_metric[metric] = {
                        "skipped": f"only {len(numeric)} numeric fold(s); CV-CI needs >=2"
                    }
                    continue
                try:
                    ci = cv_clt_ci(arr)
                    per_metric[metric] = {
                        "mean": ci.point_estimate,
                        "ci_low": ci.ci_low,
                        "ci_high": ci.ci_high,
                        "n_folds": int(len(numeric)),
                    }
                except (ValueError, RuntimeError) as exc:
                    per_metric[metric] = {"skipped": str(exc)}
            per_scorer[scorer_name] = per_metric
        summary[slice_name] = per_scorer
    return summary


def evaluate_folded(
    scorers: dict[str, Scorer],
    splitter: Splitter,
    slice_: EvalSlice,
    *,
    run_id: str,
    git_sha: str | None = None,
    seeds: Sequence[int] = (42,),
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    paired_diffs: list[tuple[str, str]] | None = None,
    leakage_checks: Sequence[LeakageCheck] = (),
    on_leakage: Literal["raise", "record", "skip"] = "raise",
    on_scorer_error: Literal["raise", "record"] = "raise",
    eval_split_names: Sequence[str] = ("test",),
    summary_metrics: Sequence[str] = ("pr_auc", "roc_auc"),
) -> RunResult:
    """Run a fold aggregator: ``Splitter × seeds → RunResult`` with CV-CI summary.

    For each ``seed`` in ``seeds``, iterate ``splitter.iter_folds(slice_)``,
    delegate to :func:`evaluate` per fold (passing only the splits named in
    ``eval_split_names``), and aggregate. Both raw per-fold results and an
    auto-computed CV-CI summary land on the returned :class:`RunResult`:

    - ``RunResult.by_fold[fold_id]`` — raw :class:`RunResult` per
      (seed, fold), keyed ``"seed=<seed>/fold=<i>"``.
    - ``RunResult.fold_summary[slice_name][scorer_name][metric]`` —
      ``{mean, ci_low, ci_high, n_folds}`` from
      :func:`eval_toolkit.bootstrap.cv_clt_ci`. Falls back to
      ``{"skipped": "<reason>"}`` when fewer than 2 numeric folds.

    Parameters
    ----------
    scorers : dict[str, Scorer]
    splitter : Splitter
        Any object implementing
        :meth:`~eval_toolkit.splits.Splitter.iter_folds`.
    slice_ : EvalSlice
        Parent dataset; the splitter partitions it.
    run_id : str
    git_sha : str or None
    seeds : sequence of int, optional
        RNG seeds for multi-seed × CV. Default ``(42,)`` (single seed).
    n_resamples, paired_diffs, leakage_checks, on_leakage, on_scorer_error :
        Forwarded to :func:`evaluate` per fold.
    eval_split_names : sequence of str, optional
        Subset of each fold-dict's keys to actually evaluate. Default
        ``("test",)`` — train sets are skipped (eval-only K-fold). Pass
        ``("val", "test")`` to evaluate both.
    summary_metrics : sequence of str, optional
        Metrics aggregated into :attr:`RunResult.fold_summary`. Default
        ``("pr_auc", "roc_auc")``.

    Returns
    -------
    RunResult
        ``by_slice`` empty (per-fold details live in ``by_fold``);
        ``fold_summary`` populated.

    Raises
    ------
    ValueError
        If ``scorers`` is empty or no ``eval_split_names`` are present in
        any fold.

    Notes
    -----
    Eval-only K-fold semantics: the same scorer instance runs on each fold's
    test partition. For "different trained model per fold" workflows, train
    K models externally and wrap each as a :class:`Scorer` whose
    ``predict_proba`` dispatches to the right underlying model based on
    the slice's content.
    """
    if not scorers:
        raise ValueError("at least one scorer required")

    by_fold: dict[str, RunResult] = {}
    fold_slice_names_seen: set[str] = set()
    for seed in seeds:
        for fold_idx, fold_dict in enumerate(splitter.iter_folds(slice_)):
            fold_id = f"seed={seed}/fold={fold_idx}"
            eval_slices = [fold_dict[name] for name in eval_split_names if name in fold_dict]
            if not eval_slices:
                raise ValueError(
                    f"fold {fold_id}: none of eval_split_names={list(eval_split_names)} "
                    f"present in fold keys={list(fold_dict.keys())}"
                )
            for s in eval_slices:
                fold_slice_names_seen.add(s.name)
            fold_result = evaluate(
                scorers,
                eval_slices,
                run_id=fold_id,
                git_sha=git_sha,
                n_resamples=n_resamples,
                paired_diffs=paired_diffs,
                seed=seed,
                leakage_checks=leakage_checks,
                on_leakage=on_leakage,
                on_scorer_error=on_scorer_error,
            )
            by_fold[fold_id] = fold_result

    fold_summary = _build_fold_summary(
        by_fold,
        slice_names=sorted(fold_slice_names_seen),
        scorer_names=list(scorers.keys()),
        summary_metrics=summary_metrics,
    )

    config: dict[str, object] = {
        "n_resamples": n_resamples,
        "seeds": list(seeds),
        "scorers": list(scorers.keys()),
        "splitter": type(splitter).__name__,
        "eval_split_names": list(eval_split_names),
        "summary_metrics": list(summary_metrics),
        "n_folds": int(len(by_fold)),
    }
    return RunResult(
        run_id=run_id,
        git_sha=git_sha,
        config=config,
        by_slice={},
        by_fold=by_fold,
        fold_summary=fold_summary,
    )


def write_run_result(result: RunResult, run_dir: Path) -> tuple[Path, Path]:
    """Write a :class:`RunResult` to ``run_dir`` as two JSON files (compact + full).

    Parameters
    ----------
    result : RunResult
    run_dir : pathlib.Path
        Directory to write into. Created if it doesn't exist.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        ``(results_json_path, results_full_json_path)``.

    Notes
    -----
    The compact ``results.json`` strips per-prompt ``scores`` arrays from the
    headline output to keep it small; the full ``results_full.json`` retains
    them.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    full_path = run_dir / "results_full.json"
    compact_path = run_dir / "results.json"
    write_json_strict(result.to_dict(), full_path)
    write_json_strict(_strip_scores(result.to_dict()), compact_path)
    return compact_path, full_path


def _strip_scores(d: dict[str, object]) -> dict[str, object]:
    """Drop the per-row ``scores`` arrays from the headline JSON."""
    out = sanitize_for_json(d)
    if not isinstance(out, dict):
        raise TypeError("_strip_scores expected a mapping payload")
    by_slice = out.get("by_slice", {})
    if isinstance(by_slice, dict):
        for slice_data in by_slice.values():
            if isinstance(slice_data, dict):
                by_scorer = slice_data.get("by_scorer", {})
                if isinstance(by_scorer, dict):
                    for scorer_data in by_scorer.values():
                        if isinstance(scorer_data, dict):
                            scorer_data.pop("scores", None)
    return out
