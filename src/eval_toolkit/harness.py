"""Slice-aware evaluation harness for binary scorers.

Public surface:

- :class:`Scorer` Protocol — anything with ``predict_proba(X) -> np.ndarray``
- :class:`SliceAwareScorer` Protocol — optional ``should_score_slice(name)`` hook
- :class:`EvalSlice` — DataFrame wrapper with configurable column names
- :class:`RunResult` — JSON-serializable run container
- :func:`evaluate_scorer_on_slice` — score one model on one slice
- :func:`evaluate` — pure orchestrator: scores × slices → RunResult (no IO)
- :func:`write_run_result` — IO wrapper: write RunResult to ``run_dir/results.json`` (and a full variant)

The pure/IO split lets callers test ``evaluate(...)`` deterministically without
touching the filesystem; ``write_run_result(...)`` is the only IO sink.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from eval_toolkit.bootstrap import bootstrap_ci, mde_from_ci, paired_bootstrap_diff
from eval_toolkit.metrics import headline_metrics, pr_auc

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "EvalSlice",
    "RunResult",
    "Scorer",
    "SliceAwareScorer",
    "evaluate",
    "evaluate_scorer_on_slice",
    "write_run_result",
]

DEFAULT_BOOTSTRAP_RESAMPLES = 1000

_logger = logging.getLogger(__name__)


class Scorer(Protocol):
    """Anything exposing ``predict_proba(X) -> np.ndarray of P(positive)``."""

    def predict_proba(  # pragma: no cover
        self, X: list[str] | pd.Series | np.ndarray
    ) -> np.ndarray:
        """Return one P(positive) score per input feature row."""
        ...


class SliceAwareScorer(Scorer, Protocol):
    """Optional scorer contract for cost-controlled slice skipping."""

    def should_score_slice(self, slice_name: str) -> bool:  # pragma: no cover
        """Return whether this scorer should run on the named slice."""
        ...


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


@dataclass(slots=True)
class RunResult:
    """Outcome of a full evaluation run."""

    run_id: str
    git_sha: str | None
    config: dict[str, object]
    by_slice: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize using the stable JSON schema."""
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "config": self.config,
            "by_slice": self.by_slice,
        }


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


def evaluate_scorer_on_slice(
    scorer: Scorer,
    slice_: EvalSlice,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
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

    Returns
    -------
    dict
        Headline metrics + ``pr_auc_ci`` + raw scores.
    """
    y_score = scorer.predict_proba(slice_.features)
    y_true = slice_.y_true
    metrics = headline_metrics(y_true, y_score, strata=slice_.strata)
    is_single_class = len({int(v) for v in y_true}) == 1
    metrics["is_single_class"] = is_single_class

    if is_single_class:
        metrics["pr_auc_ci"] = {"skipped": "single-class slice; PR-AUC is not meaningful"}
    elif len(y_true) >= 30:
        try:
            ci = bootstrap_ci(
                y_true, y_score, pr_auc, n_resamples=n_resamples, method="BCa", seed=seed
            )
            metrics["pr_auc_ci"] = ci.to_dict()
        except (ValueError, RuntimeError) as exc:
            metrics["pr_auc_ci"] = {"error": str(exc)}
    else:
        metrics["pr_auc_ci"] = {"skipped": f"n={len(y_true)} < 30"}

    metrics["scores"] = y_score.tolist()
    return dict(metrics)


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

    Returns
    -------
    RunResult
        Pure result; no filesystem touched. Pass to :func:`write_run_result`
        to persist.

    Raises
    ------
    ValueError
        If ``scorers`` or ``slices`` is empty.
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
    }
    if extra_config:
        config.update(dict(extra_config))
    result = RunResult(run_id=run_id, git_sha=git_sha, config=config)

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
                scorer, slice_, n_resamples=n_resamples, seed=seed
            )
            scores_by_scorer[sname] = np.asarray(slice_data[sname]["scores"], dtype=np.float64)
            elapsed = time.time() - t0
            pr = slice_data[sname]["pr_auc"]
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

        result.by_slice[slice_.name] = {
            "n": int(len(slice_.df)),
            "n_positive": int(slice_.y_true.sum()),
            "by_scorer": slice_data,
            "paired_diffs": diffs,
        }

    return result


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
    full_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    compact_path.write_text(json.dumps(_strip_scores(result.to_dict()), indent=2, default=str))
    return compact_path, full_path


def _strip_scores(d: dict[str, object]) -> dict[str, object]:
    """Drop the per-row ``scores`` arrays from the headline JSON."""
    out: dict[str, object] = json.loads(json.dumps(d, default=str))  # deep copy via JSON
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
