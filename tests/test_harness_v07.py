"""Tests for the v0.7.0 harness additions: leakage_checks, on_scorer_error, evaluate_folded."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import (
    RUN_RESULT_SCHEMA_VERSION,
    EvalSlice,
    RunResult,
    evaluate,
    evaluate_folded,
)
from eval_toolkit.leakage import LabelConflictCheck, NormalizedFormLeakageCheck
from eval_toolkit.splits import HoldoutSplitter, StratifiedKFoldSplitter


class _UniformScorer:
    """Deterministic scorer for tests — returns shuffled uniforms keyed on seed=42."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=len(X))


class _BrokenScorer:
    """Always raises RuntimeError. Used to exercise on_scorer_error paths."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        raise RuntimeError("intentional failure for tests")


@pytest.fixture
def big_slice() -> EvalSlice:
    """60 rows; balanced labels."""
    df = pd.DataFrame({"text": [f"t{i}" for i in range(60)], "label": [i % 2 for i in range(60)]})
    return EvalSlice(name="test", df=df)


# ---------------------------------------------------------------------------
# RunResult schema version
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_result_schema_version_default(big_slice: EvalSlice) -> None:
    result = evaluate({"u": _UniformScorer()}, [big_slice], run_id="r")
    assert result.schema_version == RUN_RESULT_SCHEMA_VERSION
    assert result.schema_version == "v1"
    assert result.to_dict()["schema_version"] == "v1"


@pytest.mark.unit
def test_run_result_to_dict_serializes_by_fold_and_summary(big_slice: EvalSlice) -> None:
    """to_dict round-trips by_fold + fold_summary additive fields."""
    result = evaluate({"u": _UniformScorer()}, [big_slice], run_id="r")
    payload = result.to_dict()
    # Non-folded run: both default-empty.
    assert payload["by_fold"] == {}
    assert payload["fold_summary"] == {}


# ---------------------------------------------------------------------------
# on_scorer_error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_scorer_error_raise_propagates(big_slice: EvalSlice) -> None:
    with pytest.raises(RuntimeError, match="intentional failure"):
        evaluate({"broken": _BrokenScorer()}, [big_slice], run_id="r", on_scorer_error="raise")


@pytest.mark.unit
def test_on_scorer_error_record_captures(big_slice: EvalSlice) -> None:
    """on_scorer_error='record' captures error per (slice, scorer); run completes."""
    result = evaluate(
        {"broken": _BrokenScorer()}, [big_slice], run_id="r", on_scorer_error="record"
    )
    entry = result.by_slice["test"]["by_scorer"]["broken"]
    assert entry["error"] == "intentional failure for tests"
    assert entry["exc_type"] == "RuntimeError"
    assert "traceback" in entry
    assert entry["scores"] == []


@pytest.mark.unit
def test_on_scorer_error_record_does_not_break_other_scorers(big_slice: EvalSlice) -> None:
    """A broken scorer should not poison the run for healthy scorers."""
    result = evaluate(
        {"broken": _BrokenScorer(), "healthy": _UniformScorer()},
        [big_slice],
        run_id="r",
        on_scorer_error="record",
    )
    assert "error" in result.by_slice["test"]["by_scorer"]["broken"]
    healthy = result.by_slice["test"]["by_scorer"]["healthy"]
    assert isinstance(healthy.get("pr_auc"), float)


# ---------------------------------------------------------------------------
# leakage_checks integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_leakage_checks_record_mode_captures_report(big_slice: EvalSlice) -> None:
    result = evaluate(
        {"u": _UniformScorer()},
        [big_slice],
        run_id="r",
        leakage_checks=[NormalizedFormLeakageCheck()],
        on_leakage="record",
    )
    assert "leakage_report" in result.config
    assert isinstance(result.config["leakage_report"], dict)


@pytest.mark.unit
def test_leakage_checks_skip_mode_omits_report(big_slice: EvalSlice) -> None:
    result = evaluate(
        {"u": _UniformScorer()},
        [big_slice],
        run_id="r",
        leakage_checks=[NormalizedFormLeakageCheck()],
        on_leakage="skip",
    )
    # In skip mode, the report is run but not recorded.
    assert "leakage_report" not in result.config


@pytest.mark.unit
def test_leakage_checks_raise_on_error_finding() -> None:
    """on_leakage='raise' with a real conflict should fail the run."""
    df = pd.DataFrame({"text": ["x", "y"], "label": [0, 1]})
    df_test = pd.DataFrame({"text": ["x", "z"], "label": [1, 0]})  # label conflict on "x"
    train = EvalSlice(name="train", df=df)
    test = EvalSlice(name="test", df=df_test)
    with pytest.raises(RuntimeError, match="Leakage checks produced"):
        evaluate(
            {"u": _UniformScorer()},
            [train, test],
            run_id="r",
            leakage_checks=[LabelConflictCheck()],
            on_leakage="raise",
        )


# ---------------------------------------------------------------------------
# evaluate_folded
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_folded_with_holdout(big_slice: EvalSlice) -> None:
    """k=1 holdout → 1 entry in by_fold."""
    result = evaluate_folded(
        {"u": _UniformScorer()},
        HoldoutSplitter(test_size=0.25, seed=42),
        big_slice,
        run_id="r",
        eval_split_names=("test",),
    )
    assert isinstance(result, RunResult)
    assert len(result.by_fold) == 1
    # by_slice empty for folded runs; fold_summary populated.
    assert result.by_slice == {}
    assert "test" in result.fold_summary


@pytest.mark.unit
def test_evaluate_folded_with_kfold(big_slice: EvalSlice) -> None:
    """k=4 stratified → 4 fold entries."""
    result = evaluate_folded(
        {"u": _UniformScorer()},
        StratifiedKFoldSplitter(k=4, seed=42),
        big_slice,
        run_id="r",
        eval_split_names=("test",),
    )
    assert len(result.by_fold) == 4
    summary = result.fold_summary["test"]["u"]["pr_auc"]
    assert "mean" in summary
    assert "ci_low" in summary
    assert "ci_high" in summary
    assert summary["n_folds"] == 4


@pytest.mark.unit
def test_evaluate_folded_multi_seed(big_slice: EvalSlice) -> None:
    """seeds=(1, 2, 3) × k=2 = 6 fold entries."""
    result = evaluate_folded(
        {"u": _UniformScorer()},
        StratifiedKFoldSplitter(k=2, seed=42),
        big_slice,
        run_id="r",
        seeds=(1, 2, 3),
        eval_split_names=("test",),
    )
    assert len(result.by_fold) == 6
    # fold_summary aggregates across all 6
    summary = result.fold_summary["test"]["u"]["pr_auc"]
    assert summary["n_folds"] == 6


@pytest.mark.unit
def test_evaluate_folded_empty_scorers_raises() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError, match="at least one scorer"):
        evaluate_folded(
            {},
            StratifiedKFoldSplitter(k=2, seed=42),
            EvalSlice(name="x", df=df),
            run_id="r",
        )


@pytest.mark.unit
def test_evaluate_folded_missing_eval_split_raises(big_slice: EvalSlice) -> None:
    """If eval_split_names doesn't intersect any fold's keys, raise."""
    with pytest.raises(ValueError, match="none of eval_split_names"):
        evaluate_folded(
            {"u": _UniformScorer()},
            StratifiedKFoldSplitter(k=3, seed=42),
            big_slice,
            run_id="r",
            eval_split_names=("nope",),
        )
