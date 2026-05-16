"""Tests for the v0.7.0 evaluate_folded orchestrator (k-fold/holdout eval).

Covers the additive RunResult.by_fold + RunResult.fold_summary fields and
every entry point into evaluate_folded (holdout, k-fold, multi-seed, and
error paths).

Extracted from the version-keyed test_harness_v07.py during the v0.30.1
hygiene split — feature-grouped instead of release-grouped naming. Every
assertion preserved verbatim from the v07 file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eval_toolkit.harness import (
    RUN_RESULT_SCHEMA_VERSION,
    EvalSlice,
    RunResult,
    evaluate,
    evaluate_folded,
)
from eval_toolkit.splits import HoldoutSplitter, StratifiedKFoldSplitter

# v0.30.0 refactor #4: shared scorer doubles moved to tests/conftest.py.
from tests.conftest import UniformScorer as _UniformScorer  # noqa: E402


@pytest.fixture
def big_slice() -> EvalSlice:
    """60 rows; balanced labels."""
    df = pd.DataFrame({"text": [f"t{i}" for i in range(60)], "label": [i % 2 for i in range(60)]})
    return EvalSlice(name="test", df=df)


# ---------------------------------------------------------------------------
# RunResult.by_fold / fold_summary additive fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_result_to_dict_serializes_by_fold_and_summary(big_slice: EvalSlice) -> None:
    """to_dict round-trips by_fold + fold_summary additive fields."""
    result = evaluate({"u": _UniformScorer()}, [big_slice], run_id="r")
    payload = result.to_dict()
    # Non-folded run: both default-empty.
    assert payload["by_fold"] == {}
    assert payload["fold_summary"] == {}


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
@pytest.mark.slow
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


# Trivial assertion to lock the schema-version constant — not folded-specific,
# but lives here because it imports the same RunResult+evaluate stack and
# requires zero extra fixtures.
@pytest.mark.unit
def test_run_result_schema_version_default(big_slice: EvalSlice) -> None:
    result = evaluate({"u": _UniformScorer()}, [big_slice], run_id="r")
    assert result.schema_version == RUN_RESULT_SCHEMA_VERSION
    assert result.schema_version == "v1"
    assert result.to_dict()["schema_version"] == "v1"
