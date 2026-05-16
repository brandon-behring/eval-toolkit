"""Smoke tests for the harness — pure/IO split, slice-aware skip, EvalSlice column flexibility."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.claims import ClaimReport, GateResult
from eval_toolkit.harness import (
    EvalSlice,
    RunResult,
    Scorer,
    evaluate,
    with_claim_report,
    write_run_result,
)

# Scorer doubles moved to tests/conftest.py (v0.30.0 refactor #4); import
# aliases preserve existing usage names with minimal diff churn.
from tests.conftest import ErrorScorer as _BrokenScorer  # noqa: E402
from tests.conftest import StubScorer as _StubScorer  # noqa: E402
from tests.conftest import UniformScorer as _UniformScorer  # noqa: E402


class _SliceAwareStub:
    def __init__(self, scores: np.ndarray, allow: set[str]) -> None:
        self._scores = scores
        self._allow = allow

    def predict_proba(self, X: object) -> np.ndarray:
        return self._scores

    def should_score_slice(self, slice_name: str) -> bool:
        return slice_name in self._allow


@pytest.fixture
def slice_with_data() -> EvalSlice:
    rng = np.random.default_rng(42)
    n = 100
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "text": [f"row{i}" for i in range(n)],
            "label": y,
            "family": rng.choice(["A", "B"], size=n),
        }
    )
    return EvalSlice(name="test", df=df, strata_col="family")


@pytest.mark.smoke
def test_evalslice_validates_columns() -> None:
    df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    with pytest.raises(KeyError, match="text"):
        EvalSlice(name="bad", df=df)


@pytest.mark.smoke
def test_evalslice_custom_columns() -> None:
    """feature_col / label_col / strata_col are configurable."""
    df = pd.DataFrame({"input": ["a", "b", "c"], "y": [0, 1, 0], "group": ["x", "y", "x"]})
    s = EvalSlice(name="custom", df=df, feature_col="input", label_col="y", strata_col="group")
    assert s.features == ["a", "b", "c"]
    assert list(s.y_true) == [0, 1, 0]


@pytest.mark.smoke
def test_evaluate_pure_no_filesystem(slice_with_data: EvalSlice, tmp_path: Path) -> None:
    """evaluate(...) does not write any files."""
    rng = np.random.default_rng(42)
    sc = _StubScorer(rng.uniform(0, 1, size=len(slice_with_data.df)))
    result = evaluate({"stub": sc}, [slice_with_data], run_id="pure-test", n_resamples=50)
    assert isinstance(result, RunResult)
    assert result.run_id == "pure-test"
    assert result.git_sha is None
    assert result.claim_report == {}
    assert result.to_dict()["claim_report"] == {}
    # Nothing should have been written into tmp_path
    assert list(tmp_path.iterdir()) == []


@pytest.mark.smoke
def test_with_claim_report_returns_new_frozen_result(slice_with_data: EvalSlice) -> None:
    rng = np.random.default_rng(42)
    sc = _StubScorer(rng.uniform(0, 1, size=len(slice_with_data.df)))
    result = evaluate({"stub": sc}, [slice_with_data], run_id="claim-test", n_resamples=20)
    report = ClaimReport(claims={"claim": [GateResult(name="gate", passed=True, message="ok")]})

    enriched = with_claim_report(result, report)

    assert result.claim_report == {}
    assert enriched.claim_report["has_failures"] is False
    assert enriched.claim_report["claims"]["claim"][0]["passed"] is True
    assert enriched.run_id == result.run_id
    with pytest.raises(FrozenInstanceError):
        enriched.claim_report = {}


@pytest.mark.smoke
def test_with_claim_report_accepts_mapping(slice_with_data: EvalSlice) -> None:
    rng = np.random.default_rng(42)
    sc = _StubScorer(rng.uniform(0, 1, size=len(slice_with_data.df)))
    result = evaluate({"stub": sc}, [slice_with_data], run_id="claim-map", n_resamples=20)

    enriched = with_claim_report(result, {"claims": {}, "has_failures": False})

    assert enriched.to_dict()["claim_report"] == {"claims": {}, "has_failures": False}


@pytest.mark.smoke
def test_evaluate_idempotent_for_same_seed(slice_with_data: EvalSlice) -> None:
    """Same inputs → identical RunResult JSON."""
    rng = np.random.default_rng(42)
    s = rng.uniform(0, 1, size=len(slice_with_data.df))
    sc = _StubScorer(s)
    r1 = evaluate({"stub": sc}, [slice_with_data], run_id="x", n_resamples=50, seed=42)
    r2 = evaluate({"stub": sc}, [slice_with_data], run_id="x", n_resamples=50, seed=42)
    assert json.dumps(r1.to_dict(), default=str) == json.dumps(r2.to_dict(), default=str)


@pytest.mark.smoke
def test_evaluate_paired_diffs(slice_with_data: EvalSlice) -> None:
    """paired_diffs are computed and labeled correctly."""
    rng = np.random.default_rng(42)
    s_a = rng.uniform(0, 1, size=len(slice_with_data.df))
    # Mix the signal in [0, 1]; v0.3.0 ECE validators reject out-of-range scores.
    s_b = np.clip(s_a + 0.1 * slice_with_data.y_true, 0.0, 1.0)
    sc = {"a": _StubScorer(s_a), "b": _StubScorer(s_b)}
    result = evaluate(sc, [slice_with_data], run_id="x", n_resamples=50, paired_diffs=[("a", "b")])
    diffs = result.by_slice["test"]["paired_diffs"]
    assert "b_minus_a" in diffs


@pytest.mark.smoke
def test_slice_aware_scorer_skips() -> None:
    """SliceAwareScorer with should_score_slice returning False causes skip."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    slice1 = EvalSlice(name="slice_1", df=df)
    slice2 = EvalSlice(name="slice_2", df=df)
    sc = _SliceAwareStub(rng.uniform(0, 1, size=4), allow={"slice_1"})
    result = evaluate({"sa": sc}, [slice1, slice2], run_id="x", n_resamples=20)
    s1 = result.by_slice["slice_1"]["by_scorer"]["sa"]
    s2 = result.by_slice["slice_2"]["by_scorer"]["sa"]
    assert "skipped" not in s1
    assert "skipped" in s2


@pytest.mark.smoke
def test_write_run_result_creates_two_jsons(slice_with_data: EvalSlice, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    sc = _StubScorer(rng.uniform(0, 1, size=len(slice_with_data.df)))
    result = evaluate({"stub": sc}, [slice_with_data], run_id="rt", n_resamples=20)
    run_dir = tmp_path / "run_rt"
    compact, full = write_run_result(result, run_dir)
    assert compact.exists()
    assert full.exists()
    # Compact should not contain raw scores
    compact_data = json.loads(compact.read_text())
    full_data = json.loads(full.read_text())
    compact_scorer = compact_data["by_slice"]["test"]["by_scorer"]["stub"]
    full_scorer = full_data["by_slice"]["test"]["by_scorer"]["stub"]
    assert "scores" not in compact_scorer
    assert "scores" in full_scorer


@pytest.mark.smoke
def test_evaluate_validates_inputs(slice_with_data: EvalSlice) -> None:
    with pytest.raises(ValueError, match="scorer"):
        evaluate({}, [slice_with_data], run_id="x")
    with pytest.raises(ValueError, match="slice"):
        evaluate({"a": _StubScorer(np.zeros(1))}, [], run_id="x")


@pytest.mark.smoke
def test_scorer_protocol_runtime_check() -> None:
    """Any object with predict_proba is accepted by the Protocol."""
    scorer: Scorer = _StubScorer(np.array([0.1, 0.9]))
    assert hasattr(scorer, "predict_proba")


# ---------------------------------------------------------------------------
# on_scorer_error contract (raise / record paths)
# Migrated from test_harness_v07.py during v0.30.1 hygiene split —
# feature-grouped instead of release-grouped.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_scorer_error_raise_propagates(slice_with_data: EvalSlice) -> None:
    with pytest.raises(RuntimeError, match="intentional failure"):
        evaluate(
            {"broken": _BrokenScorer()},
            [slice_with_data],
            run_id="r",
            on_scorer_error="raise",
        )


@pytest.mark.unit
def test_on_scorer_error_record_captures(slice_with_data: EvalSlice) -> None:
    """on_scorer_error='record' captures error per (slice, scorer); run completes."""
    result = evaluate(
        {"broken": _BrokenScorer()},
        [slice_with_data],
        run_id="r",
        on_scorer_error="record",
    )
    entry = result.by_slice["test"]["by_scorer"]["broken"]
    assert entry["error"] == "intentional failure for tests"
    assert entry["exc_type"] == "RuntimeError"
    assert "traceback" in entry
    assert entry["scores"] == []


@pytest.mark.unit
def test_on_scorer_error_record_does_not_break_other_scorers(
    slice_with_data: EvalSlice,
) -> None:
    """A broken scorer should not poison the run for healthy scorers."""
    result = evaluate(
        {"broken": _BrokenScorer(), "healthy": _UniformScorer()},
        [slice_with_data],
        run_id="r",
        on_scorer_error="record",
    )
    assert "error" in result.by_slice["test"]["by_scorer"]["broken"]
    healthy = result.by_slice["test"]["by_scorer"]["healthy"]
    assert isinstance(healthy.get("pr_auc"), float)
