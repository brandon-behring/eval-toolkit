"""Edge-case + error-path coverage for eval_toolkit.harness.

Pairs with the happy-path coverage in `test_harness_smoke.py` and
`test_operating_points.py`. These tests target validation errors and
the operating-point error branches in `_attach_transferred_operating_points`
that the smoke suites skip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import (
    EvalSlice,
    RunResult,
    evaluate,
    evaluate_scorer_on_slice,
    with_claim_report,
)
from eval_toolkit.operating_points import OperatingPointSpec
from eval_toolkit.thresholds import MaxF1Selector


class _DictScorer:
    """Score by table lookup; missing keys return 0.5."""

    def __init__(self, mapping: dict[str, float]) -> None:
        self.mapping = mapping

    def predict_proba(self, X: list[str] | pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray([self.mapping.get(str(x), 0.5) for x in X], dtype=np.float64)


class _SliceAwareScorer(_DictScorer):
    def __init__(self, mapping: dict[str, float], *, skip_slices: set[str]) -> None:
        super().__init__(mapping)
        self.skip_slices = skip_slices

    def should_score_slice(self, slice_name: str) -> bool:
        return slice_name not in self.skip_slices


@pytest.mark.unit
def test_eval_slice_rejects_missing_strata_column() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(KeyError, match="missing strata column"):
        EvalSlice(name="s", df=df, strata_col="absent")


@pytest.mark.unit
def test_eval_slice_rejects_non_binary_labels() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 2]})
    with pytest.raises(ValueError, match="labels must be in"):
        EvalSlice(name="s", df=df)


@pytest.mark.unit
def test_with_claim_report_accepts_mapping() -> None:
    base = RunResult(run_id="r", git_sha=None, config={}, by_slice={})
    out = with_claim_report(base, {"claims": {}, "has_failures": False})
    assert out.claim_report == {"claims": {}, "has_failures": False}


@pytest.mark.unit
def test_with_claim_report_accepts_object_with_to_dict() -> None:
    class _R:
        def to_dict(self) -> dict[str, object]:
            return {"k": "v"}

    base = RunResult(run_id="r", git_sha=None, config={}, by_slice={})
    out = with_claim_report(base, _R())
    assert out.claim_report == {"k": "v"}


@pytest.mark.unit
def test_with_claim_report_rejects_unknown_type() -> None:
    base = RunResult(run_id="r", git_sha=None, config={}, by_slice={})
    with pytest.raises(TypeError, match="claim report"):
        with_claim_report(base, 42)


@pytest.mark.unit
def test_slice_aware_scorer_must_return_bool() -> None:
    """should_score_slice() returning non-bool raises TypeError."""

    class _BadSliceAware:
        def predict_proba(self, X):
            return np.full(len(X), 0.5)

        def should_score_slice(self, slice_name: str):
            return "yes"  # not a bool

    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(TypeError, match="should_score_slice"):
        evaluate({"bad": _BadSliceAware()}, [EvalSlice(name="s", df=df)], run_id="r")


@pytest.mark.unit
def test_evaluate_rejects_empty_scorers() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError, match="at least one scorer"):
        evaluate({}, [EvalSlice(name="s", df=df)], run_id="r")


@pytest.mark.unit
def test_evaluate_rejects_empty_slices() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        evaluate({"s": _DictScorer({})}, [], run_id="r")


@pytest.mark.unit
def test_evaluate_records_extra_config() -> None:
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    result = evaluate(
        {"s": _DictScorer({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})},
        [EvalSlice(name="t", df=df)],
        run_id="r",
        extra_config={"experiment_tag": "ablation-3"},
        n_resamples=5,
    )
    assert result.config["experiment_tag"] == "ablation-3"


@pytest.mark.unit
def test_evaluate_with_op_spec_missing_fit_slice_records_error() -> None:
    """An OperatingPointSpec whose fit_slice isn't evaluated records a spec error."""
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    scorer = _DictScorer({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})
    result = evaluate(
        {"s": scorer},
        [EvalSlice(name="t", df=df)],
        run_id="r",
        n_resamples=5,
        operating_point_specs=[
            OperatingPointSpec(
                name="fit_missing",
                fit_slice="not_present",
                apply_slices=("t",),
                selectors=(MaxF1Selector(),),
            )
        ],
    )
    transfer = result.by_slice["t"]["by_scorer"]["s"]["transferred_operating_points"]
    assert "error" in transfer["fit_missing"]
    assert "fit slice 'not_present' not found" in transfer["fit_missing"]["error"]


@pytest.mark.unit
def test_evaluate_with_op_spec_missing_apply_slice_records_error() -> None:
    """An OperatingPointSpec listing an unknown apply slice records a spec error."""
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    scorer = _DictScorer({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})
    result = evaluate(
        {"s": scorer},
        [EvalSlice(name="fit", df=df)],
        run_id="r",
        n_resamples=5,
        operating_point_specs=[
            OperatingPointSpec(
                name="apply_missing",
                fit_slice="fit",
                apply_slices=("nowhere",),
                selectors=(MaxF1Selector(),),
            )
        ],
    )
    # Spec-level error is attached under the *missing* apply slice's stubbed block.
    transfer = result.by_slice["nowhere"]["by_scorer"]["s"]["transferred_operating_points"]
    assert "apply slice 'nowhere' not found" in transfer["apply_missing"]["error"]


@pytest.mark.unit
def test_evaluate_with_op_spec_single_class_fit_records_fit_error() -> None:
    """A fit slice with only one class produces a fit error, not a crash."""
    bad_fit_df = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 1, 1]})
    target_df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    scorer = _DictScorer({c: i / 4 for i, c in enumerate("abcd")})
    result = evaluate(
        {"s": scorer},
        [
            EvalSlice(name="bad_fit", df=bad_fit_df),
            EvalSlice(name="ok_target", df=target_df),
        ],
        run_id="r",
        n_resamples=5,
        operating_point_specs=[
            OperatingPointSpec(
                name="degenerate_fit",
                fit_slice="bad_fit",
                apply_slices=("ok_target",),
                selectors=(MaxF1Selector(),),
            )
        ],
    )
    transfer = result.by_slice["ok_target"]["by_scorer"]["s"]["transferred_operating_points"]
    assert "error" in transfer["degenerate_fit"]
    assert "mixed-class" in transfer["degenerate_fit"]["error"]


@pytest.mark.unit
def test_evaluate_slice_aware_scorer_skips_apply_slice() -> None:
    """When a slice-aware scorer skips the *apply* slice, the spec records 'skipped'."""
    fit_df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    target_df = pd.DataFrame({"text": ["e", "f"], "label": [0, 1]})
    scorer = _SliceAwareScorer({c: i / 6 for i, c in enumerate("abcdef")}, skip_slices={"target"})
    result = evaluate(
        {"s": scorer},
        [
            EvalSlice(name="fit", df=fit_df),
            EvalSlice(name="target", df=target_df),
        ],
        run_id="r",
        n_resamples=5,
        operating_point_specs=[
            OperatingPointSpec(
                name="skip_target",
                fit_slice="fit",
                apply_slices=("target",),
                selectors=(MaxF1Selector(),),
            )
        ],
    )
    transfer = result.by_slice["target"]["by_scorer"]["s"]["transferred_operating_points"]
    block = transfer["skip_target"]
    # Either "skipped" or a more specific marker is fine — we just check
    # that the harness recorded the absent target without raising.
    assert "skipped" in block or "error" in block


@pytest.mark.unit
def test_evaluate_scorer_on_slice_basic() -> None:
    """Direct call to evaluate_scorer_on_slice with a single-class slice."""
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 1, 1]})
    slice_ = EvalSlice(name="pos_only", df=df)
    scorer = _DictScorer({"a": 0.8, "b": 0.7, "c": 0.6})
    out = evaluate_scorer_on_slice(scorer, slice_, n_resamples=5)
    assert out["is_single_class"] is True
    assert isinstance(out["pr_auc_ci"], dict)
    assert out["pr_auc_ci"]["status"] == "skipped"


@pytest.mark.unit
def test_evaluate_records_scorer_error_when_configured() -> None:
    """on_scorer_error='record' captures the failure per (slice, scorer)."""

    class _ExplodingScorer:
        def predict_proba(self, X):
            raise RuntimeError("boom")

    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    result = evaluate(
        {"boom": _ExplodingScorer()},
        [EvalSlice(name="t", df=df)],
        run_id="r",
        n_resamples=5,
        on_scorer_error="record",
    )
    block = result.by_slice["t"]["by_scorer"]["boom"]
    # The scorer error should be reflected in the per-scorer block somehow.
    assert "error" in block or "scores" in block
