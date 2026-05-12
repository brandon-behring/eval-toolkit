"""Tests for cross-slice operating-point transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice, evaluate
from eval_toolkit.operating_points import (
    OperatingPointSpec,
    apply_operating_points,
    fit_operating_points,
)
from eval_toolkit.thresholds import MaxF1Selector, TargetPrecisionSelector


class _TextScoreScorer:
    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text

    def predict_proba(self, X: list[str] | pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray([self.scores_by_text[str(x)] for x in X], dtype=np.float64)


@pytest.mark.unit
def test_fit_and_apply_operating_points_mixed_and_single_class() -> None:
    y_fit = np.array([0, 0, 1, 1])
    s_fit = np.array([0.1, 0.2, 0.8, 0.9])
    fitted = fit_operating_points(
        y_fit,
        s_fit,
        [MaxF1Selector()],
        fitted_on_slice="validation",
        scorer_name="model",
    )

    mixed = apply_operating_points(
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        fitted,
        applied_to_slice="mixed_target",
        scorer_name="model",
    )
    assert mixed["max_f1"]["f1"] == pytest.approx(1.0)
    assert mixed["max_f1"]["threshold_provenance"]["fitted_on_slice"] == "validation"

    all_positive = apply_operating_points(
        np.array([1, 1]),
        np.array([0.85, 0.4]),
        fitted,
        applied_to_slice="ood_positive",
        scorer_name="model",
    )
    assert all_positive["max_f1"]["slice_class"] == "all_positive"
    assert all_positive["max_f1"]["recall@threshold"] == pytest.approx(0.5)

    all_negative = apply_operating_points(
        np.array([0, 0]),
        np.array([0.1, 0.9]),
        fitted,
        applied_to_slice="hard_negative",
        scorer_name="model",
    )
    assert all_negative["max_f1"]["slice_class"] == "all_negative"
    assert all_negative["max_f1"]["fpr@threshold"] == pytest.approx(0.5)


@pytest.mark.unit
def test_fit_operating_points_rejects_single_class_fit_slice() -> None:
    with pytest.raises(ValueError, match="mixed-class"):
        fit_operating_points(
            np.array([1, 1, 1]),
            np.array([0.2, 0.4, 0.8]),
            [MaxF1Selector()],
        )


@pytest.mark.unit
def test_fit_operating_points_surfaces_selector_failure() -> None:
    with pytest.raises(RuntimeError, match="No threshold achieves precision"):
        fit_operating_points(
            np.array([0, 1, 0, 1]),
            np.array([0.9, 0.8, 0.7, 0.6]),
            [TargetPrecisionSelector(1.0)],
        )


@pytest.mark.smoke
def test_evaluate_attaches_transferred_operating_points_sdd_shape() -> None:
    val_df = pd.DataFrame(
        {
            "text": ["v0", "v1", "v2", "v3"],
            "label": [0, 0, 1, 1],
        }
    )
    ood_pos_df = pd.DataFrame({"text": ["p0", "p1"], "label": [1, 1]})
    ood_neg_df = pd.DataFrame({"text": ["n0", "n1"], "label": [0, 0]})
    scores = {
        "v0": 0.1,
        "v1": 0.2,
        "v2": 0.8,
        "v3": 0.9,
        "p0": 0.85,
        "p1": 0.4,
        "n0": 0.1,
        "n1": 0.9,
    }
    result = evaluate(
        {"model": _TextScoreScorer(scores)},
        [
            EvalSlice(name="validation", df=val_df),
            EvalSlice(name="ood_positive", df=ood_pos_df),
            EvalSlice(name="hard_negative", df=ood_neg_df),
        ],
        run_id="transfer",
        n_resamples=10,
        operating_point_specs=[
            OperatingPointSpec(
                name="validation_fit",
                fit_slice="validation",
                apply_slices=("ood_positive", "hard_negative"),
                selectors=(MaxF1Selector(),),
            )
        ],
    )

    pos = result.by_slice["ood_positive"]["by_scorer"]["model"]["transferred_operating_points"][
        "validation_fit"
    ]["max_f1"]
    neg = result.by_slice["hard_negative"]["by_scorer"]["model"]["transferred_operating_points"][
        "validation_fit"
    ]["max_f1"]
    assert pos["recall@threshold"] == pytest.approx(0.5)
    assert neg["fpr@threshold"] == pytest.approx(0.5)
    assert pos["threshold_provenance"]["fitted_on_slice"] == "validation"
