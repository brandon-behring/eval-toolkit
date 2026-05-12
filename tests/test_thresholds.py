"""Smoke tests for the v0.7.0 ThresholdSelector Protocol + reference impls.

Each Selector must:
1. Implement the :class:`ThresholdSelector` Protocol (runtime instanceof check).
2. Produce a valid :class:`ThresholdResult` from a known-good fixture.
3. Reject malformed input with clear errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.calibration import CostMatrix
from eval_toolkit.metrics import ThresholdResult
from eval_toolkit.thresholds import (
    CISafeThresholdSelector,
    CostSensitiveSelector,
    MaxF1Selector,
    TargetFPRSelector,
    TargetPrecisionSelector,
    TargetRecallSelector,
    ThresholdPolicyMetadata,
    ThresholdSelector,
    YoudenJSelector,
    select_threshold,
    wilson_interval,
)


@pytest.fixture
def fixture() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic, informative binary data."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.25, 200), 0, 1)
    return y, s


@pytest.mark.unit
@pytest.mark.parametrize(
    "selector",
    [
        MaxF1Selector(),
        TargetRecallSelector(0.90),
        TargetPrecisionSelector(0.50),
        TargetFPRSelector(0.20),
        YoudenJSelector(),
        CostSensitiveSelector(CostMatrix(prior=0.3, fp_cost=1.0, fn_cost=2.0)),
    ],
)
def test_selectors_implement_protocol(selector: ThresholdSelector) -> None:
    """Every reference impl satisfies the runtime_checkable Protocol."""
    assert isinstance(selector, ThresholdSelector)


@pytest.mark.unit
def test_max_f1_returns_valid_result(fixture: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = fixture
    result = MaxF1Selector().select(y, s)
    assert isinstance(result, ThresholdResult)
    assert result.criterion == "max_f1"
    assert 0.0 <= result.threshold <= 1.0
    assert 0.0 <= result.f1 <= 1.0


@pytest.mark.unit
def test_target_recall_meets_target(fixture: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = fixture
    result = TargetRecallSelector(0.90).select(y, s)
    assert result.criterion == "recall_0.90"
    assert result.recall >= 0.90 - 1e-6


@pytest.mark.unit
def test_target_precision_meets_target(fixture: tuple[np.ndarray, np.ndarray]) -> None:
    """v0.7.0 NEW selector — required by prompt-injection-sdd's 'precision@0.90' workflow."""
    y, s = fixture
    result = TargetPrecisionSelector(0.50).select(y, s)
    assert result.criterion == "precision_0.50"
    assert result.precision >= 0.50 - 1e-6


@pytest.mark.unit
def test_youden_j_returns_valid_result(fixture: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = fixture
    result = YoudenJSelector().select(y, s)
    assert result.criterion == "youden_j"
    assert 0.0 <= result.threshold <= 1.0


@pytest.mark.unit
def test_cost_sensitive_uses_bayes_optimal_threshold() -> None:
    """At symmetric costs and prior=0.5, threshold should equal the prior."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    cm = CostMatrix(prior=0.5, fp_cost=1.0, fn_cost=1.0)
    result = CostSensitiveSelector(cm).select(y, s)
    assert result.threshold == 0.5


@pytest.mark.unit
def test_wilson_interval_and_ci_safe_selector() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.9])
    interval = wilson_interval(0, 4)
    selector = CISafeThresholdSelector(max_fpr=0.0, max_fpr_ci_upper=0.55, min_recall=0.5)

    selected = selector.select(y, s)
    metadata = ThresholdPolicyMetadata(
        calibration_slice="calibration",
        score_column="score",
        selector=selector.criterion,
        constraints=selector.constraints,
    )
    operating_point = selector.selected_operating_point(y, s)

    assert interval.high is not None and interval.high > 0.0
    assert selected.criterion == "ci_safe"
    assert selected.recall >= 0.5
    assert metadata.to_dict()["claim_enabled"] is False
    assert operating_point["selected_record"]["accepted"] is True  # type: ignore[index]


@pytest.mark.unit
def test_select_threshold_rejects_string_criterion() -> None:
    """v0.7.0 BREAKING — string criterion form is removed."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(TypeError, match="ThresholdSelector instance"):
        select_threshold(y, s, criterion="max_f1")  # type: ignore[arg-type]


@pytest.mark.unit
def test_target_recall_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="recall must be in"):
        TargetRecallSelector(0.0)
    with pytest.raises(ValueError, match="recall must be in"):
        TargetRecallSelector(1.5)


@pytest.mark.unit
def test_target_precision_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="precision must be in"):
        TargetPrecisionSelector(0.0)
    with pytest.raises(ValueError, match="precision must be in"):
        TargetPrecisionSelector(1.5)


@pytest.mark.unit
def test_target_fpr_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="fpr must be in"):
        TargetFPRSelector(-0.1)
    with pytest.raises(ValueError, match="fpr must be in"):
        TargetFPRSelector(1.5)
