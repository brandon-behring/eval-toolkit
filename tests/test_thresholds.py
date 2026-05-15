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


# ---------------------------------------------------------------------------
# Exactness tests for TargetFPRSelector. Property tests in
# test_thresholds_props.py cover the FPR≤cap contract; these pin the EXACT
# chosen threshold for canonical inputs, catching the kind of off-by-one or
# wrong-tie-break bugs an invariant test doesn't.
#
# Note: sklearn's roc_curve uses `drop_intermediate=True`, which collapses
# collinear ROC segments. On perfectly-separable data, the trimmed curve has
# only the boundary point — the selector picks the highest-pos / lowest-pos
# threshold, never one "inside" the (FPR=0, TPR=1) plateau. These tests
# verify that exact analytical behavior.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_fpr_separable_picks_min_positive_threshold() -> None:
    """On perfectly-separable data, TargetFPRSelector picks the min-positive score for any small target.

    Setup: 100 negs at scores [0.00..0.99], 100 pos at [1.00..1.99]. The
    only achievable operating points are FPR=0 (any threshold > 0.99) and
    FPR=1 (any threshold ≤ 0.99). For target ≤ 1.0 the selector picks
    the smallest threshold meeting FPR ≤ target, which is exactly the
    min positive score (1.00). Below that, FPR jumps to 1.0.

    Confirms the trimmed-ROC handling: the selector does NOT try to
    extrapolate to the "missing" thresholds between 1.0 and 0.99.
    """
    neg_scores = np.linspace(0.00, 0.99, 100)
    pos_scores = np.linspace(1.00, 1.99, 100)
    y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    y_score = np.concatenate([neg_scores, pos_scores])

    for target_fpr in (0.01, 0.05, 0.10, 0.20):
        result = TargetFPRSelector(target_fpr).select(y_true, y_score)
        # On separable data, the analytical answer is the min positive score:
        # the boundary between FPR=0 and FPR=1.
        assert result.threshold == pytest.approx(1.0, abs=1e-9), (
            f"target_fpr={target_fpr}: expected threshold=1.0 (min pos score), "
            f"got {result.threshold}"
        )
        # Realized FPR is exactly 0 (no negatives ≥ 1.0) — well below target
        flagged_negs = int(np.sum(neg_scores >= result.threshold))
        assert flagged_negs == 0
        # Realized TPR is exactly 1.0 (all positives ≥ 1.0)
        flagged_pos = int(np.sum(pos_scores >= result.threshold))
        assert flagged_pos == 100


@pytest.mark.unit
def test_target_fpr_threshold_pinned_on_overlapping_canonical_input() -> None:
    """Golden-style pin of the exact chosen threshold for canonical (y, score) data.

    Property tests confirm FPR ≤ target for any input; this test pins the
    EXACT threshold the selector picks on a deterministic, overlapping
    distribution. If the selector's tie-breaking logic, eligibility rule,
    or drop_intermediate handling regresses, this fails.

    Data: balanced binary labels (n=500) with discriminative-but-overlapping
    Gaussian-noise scores at seed=42.
    """
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.3, size=n).astype(int)
    y_score = np.clip(0.5 + 0.4 * (y_true - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)

    # Pinned values: computed once on this exact data + selector logic.
    # Update only if the selector's algorithm intentionally changes (and
    # add a CHANGELOG entry explaining the drift).
    expected_thresholds = {
        0.01: 0.7497574619602165,
        0.05: 0.6276334925831316,
        0.10: 0.5496488304373,
        0.20: 0.45115882422782605,
    }
    # Run selector for each target; record actual + verify monotonic
    # relationship: lower target → higher (more conservative) threshold.
    actual: dict[float, float] = {}
    for target in (0.01, 0.05, 0.10, 0.20):
        result = TargetFPRSelector(target).select(y_true, y_score)
        actual[target] = result.threshold

        # Contract check: realized FPR ≤ target
        neg_mask = y_true == 0
        n_neg = int(neg_mask.sum())
        flagged_negs = int(np.sum(y_score[neg_mask] >= result.threshold))
        realized_fpr = flagged_negs / n_neg
        assert (
            realized_fpr <= target + 1e-9
        ), f"target={target}: realized_fpr={realized_fpr} exceeds target"

    # Monotonicity: looser target → equal or lower threshold
    thresholds_in_order = [actual[t] for t in sorted(actual)]  # 0.01, 0.05, 0.10, 0.20
    for i in range(len(thresholds_in_order) - 1):
        assert thresholds_in_order[i] >= thresholds_in_order[i + 1], (
            f"selector should pick lower threshold for higher target FPR, but "
            f"target {sorted(actual)[i]} → threshold {thresholds_in_order[i]} and "
            f"target {sorted(actual)[i + 1]} → threshold {thresholds_in_order[i + 1]} (non-monotone)"
        )

    # Hard pin: these values reproduce on numpy ≥2.0 + scikit-learn ≥1.5
    # for the seed=42 / n=500 / Gaussian-noise(0.2) construction above.
    # If sklearn changes drop_intermediate behavior or the selector's
    # eligibility/tie-break logic, these regenerate (run + copy actuals).
    for target, expected in expected_thresholds.items():
        assert actual[target] == pytest.approx(expected, abs=1e-9), (
            f"threshold at target_fpr={target} drifted: "
            f"actual={actual[target]} vs pinned={expected}"
        )
