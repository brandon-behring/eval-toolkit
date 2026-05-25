"""Tests for `eval_toolkit.recall_at_fpr` + `RecallAtFprResult` (closes #9).

One-shot convenience over `TargetFPRSelector` for the recall@FPR
screening-workflow use case. Returns a typed frozen dataclass with
``.to_dict()`` for JSON / pandas-row integration.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eval_toolkit import RecallAtFprResult, recall_at_fpr


def _make_discriminative_inputs(
    n_neg: int = 100, n_pos: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced binary inputs with a clear scoring gap (positives > negatives)."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    s = np.concatenate(
        [
            rng.uniform(0.0, 0.5, size=n_neg),  # negatives in [0, 0.5)
            rng.uniform(0.5, 1.0, size=n_pos),  # positives in [0.5, 1.0)
        ]
    )
    return y, s


@pytest.mark.unit
def test_recall_at_fpr_happy_path_returns_typed_dataclass() -> None:
    """Discriminative scorer → recall > 0.9 at FPR ceiling 0.1 + dataclass shape."""
    y, s = _make_discriminative_inputs()
    result = recall_at_fpr(y, s, target_fpr=0.10)
    assert isinstance(result, RecallAtFprResult)
    assert result.recall >= 0.9, f"expected high recall on discriminative data; got {result.recall}"
    assert result.actual_fpr <= 0.10 + 1e-9, "FPR ceiling honored"
    assert result.n_val_neg == 100
    assert result.fp + result.tn == 100  # negatives partition into fp + tn
    assert 0.0 <= result.threshold <= 1.0


@pytest.mark.unit
def test_recall_at_fpr_degenerate_no_negatives_returns_zero_fpr() -> None:
    """All-positive y_true → n_val_neg=0; actual_fpr=0 by definition (no division).

    TargetFPRSelector rejects target_fpr<0 in __post_init__, so we can't force
    the "no threshold meets target" path on a well-formed scorer. The degenerate
    edge that DOES exercise the dataclass-fallback path is no-negatives input
    (FPR denominator is zero).
    """
    y_all_pos = np.ones(10, dtype=int)
    s_all_pos = np.linspace(0.1, 0.9, 10)
    result = recall_at_fpr(y_all_pos, s_all_pos, target_fpr=0.05)
    assert isinstance(result, RecallAtFprResult)
    assert result.n_val_neg == 0
    assert result.fp == 0
    assert result.tn == 0
    assert result.actual_fpr == 0.0


@pytest.mark.unit
def test_recall_at_fpr_to_dict_keys_match_v5_reference_set() -> None:
    """`.to_dict()` returns the 6-key set V5 consumers expect for pandas-row use."""
    y, s = _make_discriminative_inputs()
    result = recall_at_fpr(y, s, target_fpr=0.10)
    d = result.to_dict()
    assert set(d.keys()) == {"threshold", "recall", "actual_fpr", "n_val_neg", "fp", "tn"}
    # Type integrity
    assert isinstance(d["threshold"], float)
    assert isinstance(d["recall"], float)
    assert isinstance(d["actual_fpr"], float)
    assert isinstance(d["n_val_neg"], int)
    assert isinstance(d["fp"], int)
    assert isinstance(d["tn"], int)


@pytest.mark.unit
def test_recall_at_fpr_result_is_frozen() -> None:
    """RecallAtFprResult is immutable (frozen dataclass)."""
    y, s = _make_discriminative_inputs()
    result = recall_at_fpr(y, s, target_fpr=0.10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.recall = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_recall_at_fpr_rejects_invalid_target_fpr() -> None:
    """target_fpr outside [0, 1] raises via TargetFPRSelector validation."""
    y, s = _make_discriminative_inputs()
    with pytest.raises(ValueError, match=r"fpr"):
        recall_at_fpr(y, s, target_fpr=1.5)
    with pytest.raises(ValueError, match=r"fpr"):
        recall_at_fpr(y, s, target_fpr=-0.1)


@pytest.mark.unit
def test_recall_at_fpr_unsatisfiable_returns_inf_sentinel() -> None:
    """When no threshold meets target_fpr, return sentinel honoring the FPR ceiling.

    Round 8 audit (R8-C3) regression: the verbatim probe case
    ``y=[0,1], scores=[1.0,1.0], target_fpr=0.0`` previously returned
    ``threshold=1.0, actual_fpr=1.0, fp=1`` — silently violating the
    function's own FPR-ceiling invariant. Root cause: fallback path
    computed ``y_pred = (y_score >= 1.0)`` (inclusive comparator), which
    classified the negative-class sample with score 1.0 as predicted-positive.

    v0.51 sentinel: ``threshold=np.inf, actual_fpr=0.0, fp=0`` — the
    actual_fpr ≤ target_fpr invariant is preserved by construction
    (np.inf threshold predicts nothing positive).
    """
    result = recall_at_fpr(np.array([0, 1]), np.array([1.0, 1.0]), target_fpr=0.0)
    assert np.isinf(result.threshold), (
        "Unsatisfiable case must signal via threshold=np.inf so callers "
        "can detect it (e.g., via np.isinf)."
    )
    assert result.actual_fpr == 0.0, (
        "actual_fpr must honor target_fpr ceiling even in unsatisfiable case "
        "(pre-v0.51 returned 1.0 here, violating the contract)."
    )
    assert result.fp == 0, "No threshold = no predicted positives = fp=0."
    assert result.recall == 0.0, "No predicted positives → recall=0."
    assert result.n_val_neg == 1
    assert result.tn == 1
    # The invariant the function name itself promises:
    assert result.actual_fpr <= 0.0, "FPR ceiling honored"


@pytest.mark.unit
def test_recall_at_fpr_unsatisfiable_with_multiple_score_one_negatives() -> None:
    """Stress the sentinel path: many negative samples all scoring 1.0."""
    y = np.array([0, 0, 0, 1])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    result = recall_at_fpr(y, s, target_fpr=0.0)
    assert np.isinf(result.threshold)
    assert result.actual_fpr == 0.0
    assert result.fp == 0
    assert result.n_val_neg == 3
    assert result.tn == 3
