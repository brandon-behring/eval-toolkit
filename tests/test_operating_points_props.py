"""Hypothesis property tests for eval_toolkit.operating_points invariants.

Pins down: ``OperatingPointSpec`` field normalization, the
mixed-class precondition on ``fit_operating_points``, threshold
monotonicity through ``apply_operating_points``, fit→apply
round-trip consistency, and determinism under fixed inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit.operating_points import (
    FittedOperatingPoint,
    OperatingPointSpec,
    apply_operating_points,
    fit_operating_points,
)
from eval_toolkit.thresholds import MaxF1Selector


@pytest.mark.property
@given(
    fit_slice=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    ),
    apply_slices=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_spec_normalizes_sequences_to_tuples(fit_slice: str, apply_slices: list[str]) -> None:
    """OperatingPointSpec.__post_init__ converts list inputs to tuples."""
    spec = OperatingPointSpec(
        fit_slice=fit_slice,
        apply_slices=apply_slices,
        selectors=[MaxF1Selector()],
    )
    assert isinstance(spec.apply_slices, tuple)
    assert isinstance(spec.selectors, tuple)
    assert isinstance(spec.scorer_names, tuple)
    assert list(spec.apply_slices) == apply_slices


@pytest.mark.property
@given(n_pos=st.integers(min_value=2, max_value=20), n_neg=st.integers(min_value=2, max_value=20))
def test_fit_then_apply_same_slice_reproduces_threshold_metrics(n_pos: int, n_neg: int) -> None:
    """fit on slice A, apply to slice A → metrics_at_threshold(A, threshold) matches fit_metrics."""
    rng = np.random.default_rng(0)
    # Construct a mixed-class slice with well-separated scores so MaxF1 is unambiguous.
    pos_scores = rng.uniform(0.55, 1.0, size=n_pos)
    neg_scores = rng.uniform(0.0, 0.45, size=n_neg)
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s = np.concatenate([pos_scores, neg_scores])

    fitted = fit_operating_points(y, s, [MaxF1Selector()], fitted_on_slice="A", scorer_name="model")
    applied = apply_operating_points(y, s, fitted, applied_to_slice="A", scorer_name="model")

    fit_metrics = fitted["max_f1"].fit_metrics
    applied_max_f1 = applied["max_f1"]
    # Apply-on-fit-slice reproduces the fit-time metrics under the mixed-class key set.
    assert applied_max_f1["precision"] == pytest.approx(fit_metrics["precision"])
    assert applied_max_f1["recall"] == pytest.approx(fit_metrics["recall"])
    assert applied_max_f1["fpr"] == pytest.approx(fit_metrics["fpr"])
    # Provenance carries fit-slice identity unchanged.
    prov = applied_max_f1["threshold_provenance"]
    assert prov["fitted_on_slice"] == "A"
    assert prov["applied_to_slice"] == "A"


@pytest.mark.property
@given(
    n_pos=st.integers(min_value=3, max_value=20),
    n_neg=st.integers(min_value=3, max_value=20),
    seed=st.integers(min_value=0, max_value=999),
)
def test_fit_is_deterministic_under_identical_inputs(n_pos: int, n_neg: int, seed: int) -> None:
    """Two fits on the same data and selector yield equal FittedOperatingPoint values."""
    rng = np.random.default_rng(seed)
    pos_scores = rng.uniform(0.55, 1.0, size=n_pos)
    neg_scores = rng.uniform(0.0, 0.45, size=n_neg)
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s = np.concatenate([pos_scores, neg_scores])

    a = fit_operating_points(y, s, [MaxF1Selector()], fitted_on_slice="A", scorer_name="m")
    b = fit_operating_points(y, s, [MaxF1Selector()], fitted_on_slice="A", scorer_name="m")
    assert a["max_f1"].threshold == b["max_f1"].threshold
    assert a["max_f1"].fit_metrics == b["max_f1"].fit_metrics


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    n_pos=st.integers(min_value=2, max_value=15),
    n_neg=st.integers(min_value=2, max_value=15),
)
def test_apply_threshold_predictions_are_score_monotone(n_pos: int, n_neg: int) -> None:
    """A score above the fitted threshold ⇒ predicted positive; below ⇒ predicted negative.

    Pins the fundamental contract of a single-threshold decision rule:
    predictions are a monotone function of score with respect to threshold.
    """
    rng = np.random.default_rng(42)
    pos_scores = rng.uniform(0.55, 1.0, size=n_pos)
    neg_scores = rng.uniform(0.0, 0.45, size=n_neg)
    y_fit = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s_fit = np.concatenate([pos_scores, neg_scores])

    fitted = fit_operating_points(
        y_fit, s_fit, [MaxF1Selector()], fitted_on_slice="A", scorer_name="m"
    )
    threshold = fitted["max_f1"].threshold

    # Synthetic target: an ordered grid above and below the threshold.
    # Skip equality at threshold to dodge a single-class slice and ambiguous tie-breaking.
    grid = np.linspace(0.0, 1.0, 21)
    grid = grid[~np.isclose(grid, threshold, atol=1e-9)]
    y_target = (grid >= threshold).astype(int)
    if int(y_target.sum()) in {0, len(y_target)}:
        # If the grid lands entirely on one side, the apply still works
        # (single-class branch), but we can't probe both sides — skip.
        return
    applied = apply_operating_points(
        y_target, grid, fitted, applied_to_slice="grid", scorer_name="m"
    )
    metrics = applied["max_f1"]
    # On a perfectly score-monotone slice with labels == (score >= threshold),
    # the fitted threshold predicts every label correctly.
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


@pytest.mark.property
@given(n=st.integers(min_value=2, max_value=10), label=st.sampled_from([0, 1]))
def test_fit_rejects_single_class_slice(n: int, label: int) -> None:
    """Single-class fit slices raise ValueError ("mixed-class")."""
    y = np.full(n, label, dtype=int)
    s = np.linspace(0.1, 0.9, num=n)
    with pytest.raises(ValueError, match="mixed-class"):
        fit_operating_points(y, s, [MaxF1Selector()])


@pytest.mark.property
@given(
    n_pos=st.integers(min_value=2, max_value=15),
    n_neg=st.integers(min_value=2, max_value=15),
)
def test_apply_to_all_positive_slice_uses_single_class_branch(n_pos: int, n_neg: int) -> None:
    """Applying a fitted point to an all-positive slice reports slice_class metrics."""
    rng = np.random.default_rng(11)
    pos = rng.uniform(0.55, 1.0, size=n_pos)
    neg = rng.uniform(0.0, 0.45, size=n_neg)
    y_fit = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s_fit = np.concatenate([pos, neg])
    fitted = fit_operating_points(
        y_fit, s_fit, [MaxF1Selector()], fitted_on_slice="A", scorer_name="m"
    )

    y_target = np.ones(5, dtype=int)
    s_target = np.array([0.1, 0.4, 0.6, 0.85, 0.95])
    applied = apply_operating_points(y_target, s_target, fitted, applied_to_slice="pos_only")
    assert applied["max_f1"]["slice_class"] == "all_positive"
    assert "recall@threshold" in applied["max_f1"]


@pytest.mark.property
@given(
    n_pos=st.integers(min_value=2, max_value=15),
    n_neg=st.integers(min_value=2, max_value=15),
)
def test_apply_rejects_out_of_domain_labels(n_pos: int, n_neg: int) -> None:
    """Applying with labels outside {0, 1} raises ValueError."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.55, 1.0, size=n_pos)
    neg = rng.uniform(0.0, 0.45, size=n_neg)
    y_fit = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    s_fit = np.concatenate([pos, neg])
    fitted = fit_operating_points(y_fit, s_fit, [MaxF1Selector()])

    bad_y = np.array([0, 1, 2])
    bad_s = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="binary"):
        apply_operating_points(bad_y, bad_s, fitted)


@pytest.mark.unit
def test_fitted_operating_point_to_dict_round_trip() -> None:
    """FittedOperatingPoint.to_dict() produces a JSON-shaped mapping."""
    op = FittedOperatingPoint(
        criterion="max_f1",
        threshold=0.5,
        fitted_on_slice="A",
        scorer_name="m",
        n_fit=10,
        n_positive_fit=5,
        n_negative_fit=5,
        fit_metrics={"precision": 0.8, "recall@threshold": 0.6, "fpr@threshold": 0.1},
    )
    out = op.to_dict()
    assert out["criterion"] == "max_f1"
    assert out["threshold"] == 0.5
    assert out["fit_metrics"]["precision"] == 0.8
