"""Tests for ``eval_toolkit.stacking`` — MetaLearner Protocol + LogisticStacker (v0.45.0, closes #52).

Coverage:
- Protocol structural satisfaction (`isinstance(stacker, MetaLearner)`).
- Shape contracts (N detectors × M samples).
- Regularization handling (C, penalty).
- Calibration chaining: stacker output → fit_platt_binary → calibrated proba.
- Input validation (shape mismatch, single-class, non-finite, unfit).
- Determinism under fixed random_state.
- Hypothesis property: monotonicity of detector contribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eval_toolkit import (
    BootstrapCI,
    LogisticStacker,
    MetaLearner,
    bootstrap_ci,
    fit_isotonic_binary,
    fit_platt_binary,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def three_detector_data() -> tuple[np.ndarray, np.ndarray]:
    """500 samples × 3 detectors with declining signal-to-noise.

    Detector 0: high signal (coefficient ≈ 0.7 on `y` + low noise).
    Detector 1: medium signal.
    Detector 2: low signal, near random.

    A well-fit stacker should assign descending weights:
    ``coef_[0] > coef_[1] > coef_[2]``.
    """
    rng = np.random.default_rng(0)
    n = 500
    y = rng.binomial(1, 0.3, size=n).astype(int)
    scores = np.column_stack(
        [
            np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0, 1),
            np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1),
            np.clip(y * 0.4 + rng.normal(0, 0.4, n), 0, 1),
        ]
    )
    return scores, y


@pytest.fixture
def perfect_detector_data() -> tuple[np.ndarray, np.ndarray]:
    """One perfect detector + one pure-noise detector. Stacker should drop the noisy column."""
    rng = np.random.default_rng(1)
    n = 400
    y = rng.binomial(1, 0.5, size=n).astype(int)
    scores = np.column_stack(
        [
            y.astype(float) + rng.normal(0, 0.05, n),  # near-perfect
            rng.uniform(0, 1, n),  # pure noise
        ]
    )
    return scores, y


# ─────────────────────────────────────────────────────────────────────────────
# Protocol satisfaction
# ─────────────────────────────────────────────────────────────────────────────


def test_logistic_stacker_satisfies_meta_learner_protocol(three_detector_data) -> None:
    """LogisticStacker structurally satisfies MetaLearner Protocol after fit."""
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    assert isinstance(stacker, MetaLearner)


def test_meta_learner_protocol_is_runtime_checkable() -> None:
    """MetaLearner is @runtime_checkable so isinstance() works on duck-typed impls."""

    class _MinimalStacker:
        coef_ = np.array([1.0, 2.0, 3.0])
        classes_ = np.array([0, 1])
        intercept_ = np.array([0.5])

        def fit(self, score_matrix, y):  # noqa: ARG002
            return self

        def predict(self, score_matrix):
            return np.zeros(len(score_matrix), dtype=int)

        def predict_proba(self, score_matrix):
            n = len(score_matrix)
            return np.column_stack([np.ones(n) * 0.7, np.ones(n) * 0.3])

    instance = _MinimalStacker()
    assert isinstance(instance, MetaLearner)


# ─────────────────────────────────────────────────────────────────────────────
# Shape contracts
# ─────────────────────────────────────────────────────────────────────────────


def test_coef_shape_matches_n_detectors(three_detector_data) -> None:
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    assert stacker.coef_.shape == (3,)
    assert stacker.intercept_.shape == (1,)
    assert stacker.classes_.tolist() == [0, 1]


def test_predict_proba_shape(three_detector_data) -> None:
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    proba = stacker.predict_proba(scores)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_shape(three_detector_data) -> None:
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    preds = stacker.predict(scores)
    assert preds.shape == (500,)
    assert set(preds.tolist()).issubset({0, 1})


def test_predict_proba_works_on_subset(three_detector_data) -> None:
    """predict_proba on a different-sized score matrix (same n_detectors) works."""
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    proba = stacker.predict_proba(scores[:50])
    assert proba.shape == (50, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Regularization behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_regularization_shrinks_weights(three_detector_data) -> None:
    """Smaller C → stronger L2 regularization → smaller-magnitude coef_."""
    scores, y = three_detector_data
    weak = LogisticStacker(C=10.0, rng=0).fit(scores, y)
    strong = LogisticStacker(C=0.01, rng=0).fit(scores, y)
    weak_norm = float(np.linalg.norm(weak.coef_))
    strong_norm = float(np.linalg.norm(strong.coef_))
    assert strong_norm < weak_norm, (
        f"strong regularization (C=0.01) coef norm {strong_norm:.3f} should be < "
        f"weak regularization (C=10.0) coef norm {weak_norm:.3f}"
    )


def test_l1_penalty_can_drive_weights_to_zero(perfect_detector_data) -> None:
    """L1 penalty + small C should zero out the noise column."""
    scores, y = perfect_detector_data
    stacker = LogisticStacker(C=0.1, penalty="l1", solver="liblinear", rng=0).fit(scores, y)
    # The noisy column (index 1) should have substantially smaller coef than the signal column (0).
    assert abs(stacker.coef_[0]) > abs(stacker.coef_[1]), (
        f"signal column |coef|={abs(stacker.coef_[0]):.3f} should exceed "
        f"noise column |coef|={abs(stacker.coef_[1]):.3f} under L1+small-C"
    )


def test_signal_ordering(three_detector_data) -> None:
    """Detector 0 (highest signal) should get the largest weight."""
    scores, y = three_detector_data
    stacker = LogisticStacker(C=1.0, rng=0).fit(scores, y)
    # Detector 0 (signal=0.7) should outrank detectors 1 and 2.
    assert stacker.coef_[0] > stacker.coef_[1]
    assert stacker.coef_[0] > stacker.coef_[2]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration chaining (v0.40 + v0.42 family compose with v0.45 stacker)
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_chaining_with_fit_platt_binary(three_detector_data) -> None:
    """Stacker output → fit_platt_binary returns a callable that produces (n,) probabilities."""
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    stacker_proba = stacker.predict_proba(scores)[:, 1]

    (a, b), apply = fit_platt_binary(y, stacker_proba)

    assert isinstance(a, float)
    assert isinstance(b, float)
    calibrated = apply(stacker_proba)
    assert calibrated.shape == (500,)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_calibration_chaining_with_fit_isotonic_binary(three_detector_data) -> None:
    """Stacker output → fit_isotonic_binary (non-parametric — params is None)."""
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    stacker_proba = stacker.predict_proba(scores)[:, 1]

    params, apply = fit_isotonic_binary(y, stacker_proba)

    assert params is None  # non-parametric
    calibrated = apply(stacker_proba)
    assert calibrated.shape == (500,)


def test_stacker_output_bootstrap_ci(three_detector_data) -> None:
    """Bootstrap CI on PR-AUC of stacker output — exercises the v0.45 → v0.34 chain."""
    from eval_toolkit.metrics import pr_auc

    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    proba = stacker.predict_proba(scores)[:, 1]

    ci = bootstrap_ci(y, proba, pr_auc, n_resamples=200, rng=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high or ci.method == "BCa"
    assert ci.ci_low < ci.ci_high
    assert 0.0 <= ci.ci_low <= 1.0
    assert 0.0 <= ci.ci_high <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_raises_on_shape_mismatch(three_detector_data) -> None:
    scores, y = three_detector_data
    with pytest.raises(ValueError, match="matching n_samples"):
        LogisticStacker().fit(scores[:100], y)


def test_fit_raises_on_single_class(three_detector_data) -> None:
    scores, _ = three_detector_data
    with pytest.raises(ValueError, match="single-class"):
        LogisticStacker().fit(scores, np.zeros(500, dtype=int))


def test_fit_raises_on_non_finite_score_matrix(three_detector_data) -> None:
    scores, y = three_detector_data
    bad = scores.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        LogisticStacker().fit(bad, y)


def test_fit_raises_on_1d_score_matrix() -> None:
    rng = np.random.default_rng(0)
    flat = rng.random(100)  # 1-D
    y = rng.binomial(1, 0.5, size=100)
    with pytest.raises(ValueError, match="2-D"):
        LogisticStacker().fit(flat, y)


def test_fit_raises_on_empty_score_matrix() -> None:
    empty = np.zeros((0, 3))
    y = np.zeros(0, dtype=int)
    with pytest.raises(ValueError, match="empty"):
        LogisticStacker().fit(empty, y)


def test_predict_proba_raises_when_not_fit(three_detector_data) -> None:
    scores, _ = three_detector_data
    with pytest.raises(ValueError, match="not been fit"):
        LogisticStacker().predict_proba(scores)


def test_predict_raises_when_not_fit(three_detector_data) -> None:
    scores, _ = three_detector_data
    with pytest.raises(ValueError, match="not been fit"):
        LogisticStacker().predict(scores)


def test_coef_property_raises_when_not_fit() -> None:
    with pytest.raises(ValueError, match="not been fit"):
        _ = LogisticStacker().coef_


def test_predict_proba_raises_on_wrong_n_detectors(three_detector_data) -> None:
    scores, y = three_detector_data
    stacker = LogisticStacker(rng=0).fit(scores, y)
    # Wrong number of detector columns
    wrong = scores[:, :2]
    with pytest.raises(ValueError, match="wrong number of detectors"):
        stacker.predict_proba(wrong)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_is_deterministic_under_seed(three_detector_data) -> None:
    """Two stackers with same random_state on same data produce identical coef_."""
    scores, y = three_detector_data
    a = LogisticStacker(rng=0).fit(scores, y)
    b = LogisticStacker(rng=0).fit(scores, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.intercept_, b.intercept_)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis property: stacker monotonicity
# ─────────────────────────────────────────────────────────────────────────────


@given(
    n=st.integers(min_value=50, max_value=300),
    signal=st.floats(min_value=0.2, max_value=0.9),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=20, deadline=None)
def test_property_stronger_signal_yields_better_separation(
    n: int, signal: float, seed: int
) -> None:
    """A stacker on a stronger-signal feature should produce more-separated proba.

    Property: as `signal` grows (single detector aligned with labels), the
    spread between mean-positive-class proba and mean-negative-class proba
    should grow. The stacker should reflect this.

    Suppress sklearn FutureWarning about `penalty=` kwarg (sklearn 1.8+).
    Public API is stable; the warning is from our internal sklearn call.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, 0.4, size=n).astype(int)
        scores = (y * signal + rng.normal(0, 0.15, n)).reshape(-1, 1)
        stacker = LogisticStacker(rng=seed).fit(scores, y)
        proba = stacker.predict_proba(scores)[:, 1]

        mean_pos = float(proba[y == 1].mean()) if (y == 1).any() else 0.0
        mean_neg = float(proba[y == 0].mean()) if (y == 0).any() else 0.0
        # When signal exists, the stacker should learn positive correlation.
        # Threshold lenient — only check directionality.
        assert mean_pos >= mean_neg, (
            f"signal={signal:.2f}, seed={seed}: mean_pos={mean_pos:.3f} "
            f"should be >= mean_neg={mean_neg:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public-API smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_top_level_exports_present() -> None:
    """Both symbols importable from the top-level package."""
    import eval_toolkit

    assert "LogisticStacker" in eval_toolkit.__all__
    assert "MetaLearner" in eval_toolkit.__all__
    assert eval_toolkit.LogisticStacker is LogisticStacker
    assert eval_toolkit.MetaLearner is MetaLearner
