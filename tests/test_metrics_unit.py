"""Unit tests for metrics implementation vs scikit-learn references.

Adapted from prompt_injection_detector/tests/test_metrics.py.
Uses synthetic data (np.random.default_rng) — no PID-specific fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from eval_toolkit.metrics import (
    expected_calibration_error,
    expected_calibration_error_equal_mass,
    headline_metrics,
    metrics_at_threshold,
    pr_auc,
    precision_at_prior,
    quantile_stratified_pr_auc,
    roc_auc,
    score_distribution_summary,
    single_class_threshold_metrics,
    stratified_recall,
)
from eval_toolkit.thresholds import MaxF1Selector, TargetRecallSelector, select_threshold


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary labels with a noisy but informative score."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.3, size=200)
    y_score = y_true + rng.normal(0, 0.3, size=200)
    y_score = np.clip(y_score, 0, 1)
    return y_true.astype(int), y_score


@pytest.mark.unit
def test_pr_auc_matches_sklearn(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    """pr_auc matches sklearn.metrics.average_precision_score on a known case."""
    y_true, y_score = synthetic_data
    ours = pr_auc(y_true, y_score)
    theirs = float(average_precision_score(y_true, y_score))
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.unit
def test_roc_auc_in_unit_interval(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = synthetic_data
    score = roc_auc(y_true, y_score)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # informative score should beat random


@pytest.mark.unit
def test_threshold_tuning_max_f1(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    """max_f1 threshold's F1 ≥ F1 at any other PR-curve threshold."""
    y_true, y_score = synthetic_data
    result = select_threshold(y_true, y_score, criterion=MaxF1Selector())
    assert result.criterion == "max_f1"
    metrics = metrics_at_threshold(y_true, y_score, result.threshold)
    for cand in np.linspace(0.05, 0.95, 19):
        cand_f1 = metrics_at_threshold(y_true, y_score, cand)["f1"]
        assert metrics["f1"] >= cand_f1 - 1e-6


@pytest.mark.unit
def test_threshold_recall_target(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    """recall_0.90 threshold actually achieves recall ≥ 0.90."""
    y_true, y_score = synthetic_data
    result = select_threshold(y_true, y_score, criterion=TargetRecallSelector(0.90))
    metrics = metrics_at_threshold(y_true, y_score, result.threshold)
    assert metrics["recall"] >= 0.90 - 1e-6


@pytest.mark.unit
def test_stratified_recall_breakdown() -> None:
    """Per-stratum recall is correctly computed for a known categorical split."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_score = np.array([0.9, 0.9, 0.1, 0.1, 0.9])
    strata = np.array(["A", "A", "B", "A", "B"])
    out = stratified_recall(y_true, y_score, threshold=0.5, strata=strata)
    assert out["A"]["n"] == 2
    assert out["A"]["tp"] == 2
    assert out["A"]["recall"] == 1.0
    assert out["B"]["n"] == 1
    assert out["B"]["tp"] == 0
    assert out["B"]["recall"] == 0.0


@pytest.mark.unit
def test_input_validation() -> None:
    """Mismatched shapes and non-binary labels are rejected fail-fast."""
    with pytest.raises(ValueError, match="shape"):
        pr_auc(np.array([0, 1]), np.array([0.5, 0.6, 0.7]))
    with pytest.raises(ValueError, match="binary"):
        pr_auc(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="empty"):
        pr_auc(np.array([]), np.array([]))


@pytest.mark.unit
def test_ece_zero_when_perfectly_calibrated() -> None:
    """A perfectly calibrated score gives ECE near 0 with large n."""
    rng = np.random.default_rng(0)
    n = 10000
    y_score = rng.uniform(0, 1, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_score).astype(int)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert ece < 0.05


@pytest.mark.unit
def test_headline_metrics_bundles_everything(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = synthetic_data
    out = headline_metrics(y_true, y_score)
    assert "pr_auc" in out
    assert "roc_auc" in out
    assert "operating_points" in out
    assert "ece" in out
    assert "ece_equal_width" in out
    assert "ece_equal_mass" in out
    assert "precision_at_prior" in out
    assert set(out["operating_points"].keys()) == {"max_f1", "recall_0.90", "recall_0.95"}


@pytest.mark.unit
def test_headline_metrics_marks_single_class_pr_auc_not_meaningful() -> None:
    y_true = np.zeros(20, dtype=int)
    y_score = np.linspace(0.05, 0.95, 20)
    out = headline_metrics(y_true, y_score)
    assert out["pr_auc"] is None
    assert out["roc_auc"] is None
    assert out["single_class_label"] == 0
    assert out["precision_at_prior"] == {
        "skipped": "single-class slice cannot estimate both TPR and FPR"
    }
    assert out["operating_points"]["max_f1"] == {
        "skipped": "single-class slice; threshold must be selected on a mixed-class slice"
    }


@pytest.mark.unit
def test_single_class_threshold_metrics_all_negative_reports_fpr() -> None:
    y_true = np.zeros(4, dtype=int)
    y_score = np.array([0.1, 0.2, 0.7, 0.9])
    out = single_class_threshold_metrics(y_true, y_score, threshold=0.5)
    assert out["slice_class"] == "all_negative"
    assert out["fp"] == 2
    assert out["tn"] == 2
    assert out["fpr@threshold"] == 0.5
    assert out["specificity@threshold"] == 0.5


@pytest.mark.unit
def test_single_class_threshold_metrics_all_positive_reports_recall() -> None:
    y_true = np.ones(4, dtype=int)
    y_score = np.array([0.1, 0.2, 0.7, 0.9])
    out = single_class_threshold_metrics(y_true, y_score, threshold=0.5)
    assert out["slice_class"] == "all_positive"
    assert out["tp"] == 2
    assert out["fn"] == 2
    assert out["recall@threshold"] == 0.5


@pytest.mark.unit
def test_single_class_threshold_metrics_rejects_mixed_class() -> None:
    with pytest.raises(ValueError, match="single-class"):
        single_class_threshold_metrics(np.array([0, 1]), np.array([0.2, 0.8]), threshold=0.5)


@pytest.mark.unit
def test_ece_equal_mass_zero_when_perfectly_calibrated() -> None:
    rng = np.random.default_rng(0)
    n = 10000
    y_score = rng.uniform(0, 1, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_score).astype(int)
    ece = expected_calibration_error_equal_mass(y_true, y_score, n_bins=10)
    assert ece < 0.05


@pytest.mark.unit
def test_ece_equal_mass_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="smaller than n_bins"):
        expected_calibration_error_equal_mass(
            np.array([0, 1, 0, 1, 0]), np.array([0.1, 0.9, 0.2, 0.8, 0.5]), n_bins=10
        )


@pytest.mark.unit
def test_ece_equal_mass_uses_balanced_bin_sizes() -> None:
    """Equal-mass and equal-width ECE differ on imbalanced data."""
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.binomial(1, 0.05, size=n).astype(int)
    y_score = np.clip(0.1 * y_true + rng.beta(2, 20, size=n), 0, 1)
    eq_width = expected_calibration_error(y_true, y_score)
    eq_mass = expected_calibration_error_equal_mass(y_true, y_score)
    assert 0.0 <= eq_width < 1.0
    assert 0.0 <= eq_mass < 1.0
    assert eq_width != pytest.approx(eq_mass, abs=1e-6)


@pytest.mark.unit
def test_precision_at_prior_matches_eval_when_prior_eq_eval() -> None:
    """When assumed_prior == eval_prior, projected precision == observed precision."""
    rng = np.random.default_rng(2)
    n = 500
    y_true = rng.binomial(1, 0.3, size=n).astype(int)
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.25, n), 0, 1)
    tr = select_threshold(y_true, y_score, criterion=MaxF1Selector())
    out = precision_at_prior(y_true, y_score, tr.threshold, assumed_prior=0.3)
    assert out["precision_at_assumed_prior"] == pytest.approx(
        out["precision_at_eval_prior"], rel=0.05
    )


@pytest.mark.unit
def test_precision_at_prior_drops_when_prior_drops() -> None:
    """Lower deployment prior → much lower precision."""
    rng = np.random.default_rng(2)
    n = 500
    y_true = rng.binomial(1, 0.3, size=n).astype(int)
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.25, n), 0, 1)
    tr = select_threshold(y_true, y_score, criterion=MaxF1Selector())
    eval_p = precision_at_prior(y_true, y_score, tr.threshold, assumed_prior=0.30)
    deploy_p = precision_at_prior(y_true, y_score, tr.threshold, assumed_prior=0.001)
    assert deploy_p["precision_at_assumed_prior"] < eval_p["precision_at_assumed_prior"] / 2


@pytest.mark.unit
def test_precision_at_prior_rejects_invalid_prior() -> None:
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, size=50).astype(int)
    s = rng.uniform(0, 1, 50)
    with pytest.raises(ValueError, match="assumed_prior"):
        precision_at_prior(y, s, 0.5, assumed_prior=0.0)
    with pytest.raises(ValueError, match="assumed_prior"):
        precision_at_prior(y, s, 0.5, assumed_prior=1.0)


@pytest.mark.unit
def test_precision_at_prior_requires_both_classes() -> None:
    y_all_pos = np.ones(50, dtype=int)
    s = np.linspace(0.3, 0.9, 50)
    with pytest.raises(ValueError, match="both classes"):
        precision_at_prior(y_all_pos, s, 0.5, assumed_prior=0.01)


@pytest.mark.unit
def test_quantile_stratified_pr_auc_drops_signal_when_stratifier_dominates() -> None:
    """A stratifier-dominated score should weaken under quantile stratification."""
    rng = np.random.default_rng(42)
    n = 600
    y = rng.binomial(1, 0.3, size=n).astype(int)
    stratifier = np.where(y == 1, rng.normal(150, 30, n), rng.normal(100, 30, n)).astype(np.int64)
    score = stratifier.astype(float)
    full_pr = pr_auc(y, score)
    out = quantile_stratified_pr_auc(y, score, stratifier, q_low=0.25, q_high=0.75)
    assert out["pr_auc"] < full_pr - 0.05


@pytest.mark.unit
def test_quantile_stratified_pr_auc_preserves_real_signal() -> None:
    """A stratifier-independent score survives stratification."""
    rng = np.random.default_rng(7)
    n = 400
    y = rng.binomial(1, 0.3, size=n).astype(int)
    stratifier = rng.integers(50, 250, size=n).astype(np.int64)
    score = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0, 1)
    full_pr = pr_auc(y, score)
    out = quantile_stratified_pr_auc(y, score, stratifier, q_low=0.25, q_high=0.75)
    assert out["pr_auc"] >= full_pr - 0.10


@pytest.mark.unit
def test_quantile_stratified_pr_auc_rejects_thin_subset() -> None:
    rng = np.random.default_rng(0)
    n = 30
    y = rng.binomial(1, 0.05, size=n).astype(int)
    stratifier = np.linspace(10, 200, n).astype(np.int64)
    score = rng.uniform(0, 1, size=n)
    with pytest.raises(ValueError, match="too imbalanced"):
        quantile_stratified_pr_auc(y, score, stratifier, q_low=0.25, q_high=0.75)


@pytest.mark.unit
def test_quantile_stratified_pr_auc_rejects_bad_quantiles() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=50).astype(int)
    score = rng.uniform(0, 1, size=50)
    stratifier = rng.integers(10, 200, size=50).astype(np.int64)
    with pytest.raises(ValueError, match="0 ≤ q_low"):
        quantile_stratified_pr_auc(y, score, stratifier, q_low=0.5, q_high=0.5)
    with pytest.raises(ValueError, match="0 ≤ q_low"):
        quantile_stratified_pr_auc(y, score, stratifier, q_low=-0.1, q_high=0.5)


def test_score_distribution_summary_returns_expected_keys() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    out = score_distribution_summary(scores)
    assert set(out.keys()) == {"n", "mean", "median", "std", "q25", "q75"}
    assert out["n"] == 9
    assert out["mean"] == pytest.approx(0.5)
    assert out["median"] == pytest.approx(0.5)
    assert out["q25"] == pytest.approx(0.3)
    assert out["q75"] == pytest.approx(0.7)


def test_score_distribution_summary_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        score_distribution_summary(np.array([]))
    with pytest.raises(ValueError, match="NaN"):
        score_distribution_summary(np.array([0.1, np.nan, 0.3]))
    with pytest.raises(ValueError, match="NaN"):
        score_distribution_summary(np.array([0.1, np.inf]))
