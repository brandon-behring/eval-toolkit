"""Coverage-targeted tests for metrics error paths and defensive code.

Extracted from the v0.27.x-era ``test_coverage_gap.py`` during the
v0.30.1 hygiene split — every assertion preserved verbatim; only the
file boundary changed.

Pairs with the happy-path coverage in ``test_metrics_unit.py`` and the
invariants in ``test_metrics_props.py``. Targets input-validation
error branches that the smoke / property suites do not naturally hit
(NaN/Inf scores, single-class inputs, shape mismatches, out-of-range
probabilities, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.metrics import (
    expected_calibration_error,
    expected_calibration_error_equal_mass,
    metrics_at_threshold,
    pr_auc,
    precision_at_prior,
    quantile_stratified_pr_auc,
    single_class_threshold_metrics,
    stratified_recall,
)
from eval_toolkit.thresholds import select_threshold

# ---------------------------------------------------------------------------
# metrics: validation error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pr_auc_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        pr_auc(np.array([0, 1, 0]), np.array([0.1, 0.9]))


@pytest.mark.unit
def test_pr_auc_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        pr_auc(np.array([], dtype=int), np.array([], dtype=float))


@pytest.mark.unit
def test_pr_auc_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        pr_auc(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


@pytest.mark.unit
def test_select_threshold_rejects_non_selector_criterion() -> None:
    """v0.7.0 BREAKING — string criterion form is removed. Must pass a
    ThresholdSelector instance; anything else raises TypeError with a
    migration message."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.9])
    with pytest.raises(TypeError, match="ThresholdSelector instance"):
        select_threshold(y, s, criterion="max_f1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ThresholdSelector instance"):
        select_threshold(y, s, criterion="bogus")  # type: ignore[arg-type]


@pytest.mark.unit
def test_metrics_at_threshold_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics_at_threshold(np.array([0, 1]), np.array([0.5]), 0.5)


@pytest.mark.unit
def test_single_class_threshold_metrics_all_negative() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    out = single_class_threshold_metrics(y, s, threshold=0.5)
    assert out["slice_class"] == "all_negative"
    assert "fpr@threshold" in out


@pytest.mark.unit
def test_single_class_threshold_metrics_rejects_mixed_class() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="single-class"):
        single_class_threshold_metrics(y, s, threshold=0.5)


@pytest.mark.unit
def test_stratified_recall_rejects_shape_mismatch() -> None:
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    strata = np.array(["A"])
    with pytest.raises(ValueError, match="strata"):
        stratified_recall(y, s, threshold=0.5, strata=strata)


@pytest.mark.unit
def test_stratified_recall_handles_none_and_nan_strata() -> None:
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.1, 0.1, 0.9])
    strata = np.array([None, np.nan, "B", "B"], dtype=object)
    out = stratified_recall(y, s, threshold=0.5, strata=strata)
    assert "unlabeled" in out


@pytest.mark.unit
def test_quantile_stratified_pr_auc_rejects_shape_mismatch() -> None:
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    bad_strat = np.array([1.0])
    with pytest.raises(ValueError, match="stratifier"):
        quantile_stratified_pr_auc(y, s, bad_strat)


@pytest.mark.unit
def test_quantile_stratified_pr_auc_rejects_bad_quantiles() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    strat = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="q_low"):
        quantile_stratified_pr_auc(y, s, strat, q_low=0.9, q_high=0.1)


@pytest.mark.unit
def test_quantile_stratified_pr_auc_too_imbalanced_raises() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.01, size=100).astype(int)
    s = rng.uniform(0, 1, 100)
    strat = rng.uniform(0, 1, 100)
    if y.sum() < 10:
        with pytest.raises(ValueError, match="imbalanced"):
            quantile_stratified_pr_auc(y, s, strat)


@pytest.mark.unit
def test_expected_calibration_error_rejects_few_bins() -> None:
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(y, s, n_bins=1)


@pytest.mark.unit
def test_expected_calibration_error_equal_mass_rejects_n_lt_bins() -> None:
    y = np.zeros(5, dtype=int)
    s = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="quantile bins"):
        expected_calibration_error_equal_mass(y, s, n_bins=10)


@pytest.mark.unit
def test_precision_at_prior_rejects_invalid_prior() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="assumed_prior"):
        precision_at_prior(y, s, threshold=0.5, assumed_prior=1.5)


@pytest.mark.unit
def test_precision_at_prior_rejects_single_class() -> None:
    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="both classes"):
        precision_at_prior(y, s, threshold=0.5, assumed_prior=0.01)


# ---------------------------------------------------------------------------
# v0.3.0 C1: validation hardening — ECE family
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ece_rejects_out_of_range_scores() -> None:
    """expected_calibration_error fails fast on logit-shaped input."""
    y = np.array([0, 1, 0, 1] * 5, dtype=int)
    s = np.array([2.0, -1.0, 0.5, 1.5] * 5, dtype=float)  # logits, not probs
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        expected_calibration_error(y, s, n_bins=5)


@pytest.mark.unit
def test_ece_equal_mass_rejects_out_of_range_scores() -> None:
    """expected_calibration_error_equal_mass fails fast on logit-shaped input."""
    y = np.array([0, 1] * 25, dtype=int)
    s = np.linspace(-2.0, 2.0, 50)  # logits
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        expected_calibration_error_equal_mass(y, s, n_bins=5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "ece_fn_name",
    [
        "expected_calibration_error",
        "expected_calibration_error_debiased",
        "expected_calibration_error_l2",
        "expected_calibration_error_l2_debiased",
        "expected_calibration_error_equal_mass",
    ],
)
def test_all_ece_variants_reject_out_of_range_scores(ece_fn_name: str) -> None:
    """v0.8.0 regression: every ECE variant raises ValueError on uncalibrated logits.

    Closes v0.3 audit P1 #2 — silent meaningless ECE on logit input was the
    dominant historical failure mode for the calibration-aware metrics.
    """
    import eval_toolkit.metrics as _metrics

    fn = getattr(_metrics, ece_fn_name)
    y = np.array([0, 1] * 25, dtype=int)
    s_logits = np.linspace(-3.0, 4.0, 50)  # uncalibrated logits
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        fn(y, s_logits, n_bins=5)


@pytest.mark.unit
def test_metrics_validate_inputs_rejects_nan_inf_scores() -> None:
    """_validate_inputs (used by all metric helpers) rejects NaN/Inf in y_score."""
    y = np.array([0, 1, 0, 1])
    s_nan = np.array([0.1, np.nan, 0.5, 0.9])
    with pytest.raises(ValueError, match="NaN or inf"):
        pr_auc(y, s_nan)
    s_inf = np.array([0.1, np.inf, 0.5, 0.9])
    with pytest.raises(ValueError, match="NaN or inf"):
        pr_auc(y, s_inf)


# ---------------------------------------------------------------------------
# v0.3.0 C5: Brier score + decomposition + FPR/FNR + stratified_recall CI
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brier_score_perfect_calibration_zero() -> None:
    """Brier score = 0 when predictions match labels exactly."""
    from eval_toolkit.metrics import brier_score

    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, p) == 0.0


@pytest.mark.unit
def test_brier_score_constant_prevalence() -> None:
    """Brier score = 0.25 for the constant-prevalence forecast at p=0.5."""
    from eval_toolkit.metrics import brier_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000).astype(int)
    p = np.full(1000, 0.5)
    assert abs(brier_score(y, p) - 0.25) < 0.01


@pytest.mark.unit
def test_brier_decomposition_identity_holds_approximately() -> None:
    """BS ≈ REL - RES + UNC under equal-mass binning."""
    from eval_toolkit.metrics import brier_decomposition

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, 500), 0, 1)
    out = brier_decomposition(y, s, n_bins=10)
    approx = out["reliability"] - out["resolution"] + out["uncertainty"]
    # Identity is approximate — bins are independent of labels in expectation,
    # not strictly. 5% slack on n=500.
    assert abs(out["brier"] - approx) < 0.05


@pytest.mark.unit
def test_brier_rejects_logits() -> None:
    """Brier score also enforces probability range like ECE."""
    from eval_toolkit.metrics import brier_score

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        brier_score(np.array([0, 1, 0, 1]), np.array([2.0, -1.0, 0.5, 1.5]))


@pytest.mark.unit
def test_metrics_at_threshold_includes_fpr_fnr() -> None:
    """v0.3.0 metrics_at_threshold dict includes fpr + fnr keys."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    out = metrics_at_threshold(y, s, threshold=0.5)
    assert "fpr" in out
    assert "fnr" in out
    # All-correct case: fpr=0, fnr=0
    assert out["fpr"] == 0.0
    assert out["fnr"] == 0.0


@pytest.mark.unit
def test_stratified_recall_with_ci_attaches_wilson_bounds() -> None:
    """with_ci=True attaches ci_low + ci_high (Wilson scoring CI)."""
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.7, 0.6, 0.1, 0.2])
    strata = np.array(["A"] * 5 + ["B"] * 2)
    out = stratified_recall(y, s, threshold=0.5, strata=strata, with_ci=True)
    assert "ci_low" in out["A"]
    assert "ci_high" in out["A"]
    # Wilson CI bounds the recall point estimate.
    rec = out["A"]["recall"]
    assert out["A"]["ci_low"] <= rec <= out["A"]["ci_high"]


@pytest.mark.unit
def test_stratified_recall_no_ci_by_default() -> None:
    """Default with_ci=False; ci_low/ci_high keys absent."""
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.2, 0.1])
    strata = np.array(["A", "B", "A", "B"])
    out = stratified_recall(y, s, threshold=0.5, strata=strata)
    assert "ci_low" not in out["A"]
    assert "ci_high" not in out["A"]


# ---------------------------------------------------------------------------
# v0.4.0 C1: bias-corrected L2 ECE (Kumar 2019)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_l2_ece_bounded() -> None:
    """L2 ECE is in [0, 1] for any valid input."""
    from eval_toolkit.metrics import expected_calibration_error_l2

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(int)
    s = rng.uniform(0, 1, 200)
    out = expected_calibration_error_l2(y, s)
    assert 0.0 <= out <= 1.0


@pytest.mark.unit
def test_l2_debiased_smaller_than_plug_in_on_calibrated_data() -> None:
    """On well-calibrated data, the debiased estimate is ≤ plug-in (Kumar 2019)."""
    from eval_toolkit.metrics import (
        expected_calibration_error_l2,
        expected_calibration_error_l2_debiased,
    )

    rng = np.random.default_rng(0)
    n = 5000
    s = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < s).astype(int)  # perfectly calibrated
    plug_in = expected_calibration_error_l2(y, s)
    debiased = expected_calibration_error_l2_debiased(y, s)
    # The debiased estimator must remove positive bias on calibrated data.
    assert debiased <= plug_in + 1e-9


@pytest.mark.unit
def test_l2_debiased_zero_on_well_calibrated_large_n() -> None:
    """On n=10K perfectly-calibrated data, debiased L2 ECE is near zero."""
    from eval_toolkit.metrics import expected_calibration_error_l2_debiased

    rng = np.random.default_rng(42)
    n = 10000
    s = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < s).astype(int)
    debiased = expected_calibration_error_l2_debiased(y, s, n_bins=10)
    # 3σ tail: with bias removed, residual should be within sampling noise.
    assert debiased < 0.05


@pytest.mark.unit
def test_l2_ece_rejects_logits() -> None:
    from eval_toolkit.metrics import expected_calibration_error_l2

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        expected_calibration_error_l2(np.array([0, 1, 0, 1] * 5), np.linspace(-2, 2, 20))


# ---------------------------------------------------------------------------
# v0.5.0 C2: expected_calibration_error_debiased
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ece_debiased_smaller_than_plug_in_on_calibrated() -> None:
    """On well-calibrated data, simulated-H0 debiased L1 ECE ≤ plug-in."""
    from eval_toolkit.metrics import (
        expected_calibration_error_debiased,
        expected_calibration_error_equal_mass,
    )

    rng = np.random.default_rng(0)
    n = 2000
    s = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < s).astype(int)
    plug_in = expected_calibration_error_equal_mass(y, s)
    debiased = expected_calibration_error_debiased(y, s, n_sweep=100, rng=0)
    assert debiased <= plug_in + 1e-9


@pytest.mark.unit
def test_ece_debiased_validates() -> None:
    from eval_toolkit.metrics import expected_calibration_error_debiased

    y = np.array([0, 1] * 5)
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="n_sweep"):
        expected_calibration_error_debiased(y, s, n_sweep=5)
