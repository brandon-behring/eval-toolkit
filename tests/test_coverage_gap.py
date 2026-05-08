"""Coverage-targeted tests for error paths and defensive code.

Adds targeted tests to reach the library-grade ≥ 90% coverage bar by
exercising input-validation error branches that the smoke / property
suites do not naturally hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    BootstrapCI,
    PairedBootstrapCI,
    bootstrap_ci,
    mde_from_ci,
    paired_bootstrap_diff,
    paired_bootstrap_ece_diff,
    paired_bootstrap_op_point_diff,
    paired_mde,
)
from eval_toolkit.calibration import (
    bayes_optimal_threshold,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    fit_temperature,
    fit_temperature_oracle,
    reliability_curve,
)
from eval_toolkit.harness import EvalSlice, Scorer
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
from eval_toolkit.plotting import (
    plot_confusion_matrix_grid,
    plot_lift_ci,
    plot_metric_bars,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_score_histograms,
    save_figure,
)
from eval_toolkit.thresholds import select_threshold


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


@dataclass(frozen=True, slots=True)
class _StubCI:
    """Duck-typed CI for plot_lift_ci tests."""

    point_estimate: float
    ci_low: float
    ci_high: float


@pytest.fixture
def informative() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = np.clip(y + rng.normal(0, 0.3, size=200), 0, 1)
    return y, s


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
# bootstrap: validation paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_ci_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        bootstrap_ci(np.zeros(20, dtype=int), np.zeros(10), pr_auc)


@pytest.mark.unit
def test_bootstrap_ci_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        bootstrap_ci(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), pr_auc)


@pytest.mark.unit
def test_bootstrap_ci_rejects_invalid_confidence() -> None:
    y = np.array([0, 1] * 10)
    s = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(y, s, pr_auc, confidence=1.5)


@pytest.mark.unit
def test_paired_bootstrap_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        paired_bootstrap_diff(
            np.zeros(20, dtype=int),
            np.zeros(10),
            np.zeros(20),
            pr_auc,
        )


@pytest.mark.unit
def test_paired_bootstrap_diff_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        paired_bootstrap_diff(
            np.array([0, 1, 0]),
            np.array([0.1, 0.5, 0.2]),
            np.array([0.2, 0.6, 0.3]),
            pr_auc,
        )


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        paired_bootstrap_ece_diff(
            np.zeros(20, dtype=int),
            np.zeros(10),
            np.zeros(20),
            ece_fn=expected_calibration_error,
        )


@pytest.mark.unit
def test_paired_bootstrap_ece_diff_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        paired_bootstrap_ece_diff(
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            ece_fn=expected_calibration_error,
        )


@pytest.mark.unit
def test_paired_bootstrap_op_point_diff_rejects_shape_mismatch(
    informative: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = informative
    val_y = y[:100]
    val_a = s[:100]
    with pytest.raises(ValueError, match="val shape mismatch"):
        paired_bootstrap_op_point_diff(
            val_y=val_y,
            val_score_a=val_a,
            val_score_b=val_a[:50],
            test_y=y[100:],
            test_score_a=s[100:],
            test_score_b=s[100:],
            threshold_fn=lambda yt, ys: 0.5,
            metric_fn=lambda yt, ys, t: 0.0,
        )


@pytest.mark.unit
def test_mde_from_ci_rejects_zero_width() -> None:
    fake = PairedBootstrapCI(
        delta=0.0,
        ci_low=0.05,
        ci_high=0.05,
        overlaps_zero=False,
        confidence=0.95,
        n_resamples=100,
    )
    with pytest.raises(RuntimeError, match="non-positive"):
        mde_from_ci(fake)


@pytest.mark.unit
def test_paired_mde_returns_estimate(
    informative: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = informative
    s_b = s + 0.05 * y.astype(float)
    est = paired_mde(y, s, s_b, pr_auc, n_resamples=50, seed=0)
    assert est.mde >= 0
    assert est.n == len(y)


@pytest.mark.unit
def test_bootstrap_ci_to_dict_schema() -> None:
    ci = BootstrapCI(0.5, 0.4, 0.6, 0.95, 100, "BCa")
    d = ci.to_dict()
    assert set(d.keys()) == {"point_estimate", "ci_95", "confidence", "n_resamples", "method"}


@pytest.mark.unit
def test_paired_bootstrap_ci_to_dict_schema() -> None:
    pci = PairedBootstrapCI(0.05, 0.02, 0.08, False, 0.95, 100)
    d = pci.to_dict()
    assert set(d.keys()) == {"delta", "ci_95", "overlaps_zero", "confidence", "n_resamples"}


# ---------------------------------------------------------------------------
# calibration: validation paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bayes_optimal_threshold_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="prior"):
        bayes_optimal_threshold(1.5, c_fp=1.0, c_fn=1.0)
    with pytest.raises(ValueError, match="c_fp"):
        bayes_optimal_threshold(0.5, c_fp=0.0, c_fn=1.0)
    with pytest.raises(ValueError, match="c_fn"):
        bayes_optimal_threshold(0.5, c_fp=1.0, c_fn=-1.0)


@pytest.mark.unit
def test_reliability_curve_handles_single_class() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    out = reliability_curve(y, s, n_bins=5)
    assert "skipped" in out


@pytest.mark.unit
def test_reliability_curve_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        reliability_curve(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        reliability_curve(np.array([], dtype=int), np.array([], dtype=float))
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(np.array([0, 1]), np.array([0.4, 0.6]), n_bins=1)
    with pytest.raises(ValueError, match="strategy"):
        reliability_curve(np.array([0, 1]), np.array([0.4, 0.6]), strategy="bogus")  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_isotonic_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_isotonic_calibrator(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        fit_isotonic_calibrator(np.array([], dtype=int), np.array([], dtype=float))
    with pytest.raises(ValueError, match="NaN"):
        fit_isotonic_calibrator(np.array([0, 1]), np.array([np.nan, 0.5]))
    with pytest.raises(ValueError, match="both classes"):
        fit_isotonic_calibrator(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


@pytest.mark.unit
def test_fit_isotonic_apply_rejects_nan_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0, 1)
    g = fit_isotonic_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))


@pytest.mark.unit
def test_fit_platt_apply_rejects_nan_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = y + rng.normal(0, 0.3, 50)
    g = fit_platt_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))


@pytest.mark.unit
def test_fit_temperature_validates_logits_shape() -> None:
    with pytest.raises(ValueError, match="must be"):
        fit_temperature(np.zeros((10,)), np.zeros(10, dtype=int))


@pytest.mark.unit
def test_fit_temperature_validates_length_match() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        fit_temperature(np.zeros((10, 2)), np.zeros(5, dtype=int))


@pytest.mark.unit
def test_fit_temperature_validates_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        fit_temperature(np.zeros((10, 2)), np.array([0, 1, 2] + [0] * 7))


@pytest.mark.unit
def test_fit_temperature_oracle_apply_rejects_nan() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0.01, 0.99)
    _, apply = fit_temperature_oracle(y, s)
    with pytest.raises(ValueError, match="NaN"):
        apply(np.array([np.nan, 0.5]))


# ---------------------------------------------------------------------------
# plotting: validation + branch coverage
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_save_figure_rejects_non_positive_dpi(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, tmp_path / "x.png", dpi=0)


@pytest.mark.smoke
def test_save_figure_skip_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_TOOLKIT_SKIP_SAVEFIG", "1")
    fig, _ = plt.subplots()
    target = tmp_path / "skipped.png"
    out = save_figure(fig, target)
    assert out == target.resolve()
    assert not target.exists()


@pytest.mark.smoke
def test_plot_pr_curve_with_baseline_curve_and_threshold() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100).astype(np.int64)
    s = (y + rng.normal(0, 0.3, 100)).clip(0, 1)
    bl = (np.linspace(0, 1, 50), np.linspace(0.5, 0.5, 50))
    fig = plot_pr_curve(
        y, s, threshold=0.5, prevalence=0.3, baseline_curve=bl, baseline_label="rand"
    )
    assert fig.axes


@pytest.mark.smoke
def test_plot_pr_curve_rejects_invalid_threshold() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="threshold"):
        plot_pr_curve(y, s, threshold=1.5)


@pytest.mark.smoke
def test_plot_pr_curve_rejects_invalid_prevalence() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="prevalence"):
        plot_pr_curve(y, s, prevalence=1.5)


@pytest.mark.smoke
def test_plot_pr_curve_rejects_bad_baseline_curve() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="baseline_curve"):
        plot_pr_curve(y, s, baseline_curve=(np.array([0.1]),))  # type: ignore[arg-type]


@pytest.mark.smoke
def test_plot_pr_curve_baseline_shape_mismatch_raises() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    bad = (np.array([0.0, 1.0]), np.array([0.5]))
    with pytest.raises(ValueError, match="same shape"):
        plot_pr_curve(y, s, baseline_curve=bad)


@pytest.mark.smoke
def test_plot_reliability_rejects_single_class() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="single class"):
        plot_reliability_diagram(y, s)


@pytest.mark.smoke
def test_plot_confusion_matrix_grid_validates() -> None:
    with pytest.raises(ValueError):
        plot_confusion_matrix_grid({})


@pytest.mark.smoke
def test_plot_metric_bars_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_metric_bars({})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_score_histograms({})


@pytest.mark.smoke
def test_plot_score_histograms_validates_array_shape() -> None:
    with pytest.raises(ValueError, match="1D"):
        plot_score_histograms({"x": np.zeros((3, 3))})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_score_histograms({"x": np.array([], dtype=float)})


@pytest.mark.smoke
def test_plot_score_histograms_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="NaN"):
        plot_score_histograms({"x": np.array([0.1, np.nan, 0.5])})


@pytest.mark.smoke
def test_plot_lift_ci_handles_empty_dict() -> None:
    with pytest.raises(ValueError):
        plot_lift_ci({})


@pytest.mark.smoke
def test_plot_lift_ci_renders_zero_line() -> None:
    cis = {
        "A": _StubCI(0.05, 0.02, 0.08),
        "B": _StubCI(-0.01, -0.05, 0.03),
    }
    fig = plot_lift_ci(cis)
    assert fig.axes


# ---------------------------------------------------------------------------
# harness: edge cases + Scorer protocol exercise
# ---------------------------------------------------------------------------


class _ConstantScorer:
    """Scorer Protocol implementation returning a constant score."""

    def __init__(self, value: float = 0.5) -> None:
        self._value = value

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return np.full(len(X), self._value, dtype=float)


@pytest.mark.unit
def test_scorer_protocol_accepts_duck_typed() -> None:
    scorer: Scorer = _ConstantScorer(0.7)
    assert isinstance(scorer.predict_proba(["a", "b", "c"]), np.ndarray)


@pytest.mark.unit
def test_eval_slice_strata_returns_none_when_unset() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    sl = EvalSlice(name="t", df=df)
    assert sl.strata is None


@pytest.mark.unit
def test_eval_slice_strata_returns_array() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1], "stratum": ["x", "y"]})
    sl = EvalSlice(name="t", df=df, strata_col="stratum")
    arr = sl.strata
    assert arr is not None and arr.shape == (2,)


# ---------------------------------------------------------------------------
# seeds: exercise the public branch; torch branch optional
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_global_seeds_makes_numpy_deterministic() -> None:
    from eval_toolkit.seeds import set_global_seeds

    set_global_seeds(42)
    a = np.random.rand(10)
    set_global_seeds(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_set_global_seeds_rejects_invalid() -> None:
    from eval_toolkit.seeds import set_global_seeds

    with pytest.raises(TypeError, match="int"):
        set_global_seeds("42")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seeds(-1)


# ---------------------------------------------------------------------------
# v0.3.0 C1: validation hardening
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


@pytest.mark.unit
def test_fit_temperature_rejects_single_class() -> None:
    """fit_temperature is now consistent with peer calibrators on single-class input."""
    from eval_toolkit.calibration import fit_temperature

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 2))
    labels = np.zeros(50, dtype=int)  # single-class
    with pytest.raises(ValueError, match="both classes"):
        fit_temperature(logits, labels)


@pytest.mark.unit
def test_embedding_cosine_pairs_across_rejects_dim_mismatch() -> None:
    """EmbeddingCosineStrategy.pairs_across catches buggy embedders that return
    different feature dimensions for query vs reference."""
    from eval_toolkit.text_dedup import EmbeddingCosineStrategy

    call_count = {"n": 0}

    def buggy_embedder(texts: list[str]) -> np.ndarray:
        call_count["n"] += 1
        # First call (refs) returns d=4; second call (queries) returns d=8.
        d = 4 if call_count["n"] == 1 else 8
        return np.zeros((len(texts), d), dtype=np.float64)

    strat = EmbeddingCosineStrategy(buggy_embedder)
    with pytest.raises(ValueError, match="inconsistent feature dimensions"):
        strat.pairs_across(["q1", "q2"], ["r1", "r2", "r3"], k=2)


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
# v0.3.0 C6: expected_cost + Beta calibration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cost_matrix_expected_cost_known_value() -> None:
    """expected_cost on a fixed scenario."""
    from eval_toolkit.calibration import CostMatrix

    cm = CostMatrix(prior=0.5, fp_cost=1.0, fn_cost=10.0)
    y = np.array([0, 1, 0, 1])
    s = np.array([0.6, 0.4, 0.1, 0.9])
    # At threshold=0.5: pred = [1, 0, 0, 1]; FP at idx 0, FN at idx 1
    # Cost = (1*1.0 + 1*10.0) / 4 = 2.75
    assert cm.expected_cost(y, s, threshold=0.5) == 2.75


@pytest.mark.unit
def test_cost_matrix_expected_cost_uses_bayes_threshold_by_default() -> None:
    from eval_toolkit.calibration import CostMatrix

    cm = CostMatrix(prior=0.01, fp_cost=1.0, fn_cost=10.0)
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.01, size=200)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, 200), 0, 1)
    cost_default = cm.expected_cost(y, s)
    cost_explicit = cm.expected_cost(y, s, threshold=cm.bayes_threshold)
    assert cost_default == cost_explicit


@pytest.mark.unit
def test_fit_beta_calibrator_returns_unit_interval() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300).astype(int)
    s = (y + rng.normal(0, 0.4, 300)).clip(0.01, 0.99)
    g = fit_beta_calibrator(y, s)
    out = g(s)
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.unit
def test_fit_beta_calibrator_validates_single_class() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    y = np.zeros(50, dtype=int)
    s = np.linspace(0.1, 0.9, 50)
    with pytest.raises(ValueError, match="both classes"):
        fit_beta_calibrator(y, s)


@pytest.mark.unit
def test_fit_beta_calibrator_apply_rejects_nan() -> None:
    from eval_toolkit.calibration import fit_beta_calibrator

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    s = (y + rng.normal(0, 0.3, 50)).clip(0.01, 0.99)
    g = fit_beta_calibrator(y, s)
    with pytest.raises(ValueError, match="NaN"):
        g(np.array([np.nan, 0.5]))


# ---------------------------------------------------------------------------
# v0.3.0 C8: plot_bootstrap_distribution + pdf/svg + immutable PALETTE
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_plot_bootstrap_distribution_runs() -> None:
    from eval_toolkit.plotting import plot_bootstrap_distribution

    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.02, size=500)
    fig = plot_bootstrap_distribution(deltas, ci_low=0.01, ci_high=0.09, title="lift dist")
    assert fig.axes


@pytest.mark.smoke
def test_plot_bootstrap_distribution_validates() -> None:
    from eval_toolkit.plotting import plot_bootstrap_distribution

    with pytest.raises(ValueError, match="empty"):
        plot_bootstrap_distribution(np.array([], dtype=float))
    with pytest.raises(ValueError, match="NaN"):
        plot_bootstrap_distribution(np.array([0.1, np.nan, 0.2]))
    with pytest.raises(ValueError, match="must both"):
        plot_bootstrap_distribution(np.array([0.1, 0.2]), ci_low=0.0)


@pytest.mark.smoke
def test_palette_is_immutable() -> None:
    from eval_toolkit.plotting import PALETTE

    with pytest.raises(TypeError):
        PALETTE["new_role"] = "#000000"  # type: ignore[index]


@pytest.mark.smoke
def test_save_figure_supports_pdf_svg(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    out_pdf = save_figure(fig, tmp_path / "fig.pdf", provenance={"git_sha": "abc"})
    assert out_pdf.suffix == ".pdf"
    sidecar_pdf = (tmp_path / "fig.pdf").with_suffix(".meta.json")
    assert sidecar_pdf.exists()

    out_svg = save_figure(fig, tmp_path / "fig.svg", provenance={"git_sha": "abc"})
    assert out_svg.suffix == ".svg"


@pytest.mark.smoke
def test_save_figure_rejects_unknown_suffix(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match=r"\.png|\.pdf|\.svg|sorted"):
        save_figure(fig, tmp_path / "fig.jpg")


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
# v0.4.0 C2: studentized bootstrap-t
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_ci_studentized_runs() -> None:
    """method='studentized' returns a valid BootstrapCI."""
    from eval_toolkit.metrics import pr_auc

    rng = np.random.default_rng(0)
    n = 60  # smaller n so jackknife is fast
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    ci = bootstrap_ci(y, s, pr_auc, n_resamples=100, method="studentized", seed=42)
    assert ci.method == "studentized"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high
    assert ci.ci_high - ci.ci_low > 0  # non-degenerate


@pytest.mark.unit
def test_bootstrap_ci_studentized_deterministic() -> None:
    """Same seed → identical studentized CI."""
    from eval_toolkit.metrics import pr_auc

    rng = np.random.default_rng(0)
    n = 60
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    ci1 = bootstrap_ci(y, s, pr_auc, n_resamples=80, method="studentized", seed=7)
    ci2 = bootstrap_ci(y, s, pr_auc, n_resamples=80, method="studentized", seed=7)
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high


# ---------------------------------------------------------------------------
# v0.4.0 C3: cv_clt_ci helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cv_clt_ci_known_value() -> None:
    """CV-CLT CI for fixed inputs equals the closed-form Bayle 2020 formula."""
    from eval_toolkit.bootstrap import cv_clt_ci

    # 5-fold CV PR-AUC: mean=0.82, std (ddof=1)≈0.0245, z_{0.975}=1.96
    folds = np.array([0.83, 0.81, 0.85, 0.79, 0.82])
    ci = cv_clt_ci(folds, confidence=0.95)
    assert ci.method == "cv_clt"
    assert ci.n_resamples == 5
    assert ci.point_estimate == pytest.approx(0.82, abs=1e-9)
    expected_margin = 1.959963984540054 * float(np.std(folds, ddof=1)) / np.sqrt(5)
    assert ci.ci_low == pytest.approx(0.82 - expected_margin, abs=1e-9)
    assert ci.ci_high == pytest.approx(0.82 + expected_margin, abs=1e-9)


@pytest.mark.unit
def test_cv_clt_ci_validates() -> None:
    from eval_toolkit.bootstrap import cv_clt_ci

    with pytest.raises(ValueError, match="≥ 2 entries"):
        cv_clt_ci(np.array([0.5]))
    with pytest.raises(ValueError, match="NaN or inf"):
        cv_clt_ci(np.array([0.5, np.nan, 0.6]))
    with pytest.raises(ValueError, match="confidence"):
        cv_clt_ci(np.array([0.5, 0.6, 0.7]), confidence=0.0)


@pytest.mark.unit
def test_cv_clt_ci_widens_with_variance() -> None:
    """Higher across-fold variance → wider CI at fixed K."""
    from eval_toolkit.bootstrap import cv_clt_ci

    tight = cv_clt_ci(np.array([0.80, 0.81, 0.79, 0.80, 0.81]))
    wide = cv_clt_ci(np.array([0.70, 0.90, 0.60, 0.95, 0.80]))
    assert (wide.ci_high - wide.ci_low) > (tight.ci_high - tight.ci_low)


# ---------------------------------------------------------------------------
# v0.4.0 C4: MinHashLSHStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minhash_lsh_satisfies_protocol() -> None:
    """MinHashLSHStrategy is a runtime-checkable SimilarityStrategy."""
    from eval_toolkit.text_dedup import MinHashLSHStrategy, SimilarityStrategy

    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16)
    assert isinstance(strat, SimilarityStrategy)


@pytest.mark.unit
def test_minhash_lsh_pairs_within_shape() -> None:
    from eval_toolkit.text_dedup import MinHashLSHStrategy

    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16, seed=0)
    texts = ["alpha bravo", "alpha bravo charlie", "delta echo", "foxtrot golf"]
    sims, idx = strat.pairs_within(texts, k=3)
    assert sims.shape == idx.shape == (4, 3)


@pytest.mark.unit
def test_minhash_lsh_self_similarity_is_one() -> None:
    """For pairs_within, each text's most-similar neighbor is itself with sim=1."""
    from eval_toolkit.text_dedup import MinHashLSHStrategy

    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16, seed=0)
    texts = ["alpha bravo charlie", "delta echo foxtrot"]
    sims, idx = strat.pairs_within(texts, k=2)
    for i in range(2):
        assert int(idx[i, 0]) == i
        assert sims[i, 0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_minhash_lsh_finds_near_duplicate() -> None:
    """Near-duplicate pair should be discovered + scored ≥ 0.5 Jaccard."""
    from eval_toolkit.text_dedup import MinHashLSHStrategy

    strat = MinHashLSHStrategy(n=3, num_perm=128, bands=16, seed=0)
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy doggo!",  # near-dup
        "completely unrelated lorem ipsum text content",
    ]
    sims, idx = strat.pairs_within(texts, k=2)
    # Top-1 (other than self) for text 0 should be text 1
    assert int(idx[0, 1]) == 1
    assert sims[0, 1] > 0.5  # high Jaccard between the near-dups


@pytest.mark.unit
def test_minhash_lsh_validates_args() -> None:
    from eval_toolkit.text_dedup import MinHashLSHStrategy

    with pytest.raises(ValueError, match="n must be"):
        MinHashLSHStrategy(n=0)
    with pytest.raises(ValueError, match="num_perm"):
        MinHashLSHStrategy(num_perm=4)
    with pytest.raises(ValueError, match="bands"):
        MinHashLSHStrategy(num_perm=128, bands=0)
    with pytest.raises(ValueError, match="bands"):
        MinHashLSHStrategy(num_perm=128, bands=200)
    with pytest.raises(ValueError, match="divisible"):
        MinHashLSHStrategy(num_perm=128, bands=15)  # 128 not divisible by 15


@pytest.mark.unit
def test_minhash_lsh_handles_empty_input() -> None:
    from eval_toolkit.text_dedup import MinHashLSHStrategy

    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16)
    sims_w, idx_w = strat.pairs_within([], k=5)
    assert sims_w.shape == idx_w.shape == (0, 0)
    sims_a, idx_a = strat.pairs_across([], ["a"], k=5)
    assert sims_a.shape == idx_a.shape == (0, 0)


@pytest.mark.unit
def test_minhash_lsh_in_near_dedup_orchestrator() -> None:
    """Plug-in contract: near_dedup accepts MinHashLSHStrategy via strategy=."""
    from eval_toolkit.text_dedup import MinHashLSHStrategy, near_dedup

    texts = [
        "the quick brown fox",
        "the quick brown fox!",  # near-dup
        "lorem ipsum dolor sit amet",
    ]
    strat = MinHashLSHStrategy(n=3, num_perm=128, bands=16, seed=0)
    report = near_dedup(texts, threshold=0.5, strategy=strat)
    # The near-dup pair should collapse to 1 entry
    assert report.n_kept == 2


# ---------------------------------------------------------------------------
# v0.5.0 C1: cross_validate_metric eval-only orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cross_validate_metric_returns_per_fold_values() -> None:
    from eval_toolkit.bootstrap import cross_validate_metric
    from eval_toolkit.metrics import pr_auc

    rng = np.random.default_rng(42)
    n = 200
    y = rng.binomial(1, 0.3, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.3, n), 0, 1)
    folds = cross_validate_metric(y, s, metric=pr_auc, k=5, seed=42)
    assert folds.shape == (5,)
    valid = folds[~np.isnan(folds)]
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


@pytest.mark.unit
def test_cross_validate_metric_pairs_with_cv_clt_ci() -> None:
    """End-to-end: cross_validate_metric → cv_clt_ci."""
    from eval_toolkit.bootstrap import cross_validate_metric, cv_clt_ci
    from eval_toolkit.metrics import pr_auc

    rng = np.random.default_rng(0)
    n = 300
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1)
    folds = cross_validate_metric(y, s, metric=pr_auc, k=5, seed=0)
    valid = folds[~np.isnan(folds)]
    assert valid.size >= 2  # Need ≥ 2 folds for cv_clt_ci
    ci = cv_clt_ci(valid)
    assert ci.method == "cv_clt"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_cross_validate_metric_validates() -> None:
    from eval_toolkit.bootstrap import cross_validate_metric
    from eval_toolkit.metrics import pr_auc

    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="shape"):
        cross_validate_metric(y, np.array([0.5]), metric=pr_auc)
    with pytest.raises(ValueError, match="k must be"):
        cross_validate_metric(y, s, metric=pr_auc, k=1)
    with pytest.raises(ValueError, match="exceeds n"):
        cross_validate_metric(y, s, metric=pr_auc, k=10)


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
    debiased = expected_calibration_error_debiased(y, s, n_sweep=100, seed=0)
    assert debiased <= plug_in + 1e-9


@pytest.mark.unit
def test_ece_debiased_validates() -> None:
    from eval_toolkit.metrics import expected_calibration_error_debiased

    y = np.array([0, 1] * 5)
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="n_sweep"):
        expected_calibration_error_debiased(y, s, n_sweep=5)
