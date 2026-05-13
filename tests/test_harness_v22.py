"""Tests for the v0.22.0 expanded evaluate_scorer_on_slice kwargs (C11 / F8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.calibration import fit_platt_calibrator
from eval_toolkit.harness import EvalSlice, evaluate_scorer_on_slice


class _SignalScorer:
    """Score returns rng.uniform but biased toward label so AUC > 0.5."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        n = len(X)
        return rng.uniform(0, 1, size=n)


class _RaisingScorer:
    """Always raises — exercises the precomputed_scores fast-path."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        raise RuntimeError("should NOT be called when precomputed_scores is set")


@pytest.fixture
def signal_slice() -> EvalSlice:
    """80 rows; class-conditional gaussian scores. Mixed-class."""
    rng = np.random.default_rng(13)
    n = 80
    labels = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
    df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": labels})
    return EvalSlice(name="signal", df=df)


@pytest.mark.unit
def test_precomputed_scores_skips_inference(signal_slice: EvalSlice) -> None:
    """precomputed_scores=arr bypasses scorer.predict_proba."""
    n = len(signal_slice.df)
    scores = np.linspace(0.0, 1.0, n)
    # _RaisingScorer would blow up if predict_proba were called.
    result = evaluate_scorer_on_slice(
        _RaisingScorer(), signal_slice, precomputed_scores=scores, n_resamples=0
    )
    assert result["scores"] == scores.tolist()


@pytest.mark.unit
def test_precomputed_scores_shape_mismatch_raises(signal_slice: EvalSlice) -> None:
    """Mismatched shape is a programmer error, not a metric issue."""
    bad = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="precomputed_scores shape"):
        evaluate_scorer_on_slice(_SignalScorer(), signal_slice, precomputed_scores=bad)


@pytest.mark.unit
def test_attack_style_passes_through(signal_slice: EvalSlice) -> None:
    """attack_style is a label only — no metric effect."""
    result = evaluate_scorer_on_slice(
        _SignalScorer(), signal_slice, attack_style="goal_hijack", n_resamples=0
    )
    assert result["attack_style"] == "goal_hijack"


@pytest.mark.unit
def test_attack_style_threads_through_error_path(signal_slice: EvalSlice) -> None:
    """When the scorer raises, attack_style still lands in the recorded dict."""
    result = evaluate_scorer_on_slice(
        _RaisingScorer(),
        signal_slice,
        attack_style="prompt_injection",
        on_scorer_error="record",
        n_resamples=0,
    )
    assert result["attack_style"] == "prompt_injection"
    assert "error" in result


@pytest.mark.unit
def test_fpr_ladder_emits_tpr_at_fpr_dict(signal_slice: EvalSlice) -> None:
    """fpr_ladder=[0.01, 0.05] emits a dict keyed by str(fpr)."""
    scores = np.linspace(0.0, 1.0, len(signal_slice.df))
    result = evaluate_scorer_on_slice(
        _SignalScorer(),
        signal_slice,
        precomputed_scores=scores,
        fpr_ladder=[0.01, 0.05],
        n_resamples=0,
    )
    assert "tpr_at_fpr" in result
    tpr_dict = result["tpr_at_fpr"]
    assert isinstance(tpr_dict, dict)
    assert set(tpr_dict.keys()) == {"0.01", "0.05"}


@pytest.mark.unit
def test_compute_mce_and_brier_emit_fields(signal_slice: EvalSlice) -> None:
    """compute_mce / compute_brier add the respective keys; default-off back-compat."""
    scores = np.linspace(0.0, 1.0, len(signal_slice.df))
    result = evaluate_scorer_on_slice(
        _SignalScorer(),
        signal_slice,
        precomputed_scores=scores,
        compute_mce=True,
        compute_brier=True,
        n_resamples=0,
    )
    assert "mce" in result
    assert "brier_score" in result

    # Defaults-off: no new keys.
    bare = evaluate_scorer_on_slice(
        _SignalScorer(), signal_slice, precomputed_scores=scores, n_resamples=0
    )
    assert "mce" not in bare
    assert "brier_score" not in bare
    assert "tpr_at_fpr" not in bare
    assert "roc_auc_ci" not in bare


@pytest.mark.unit
def test_bootstrap_roc_auc_emits_ci_dict(signal_slice: EvalSlice) -> None:
    """bootstrap_roc_auc=True (with n_resamples>0) emits roc_auc_ci alongside pr_auc_ci."""
    scores = np.linspace(0.0, 1.0, len(signal_slice.df))
    result = evaluate_scorer_on_slice(
        _SignalScorer(),
        signal_slice,
        precomputed_scores=scores,
        bootstrap_roc_auc=True,
        n_resamples=100,
        seed=1,
    )
    assert "pr_auc_ci" in result
    assert "roc_auc_ci" in result


@pytest.mark.unit
def test_calibrator_emits_calibrated_keys(signal_slice: EvalSlice) -> None:
    """calibrator=PlattFit triggers the *_calibrated metric block."""
    n = len(signal_slice.df)
    rng = np.random.default_rng(7)
    # Use sigmoid-shaped raw scores in [0,1] so headline_metrics' ECE check
    # doesn't reject the raw side. The calibrator path applies Platt on top.
    raw_logits = np.concatenate([rng.normal(-0.5, 0.8, 40), rng.normal(0.8, 0.8, 40)])
    raw_scores = 1.0 / (1.0 + np.exp(-raw_logits))  # sigmoid into [0, 1]
    calibrator = fit_platt_calibrator(signal_slice.y_true, raw_scores)
    result = evaluate_scorer_on_slice(
        _SignalScorer(),
        signal_slice,
        precomputed_scores=raw_scores,
        calibrator=calibrator,
        compute_mce=True,
        compute_brier=True,
        fpr_ladder=[0.05],
        n_resamples=0,
    )
    assert "pr_auc_calibrated" in result
    assert "roc_auc_calibrated" in result
    assert "brier_score_calibrated" in result
    assert "mce_calibrated" in result
    assert "tpr_at_fpr_calibrated" in result
    assert "scores_calibrated" in result
    # Calibrated scores live in [0, 1]; check shape only.
    assert len(result["scores_calibrated"]) == n


@pytest.mark.unit
def test_backward_compat_existing_kwargs_only(signal_slice: EvalSlice) -> None:
    """Pre-v0.22 contract: passing only the old kwargs still works unchanged."""
    result = evaluate_scorer_on_slice(_SignalScorer(), signal_slice, n_resamples=0)
    assert "n" in result
    assert "n_positive" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert "pr_auc_ci" in result
    assert "scores" in result
    # No v0.22 additions appear unless requested.
    assert "attack_style" not in result
    assert "tpr_at_fpr" not in result
    assert "mce" not in result
    assert "brier_score" not in result
    assert "roc_auc_ci" not in result
