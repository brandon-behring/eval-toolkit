"""Edge-case error-path coverage for ``eval_toolkit.bootstrap`` (v0.26.0).

Closes audit gaps from the v0.26.0 toolkit completeness sweep around
small-n and rare-positive bootstrap paths. Three categories:

1. **n < 10 guards**: ``bootstrap_ci``, ``paired_bootstrap_diff``,
   and ``paired_bootstrap_ece_diff`` all enforce a minimum sample
   size; existing tests cover ``bootstrap_ci`` but the paired
   variants' guards are unasserted.
2. **No usable resamples**: paired bootstraps catch metric-raising
   resamples (rare positives in PR-AUC / ECE diff) and refuse to
   compute a CI on > 5% degenerate resamples; the "all degenerate"
   path was untested.
3. **>5% degenerate resamples**: a partial-degeneracy sister case
   to (2) — the threshold-crossing path.

These tests use small-n + rare-positive constructions that
deterministically hit the relevant raise sites, and pin the error
message phrasing so a regression that silently changes the message
(used by downstream consumers parsing it) would fail the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_diff,
    paired_bootstrap_ece_diff,
)
from eval_toolkit.metrics import expected_calibration_error, pr_auc


def test_bootstrap_ci_raises_on_n_lt_10() -> None:
    """``bootstrap_ci`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:256``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    with pytest.raises(ValueError, match="n=6 too small for bootstrap"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100)


def test_paired_bootstrap_diff_raises_on_n_lt_10() -> None:
    """``paired_bootstrap_diff`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:448``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    with pytest.raises(ValueError, match="n=6 too small for paired bootstrap"):
        paired_bootstrap_diff(y, a, b, pr_auc, n_resamples=100)


def test_paired_bootstrap_ece_diff_raises_on_n_lt_10() -> None:
    """``paired_bootstrap_ece_diff`` raises ValueError when ``n < 10``. Covers ``bootstrap.py:547``."""
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    with pytest.raises(ValueError, match="n=6 too small for paired bootstrap"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=expected_calibration_error, n_resamples=100)


def _strict_single_class_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """A metric that explicitly raises on single-class y_true (no NaN return).

    sklearn's pr_auc / roc_auc return 0 / NaN on single-class inputs
    rather than raising, so they don't trigger the bootstrap's
    "degenerate resample" counter. This helper raises the explicit
    ValueError that ``paired_bootstrap_diff`` catches and counts.
    """
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError("metric undefined on single-class input")
    # Trivial well-behaved metric on mixed-class inputs: mean of positive scores.
    return float(np.mean(y_score[y_true == 1]))


def test_paired_bootstrap_diff_raises_on_too_many_degenerate_resamples() -> None:
    """``paired_bootstrap_diff`` raises when > 5% of resamples raise.

    Covers ``bootstrap.py:467``. Construction: rare-positive data
    (n=20, 1 positive) + a metric that explicitly raises on single-
    class resamples. With only 1 positive in n=20, most bootstrap
    resamples will draw 0 positives → metric raises → degenerate
    counter increments past 5% → assertion fires.
    """
    rng = np.random.default_rng(42)
    n = 20
    y = np.zeros(n, dtype=int)
    y[0] = 1  # exactly one positive
    a = rng.uniform(0, 1, size=n)
    b = rng.uniform(0, 1, size=n)
    with pytest.raises(ValueError, match="raised the metric function|no usable resamples"):
        paired_bootstrap_diff(y, a, b, _strict_single_class_metric, n_resamples=200, seed=42)


def test_paired_bootstrap_diff_shape_mismatch_raises() -> None:
    """``paired_bootstrap_diff`` raises ValueError on shape mismatch. Covers ``bootstrap.py:445``."""
    y = np.array([0, 1] * 10)
    a = np.array([0.5] * 20)
    b = np.array([0.5] * 19)  # one short
    with pytest.raises(ValueError, match="shapes mismatch"):
        paired_bootstrap_diff(y, a, b, pr_auc, n_resamples=100)


def test_paired_bootstrap_ece_diff_shape_mismatch_raises() -> None:
    """``paired_bootstrap_ece_diff`` raises ValueError on shape mismatch. Covers ``bootstrap.py:544``."""
    y = np.array([0, 1] * 10)
    a = np.array([0.5] * 20)
    b = np.array([0.5] * 19)
    with pytest.raises(ValueError, match="shapes mismatch"):
        paired_bootstrap_ece_diff(y, a, b, ece_fn=expected_calibration_error, n_resamples=100)


def test_bootstrap_ci_invalid_confidence_raises() -> None:
    """``bootstrap_ci`` raises ValueError when ``confidence`` is out of (0, 1).

    Covers ``bootstrap.py:258``.
    """
    y = np.array([0, 1] * 10)
    s = np.array([0.5] * 20)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100, confidence=1.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=100, confidence=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
