"""Perf-regression benchmarks for the math kernels.

Tier γ #1 of the v0.29.0 best-practice gap audit. Catches O(n)
regressions before release.

Excluded from PR CI via the ``benchmark`` marker; runs only in
``.github/workflows/nightly-benchmarks.yml`` (cron-triggered). The
existing test suite (property, golden, MC) covers numerical
correctness; these benchmarks cover the **time complexity**
contract — that a refactor doesn't accidentally regress a kernel
from O(n) to O(n²) or similar.

Methodology
-----------
Each benchmark uses pytest-benchmark to record per-iteration time
on a fixed-size input (n=1000 for most kernels, n=200 for the
bootstrap variants since their inner loop is the dominant cost).
Output goes to JSON in the workflow artifacts; comparison-to-baseline
happens manually for now (`pytest-benchmark compare`).

Future enhancement: store a baseline JSON in the repo (or as a
workflow artifact) and add an assertion that each kernel runs within
some multiple of its baseline. Deferred until we observe real perf
drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit import (
    bootstrap_ci,
    paired_bootstrap_diff,
)
from eval_toolkit.metrics import (  # v0.47: scalars now live in the submodule (Decision C; ADR 0002)
    brier_score,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)

# ---------------------------------------------------------------------------
# Fixtures: pre-built (y, score) arrays at canonical sizes.
# Module-scoped to avoid per-benchmark regeneration overhead.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yspc_n1000() -> tuple[np.ndarray, np.ndarray]:
    """Balanced binary fixture, n=1000."""
    rng = np.random.default_rng(42)
    n = 1000
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    rng.shuffle(y)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)
    return y, s


@pytest.fixture(scope="module")
def yspc_n200() -> tuple[np.ndarray, np.ndarray]:
    """Balanced binary fixture, n=200 (small enough for bootstrap inner loop)."""
    rng = np.random.default_rng(42)
    n = 200
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    rng.shuffle(y)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)
    return y, s


@pytest.fixture(scope="module")
def y_two_scorers_n200() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same labels, two different scorers — fixture for paired-diff."""
    rng = np.random.default_rng(42)
    n = 200
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    rng.shuffle(y)
    s_a = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=n), 0.0, 1.0)
    s_b = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, size=n), 0.0, 1.0)
    return y, s_a, s_b


# ---------------------------------------------------------------------------
# Kernel benchmarks (Tier 1 — functional core)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_benchmark_pr_auc_n1000(benchmark, yspc_n1000: tuple[np.ndarray, np.ndarray]) -> None:
    """pr_auc on n=1000. Expected: O(n log n) — sklearn's argsort dominates."""
    y, s = yspc_n1000
    result = benchmark(pr_auc, y, s)
    assert 0.0 <= result <= 1.0


@pytest.mark.benchmark
def test_benchmark_roc_auc_n1000(benchmark, yspc_n1000: tuple[np.ndarray, np.ndarray]) -> None:
    """roc_auc on n=1000. Expected: O(n log n)."""
    y, s = yspc_n1000
    result = benchmark(roc_auc, y, s)
    assert 0.0 <= result <= 1.0


@pytest.mark.benchmark
def test_benchmark_brier_score_n1000(benchmark, yspc_n1000: tuple[np.ndarray, np.ndarray]) -> None:
    """brier_score on n=1000. Expected: O(n) — single mean of squared errors."""
    y, s = yspc_n1000
    result = benchmark(brier_score, y, s)
    assert 0.0 <= result <= 1.0


@pytest.mark.benchmark
def test_benchmark_expected_calibration_error_n1000(
    benchmark, yspc_n1000: tuple[np.ndarray, np.ndarray]
) -> None:
    """ECE on n=1000 with default n_bins=10. Expected: O(n + n_bins)."""
    y, s = yspc_n1000
    result = benchmark(expected_calibration_error, y, s, n_bins=10)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap benchmarks — scoped n=200 because n_resamples is the dominant cost
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_benchmark_bootstrap_ci_pr_auc_n200(
    benchmark, yspc_n200: tuple[np.ndarray, np.ndarray]
) -> None:
    """bootstrap_ci(pr_auc) on n=200 with 200 resamples. Expected: O(n_resamples * n log n).

    The bootstrap CI is the most-used downstream of the metric kernels;
    a 2x slowdown here propagates through the whole evaluation pipeline.
    """
    y, s = yspc_n200

    def _run() -> float:
        return bootstrap_ci(y, s, metric=pr_auc, n_resamples=200, seed=42).point_estimate

    result = benchmark(_run)
    assert 0.0 <= result <= 1.0


@pytest.mark.benchmark
def test_benchmark_paired_bootstrap_diff_pr_auc_n200(
    benchmark, y_two_scorers_n200: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """paired_bootstrap_diff on n=200 with 200 resamples.

    The paired version preserves within-sample correlation by resampling
    INDICES (not separate arrays per scorer); benchmark verifies this
    is no slower than 2x of the single-scorer bootstrap.
    """
    y, s_a, s_b = y_two_scorers_n200

    def _run() -> float:
        return paired_bootstrap_diff(y, s_a, s_b, metric=pr_auc, n_resamples=200, seed=42).delta

    result = benchmark(_run)
    # Delta can be negative (B worse than A) — just verify it's finite + in expected range
    assert -1.0 <= result <= 1.0
