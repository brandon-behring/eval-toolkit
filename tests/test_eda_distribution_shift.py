"""Tests for ``eval_toolkit.eda.distribution_shift`` (PAD + MMD + kNN purity).

All fixtures are tiny in-memory NumPy arrays (no network, no model downloads).
Separated Gaussian blobs stand in for "shifted" populations; a single blob split
in half stands in for the "no-shift" floor. Permutation / bootstrap counts are
kept small for speed (the statistics are exercised, not their precision).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval_toolkit.eda import (
    DistributionShiftResult,
    KnnPurityResult,
    MmdResult,
    PadResult,
    distribution_shift,
    knn_purity,
    maximum_mean_discrepancy,
    median_bandwidth,
    proxy_a_distance,
)


@pytest.fixture
def separated() -> tuple[np.ndarray, np.ndarray]:
    """Two clearly-separated Gaussian blobs (real shift)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=(40, 5))
    b = rng.normal(6.0, 1.0, size=(40, 5))
    return a, b


@pytest.fixture
def same_dist() -> tuple[np.ndarray, np.ndarray]:
    """Two halves of one blob (no shift — the floor case)."""
    rng = np.random.default_rng(1)
    pooled = rng.normal(0.0, 1.0, size=(80, 5))
    return pooled[:40], pooled[40:]


# --- _validate_pair (via public functions) ---


@pytest.mark.unit
def test_validate_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        knn_purity(np.zeros(5), np.zeros((3, 5)))


@pytest.mark.unit
def test_validate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        knn_purity(np.zeros((0, 5)), np.zeros((3, 5)))


@pytest.mark.unit
def test_validate_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="feature-dimension mismatch"):
        knn_purity(np.zeros((4, 5)), np.zeros((4, 6)))


# --- median_bandwidth ---


@pytest.mark.unit
def test_median_bandwidth_positive(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, _ = separated
    sigma = median_bandwidth(a)
    assert sigma > 0.0


@pytest.mark.unit
def test_median_bandwidth_subsample_path(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, _ = separated  # 40 rows
    # max_samples below n_rows forces the deterministic subsample branch.
    sigma = median_bandwidth(a, max_samples=10, rng=3)
    assert sigma > 0.0


@pytest.mark.unit
def test_median_bandwidth_too_few_rows() -> None:
    with pytest.raises(ValueError, match=">= 2 rows"):
        median_bandwidth(np.zeros((1, 5)))


@pytest.mark.unit
def test_median_bandwidth_identical_points_raises() -> None:
    with pytest.raises(ValueError, match="all pooled points identical"):
        median_bandwidth(np.ones((6, 4)))


# --- proxy_a_distance ---


@pytest.mark.unit
def test_pad_separated_is_high(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = proxy_a_distance(a, b, n_folds=4)
    assert isinstance(r, PadResult)
    assert r.pad > 1.5
    assert 0.0 <= r.pad <= 2.0
    assert r.n_folds == 4
    assert r.ci_low is None and r.ci_high is None


@pytest.mark.unit
def test_pad_same_dist_is_low(same_dist: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = same_dist
    r = proxy_a_distance(a, b, n_folds=4)
    # No real shift → domain classifier near chance → PAD near 0.
    assert r.pad < 0.7
    assert 0.0 <= r.pad <= 2.0


@pytest.mark.unit
def test_pad_folds_clamped_to_corpus_size() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=(3, 4))
    b = rng.normal(5.0, 1.0, size=(3, 4))
    # Requested 5 folds but only 3 per corpus → clamped to 3.
    r = proxy_a_distance(a, b, n_folds=5)
    assert r.n_folds == 3


@pytest.mark.unit
def test_pad_too_few_samples_raises() -> None:
    a = np.zeros((1, 4))
    b = np.ones((3, 4))
    with pytest.raises(ValueError, match=">= 2 CV folds"):
        proxy_a_distance(a, b)


@pytest.mark.unit
def test_pad_bootstrap_ci(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = proxy_a_distance(a, b, n_folds=4, n_resamples=15, rng=0)
    assert r.ci_low is not None and r.ci_high is not None
    assert r.ci_low <= r.ci_high
    assert r.ci_low >= 0.0 and r.ci_high <= 2.0


# --- maximum_mean_discrepancy ---


@pytest.mark.unit
def test_mmd_separated_significant(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = maximum_mean_discrepancy(a, b, n_permutations=100, rng=0)
    assert isinstance(r, MmdResult)
    assert r.mmd_squared > 0.1
    assert r.p_value < 0.05
    assert r.bandwidth > 0.0
    assert r.n_permutations == 100


@pytest.mark.unit
def test_mmd_same_dist_not_significant(same_dist: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = same_dist
    r = maximum_mean_discrepancy(a, b, n_permutations=100, rng=0)
    assert r.p_value > 0.05


@pytest.mark.unit
def test_mmd_explicit_bandwidth(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = maximum_mean_discrepancy(a, b, bandwidth=2.0, n_permutations=50)
    assert r.bandwidth == pytest.approx(2.0)


@pytest.mark.unit
def test_mmd_rejects_bad_bandwidth(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    with pytest.raises(ValueError, match="bandwidth must be finite and > 0"):
        maximum_mean_discrepancy(a, b, bandwidth=0.0)


@pytest.mark.unit
def test_mmd_rejects_bad_permutations(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    with pytest.raises(ValueError, match="n_permutations must be >= 1"):
        maximum_mean_discrepancy(a, b, n_permutations=0)


@pytest.mark.unit
def test_mmd_rejects_too_few_rows() -> None:
    a = np.zeros((1, 4))
    b = np.ones((5, 4))
    with pytest.raises(ValueError, match=">= 2 rows"):
        maximum_mean_discrepancy(a, b)


@pytest.mark.unit
def test_mmd_p_value_floor(separated: tuple[np.ndarray, np.ndarray]) -> None:
    # p = (1 + count) / (B + 1) is never zero.
    a, b = separated
    r = maximum_mean_discrepancy(a, b, n_permutations=20, rng=0)
    assert r.p_value >= 1.0 / 21.0


@pytest.mark.unit
def test_mmd_bootstrap_ci(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = maximum_mean_discrepancy(a, b, n_permutations=20, n_resamples=15, rng=0)
    assert r.ci_low is not None and r.ci_high is not None
    assert r.ci_low <= r.ci_high


# --- knn_purity ---


@pytest.mark.unit
def test_knn_purity_separated_high(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    r = knn_purity(a, b, k=5)
    assert isinstance(r, KnnPurityResult)
    assert r.mean_purity > 0.95
    assert r.k == 5


@pytest.mark.unit
def test_knn_purity_same_dist_near_half(same_dist: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = same_dist
    r = knn_purity(a, b, k=5)
    # Mixed populations → purity near 0.5 (balanced).
    assert 0.35 < r.mean_purity < 0.65


@pytest.mark.unit
def test_knn_purity_k_clamped() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, size=(5, 3))
    b = rng.normal(9.0, 1.0, size=(5, 3))
    # k far larger than the pool → clamped to total - 1 = 9.
    r = knn_purity(a, b, k=100)
    assert r.k == 9


@pytest.mark.unit
def test_knn_purity_rejects_bad_k(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    with pytest.raises(ValueError, match="k must be >= 1"):
        knn_purity(a, b, k=0)


# --- distribution_shift orchestrator ---


@pytest.mark.unit
def test_distribution_shift_combines_all(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    res = distribution_shift(a, b, pad_folds=4, n_permutations=100, knn_k=5)
    assert isinstance(res, DistributionShiftResult)
    assert res.pad.pad > 1.0
    assert res.mmd.p_value < 0.05
    assert res.knn.mean_purity > 0.9


@pytest.mark.unit
def test_distribution_shift_with_bootstrap(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    res = distribution_shift(a, b, pad_folds=4, n_permutations=20, n_resamples=10, rng=0)
    assert res.pad.ci_low is not None
    assert res.mmd.ci_low is not None


# --- serialization ---


@pytest.mark.unit
def test_results_to_dict_json_round_trip(separated: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = separated
    res = distribution_shift(a, b, pad_folds=4, n_permutations=20)
    restored = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert set(restored) == {"pad", "mmd", "knn"}
    assert set(restored["pad"]) == {
        "pad",
        "domain_classifier_error",
        "n_folds",
        "c",
        "n_a",
        "n_b",
        "ci_low",
        "ci_high",
    }
    assert set(restored["mmd"]) == {
        "mmd_squared",
        "bandwidth",
        "p_value",
        "n_permutations",
        "n_a",
        "n_b",
        "ci_low",
        "ci_high",
    }
    assert set(restored["knn"]) == {"mean_purity", "k", "n_a", "n_b"}
    # null CI fields survive the round-trip as JSON null.
    assert restored["pad"]["ci_low"] is None


# --- silent-NaN hardening (#96, v1.9.0) ---


def test_median_bandwidth_non_finite_input_raises() -> None:
    """NaN/inf input raises at entry (pre-#96 a NaN σ escaped: NaN <= 0.0 is False)."""
    x = np.ones((6, 4))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite value"):
        median_bandwidth(x)


def test_median_bandwidth_non_finite_outside_subsample_raises() -> None:
    """The finiteness check covers the FULL input, not just the subsampled rows."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    x[49, 0] = np.nan  # likely excluded from a 10-row subsample draw
    with pytest.raises(ValueError, match="non-finite value"):
        median_bandwidth(x, max_samples=10, rng=3)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_mmd_non_finite_bandwidth_raises(
    separated: tuple[np.ndarray, np.ndarray], bad: float
) -> None:
    """inf bandwidth → γ=0 → all-ones Gram → MMD²=0 → p=1.0 silently reads 'no shift'."""
    a, b = separated
    with pytest.raises(ValueError, match="bandwidth must be finite"):
        maximum_mean_discrepancy(a, b, bandwidth=bad, n_permutations=5)


def test_validate_pair_non_finite_embedding_raises(
    separated: tuple[np.ndarray, np.ndarray],
) -> None:
    """NaN/inf embeddings raise at the boundary, not deep inside sklearn's check_array."""
    a, b = separated
    b = b.copy()
    b[2, 1] = np.inf
    with pytest.raises(ValueError, match="x_b contains non-finite"):
        proxy_a_distance(a, b)


# --- SPEC 7 rng contract (v1.12.0 rename) ---


@pytest.mark.unit
def test_rng_accepts_generator(separated: tuple[np.ndarray, np.ndarray]) -> None:
    """SPEC 7: ``rng`` accepts a ``np.random.Generator``, not just an int seed."""
    a, b = separated
    r = proxy_a_distance(a, b, n_folds=4, rng=np.random.default_rng(0))
    assert 0.0 <= r.pad <= 2.0
    m = maximum_mean_discrepancy(a, b, n_permutations=20, rng=np.random.default_rng(0))
    assert 0.0 < m.p_value <= 1.0


@pytest.mark.unit
def test_rng_int_seed_is_deterministic(same_dist: tuple[np.ndarray, np.ndarray]) -> None:
    """Same int seed → identical results; a different seed changes them.

    Determinism (same seed → identical output) is checked for both PAD and
    MMD. Cross-seed *sensitivity* is witnessed by MMD's permutation p-value:
    on ``same_dist`` PAD is degenerate (the domain classifier sits at chance,
    so PAD clamps to ~0 and every bootstrap resample clamps too), which makes
    a PAD cross-seed inequality fixture-fragile — so MMD carries that check,
    and PAD gets only the same-seed determinism assertion. (Pinning the PAD
    inequality on the whole dataclass was the fragile form flagged by the
    2026-06-13 review.)
    """
    a, b = same_dist
    r1 = proxy_a_distance(a, b, n_folds=4, n_resamples=10, rng=7)
    r2 = proxy_a_distance(a, b, n_folds=4, n_resamples=10, rng=7)
    assert r1 == r2  # PAD: same seed → identical
    m1 = maximum_mean_discrepancy(a, b, n_permutations=50, rng=7)
    m2 = maximum_mean_discrepancy(a, b, n_permutations=50, rng=7)
    assert m1 == m2  # MMD: same seed → identical
    # Cross-seed sensitivity: MMD's permutation p-value is seed-driven and
    # non-degenerate on same_dist (unlike PAD, which clamps).
    assert maximum_mean_discrepancy(a, b, n_permutations=50, rng=8) != m1


@pytest.mark.unit
def test_negative_n_resamples_raises(same_dist: tuple[np.ndarray, np.ndarray]) -> None:
    """A negative n_resamples raises instead of silently skipping the bootstrap."""
    a, b = same_dist
    with pytest.raises(ValueError, match="n_resamples must be >= 0"):
        proxy_a_distance(a, b, n_folds=4, n_resamples=-3)
    with pytest.raises(ValueError, match="n_resamples must be >= 0"):
        maximum_mean_discrepancy(a, b, n_permutations=5, n_resamples=-3)
