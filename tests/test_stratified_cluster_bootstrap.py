"""Tests for :func:`eval_toolkit.bootstrap.stratified_cluster_bootstrap_ci`.

Covers the public contract, the **single-stratum identity reduction == `cluster_bootstrap_ci`**
equivalence, the **seed-averaged gap** shape (`Gx = val − mean_seed(metric)`), a **composite
top−bottom** statistic over strata, the v0.34.0 n_jobs-reproducibility contract, and validation.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    BootstrapCI,
    cluster_bootstrap_ci,
    stratified_cluster_bootstrap_ci,
)
from eval_toolkit.metrics import roc_auc


def _stratum(
    seed: int, *, n_clusters: int = 30, per: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster-pure labels + a separable score with cluster-level noise (one resample-unit)."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_clusters), per)
    y = (g % 2).astype(int)
    s = y + rng.normal(0, 0.3, size=y.size) + rng.normal(0, 0.3, size=n_clusters)[g]
    return y, s, g


def _mean_combine(m: dict) -> float:  # picklable (module-level)
    return float(np.mean(list(m.values())))


def _gap_combine(m: dict, *, val: float) -> float:  # picklable via functools.partial
    return float(val - np.mean(list(m.values())))


def _top_bottom_combine(m: dict, *, top: tuple, bottom: tuple) -> float:  # picklable via partial
    return float(np.mean([m[k] for k in bottom]) - np.mean([m[k] for k in top]))


@pytest.mark.unit
def test_returns_bootstrap_ci_with_stratified_method() -> None:
    """Basic contract: BootstrapCI, stratified_cluster_percentile method, ordered CI."""
    strata = {0: _stratum(0), 1: _stratum(1)}
    ci = stratified_cluster_bootstrap_ci(strata, roc_auc, _mean_combine, n_resamples=300, rng=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.method == "stratified_cluster_percentile"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_single_stratum_identity_equals_cluster_bootstrap_ci() -> None:
    """One stratum + identity reduce reproduces cluster_bootstrap_ci bit-for-bit (same rng path)."""
    y, s, g = _stratum(7)
    strat = stratified_cluster_bootstrap_ci(
        {0: (y, s, g)}, roc_auc, lambda m: m[0], n_resamples=400, rng=11
    )
    single = cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=400, rng=11)
    assert (strat.point_estimate, strat.ci_low, strat.ci_high) == (
        single.point_estimate,
        single.ci_low,
        single.ci_high,
    )


@pytest.mark.unit
def test_seed_averaged_gap_shape() -> None:
    """The dialect/carrier shape: Gx = val − mean_seed(roc_auc) folded into combine."""
    strata = {s: _stratum(s) for s in (0, 1, 2)}
    val = 0.95
    point_metric = float(np.mean([roc_auc(strata[s][0], strata[s][1]) for s in strata]))
    ci = stratified_cluster_bootstrap_ci(
        strata, roc_auc, functools.partial(_gap_combine, val=val), n_resamples=400, rng=0
    )
    assert ci.method == "stratified_cluster_percentile"
    assert np.isclose(ci.point_estimate, val - point_metric)  # combine reduces correctly
    assert ci.ci_low <= ci.ci_high


@pytest.mark.unit
def test_composite_top_bottom_statistic() -> None:
    """The §6.5 shape: a top−bottom contrast over strata keyed by group (positives-only resample)."""
    # 4 strata, separability ordered worst->best by key, positives resampled, negatives fixed.
    strata = {k: _stratum(k, n_clusters=20) for k in range(4)}
    ci = stratified_cluster_bootstrap_ci(
        strata,
        roc_auc,
        functools.partial(_top_bottom_combine, top=(0, 1), bottom=(2, 3)),
        resample_labels=(1,),
        n_resamples=300,
        rng=0,
    )
    assert ci.method == "stratified_cluster_percentile"
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
@pytest.mark.slow
def test_njobs_reproducibility() -> None:
    """Same seed → bit-for-bit-identical CI across n_jobs (spawn_seed_sequences)."""
    strata = {0: _stratum(0), 1: _stratum(1), 2: _stratum(2)}
    r1 = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=200, rng=42, n_jobs=1
    )
    r2 = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=200, rng=42, n_jobs=2
    )
    assert (r1.point_estimate, r1.ci_low, r1.ci_high) == (r2.point_estimate, r2.ci_low, r2.ci_high)


@pytest.mark.unit
@pytest.mark.slow
def test_njobs_minus_one_runs() -> None:
    """n_jobs=-1 (all cores) completes without error."""
    strata = {0: _stratum(0), 1: _stratum(1)}
    ci = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=100, rng=42, n_jobs=-1
    )
    assert ci.ci_low <= ci.ci_high


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "strata_override", "match"),
    [
        ({}, {}, "strata must be non-empty"),
        ({"confidence": 1.5}, None, r"confidence must be in \(0, 1\)"),
        ({"resample_labels": ()}, None, "non-empty"),
        ({"n_jobs": 0}, None, "n_jobs"),
    ],
)
def test_validation_errors(kwargs: dict, strata_override, match: str) -> None:
    """Invalid parameters raise ValueError with a diagnostic message."""
    strata = {} if strata_override == {} else {0: _stratum(0), 1: _stratum(1)}
    with pytest.raises(ValueError, match=match):
        stratified_cluster_bootstrap_ci(
            strata, roc_auc, _mean_combine, n_resamples=50, rng=0, **kwargs
        )


@pytest.mark.unit
def test_shape_mismatch_in_a_stratum_raises() -> None:
    """A stratum with misaligned arrays raises ValueError naming the stratum."""
    y, s, g = _stratum(0)
    with pytest.raises(ValueError, match="aligned 1-D"):
        stratified_cluster_bootstrap_ci(
            {"bad": (y, s[:-1], g)}, roc_auc, _mean_combine, n_resamples=50, rng=0
        )


# ─────────────────────────────────────────────────────────────────────────────
# Silent-NaN hardening (#96, v1.9.0)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_nan_stratum_scores_rejected_at_boundary() -> None:
    """NaN in any stratum's scores raises at the validation boundary (was shape-only)."""
    y0, s0, g0 = _stratum(0)
    s0[5] = np.nan
    with pytest.raises(ValueError, match="stratum 0: y_score contains NaN or inf"):
        stratified_cluster_bootstrap_ci(
            {0: (y0, s0, g0)}, roc_auc, _mean_combine, n_resamples=50, rng=0
        )


@pytest.mark.unit
def test_non_finite_point_estimate_raises() -> None:
    """``combine`` returning NaN on the full data raises instead of a silent NaN CI."""
    strata = {0: _stratum(0), 1: _stratum(1)}

    def nan_combine(m: dict) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="non-finite point estimate"):
        stratified_cluster_bootstrap_ci(strata, roc_auc, nan_combine, n_resamples=50, rng=0)


@pytest.mark.unit
def test_nan_resamples_count_as_degenerate() -> None:
    """NaN-returning ``combine`` draws hit the >5% degenerate gate (pre-#96: silent NaN CI)."""
    strata = {0: _stratum(0), 1: _stratum(1)}
    point_values = {k: float(roc_auc(v[0], v[1])) for k, v in strata.items()}

    def nan_on_resamples(m: dict) -> float:
        # Finite on the point-estimate call (per-stratum metrics match the full
        # data), NaN on every resampled draw.
        if all(m[k] == point_values[k] for k in m):
            return 0.5
        return float("nan")

    with pytest.raises(ValueError, match="degenerate"):
        stratified_cluster_bootstrap_ci(strata, roc_auc, nan_on_resamples, n_resamples=50, rng=0)


# ─────────────────────────────────────────────────────────────────────────────
# return_samples — resample-distribution exposure (#93, v1.9.0)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_samples_default_none() -> None:
    strata = {0: _stratum(0), 1: _stratum(1)}
    ci = stratified_cluster_bootstrap_ci(strata, roc_auc, _mean_combine, n_resamples=50, rng=0)
    assert ci.samples is None


@pytest.mark.unit
def test_return_samples_exposes_consistent_distribution() -> None:
    strata = {0: _stratum(0), 1: _stratum(1)}
    ci = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=200, rng=0, return_samples=True
    )
    assert ci.samples is not None
    assert ci.samples.shape == (ci.n_resamples,)
    assert np.isfinite(ci.samples).all()
    assert not ci.samples.flags.writeable
    alpha = 1.0 - ci.confidence
    lo, hi = np.quantile(ci.samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    assert ci.ci_low == pytest.approx(float(lo))
    assert ci.ci_high == pytest.approx(float(hi))


@pytest.mark.unit
def test_return_samples_bit_identical_across_njobs() -> None:
    strata = {0: _stratum(0), 1: _stratum(1)}
    ci1 = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=100, rng=7, return_samples=True
    )
    ci2 = stratified_cluster_bootstrap_ci(
        strata, roc_auc, _mean_combine, n_resamples=100, rng=7, n_jobs=2, return_samples=True
    )
    assert np.array_equal(ci1.samples, ci2.samples)
