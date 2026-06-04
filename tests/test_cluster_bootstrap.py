"""Tests for :func:`eval_toolkit.bootstrap.cluster_bootstrap_ci`.

Covers the public contract (percentile CI, method tag, validation), the ``(label, group)``
resample-unit semantics (mixed-label groups split by label), the ``resample_labels`` knob, the
v0.34.0 n_jobs-reproducibility contract (same seed → identical CI across worker counts), and the
methodological reason the function exists: under strong intra-cluster correlation the cluster
bootstrap CI is **wider** than a naive row bootstrap (which would under-cover).
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.bootstrap import (
    BootstrapCI,
    _label_cluster_units,
    bootstrap_ci,
    cluster_bootstrap_ci,
)
from eval_toolkit.metrics import roc_auc


def _clustered_inputs(
    n_clusters: int = 40, rows_per_cluster: int = 5, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster-pure labels (cluster parity) + a separable score with cluster-level noise."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_clusters), rows_per_cluster)
    y = (groups % 2).astype(int)
    cluster_offset = rng.normal(0, 0.5, size=n_clusters)[groups]
    s = y + cluster_offset + rng.normal(0, 0.2, size=y.size)
    return y, s, groups


@pytest.mark.unit
def test_returns_bootstrap_ci_with_cluster_method() -> None:
    """Basic contract: BootstrapCI, cluster_percentile method, ordered CI, point in range."""
    y, s, g = _clustered_inputs()
    ci = cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=300, rng=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.method == "cluster_percentile"
    assert ci.n_resamples == 300
    assert 0.0 <= ci.ci_low <= ci.ci_high <= 1.0
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


@pytest.mark.unit
def test_label_cluster_units_splits_mixed_label_group() -> None:
    """A group id present under both labels splits into one positive unit and one negative unit."""
    y = np.array([1, 1, 0, 0, 1])
    groups = np.array(["g0", "g0", "g0", "g1", "g2"])  # g0 is mixed-label
    units = _label_cluster_units(y, groups)
    assert set(np.concatenate(units[1]).tolist()) == {0, 1, 4}  # positives: g0 + g2
    assert set(np.concatenate(units[0]).tolist()) == {2, 3}  # negatives: g0 + g1
    assert len(units[1]) == 2 and len(units[0]) == 2  # g0 counted on both sides


@pytest.mark.unit
def test_resample_labels_positive_only_holds_negatives_fixed() -> None:
    """resample_labels=(1,) resamples positive clusters and keeps all negatives fixed."""
    y, s, g = _clustered_inputs()
    ci_pos = cluster_bootstrap_ci(y, s, g, roc_auc, resample_labels=(1,), n_resamples=300, rng=0)
    ci_both = cluster_bootstrap_ci(y, s, g, roc_auc, resample_labels=(0, 1), n_resamples=300, rng=0)
    # Both valid CIs; holding negatives fixed removes one variance source → no wider than both-sides.
    assert ci_pos.ci_low <= ci_pos.ci_high
    assert (ci_pos.ci_high - ci_pos.ci_low) <= (ci_both.ci_high - ci_both.ci_low) + 1e-9


@pytest.mark.unit
@pytest.mark.slow
def test_njobs_reproducibility() -> None:
    """Same seed produces a bit-for-bit-identical CI regardless of n_jobs (spawn_seed_sequences)."""
    y, s, g = _clustered_inputs()
    r1 = cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=200, rng=42, n_jobs=1)
    r2 = cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=200, rng=42, n_jobs=2)
    assert (r1.point_estimate, r1.ci_low, r1.ci_high) == (r2.point_estimate, r2.ci_low, r2.ci_high)


@pytest.mark.unit
@pytest.mark.slow
def test_njobs_minus_one_runs() -> None:
    """n_jobs=-1 (all cores) completes without error."""
    y, s, g = _clustered_inputs()
    ci = cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=100, rng=42, n_jobs=-1)
    assert ci.ci_low <= ci.ci_high


@pytest.mark.unit
def test_cluster_ci_wider_than_row_under_intracluster_correlation() -> None:
    """The reason the function exists: with few, strongly-correlated clusters the cluster bootstrap
    CI is wider than a naive row bootstrap (which would under-cover by treating rows as i.i.d.)."""
    # 10 clusters × 30 rows; the cluster-level offset dominates → rows within a cluster are highly
    # correlated, so the effective sample size is ~10 clusters, not 300 rows.
    rng = np.random.default_rng(7)
    n_clusters, per = 10, 30
    groups = np.repeat(np.arange(n_clusters), per)
    y = (groups % 2).astype(int)
    s = y + rng.normal(0, 0.8, size=n_clusters)[groups] + rng.normal(0, 0.05, size=y.size)
    cluster = cluster_bootstrap_ci(y, s, groups, roc_auc, n_resamples=400, rng=0)
    row = bootstrap_ci(y, s, roc_auc, n_resamples=400, rng=0, method="percentile")
    cluster_width = cluster.ci_high - cluster.ci_low
    row_width = row.ci_high - row.ci_low
    assert cluster_width > row_width, f"cluster {cluster_width:.3f} !> row {row_width:.3f}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resample_labels": ()}, "non-empty"),
        ({"resample_labels": (2,)}, "absent from y_true"),
        ({"confidence": 1.5}, r"confidence must be in \(0, 1\)"),
        ({"n_jobs": 0}, "n_jobs"),
    ],
)
def test_validation_errors(kwargs: dict[str, object], match: str) -> None:
    """Invalid parameters raise ValueError with a diagnostic message."""
    y, s, g = _clustered_inputs()
    with pytest.raises(ValueError, match=match):
        cluster_bootstrap_ci(y, s, g, roc_auc, n_resamples=50, rng=0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_shape_and_size_validation() -> None:
    """Shape mismatch and n < 10 raise ValueError."""
    y, s, g = _clustered_inputs(n_clusters=4, rows_per_cluster=3)  # n=12
    with pytest.raises(ValueError, match="shapes mismatch"):
        cluster_bootstrap_ci(y, s[:-1], g, roc_auc, n_resamples=50, rng=0)
    with pytest.raises(ValueError, match="too small"):
        cluster_bootstrap_ci(y[:8], s[:8], g[:8], roc_auc, n_resamples=50, rng=0)
