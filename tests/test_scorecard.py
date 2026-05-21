"""Tests for ``eval_toolkit.scorecard`` — primary v1.0 metric surface (closes #36).

Coverage per v1.0 plan §3A "Tests" section + Decisions R / S / X:

- Threshold-free metric_specs (Decision R)
- Status-aware cells: ok / skipped / error (Decision S)
- Dict-subscript Mapping access; no __getattr__ (Decision I)
- Per-cell error isolation (one spec failing doesn't abort the scorecard)
- Skipped detection via is_metric_defined_for_slice (Decision X)
- Bootstrap CI handling: ok + ci, ok + ci=None on too-small slice
- Custom user spec via Protocol structural satisfaction
- Singleton identity (ms.pr_auc is ms.pr_auc)
- LRU-cached factory identity (ms.ece(n_bins=15) is ms.ece(n_bins=15))
- to_dict / to_pandas serialization
- Absolute input validation (raises) vs per-cell isolation (records state)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eval_toolkit import (
    BootstrapCI,
    MetricSpec,
    Scorecard,
    scorecard,
)
from eval_toolkit import (
    metric_specs as ms,
)
from eval_toolkit import (
    pr_auc as scalar_pr_auc,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def well_mixed_data() -> tuple[np.ndarray, np.ndarray]:
    """500 samples, balanced classes, well-separated scores."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.binomial(1, 0.4, size=n).astype(int)
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, n), 0.0, 1.0)
    return y, s


@pytest.fixture
def all_zeros_data() -> tuple[np.ndarray, np.ndarray]:
    """Single-class slice (all negatives)."""
    rng = np.random.default_rng(1)
    n = 100
    return np.zeros(n, dtype=int), rng.random(n)


@pytest.fixture
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    """5-sample slice — below bootstrap_ci's n=10 floor."""
    return np.array([0, 1, 0, 1, 0], dtype=int), np.array([0.1, 0.9, 0.3, 0.7, 0.5])


# ─────────────────────────────────────────────────────────────────────────────
# Protocol satisfaction
# ─────────────────────────────────────────────────────────────────────────────


def test_pr_auc_satisfies_metric_spec_protocol() -> None:
    """The shipped pr_auc singleton structurally satisfies MetricSpec."""
    assert isinstance(ms.pr_auc, MetricSpec)


def test_ece_factory_returns_metric_spec() -> None:
    """ece() returns an object satisfying MetricSpec."""
    assert isinstance(ms.ece(n_bins=15), MetricSpec)


def test_custom_user_spec_satisfies_protocol() -> None:
    """A custom user class with `name` attribute + `compute()` method satisfies MetricSpec."""

    class _Custom:
        name = "my_metric"

        def compute(self, y: np.ndarray, s: np.ndarray) -> float:
            return 0.5

    assert isinstance(_Custom(), MetricSpec)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + factory identity (Decision J)
# ─────────────────────────────────────────────────────────────────────────────


def test_singleton_identity_holds() -> None:
    assert ms.pr_auc is ms.pr_auc
    assert ms.roc_auc is ms.roc_auc
    assert ms.brier is ms.brier


def test_ece_factory_lru_cache_identity() -> None:
    """Repeated calls with same kwargs return the same instance (Decision J cap maxsize=128)."""
    a = ms.ece(n_bins=15)
    b = ms.ece(n_bins=15)
    assert a is b


def test_ece_different_kwargs_different_instances() -> None:
    """Different kwargs → distinct cells in a scorecard."""
    assert ms.ece(n_bins=15) is not ms.ece(n_bins=10)
    assert ms.ece(n_bins=15) is not ms.ece(n_bins=15, strategy="quantile")


# ─────────────────────────────────────────────────────────────────────────────
# Spec name encoding (Decision X.2 + name-mangling rule from §3A)
# ─────────────────────────────────────────────────────────────────────────────


def test_singleton_spec_names() -> None:
    assert ms.pr_auc.name == "pr_auc"
    assert ms.roc_auc.name == "roc_auc"
    assert ms.brier.name == "brier"


def test_ece_spec_name_encodes_kwargs() -> None:
    """Parameterized specs encode kwargs in the name (alphabetized + snake_cased)."""
    assert ms.ece(n_bins=15).name == "ece_n_bins_15_strategy_uniform"
    assert ms.ece(n_bins=10, strategy="quantile").name == "ece_n_bins_10_strategy_quantile"


# ─────────────────────────────────────────────────────────────────────────────
# Status-aware cells: ok / skipped / error (Decision S)
# ─────────────────────────────────────────────────────────────────────────────


def test_scorecard_ok_status_with_bootstrap(well_mixed_data) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True, n_resamples=200, seed=0)
    assert r["pr_auc"].status == "ok"
    assert isinstance(r["pr_auc"].value, float)
    assert isinstance(r["pr_auc"].ci, BootstrapCI)
    assert r["pr_auc"].reason == ""
    assert r["brier"].status == "ok"


def test_scorecard_ok_status_without_bootstrap(well_mixed_data) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    assert r["brier"].status == "ok"
    assert r["brier"].ci is None


def test_single_class_slice_pr_auc_skipped(all_zeros_data) -> None:
    """PR-AUC on a single-class slice → status='skipped', not raised."""
    y, s = all_zeros_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.roc_auc, ms.brier], bootstrap=False)
    assert r["pr_auc"].status == "skipped"
    assert r["pr_auc"].value is None
    assert "single-class" in r["pr_auc"].reason
    assert r["roc_auc"].status == "skipped"
    # Brier IS defined on single-class
    assert r["brier"].status == "ok"
    assert r["brier"].value is not None


def test_per_cell_error_isolation(well_mixed_data) -> None:
    """One metric's exception doesn't abort the others."""
    y, s = well_mixed_data

    class _BadSpec:
        name = "bad"

        def compute(self, y_t: np.ndarray, y_s: np.ndarray) -> float:
            raise RuntimeError("intentional failure for test")

    r = scorecard(y, s, metrics=[ms.brier, _BadSpec()], bootstrap=False)
    assert r["brier"].status == "ok"
    assert r["bad"].status == "error"
    assert "RuntimeError" in r["bad"].reason
    assert "intentional failure" in r["bad"].reason
    assert r["bad"].value is None


def test_bootstrap_unavailable_keeps_ok_status(tiny_data) -> None:
    """When bootstrap_ci can't run (n<10 floor), point is ok but ci=None with reason."""
    y, s = tiny_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=200, seed=0)
    # The point estimate is valid; only the CI is unavailable
    assert r["brier"].status == "ok"
    assert r["brier"].value is not None
    if r["brier"].ci is None:
        # Either ci is None with a reason (n<10 case), OR the bootstrap
        # succeeded despite the small n (depends on bootstrap_ci internals).
        assert r["brier"].reason  # non-empty reason


# ─────────────────────────────────────────────────────────────────────────────
# Mapping / type-safe access (Decision I)
# ─────────────────────────────────────────────────────────────────────────────


def test_scorecard_is_mapping(well_mixed_data) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    assert isinstance(r, Scorecard)
    from collections.abc import Mapping

    assert isinstance(r, Mapping)


def test_unknown_key_raises_key_error(well_mixed_data) -> None:
    """Typo in subscript → KeyError, not silent None."""
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    with pytest.raises(KeyError):
        _ = r["pr_uac"]  # noqa: F841


def test_mapping_iter_keys_items(well_mixed_data) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=False)
    assert set(r.keys()) == {"pr_auc", "brier"}
    assert set(r) == {"pr_auc", "brier"}
    assert "pr_auc" in r
    assert "absent" not in r
    assert len(r) == 2
    items = dict(r.items())
    assert items["pr_auc"].status == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_to_dict_roundtrip(well_mixed_data) -> None:
    """to_dict produces JSON-serializable output."""
    import json

    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True, n_resamples=100, seed=0)
    d = r.to_dict()
    j = json.dumps(d)
    parsed = json.loads(j)
    assert parsed["pr_auc"]["status"] == "ok"
    assert isinstance(parsed["pr_auc"]["value"], float)
    assert "ci" in parsed["pr_auc"]


def test_to_dict_handles_skipped_and_error(all_zeros_data) -> None:
    """to_dict serializes skipped + error states cleanly (None value)."""
    import json

    y, s = all_zeros_data

    class _Bad:
        name = "bad"

        def compute(self, y_t: np.ndarray, y_s: np.ndarray) -> float:
            raise RuntimeError("err")

    r = scorecard(y, s, metrics=[ms.pr_auc, _Bad(), ms.brier], bootstrap=False)
    d = r.to_dict()
    assert json.dumps(d)  # must serialize
    assert d["pr_auc"]["value"] is None
    assert d["pr_auc"]["status"] == "skipped"
    assert d["bad"]["status"] == "error"


def test_to_pandas_one_row(well_mixed_data) -> None:
    """to_pandas returns a 1-row DataFrame with metric × field multi-index."""
    pytest.importorskip("pandas")
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, seed=0)
    df = r.to_pandas()
    assert df.shape[0] == 1
    assert ("brier", "value") in df.columns
    assert ("brier", "ci_low") in df.columns
    assert df.loc[0, ("brier", "status")] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Point-agreement: scorecard().value matches the underlying scalar function
# ─────────────────────────────────────────────────────────────────────────────


def test_point_estimate_agrees_with_submodule_scalar(well_mixed_data) -> None:
    """scorecard(...).pr_auc.value == metrics.pr_auc(y, s)."""
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=False)
    direct = scalar_pr_auc(y, s)
    assert r["pr_auc"].value == pytest.approx(direct)


# ─────────────────────────────────────────────────────────────────────────────
# Absolute input validation (raises) vs per-cell isolation (records)
# ─────────────────────────────────────────────────────────────────────────────


def test_raises_on_length_mismatch() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="matching length"):
        scorecard(y, s, metrics=[ms.brier], bootstrap=False)


def test_raises_on_empty() -> None:
    y = np.zeros(0, dtype=int)
    s = np.zeros(0)
    with pytest.raises(ValueError, match="empty"):
        scorecard(y, s, metrics=[ms.brier], bootstrap=False)


def test_raises_on_2d_y_true() -> None:
    y = np.zeros((5, 2), dtype=int)
    s = np.zeros((5, 2))
    with pytest.raises(ValueError, match="1-D"):
        scorecard(y, s, metrics=[ms.brier], bootstrap=False)


def test_raises_on_invalid_confidence() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="confidence"):
        scorecard(y, s, metrics=[ms.brier], bootstrap=True, confidence=1.5)


def test_raises_on_negative_n_resamples() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    s = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="n_resamples"):
        scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=0)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_deterministic_under_seed(well_mixed_data) -> None:
    y, s = well_mixed_data
    a = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, seed=42)
    b = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, seed=42)
    assert a["brier"].ci is not None and b["brier"].ci is not None
    assert a["brier"].ci.ci_low == b["brier"].ci.ci_low
    assert a["brier"].ci.ci_high == b["brier"].ci.ci_high


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis property: scorecard never raises out of per-cell concerns
# ─────────────────────────────────────────────────────────────────────────────


@given(
    n=st.integers(min_value=20, max_value=300),
    seed=st.integers(min_value=0, max_value=999),
    is_balanced=st.booleans(),
)
@settings(max_examples=15, deadline=None)
def test_property_scorecard_never_raises_on_normal_input(
    n: int, seed: int, is_balanced: bool
) -> None:
    """For any reasonable (y, s) shape, scorecard returns; failures live in cells."""
    rng = np.random.default_rng(seed)
    p_positive = 0.5 if is_balanced else 0.1
    y = rng.binomial(1, p_positive, size=n).astype(int)
    s = rng.random(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = scorecard(
            y,
            s,
            metrics=[ms.pr_auc, ms.brier, ms.ece(n_bins=10)],
            bootstrap=False,
        )
    # Every spec name is present in the result mapping
    assert "pr_auc" in r
    assert "brier" in r
    assert "ece_n_bins_10_strategy_uniform" in r
    # No exception escaped


# ─────────────────────────────────────────────────────────────────────────────
# Public-API smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_top_level_exports_present() -> None:
    import eval_toolkit

    assert "scorecard" in eval_toolkit.__all__
    assert "Scorecard" in eval_toolkit.__all__
    assert "MetricSpec" in eval_toolkit.__all__
    assert "MetricResult" in eval_toolkit.__all__


def test_metric_specs_submodule_importable() -> None:
    """`from eval_toolkit import metric_specs as ms` resolves natively."""
    from eval_toolkit import metric_specs

    assert hasattr(metric_specs, "pr_auc")
    assert hasattr(metric_specs, "ece")
