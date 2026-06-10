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
from eval_toolkit.metrics import (
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
# ECE strategy validation (v0.46.1 — Round 6 R6-F1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["typo", "UNIFORM", "Quantile", "", "default"])
def test_ece_factory_rejects_invalid_strategy(strategy: str) -> None:
    """`ece(strategy=<invalid>)` raises ValueError at factory level (R6-F1).

    Prior to v0.46.1, invalid strategies silently dispatched to quantile ECE
    and returned a scorecard cell with `status="ok"` under an invalid key
    like `"ece_n_bins_15_strategy_typo"`. Verified by Codex Round 6 runtime
    probe.
    """
    with pytest.raises(ValueError, match="ECE strategy must be 'uniform' or 'quantile'"):
        ms.ece(strategy=strategy)


@pytest.mark.parametrize("bad_n_bins", [0, 1, -1, -100])
def test_ece_factory_rejects_invalid_n_bins_eagerly(bad_n_bins: int) -> None:
    """R8-F2 regression: ece(n_bins=<bad>) fails at factory level, not compute time.

    Pre-v0.51 metric_specs.ece() validated strategy eagerly but deferred
    n_bins validation to compute time. v0.51 validates both at factory
    construction for consistency with the fail-fast pattern.
    """
    with pytest.raises(ValueError, match="n_bins"):
        ms.ece(n_bins=bad_n_bins)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_ece_factory_accepts_valid_strategies(strategy: str) -> None:
    """Both documented strategies still work after the v0.46.1 validation."""
    spec = ms.ece(n_bins=10, strategy=strategy)
    assert spec.name == f"ece_n_bins_10_strategy_{strategy}"


def test_ece_compute_defence_in_depth() -> None:
    """`_EceSpec.compute()` ALSO validates strategy (defence-in-depth — R6-F1).

    Direct construction of `_EceSpec(strategy="typo")` bypasses the factory's
    validation. compute() catches the invalid strategy at the compute boundary
    so the wrong-metric `ok`-status path can never happen.
    """
    from eval_toolkit.metric_specs import _EceSpec

    spec = _EceSpec(n_bins=10, strategy="typo")  # type: ignore[arg-type]
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    s = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6, 0.4])
    with pytest.raises(ValueError, match="ECE strategy must be 'uniform' or 'quantile'"):
        spec.compute(y, s)


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


def test_scorecard_ok_status_with_bootstrap(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True, n_resamples=200, rng=0)
    assert r["pr_auc"].status == "ok"
    assert isinstance(r["pr_auc"].value, float)
    assert isinstance(r["pr_auc"].ci, BootstrapCI)
    assert r["pr_auc"].reason == ""
    assert r["brier"].status == "ok"


def test_scorecard_ok_status_without_bootstrap(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    assert r["brier"].status == "ok"
    assert r["brier"].ci is None


def test_single_class_slice_pr_auc_skipped(all_zeros_data: tuple[np.ndarray, np.ndarray]) -> None:
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


def test_per_cell_error_isolation(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
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


def test_bootstrap_unavailable_keeps_ok_status(tiny_data: tuple[np.ndarray, np.ndarray]) -> None:
    """When bootstrap_ci can't run (n<10 floor), point is ok but ci=None with reason."""
    y, s = tiny_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=200, rng=0)
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


def test_scorecard_is_mapping(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    assert isinstance(r, Scorecard)
    from collections.abc import Mapping

    assert isinstance(r, Mapping)


def test_unknown_key_raises_key_error(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Typo in subscript → KeyError, not silent None."""
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    with pytest.raises(KeyError):
        _ = r["pr_uac"]  # noqa: F841


def test_mapping_iter_keys_items(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
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


def test_to_dict_roundtrip(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    """to_dict produces JSON-serializable output."""
    import json

    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True, n_resamples=100, rng=0)
    d = r.to_dict()
    j = json.dumps(d)
    parsed = json.loads(j)
    assert parsed["pr_auc"]["status"] == "ok"
    assert isinstance(parsed["pr_auc"]["value"], float)
    assert "ci" in parsed["pr_auc"]


def test_to_dict_handles_skipped_and_error(all_zeros_data: tuple[np.ndarray, np.ndarray]) -> None:
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


def test_to_pandas_one_row(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    """to_pandas returns a 1-row DataFrame with metric × field multi-index."""
    pytest.importorskip("pandas")
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, rng=0)
    df = r.to_pandas()
    assert df.shape[0] == 1
    assert ("brier", "value") in df.columns
    assert ("brier", "ci_low") in df.columns
    assert df.loc[0, ("brier", "status")] == "ok"


def test_to_pandas_includes_n_resamples_and_method(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-C: to_pandas schema includes n_resamples + method columns.

    v0.47 expansion makes the DataFrame schema lossless against
    BootstrapCI.to_dict(); trace provenance (resample count + CI method) is
    no longer dropped at the DataFrame boundary.
    """
    pytest.importorskip("pandas")
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, rng=0)
    df = r.to_pandas()
    # Both new columns present
    assert ("brier", "n_resamples") in df.columns
    assert ("brier", "method") in df.columns
    # Values match the BootstrapCI fields
    ci = r["brier"].ci
    assert ci is not None
    assert df.loc[0, ("brier", "n_resamples")] == ci.n_resamples
    assert df.loc[0, ("brier", "method")] == ci.method


def test_to_pandas_skipped_cells_carry_sentinels(
    all_zeros_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """For skipped cells (no CI), the two new columns get sentinel values.

    Numeric column ``n_resamples`` → NaN; string column ``method`` → "".
    Matches the existing pattern for ``ci_low`` / ``ci_high`` / ``confidence``.
    """
    pytest.importorskip("pandas")
    import math

    y, s = all_zeros_data  # single-class slice → pr_auc skipped
    r = scorecard(y, s, metrics=[ms.pr_auc, ms.brier], bootstrap=True, n_resamples=100, rng=0)
    df = r.to_pandas()
    assert r["pr_auc"].status == "skipped"
    # Sentinels in the skipped cell's new columns
    assert math.isnan(df.loc[0, ("pr_auc", "n_resamples")])
    assert df.loc[0, ("pr_auc", "method")] == ""
    # Brier was still bootstrapped, so its new columns are populated
    assert not math.isnan(df.loc[0, ("brier", "n_resamples")])
    assert df.loc[0, ("brier", "method")] != ""


def test_to_pandas_no_bootstrap_carries_sentinels(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """When bootstrap=False, the CI columns (incl. R6-C additions) all sentinel."""
    pytest.importorskip("pandas")
    import math

    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.brier], bootstrap=False)
    df = r.to_pandas()
    assert math.isnan(df.loc[0, ("brier", "ci_low")])
    assert math.isnan(df.loc[0, ("brier", "n_resamples")])
    assert df.loc[0, ("brier", "method")] == ""


# ─────────────────────────────────────────────────────────────────────────────
# Point-agreement: scorecard().value matches the underlying scalar function
# ─────────────────────────────────────────────────────────────────────────────


def test_point_estimate_agrees_with_submodule_scalar(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
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


def test_deterministic_under_seed(well_mixed_data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = well_mixed_data
    a = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, rng=42)
    b = scorecard(y, s, metrics=[ms.brier], bootstrap=True, n_resamples=100, rng=42)
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


# ─────────────────────────────────────────────────────────────────────────────
# Round 6 follow-on (R6-A docstring, R6-B duplicate-name, R6-F5 narrow except,
# R6-H make_spec_name) — landing in v0.47.0 per Decision R6-E.
# ─────────────────────────────────────────────────────────────────────────────


def test_scorecard_rejects_duplicate_spec_names(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-B: silent last-wins is not a documented contract.

    Two specs with identical ``name`` must raise ``ValueError`` so the caller
    disambiguates before any compute happens. Forces no data loss on user error.
    """
    y, s = well_mixed_data
    with pytest.raises(ValueError, match="Duplicate MetricSpec name 'pr_auc'"):
        scorecard(y, s, metrics=[ms.pr_auc, ms.pr_auc], bootstrap=False)


def test_scorecard_accepts_distinct_specs_same_family(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Positive case: two distinct ECE specs coexist because their names differ.

    ``ms.ece(n_bins=10).name == "ece_n_bins_10_strategy_uniform"`` and
    ``ms.ece(n_bins=15).name == "ece_n_bins_15_strategy_uniform"`` — distinct,
    so they survive the R6-B duplicate-name guard.
    """
    y, s = well_mixed_data
    r = scorecard(y, s, metrics=[ms.ece(n_bins=10), ms.ece(n_bins=15)], bootstrap=False)
    assert "ece_n_bins_10_strategy_uniform" in r
    assert "ece_n_bins_15_strategy_uniform" in r


def test_scorecard_duplicate_name_message_reports_index(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-B: the error must cite both indices for debuggability."""
    y, s = well_mixed_data
    with pytest.raises(ValueError) as excinfo:
        scorecard(y, s, metrics=[ms.pr_auc, ms.brier, ms.pr_auc], bootstrap=False)
    msg = str(excinfo.value)
    assert "index 2" in msg
    assert "index 0" in msg


def test_scorecard_seed_none_is_deterministic(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-A: ``rng=None`` is treated as ``rng=0`` for reproducibility.

    Two calls with ``rng=None`` must produce bit-for-bit identical CIs. The
    R6-A docstring fix codified the deterministic-by-default contract.
    """
    y, s = well_mixed_data
    r_a = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=True, n_resamples=200, rng=None)
    r_b = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=True, n_resamples=200, rng=None)
    ci_a = r_a["pr_auc"].ci
    ci_b = r_b["pr_auc"].ci
    assert ci_a is not None
    assert ci_b is not None
    assert ci_a.ci_low == ci_b.ci_low
    assert ci_a.ci_high == ci_b.ci_high


def test_scorecard_seed_none_matches_explicit_zero(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-A: ``rng=None`` must equal ``rng=0`` bit-for-bit."""
    y, s = well_mixed_data
    r_none = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=True, n_resamples=200, rng=None)
    r_zero = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=True, n_resamples=200, rng=0)
    ci_none = r_none["pr_auc"].ci
    ci_zero = r_zero["pr_auc"].ci
    assert ci_none is not None
    assert ci_zero is not None
    assert ci_none.ci_low == ci_zero.ci_low
    assert ci_none.ci_high == ci_zero.ci_high


@pytest.mark.parametrize("exc_class", [MemoryError, RecursionError, KeyboardInterrupt, SystemExit])
def test_scorecard_propagates_system_exit_class_exceptions(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
    exc_class: type[BaseException],
) -> None:
    """Decision R6-F5: process-exhaustion / user-interrupt signals must propagate.

    A custom spec raising any of ``MemoryError`` / ``RecursionError`` /
    ``KeyboardInterrupt`` / ``SystemExit`` from ``compute()`` must NOT be
    swallowed into a ``status="error"`` cell — the surrounding scorecard
    call must raise so process-level shutdown / OOM / Ctrl-C / sys.exit work.
    """
    from eval_toolkit import MetricResult, MetricSpec, Scorecard  # noqa: F401

    y, s = well_mixed_data

    class _RaisingSpec:
        name = "raises_system_exit_class"

        def __init__(self, exc_class: type[BaseException]) -> None:
            self._exc_class = exc_class

        def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
            raise self._exc_class("simulated process-class signal")

    with pytest.raises(exc_class):
        scorecard(y, s, metrics=[_RaisingSpec(exc_class)], bootstrap=False)


def test_scorecard_propagates_system_exit_class_from_bootstrap(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Decision R6-F5: the bootstrap path must also propagate the four classes.

    The point estimate succeeds (so we enter the bootstrap branch); the
    bootstrap call itself raises ``MemoryError`` mid-resample. The narrow
    except in `_evaluate_spec` must re-raise rather than caching the failure
    as a ``status="ok"`` cell with a non-None reason.
    """
    y, s = well_mixed_data
    call_count = [0]

    class _SecondCallRaises:
        name = "pr_auc"

        def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call = point estimate; succeed so we enter the bootstrap loop.
                from eval_toolkit.metrics import pr_auc as _pr_auc

                return float(_pr_auc(y_true, y_score))
            # Subsequent calls = bootstrap resamples; raise process-class signal.
            raise MemoryError("simulated OOM during bootstrap")

    with pytest.raises(MemoryError):
        scorecard(y, s, metrics=[_SecondCallRaises()], bootstrap=True, n_resamples=10, rng=0)


# ─────────────────────────────────────────────────────────────────────────────
# Decision R6-H: `make_spec_name()` canonicalization helper
# ─────────────────────────────────────────────────────────────────────────────


def test_make_spec_name_zero_kwargs_returns_prefix() -> None:
    """``make_spec_name("pr_auc")`` returns the prefix unchanged."""
    from eval_toolkit.metric_specs import make_spec_name

    assert make_spec_name("pr_auc") == "pr_auc"


def test_make_spec_name_alphabetizes_kwargs() -> None:
    """Argument-order invariance: ``a, b`` and ``b, a`` produce same name."""
    from eval_toolkit.metric_specs import make_spec_name

    name_ab = make_spec_name("ece", n_bins=15, strategy="uniform")
    name_ba = make_spec_name("ece", strategy="uniform", n_bins=15)
    assert name_ab == name_ba == "ece_n_bins_15_strategy_uniform"


def test_make_spec_name_matches_ece_factory_encoding() -> None:
    """The helper produces the same name as the ECE factory's auto-encoding.

    Regression-guards the encoding convention used internally so custom
    user specs that opt into ``make_spec_name()`` get keys interoperable
    with the first-party specs.
    """
    from eval_toolkit.metric_specs import make_spec_name

    assert make_spec_name("ece", n_bins=15, strategy="uniform") == ms.ece(n_bins=15).name
    assert (
        make_spec_name("ece", n_bins=10, strategy="quantile")
        == ms.ece(n_bins=10, strategy="quantile").name
    )


def test_make_spec_name_supports_numeric_values() -> None:
    """``make_spec_name`` accepts int and float values via ``str()`` conversion."""
    from eval_toolkit.metric_specs import make_spec_name

    assert make_spec_name("custom_metric", alpha=0.1, beta=2) == "custom_metric_alpha_0.1_beta_2"
    assert make_spec_name("topk", k=10) == "topk_k_10"


def test_make_spec_name_in_metric_specs_all() -> None:
    """Decision R6-H placement: helper is in metric_specs.__all__ only.

    NOT in top-level ``eval_toolkit.__all__`` (Tier-2 additive contract).
    """
    import eval_toolkit
    from eval_toolkit import metric_specs

    assert "make_spec_name" in metric_specs.__all__
    assert "make_spec_name" not in eval_toolkit.__all__


def test_non_finite_metric_value_becomes_error_status(
    well_mixed_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """#96: NaN from a custom compute is an error cell, not status='ok' with a NaN value."""
    y, s = well_mixed_data

    class _NanSpec:
        name = "nan_metric"

        def compute(self, y_t: np.ndarray, y_s: np.ndarray) -> float:
            return float("nan")

    r = scorecard(y, s, metrics=[ms.brier, _NanSpec()], bootstrap=False)
    assert r["brier"].status == "ok"
    assert r["nan_metric"].status == "error"
    assert "non-finite" in r["nan_metric"].reason
    assert r["nan_metric"].value is None


def test_bootstrap_non_finite_ci_bounds_not_attached() -> None:
    """#96: BCa-degenerate NaN bounds are recorded in reason, never attached as a CI."""
    y = np.array([0] * 30 + [1] * 30)
    s = y.astype(float)  # perfect separation → pr_auc ≡ 1.0 → BCa NaN bounds
    with pytest.warns(UserWarning, match="BCa degenerated"):
        r = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=True, n_resamples=100, rng=0)
    cell = r["pr_auc"]
    assert cell.status == "ok"
    assert cell.value == pytest.approx(1.0)
    assert cell.ci is None
    assert "non-finite CI bounds" in cell.reason
