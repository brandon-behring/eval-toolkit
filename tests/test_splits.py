"""Smoke tests for the v0.7.0 Splitter Protocol + reference impls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.splits import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    PurgedKFoldSplitter,
    SourceDisjointKFoldSplitter,
    Splitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    compute_label_overlap,
    iter_folds_with_pool,
)


@pytest.fixture
def parent_slice() -> EvalSlice:
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
            "group": [i // 4 for i in range(40)],
            "source": [f"s{i % 5}" for i in range(40)],
            "t": np.arange(40),
        }
    )
    return EvalSlice(name="all", df=df)


@pytest.mark.unit
@pytest.mark.parametrize(
    "splitter",
    [
        HoldoutSplitter(test_size=0.25, seed=42),
        StratifiedKFoldSplitter(k=5, seed=42),
        GroupKFoldSplitter(k=5, group_col="group"),
        SourceDisjointKFoldSplitter(source_col="source", k=3, seed=42),
        TimeSeriesSplitter(k=4, time_col="t"),
    ],
)
def test_splitters_implement_protocol(splitter: Splitter) -> None:
    assert isinstance(splitter, Splitter)


@pytest.mark.unit
def test_holdout_splitter_yields_one_fold(parent_slice: EvalSlice) -> None:
    spl = HoldoutSplitter(test_size=0.25, seed=42)
    folds = list(spl.iter_folds(parent_slice))
    assert len(folds) == 1
    fold = folds[0]
    assert "train" in fold and "test" in fold
    assert len(fold["train"].df) + len(fold["test"].df) == len(parent_slice.df)


@pytest.mark.unit
def test_stratified_kfold_yields_k_folds(parent_slice: EvalSlice) -> None:
    spl = StratifiedKFoldSplitter(k=5, seed=42)
    folds = list(spl.iter_folds(parent_slice))
    assert len(folds) == 5
    for fold in folds:
        assert len(fold["train"].df) + len(fold["test"].df) == len(parent_slice.df)


@pytest.mark.unit
def test_source_disjoint_kfold_keeps_sources_disjoint(parent_slice: EvalSlice) -> None:
    spl = SourceDisjointKFoldSplitter(source_col="source", k=3, seed=42)
    for fold in spl.iter_folds(parent_slice):
        train_sources = set(fold["train"].df["source"].tolist())
        test_sources = set(fold["test"].df["source"].tolist())
        assert not (
            train_sources & test_sources
        ), f"sources overlapped: {train_sources & test_sources}"


@pytest.mark.unit
def test_group_kfold_keeps_groups_disjoint(parent_slice: EvalSlice) -> None:
    spl = GroupKFoldSplitter(k=5, group_col="group")
    for fold in spl.iter_folds(parent_slice):
        train_groups = set(fold["train"].df["group"].tolist())
        test_groups = set(fold["test"].df["group"].tolist())
        assert not (train_groups & test_groups)


@pytest.mark.unit
def test_time_series_splitter_respects_ordering(parent_slice: EvalSlice) -> None:
    spl = TimeSeriesSplitter(k=4, time_col="t")
    for fold in spl.iter_folds(parent_slice):
        # Every train timestamp must be ≤ every test timestamp (monotone).
        max_train = fold["train"].df["t"].max()
        min_test = fold["test"].df["t"].min()
        assert max_train < min_test


@pytest.mark.unit
def test_splitter_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        StratifiedKFoldSplitter(k=1)
    with pytest.raises(ValueError, match="k must be"):
        GroupKFoldSplitter(k=1)


@pytest.mark.unit
def test_holdout_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError, match="test_size must be"):
        HoldoutSplitter(test_size=0.0)
    with pytest.raises(ValueError, match="test_size must be"):
        HoldoutSplitter(test_size=1.0)


# --- v0.19.0: PoolBuilder Protocol + iter_folds_with_pool (F7.1) ---


@pytest.mark.unit
def test_iter_folds_with_pool_attaches_test_and_forwards_pool_keys(
    parent_slice: EvalSlice,
) -> None:
    """v0.19.0 — composition helper yields {train, val, test} per fold."""

    class _Pool:
        """50/50 train/val carve from the splitter's train slice."""

        def build(self, train: EvalSlice, *, fold_idx: int) -> dict[str, EvalSlice]:
            n = len(train.df)
            half = n // 2
            return {
                "train": EvalSlice(name=f"fold{fold_idx}_train", df=train.df.iloc[half:].copy()),
                "val": EvalSlice(name=f"fold{fold_idx}_val", df=train.df.iloc[:half].copy()),
            }

    folds = list(
        iter_folds_with_pool(
            StratifiedKFoldSplitter(k=2, seed=0),
            parent_slice,
            pool_builder=_Pool(),
        )
    )
    assert len(folds) == 2
    for fold in folds:
        assert sorted(fold.keys()) == ["test", "train", "val"]


@pytest.mark.unit
def test_iter_folds_with_pool_rejects_missing_val(parent_slice: EvalSlice) -> None:
    """PoolBuilder.build must return at least train + val."""

    class _BadPool:
        def build(self, train: EvalSlice, *, fold_idx: int) -> dict[str, EvalSlice]:
            return {"train": train}  # missing 'val'

    with pytest.raises(ValueError, match="must return at least"):
        list(
            iter_folds_with_pool(
                StratifiedKFoldSplitter(k=2, seed=0),
                parent_slice,
                pool_builder=_BadPool(),
            )
        )


# ---------------------------------------------------------------------------
# PurgedKFoldSplitter + compute_label_overlap (v0.28.0)
# ---------------------------------------------------------------------------


@pytest.fixture
def time_ordered_slice() -> EvalSlice:
    """Time-ordered 40-row slice with monotonic timestamp column."""
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
            "timestamp": np.arange(40),
        }
    )
    return EvalSlice(name="all", df=df)


@pytest.mark.unit
def test_purged_kfold_implements_splitter_protocol() -> None:
    """PurgedKFoldSplitter is a runtime_checkable Splitter."""
    spl = PurgedKFoldSplitter(n_splits=4)
    assert isinstance(spl, Splitter)


@pytest.mark.unit
def test_purged_kfold_yields_n_folds(time_ordered_slice: EvalSlice) -> None:
    """iter_folds yields exactly n_splits fold-dicts, each {train, test}."""
    spl = PurgedKFoldSplitter(n_splits=4, time_col="timestamp")
    folds = list(spl.iter_folds(time_ordered_slice))
    assert len(folds) == 4
    for fold in folds:
        assert sorted(fold.keys()) == ["test", "train"]
        train_t = set(fold["train"].df["timestamp"].tolist())
        test_t = set(fold["test"].df["timestamp"].tolist())
        assert train_t.isdisjoint(test_t)


@pytest.mark.unit
def test_purged_kfold_purge_gap_drops_correct_samples(
    time_ordered_slice: EvalSlice,
) -> None:
    """purge_gap removes that many samples on each side of test."""
    spl = PurgedKFoldSplitter(n_splits=4, purge_gap=3, time_col="timestamp")
    folds = list(spl.iter_folds(time_ordered_slice))
    fold1 = folds[1]
    train_t = sorted(fold1["train"].df["timestamp"].tolist())
    test_t = sorted(fold1["test"].df["timestamp"].tolist())
    test_min, test_max = test_t[0], test_t[-1]
    for t in train_t:
        assert not (
            test_min - 3 <= t <= test_max + 3
        ), f"train sample at t={t} is within purge_gap=3 of test [{test_min}, {test_max}]"


@pytest.mark.unit
def test_purged_kfold_embargo_drops_post_test_samples(
    time_ordered_slice: EvalSlice,
) -> None:
    """embargo_pct removes samples immediately after the test fold (post-test only)."""
    spl = PurgedKFoldSplitter(n_splits=4, purge_gap=0, embargo_pct=0.1, time_col="timestamp")
    folds = list(spl.iter_folds(time_ordered_slice))
    # Fold 0 (test=[0..9]): n_embargo = int(0.1*40) = 4 → drops [10..13]
    fold0 = folds[0]
    train_t = set(fold0["train"].df["timestamp"].tolist())
    for t in (10, 11, 12, 13):
        assert t not in train_t, f"train should NOT contain post-test embargo sample t={t}"
    assert 14 in train_t


@pytest.mark.unit
def test_purged_kfold_sorts_by_time_col(time_ordered_slice: EvalSlice) -> None:
    """Shuffled slice gets sorted by time_col before splitting."""
    rng = np.random.default_rng(42)
    shuffled_df = time_ordered_slice.df.iloc[rng.permutation(40)].reset_index(drop=True)
    shuffled_slice = EvalSlice(name="shuffled", df=shuffled_df)
    spl = PurgedKFoldSplitter(n_splits=4, time_col="timestamp")
    folds = list(spl.iter_folds(shuffled_slice))
    fold0_test_t = sorted(folds[0]["test"].df["timestamp"].tolist())
    assert fold0_test_t == list(range(10))


@pytest.mark.unit
def test_purged_kfold_no_time_col_uses_existing_order() -> None:
    """time_col=None splits in the slice's existing row order."""
    df = pd.DataFrame({"text": [f"r{i}" for i in range(20)], "label": [i % 2 for i in range(20)]})
    slice_ = EvalSlice(name="ordered", df=df)
    spl = PurgedKFoldSplitter(n_splits=4, time_col=None)
    folds = list(spl.iter_folds(slice_))
    assert folds[0]["test"].df["text"].tolist() == ["r0", "r1", "r2", "r3", "r4"]


@pytest.mark.unit
def test_purged_kfold_get_n_splits(time_ordered_slice: EvalSlice) -> None:
    """get_n_splits returns the configured count."""
    assert PurgedKFoldSplitter(n_splits=4).get_n_splits(time_ordered_slice) == 4
    assert PurgedKFoldSplitter(n_splits=7).get_n_splits(time_ordered_slice) == 7


@pytest.mark.unit
def test_purged_kfold_validates_construction() -> None:
    """Construction rejects invalid parameter values."""
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        PurgedKFoldSplitter(n_splits=1)
    with pytest.raises(ValueError, match="purge_gap must be >= 0"):
        PurgedKFoldSplitter(n_splits=4, purge_gap=-1)
    with pytest.raises(ValueError, match="embargo_pct must be in"):
        PurgedKFoldSplitter(n_splits=4, embargo_pct=-0.1)
    with pytest.raises(ValueError, match="embargo_pct must be in"):
        PurgedKFoldSplitter(n_splits=4, embargo_pct=1.0)


@pytest.mark.unit
def test_purged_kfold_missing_time_col_raises(time_ordered_slice: EvalSlice) -> None:
    """time_col not in DataFrame raises KeyError at iter_folds time."""
    spl = PurgedKFoldSplitter(n_splits=4, time_col="missing_col")
    with pytest.raises(KeyError, match="missing_col"):
        list(spl.iter_folds(time_ordered_slice))


@pytest.mark.unit
def test_purged_kfold_n_splits_too_large_raises() -> None:
    """n_splits >= n_samples raises ValueError at iter_folds time."""
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    slice_ = EvalSlice(name="tiny", df=df)
    spl = PurgedKFoldSplitter(n_splits=5, time_col=None)
    with pytest.raises(ValueError, match="n_splits .* must be <"):
        list(spl.iter_folds(slice_))


# compute_label_overlap


@pytest.mark.unit
def test_compute_label_overlap_basic_correctness() -> None:
    """Pairwise |t_train - t_test| < horizon comparison."""
    t_train = np.array([0, 1, 5, 6])
    t_test = np.array([3, 4])
    overlap = compute_label_overlap(t_train, t_test, horizon=3)
    expected = np.array(
        [
            [False, False],
            [True, False],
            [True, True],
            [False, True],
        ]
    )
    np.testing.assert_array_equal(overlap, expected)


@pytest.mark.unit
def test_compute_label_overlap_horizon_zero_returns_all_false() -> None:
    """horizon=0 means no labels can overlap by definition."""
    t_train = np.arange(10)
    t_test = np.arange(10)
    overlap = compute_label_overlap(t_train, t_test, horizon=0)
    assert overlap.shape == (10, 10)
    assert not overlap.any()


@pytest.mark.unit
def test_compute_label_overlap_rejects_negative_horizon() -> None:
    """Negative horizon is a programming error."""
    with pytest.raises(ValueError, match="horizon must be >= 0"):
        compute_label_overlap(np.array([0, 1]), np.array([2, 3]), horizon=-1)


@pytest.mark.unit
def test_compute_label_overlap_shape_matches_inputs() -> None:
    """Output shape is (len(t_train), len(t_test))."""
    overlap = compute_label_overlap(np.arange(7), np.arange(3), horizon=2)
    assert overlap.shape == (7, 3)


@pytest.mark.unit
def test_purged_kfold_prevents_label_overlap_for_horizon() -> None:
    """When purge_gap=horizon, compute_label_overlap on the resulting folds is all-False.

    LOAD-BEARING: asserts the splitter delivers on its design promise
    (zero train/test label overlap when purge_gap is sized to the horizon).
    """
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
            "timestamp": np.arange(100),
        }
    )
    parent = EvalSlice(name="series", df=df)
    horizon = 5
    spl = PurgedKFoldSplitter(n_splits=5, purge_gap=horizon, embargo_pct=0.0, time_col="timestamp")
    for fold in spl.iter_folds(parent):
        t_train = fold["train"].df["timestamp"].to_numpy()
        t_test = fold["test"].df["timestamp"].to_numpy()
        overlap = compute_label_overlap(t_train, t_test, horizon=horizon)
        assert not overlap.any(), (
            "PurgedKFoldSplitter with purge_gap=horizon must produce "
            "zero label overlap, but found overlapping (train, test) pair"
        )
