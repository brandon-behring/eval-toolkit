"""Smoke tests for the v0.7.0 Splitter Protocol + reference impls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.splits import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    SourceDisjointKFoldSplitter,
    Splitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
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
