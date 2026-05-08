"""Hypothesis property tests for v0.7.0 Splitter reference impls.

Restores coverage on `src/eval_toolkit/splits.py` toward the 90 % gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit import EvalSlice
from eval_toolkit.splits import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    SourceDisjointKFoldSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
)


def _balanced_slice(n: int, seed: int = 0) -> EvalSlice:
    """Build an n-row slice with balanced labels + group + source + time columns."""
    rng = np.random.default_rng(seed)
    return EvalSlice(
        name="all",
        df=pd.DataFrame(
            {
                "text": [f"r_{i}" for i in range(n)],
                "label": rng.permutation([0] * (n // 2) + [1] * (n - n // 2)).tolist(),
                "group": [i // 4 for i in range(n)],
                "source": [f"s_{i % 5}" for i in range(n)],
                "t": list(range(n)),
            }
        ),
    )


# ---------------------------------------------------------------------------
# HoldoutSplitter
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(20, 200),
    test_size=st.floats(0.10, 0.50, allow_nan=False),
    seed=st.integers(0, 99),
)
@settings(deadline=None, max_examples=15)
def test_holdout_yields_one_fold_with_disjoint_indices(n: int, test_size: float, seed: int) -> None:
    """HoldoutSplitter yields exactly 1 fold; train/test row counts sum to n."""
    parent = _balanced_slice(n, seed=seed)
    spl = HoldoutSplitter(test_size=test_size, seed=seed)
    folds = list(spl.iter_folds(parent))
    assert len(folds) == 1
    fold = folds[0]
    assert "train" in fold and "test" in fold
    assert len(fold["train"].df) + len(fold["test"].df) == n
    assert spl.get_n_splits(parent) == 1


# ---------------------------------------------------------------------------
# StratifiedKFoldSplitter
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(40, 200),
    k=st.integers(2, 6),
    seed=st.integers(0, 99),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_stratified_kfold_test_partitions_disjoint_and_complete(n: int, k: int, seed: int) -> None:
    """K folds' test sets are disjoint and union to the full row set."""
    if n < k * 2:
        return
    parent = _balanced_slice(n, seed=seed)
    spl = StratifiedKFoldSplitter(k=k, seed=seed)
    test_texts: list[set[str]] = []
    for fold in spl.iter_folds(parent):
        test_texts.append(set(fold["test"].df["text"].tolist()))
        assert len(fold["train"].df) + len(fold["test"].df) == n
    # Pairwise disjoint
    for i, a in enumerate(test_texts):
        for b in test_texts[i + 1 :]:
            assert not (a & b)
    # Union = full set
    union = set()
    for s in test_texts:
        union |= s
    assert len(union) == n


# ---------------------------------------------------------------------------
# GroupKFoldSplitter
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(40, 200),
    k=st.integers(2, 5),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_group_kfold_keeps_groups_within_a_fold_disjoint(n: int, k: int) -> None:
    """Within each fold, no group spans train and test."""
    if n < k * 2:
        return
    parent = _balanced_slice(n)
    spl = GroupKFoldSplitter(k=k, group_col="group")
    for fold in spl.iter_folds(parent):
        train_groups = set(fold["train"].df["group"].tolist())
        test_groups = set(fold["test"].df["group"].tolist())
        assert not (train_groups & test_groups)


# ---------------------------------------------------------------------------
# SourceDisjointKFoldSplitter
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(40, 200),
    k=st.integers(2, 4),
    seed=st.integers(0, 99),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_source_disjoint_kfold_train_test_source_disjoint(n: int, k: int, seed: int) -> None:
    """For every fold, train sources and test sources are disjoint."""
    if n < k * 2:
        return
    parent = _balanced_slice(n, seed=seed)
    spl = SourceDisjointKFoldSplitter(source_col="source", k=k, seed=seed)
    for fold in spl.iter_folds(parent):
        train_sources = set(fold["train"].df["source"].tolist())
        test_sources = set(fold["test"].df["source"].tolist())
        assert not (train_sources & test_sources)


# ---------------------------------------------------------------------------
# TimeSeriesSplitter
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(40, 200),
    k=st.integers(2, 5),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_time_series_splitter_train_max_lt_test_min(n: int, k: int) -> None:
    """Every fold satisfies max(train_t) < min(test_t)."""
    if n < k * 2:
        return
    parent = _balanced_slice(n)
    spl = TimeSeriesSplitter(k=k, time_col="t")
    for fold in spl.iter_folds(parent):
        max_train = fold["train"].df["t"].max()
        min_test = fold["test"].df["t"].min()
        assert max_train < min_test


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(bad_k=st.integers(max_value=1))
@settings(deadline=None, max_examples=10)
def test_kfold_rejects_k_lt_2(bad_k: int) -> None:
    """Every K-fold splitter rejects k < 2."""
    with pytest.raises(ValueError, match="k must be"):
        StratifiedKFoldSplitter(k=bad_k)
    with pytest.raises(ValueError, match="k must be"):
        GroupKFoldSplitter(k=bad_k)
    with pytest.raises(ValueError, match="k must be"):
        SourceDisjointKFoldSplitter(source_col="source", k=bad_k)
    with pytest.raises(ValueError, match="k must be"):
        TimeSeriesSplitter(k=bad_k)


@pytest.mark.property
@given(bad_size=st.one_of(st.floats(max_value=0.0), st.floats(min_value=1.0)))
@settings(deadline=None, max_examples=10)
def test_holdout_rejects_test_size_outside_unit_interval(bad_size: float) -> None:
    """HoldoutSplitter rejects test_size outside (0, 1)."""
    if np.isnan(bad_size):
        return
    with pytest.raises(ValueError, match="test_size must be"):
        HoldoutSplitter(test_size=bad_size)
