"""Hypothesis property tests for v0.7.0 DatasetLoader reference impls.

Restores coverage on `src/eval_toolkit/loaders.py` toward the 90 % gate.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit import EvalSlice
from eval_toolkit.loaders import DataFrameLoader, SingleSliceLoader

CROISSANT_KEYS = {"name", "description", "citeAs", "license", "url", "distribution"}


# ---------------------------------------------------------------------------
# DataFrameLoader
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    splits_def=st.lists(
        st.tuples(
            st.sampled_from(["train", "val", "test", "ood"]),
            st.integers(2, 30),
        ),
        min_size=1,
        max_size=4,
        unique_by=lambda x: x[0],
    )
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_dataframe_loader_split_keys_match_split_col(splits_def: list[tuple[str, int]]) -> None:
    """load_splits() keys = unique values of split_col; row counts preserved."""
    rows = []
    for split_name, n in splits_def:
        for i in range(n):
            rows.append({"split": split_name, "text": f"{split_name}_{i}", "label": i % 2})
    df = pd.DataFrame(rows)
    loader = DataFrameLoader(df=df, split_col="split")
    splits = loader.load_splits()
    # Keys match unique split values.
    assert set(splits.keys()) == {name for name, _ in splits_def}
    # Per-split row counts match the input.
    for split_name, expected_n in splits_def:
        assert len(splits[split_name].df) == expected_n
    # Total preserved.
    assert sum(len(s.df) for s in splits.values()) == len(df)


@pytest.mark.property
@given(
    n=st.integers(2, 50),
    name=st.text(min_size=0, max_size=20),
    license_=st.text(min_size=0, max_size=20),
)
@settings(deadline=None, max_examples=10)
def test_dataframe_loader_describe_returns_croissant_subset(
    n: int, name: str, license_: str
) -> None:
    """describe() always returns the Croissant key set regardless of metadata."""
    df = pd.DataFrame({"split": ["x"] * n, "text": [f"r_{i}" for i in range(n)], "label": [0] * n})
    loader = DataFrameLoader(df=df, split_col="split", name=name, license=license_)
    desc = loader.describe()
    for key in CROISSANT_KEYS:
        assert key in desc
    assert desc["license"] == license_
    assert desc["n_total_rows"] == n


@pytest.mark.property
def test_dataframe_loader_rejects_missing_columns() -> None:
    """Constructor rejects when feature_col / label_col / split_col is absent."""
    df = pd.DataFrame({"split": ["a"], "text": ["x"], "label": [0]})
    with pytest.raises(KeyError, match="missing column"):
        DataFrameLoader(df=df, split_col="missing")
    with pytest.raises(KeyError, match="missing column"):
        DataFrameLoader(df=df, split_col="split", feature_col="missing")
    with pytest.raises(KeyError, match="missing column"):
        DataFrameLoader(df=df, split_col="split", label_col="missing")


# ---------------------------------------------------------------------------
# SingleSliceLoader
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(1, 50),
    parent_name=st.text(min_size=0, max_size=20),
)
@settings(deadline=None, max_examples=15)
def test_single_slice_loader_emits_only_all_key(n: int, parent_name: str) -> None:
    """SingleSliceLoader.load_splits() always emits {'all': ...} regardless of parent."""
    parent = EvalSlice(
        name=parent_name or "anything",
        df=pd.DataFrame({"text": [f"r_{i}" for i in range(n)], "label": [i % 2 for i in range(n)]}),
    )
    splits = SingleSliceLoader(slice_=parent).load_splits()
    assert list(splits.keys()) == ["all"]
    assert splits["all"].name == "all"
    # Original df preserved (same row count, same column structure).
    assert len(splits["all"].df) == n


@pytest.mark.property
@given(n=st.integers(1, 50))
@settings(deadline=None, max_examples=10)
def test_single_slice_loader_describe_returns_croissant_subset(n: int) -> None:
    """SingleSliceLoader.describe() returns the Croissant key set."""
    parent = EvalSlice(
        name="anything",
        df=pd.DataFrame({"text": ["x"] * n, "label": [0] * n}),
    )
    desc = SingleSliceLoader(slice_=parent).describe()
    for key in CROISSANT_KEYS:
        assert key in desc
    assert desc["n_total_rows"] == n
