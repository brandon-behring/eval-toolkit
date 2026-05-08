"""Smoke tests for the v0.7.0 DatasetLoader Protocol + reference impls."""

from __future__ import annotations

import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.loaders import (
    DataFrameLoader,
    DatasetLoader,
    HFDatasetsLoader,
    SingleSliceLoader,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train"] * 12 + ["val"] * 4 + ["test"] * 4,
            "text": [f"t{i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "loader_factory",
    [
        lambda df: DataFrameLoader(df=df, split_col="split"),
        lambda _df: SingleSliceLoader(
            slice_=EvalSlice(name="x", df=pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}))
        ),
        lambda _df: HFDatasetsLoader(repo_id="dummy/example"),
    ],
)
def test_loaders_implement_protocol(loader_factory, df: pd.DataFrame) -> None:
    loader = loader_factory(df)
    assert isinstance(loader, DatasetLoader)


@pytest.mark.unit
def test_dataframe_loader_splits_correctly(df: pd.DataFrame) -> None:
    splits = DataFrameLoader(df=df, split_col="split").load_splits()
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"].df) == 12
    assert len(splits["val"].df) == 4
    assert len(splits["test"].df) == 4
    # Column metadata propagates.
    assert splits["train"].feature_col == "text"
    assert splits["train"].label_col == "label"


@pytest.mark.unit
def test_dataframe_loader_describe_returns_croissant_subset(df: pd.DataFrame) -> None:
    desc = DataFrameLoader(
        df=df, split_col="split", name="test", cite_as="arXiv:0000.0000", license="MIT"
    ).describe()
    assert desc["name"] == "test"
    # Croissant-compatible 'citeAs' camelCase key in the output dict
    assert desc["citeAs"] == "arXiv:0000.0000"
    assert desc["license"] == "MIT"
    assert "distribution" in desc
    assert desc["n_total_rows"] == len(df)


@pytest.mark.unit
def test_dataframe_loader_rejects_missing_columns(df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="missing column"):
        DataFrameLoader(df=df, split_col="split", feature_col="nope")


@pytest.mark.unit
def test_single_slice_loader_returns_all_key() -> None:
    parent = EvalSlice(name="anything", df=pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}))
    loader = SingleSliceLoader(slice_=parent)
    splits = loader.load_splits()
    assert list(splits.keys()) == ["all"]
    assert splits["all"].name == "all"


@pytest.mark.unit
def test_hf_datasets_loader_raises_clear_error_when_not_installed() -> None:
    """If 'datasets' isn't installed (CI default), error message points at install."""
    loader = HFDatasetsLoader(repo_id="dummy/example")
    try:
        import datasets  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional 'datasets' package"):
            loader.load_splits()


@pytest.mark.unit
def test_parquet_glob_loader_loads_and_hashes(tmp_path) -> None:
    """ParquetGlobLoader reads parquet files and emits sha256 in describe()."""
    pytest.importorskip("pyarrow")
    from eval_toolkit.loaders import ParquetGlobLoader

    train_df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    test_df = pd.DataFrame({"text": ["d", "e"], "label": [1, 0]})
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    loader = ParquetGlobLoader(
        splits={"train": str(train_path), "test": str(test_path)},
        name="test_dataset",
    )
    splits = loader.load_splits()
    assert set(splits.keys()) == {"train", "test"}
    assert len(splits["train"].df) == 3
    assert len(splits["test"].df) == 2

    desc = loader.describe()
    assert desc["name"] == "test_dataset"
    distribution = desc["distribution"]
    assert isinstance(distribution, list)
    assert len(distribution) == 2
    for entry in distribution:
        assert entry["sha256"]  # non-empty hash
        assert entry["contentSize"] > 0


@pytest.mark.unit
def test_parquet_glob_loader_rejects_missing_files(tmp_path) -> None:
    from eval_toolkit.loaders import ParquetGlobLoader

    loader = ParquetGlobLoader(splits={"x": str(tmp_path / "nope*.parquet")})
    with pytest.raises(FileNotFoundError, match="matched no files"):
        loader.load_splits()


@pytest.mark.unit
def test_dataframe_loader_with_strata_col() -> None:
    df = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "text": ["a", "b", "c"],
            "label": [0, 1, 0],
            "source": ["s1", "s2", "s1"],
        }
    )
    loader = DataFrameLoader(df=df, split_col="split", strata_col="source")
    splits = loader.load_splits()
    assert splits["train"].strata_col == "source"


@pytest.mark.unit
def test_dataframe_loader_rejects_missing_strata() -> None:
    df = pd.DataFrame({"split": ["train"], "text": ["a"], "label": [0]})
    with pytest.raises(KeyError, match="strata column"):
        DataFrameLoader(df=df, split_col="split", strata_col="missing")
