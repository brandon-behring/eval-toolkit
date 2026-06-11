"""Coverage-focused tests for eval_toolkit.loaders error paths.

Pairs with the happy-path coverage in ``test_loaders.py``. Targets
``ParquetGlobLoader.load_splits`` column validation and the
``HFDatasetsLoader`` soft-import + per-split column validation, both
mocked so this file doesn't need ``datasets`` installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from eval_toolkit.loaders import HFDatasetsLoader, ParquetGlobLoader


@pytest.fixture(autouse=True)
def _restore_datasets_module():
    """Snapshot/restore sys.modules['datasets'] so mocking leaks don't pollute later tests."""
    sentinel = object()
    original = sys.modules.get("datasets", sentinel)
    yield
    if original is sentinel:
        sys.modules.pop("datasets", None)
    else:
        sys.modules["datasets"] = original


# ---------------------------------------------------------------------------
# ParquetGlobLoader.load_splits: missing-column branch (line 272)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parquet_glob_loader_rejects_missing_feature_column(tmp_path: Path) -> None:
    """ParquetGlobLoader.load_splits raises KeyError when feature_col is absent."""
    df = pd.DataFrame({"wrong_name": ["a", "b"], "label": [0, 1]})
    parquet_path = tmp_path / "x.parquet"
    df.to_parquet(parquet_path)
    loader = ParquetGlobLoader(splits={"train": str(parquet_path)}, feature_col="text")
    with pytest.raises(KeyError, match="missing column 'text'"):
        loader.load_splits()


@pytest.mark.unit
def test_parquet_glob_loader_rejects_missing_label_column(tmp_path: Path) -> None:
    """Missing label_col → KeyError mentioning the column name."""
    df = pd.DataFrame({"text": ["a", "b"]})
    parquet_path = tmp_path / "x.parquet"
    df.to_parquet(parquet_path)
    loader = ParquetGlobLoader(splits={"train": str(parquet_path)})
    with pytest.raises(KeyError, match="missing column 'label'"):
        loader.load_splits()


# ---------------------------------------------------------------------------
# HFDatasetsLoader.load_splits: soft-import + load + validation branches
# ---------------------------------------------------------------------------


def _install_fake_datasets_module(splits: dict[str, pd.DataFrame]) -> MagicMock:
    """Inject a fake ``datasets`` module into ``sys.modules``.

    Returns the captured ``load_dataset`` mock so tests can assert on call
    arguments. Each split's value is a small DataFrame that the loader
    converts via the fake ``Dataset.to_pandas()``.
    """

    class _FakeDataset:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    fake_dataset_dict = {name: _FakeDataset(df) for name, df in splits.items()}
    load_dataset_mock = MagicMock(return_value=fake_dataset_dict)
    fake_module = MagicMock()
    fake_module.load_dataset = load_dataset_mock
    sys.modules["datasets"] = fake_module
    return load_dataset_mock


@pytest.mark.unit
def test_hf_datasets_loader_passes_config_name() -> None:
    """When config_name is set, load_dataset is called with the name kwarg (line 366)."""
    fake_load = _install_fake_datasets_module(
        {"train": pd.DataFrame({"text": ["a"], "label": [0]})}
    )
    loader = HFDatasetsLoader(repo_id="dummy/example", config_name="default")
    loader.load_splits()
    fake_load.assert_called_once_with("dummy/example", name="default")


@pytest.mark.unit
def test_hf_datasets_loader_default_no_config_name() -> None:
    """When config_name is None, load_dataset is called positionally (line 367)."""
    fake_load = _install_fake_datasets_module(
        {"train": pd.DataFrame({"text": ["a"], "label": [0]})}
    )
    loader = HFDatasetsLoader(repo_id="dummy/example")
    loader.load_splits()
    fake_load.assert_called_once_with("dummy/example")


@pytest.mark.unit
def test_hf_datasets_loader_loads_all_splits_by_default() -> None:
    """When splits is None, every key of the loaded DatasetDict is consumed."""
    _install_fake_datasets_module(
        {
            "train": pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}),
            "test": pd.DataFrame({"text": ["c"], "label": [0]}),
        }
    )
    loader = HFDatasetsLoader(repo_id="dummy/example")
    splits = loader.load_splits()
    assert set(splits.keys()) == {"train", "test"}


@pytest.mark.unit
def test_hf_datasets_loader_rejects_missing_feature_column() -> None:
    """Per-split missing feature_col → KeyError (line 380)."""
    _install_fake_datasets_module({"train": pd.DataFrame({"wrong_name": ["a"], "label": [0]})})
    loader = HFDatasetsLoader(repo_id="dummy/example", feature_col="text")
    with pytest.raises(KeyError, match="missing column 'text'"):
        loader.load_splits()


@pytest.mark.unit
def test_hf_datasets_loader_rejects_missing_label_column() -> None:
    """Per-split missing label_col → KeyError."""
    _install_fake_datasets_module({"train": pd.DataFrame({"text": ["a"]})})
    loader = HFDatasetsLoader(repo_id="dummy/example")
    with pytest.raises(KeyError, match="missing column 'label'"):
        loader.load_splits()


@pytest.mark.unit
def test_hf_datasets_loader_subset_splits() -> None:
    """Requesting a subset of splits filters the DatasetDict."""
    _install_fake_datasets_module(
        {
            "train": pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}),
            "test": pd.DataFrame({"text": ["c"], "label": [0]}),
            "validation": pd.DataFrame({"text": ["d"], "label": [1]}),
        }
    )
    loader = HFDatasetsLoader(repo_id="dummy/example", splits=("train", "test"))
    splits = loader.load_splits()
    assert set(splits.keys()) == {"train", "test"}


@pytest.mark.unit
def test_hf_datasets_loader_describe_uses_url_or_default() -> None:
    """Describe(): url falls back to huggingface.co/datasets/<repo_id>.

    ``fetch_remote_metadata=False`` keeps this a unit test (no network).
    The network-enabled path is exercised in test_croissant_e2e.py.
    """
    loader = HFDatasetsLoader(repo_id="dummy/example", fetch_remote_metadata=False)
    out = loader.describe()
    assert out["url"] == "https://huggingface.co/datasets/dummy/example"
    assert out["distribution"][0]["sha256"] == ""


@pytest.mark.unit
def test_hf_datasets_loader_describe_with_explicit_url() -> None:
    loader = HFDatasetsLoader(
        repo_id="dummy/example",
        url="https://example.com",
        fetch_remote_metadata=False,
    )
    out = loader.describe()
    assert out["url"] == "https://example.com"


@pytest.mark.unit
def test_hf_datasets_loader_describe_includes_revision() -> None:
    """#98 Tier-2 ADDITIVE: revision-pinned loads are distinguishable in provenance output."""
    loader = HFDatasetsLoader(
        repo_id="dummy/example", revision="abc123", fetch_remote_metadata=False
    )
    assert loader.describe()["revision"] == "abc123"


@pytest.mark.unit
def test_hf_datasets_loader_describe_revision_defaults_none() -> None:
    loader = HFDatasetsLoader(repo_id="dummy/example", fetch_remote_metadata=False)
    assert loader.describe()["revision"] is None
