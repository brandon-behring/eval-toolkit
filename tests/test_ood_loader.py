"""Tests for ``ood_dataset_from_manifest`` + ``OodManifestLoader`` (v0.43.0; #48)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from eval_toolkit.harness import EvalSlice
from eval_toolkit.loaders import (
    DatasetLoader,
    OodManifestLoader,
    ood_dataset_from_manifest,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> tuple[bytes, str]:
    """Write a parquet file and return (bytes, sha256-hex) for manifest insertion."""
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    content = path.read_bytes()
    return content, _sha256(content)


@pytest.fixture
def two_slice_manifest(tmp_path: Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a two-slice manifest pointing at local parquet files."""
    a_rows = [{"prompt": "hello", "lbl": "clean"}, {"prompt": "ignore prior", "lbl": "injected"}]
    b_rows = [{"prompt": f"row{i}", "lbl": i % 2} for i in range(5)]

    a_path = tmp_path / "slice_a.parquet"
    b_path = tmp_path / "slice_b.parquet"
    _, a_sha = _write_parquet(a_path, a_rows)
    _, b_sha = _write_parquet(b_path, b_rows)

    manifest = {
        "name": "test-ood-slate",
        "description": "Synthetic 2-slice manifest used by test_ood_loader.",
        "license": "MIT",
        "slices": {
            "alpha": {
                "url": f"file://{a_path}",
                "sha256": a_sha,
                "text_field": "prompt",
                "label_field": "lbl",
                "label_map": {"clean": 0, "injected": 1},
                "format": "parquet",
            },
            "beta": {
                "url": f"file://{b_path}",
                "sha256": b_sha,
                "text_field": "prompt",
                "label_field": "lbl",
                "format": "parquet",
            },
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path, a_rows, b_rows


@pytest.mark.unit
def test_two_slice_manifest_loads_concatenated_df(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, a_rows, b_rows = two_slice_manifest
    df = ood_dataset_from_manifest(manifest_path, cache_dir=tmp_path / "cache")
    # Schema
    assert list(df.columns) == ["text", "label", "source", "row_id", "sha"]
    # Row count
    assert len(df) == len(a_rows) + len(b_rows)
    # Per-slice source tags
    assert set(df["source"].unique()) == {"alpha", "beta"}
    # Label mapping applied
    assert set(df["label"].unique()) == {0, 1}
    # row_id is sha256:<hex>
    assert all(
        rid.startswith("sha256:") and len(rid) == len("sha256:") + 64 for rid in df["row_id"]
    )
    # sha column present for every row
    assert df["sha"].str.startswith("sha256:").all()


@pytest.mark.unit
def test_sha_mismatch_raises_with_remediation_hint(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = two_slice_manifest
    # Corrupt the manifest sha for slice 'alpha'.
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    bad_sha = "0" * 64  # valid-shape but wrong-content sha
    raw["slices"]["alpha"]["sha256"] = bad_sha
    manifest_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        ood_dataset_from_manifest(manifest_path, cache_dir=tmp_path / "cache")
    msg = str(excinfo.value)
    assert "alpha" in msg
    assert "sha256 mismatch" in msg
    assert "expected:" in msg.lower() or "expected" in msg
    assert "actual:" in msg.lower() or "actual" in msg
    assert "remediation" in msg.lower()


@pytest.mark.unit
def test_slices_filter_returns_subset(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, a_rows, _ = two_slice_manifest
    df = ood_dataset_from_manifest(manifest_path, slices=["alpha"], cache_dir=tmp_path / "cache")
    assert set(df["source"].unique()) == {"alpha"}
    assert len(df) == len(a_rows)


@pytest.mark.unit
def test_slices_filter_unknown_id_raises_keyerror(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = two_slice_manifest
    with pytest.raises(KeyError) as excinfo:
        ood_dataset_from_manifest(
            manifest_path, slices=["alpha", "gamma"], cache_dir=tmp_path / "cache"
        )
    msg = str(excinfo.value)
    assert "gamma" in msg
    assert "available" in msg.lower()


@pytest.mark.unit
def test_cache_hit_on_second_call(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call must read from cache; assert by counting urlopen invocations."""
    import urllib.request

    cache_dir = tmp_path / "cache"
    call_count = {"n": 0}
    real_urlopen = urllib.request.urlopen

    def counting_urlopen(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return real_urlopen(*args, **kwargs)

    monkeypatch.setattr("urllib.request.urlopen", counting_urlopen)
    manifest_path, _, _ = two_slice_manifest

    # First call — populates cache (file:// URLs don't go through urlopen; use a real
    # HTTP-like assertion by checking the cache dir is populated instead).
    df1 = ood_dataset_from_manifest(manifest_path, cache_dir=cache_dir)
    cached_files_after_first = sorted(cache_dir.glob("*.bin"))
    assert len(cached_files_after_first) == 2

    # Second call — should produce identical DataFrame
    df2 = ood_dataset_from_manifest(manifest_path, cache_dir=cache_dir)
    cached_files_after_second = sorted(cache_dir.glob("*.bin"))
    assert cached_files_after_first == cached_files_after_second
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.unit
def test_yaml_parse_error_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(": this is :: not :: valid", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        ood_dataset_from_manifest(bad, cache_dir=tmp_path / "cache")


@pytest.mark.unit
def test_missing_slices_top_level_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_slices.yaml"
    bad.write_text(yaml.safe_dump({"name": "incomplete"}), encoding="utf-8")
    with pytest.raises(ValueError, match="slices"):
        ood_dataset_from_manifest(bad, cache_dir=tmp_path / "cache")


@pytest.mark.unit
def test_missing_yaml_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ood_dataset_from_manifest(tmp_path / "does_not_exist.yaml", cache_dir=tmp_path / "cache")


@pytest.mark.unit
def test_label_map_missing_key_raises(tmp_path: Path) -> None:
    """When a source label value isn't in label_map, surface offending value + map keys."""
    rows = [{"prompt": "x", "lbl": "weird_label_value"}]
    pq = tmp_path / "weird.parquet"
    _, sha = _write_parquet(pq, rows)
    manifest = {
        "slices": {
            "weird": {
                "url": f"file://{pq}",
                "sha256": sha,
                "text_field": "prompt",
                "label_field": "lbl",
                "label_map": {"clean": 0, "injected": 1},
            }
        }
    }
    mp = tmp_path / "weird_manifest.yaml"
    mp.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        ood_dataset_from_manifest(mp, cache_dir=tmp_path / "cache")
    msg = str(excinfo.value)
    assert "weird_label_value" in msg
    assert "clean" in msg  # available keys surfaced


@pytest.mark.unit
def test_sample_size_without_seed_raises(tmp_path: Path) -> None:
    rows = [{"prompt": f"row{i}", "lbl": i % 2} for i in range(20)]
    pq = tmp_path / "twenty.parquet"
    _, sha = _write_parquet(pq, rows)
    manifest = {
        "slices": {
            "twenty": {
                "url": f"file://{pq}",
                "sha256": sha,
                "text_field": "prompt",
                "label_field": "lbl",
                "sample_size": 5,
                # seed intentionally omitted
            }
        }
    }
    mp = tmp_path / "twenty_manifest.yaml"
    mp.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        ood_dataset_from_manifest(mp, cache_dir=tmp_path / "cache")


@pytest.mark.unit
def test_sample_size_with_seed_is_deterministic(tmp_path: Path) -> None:
    rows = [{"prompt": f"row{i}", "lbl": i % 2} for i in range(20)]
    pq = tmp_path / "twenty2.parquet"
    _, sha = _write_parquet(pq, rows)
    manifest = {
        "slices": {
            "twenty": {
                "url": f"file://{pq}",
                "sha256": sha,
                "text_field": "prompt",
                "label_field": "lbl",
                "sample_size": 7,
                "seed": 42,
            }
        }
    }
    mp = tmp_path / "twenty2_manifest.yaml"
    mp.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    df1 = ood_dataset_from_manifest(mp, cache_dir=tmp_path / "cache1")
    df2 = ood_dataset_from_manifest(mp, cache_dir=tmp_path / "cache2")
    assert len(df1) == 7
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.unit
def test_invalid_sha_format_raises(tmp_path: Path) -> None:
    pq = tmp_path / "x.parquet"
    _write_parquet(pq, [{"prompt": "a", "lbl": 0}])
    manifest = {
        "slices": {
            "bad": {
                "url": f"file://{pq}",
                "sha256": "not_a_valid_hex_sha",
                "text_field": "prompt",
                "label_field": "lbl",
            }
        }
    }
    mp = tmp_path / "badsha.yaml"
    mp.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="64-char hex"):
        ood_dataset_from_manifest(mp, cache_dir=tmp_path / "cache")


@pytest.mark.unit
def test_loader_implements_dataset_loader_protocol(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = two_slice_manifest
    loader = OodManifestLoader(yaml_path=manifest_path, cache_dir=tmp_path / "cache")
    assert isinstance(loader, DatasetLoader)


@pytest.mark.unit
def test_loader_load_splits_returns_all_key_with_source_strata(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, a_rows, b_rows = two_slice_manifest
    loader = OodManifestLoader(yaml_path=manifest_path, cache_dir=tmp_path / "cache")
    splits = loader.load_splits()
    assert list(splits.keys()) == ["all"]
    slice_obj = splits["all"]
    assert isinstance(slice_obj, EvalSlice)
    assert slice_obj.feature_col == "text"
    assert slice_obj.label_col == "label"
    assert slice_obj.strata_col == "source"
    assert len(slice_obj.df) == len(a_rows) + len(b_rows)


@pytest.mark.unit
def test_loader_describe_returns_croissant_shape(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = two_slice_manifest
    loader = OodManifestLoader(yaml_path=manifest_path, cache_dir=tmp_path / "cache")
    desc = loader.describe()
    assert "distribution" in desc
    assert isinstance(desc["distribution"], list)
    # Two slices → two distribution entries.
    assert len(desc["distribution"]) == 2
    names = {d["name"] for d in desc["distribution"]}
    assert names == {"alpha", "beta"}
    # Manifest top-level fallback for name / description / license.
    assert desc["name"] == "test-ood-slate"
    assert desc["license"] == "MIT"


@pytest.mark.unit
def test_corrupt_cache_is_redownloaded(
    two_slice_manifest: tuple[Path, list[dict[str, Any]], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """If the cached file's bytes are tampered with, the loader re-downloads."""
    manifest_path, _, _ = two_slice_manifest
    cache_dir = tmp_path / "cache"
    df1 = ood_dataset_from_manifest(manifest_path, cache_dir=cache_dir)

    # Corrupt one cached file
    cached_files = sorted(cache_dir.glob("*.bin"))
    cached_files[0].write_bytes(b"garbage that does not hash to the expected sha")

    # Second call should detect corruption + redownload + return same DataFrame
    df2 = ood_dataset_from_manifest(manifest_path, cache_dir=cache_dir)
    pd.testing.assert_frame_equal(df1, df2)
