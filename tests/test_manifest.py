"""Smoke tests for the v0.7.0 RunManifest dataclass + build/write helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval_toolkit.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    build_manifest,
    write_manifest,
)


@pytest.mark.unit
def test_build_manifest_defaults() -> None:
    m = build_manifest(run_id="demo", config={"k": 5})
    assert isinstance(m, RunManifest)
    assert m.run_id == "demo"
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.config_hash.startswith("sha256:")


@pytest.mark.unit
def test_build_manifest_captures_env() -> None:
    m = build_manifest(run_id="demo", config={})
    assert "python" in m.env
    assert "platform" in m.env
    assert "eval_toolkit" in m.env
    assert "eval_toolkit" in m.code_versions


@pytest.mark.unit
def test_build_manifest_hashes_data_files(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    p.write_bytes(b"some bytes")
    m = build_manifest(run_id="demo", config={}, data_files={"data": p})
    assert "data" in m.data_hashes
    assert m.data_hashes["data"].startswith("sha256:")


@pytest.mark.unit
def test_build_manifest_collects_versioned_objects() -> None:
    class WithVersion:
        version = "1.2.3"

    class WithoutVersion:
        pass

    m = build_manifest(
        run_id="demo",
        config={},
        versioned={"my_scorer": WithVersion(), "no_v": WithoutVersion()},
    )
    assert m.versioned_objects == {"my_scorer": "1.2.3"}


@pytest.mark.unit
def test_write_manifest_roundtrip() -> None:
    m = build_manifest(run_id="demo", config={"k": 5})
    with tempfile.TemporaryDirectory() as d:
        path = write_manifest(m, d)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == "demo"
        assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION


@pytest.mark.unit
def test_config_hash_is_deterministic() -> None:
    """Same config -> same hash; different config -> different hash."""
    m1 = build_manifest(run_id="r1", config={"a": 1, "b": 2})
    m2 = build_manifest(run_id="r2", config={"b": 2, "a": 1})  # key order swapped
    m3 = build_manifest(run_id="r3", config={"a": 1, "b": 3})
    assert m1.config_hash == m2.config_hash
    assert m1.config_hash != m3.config_hash
