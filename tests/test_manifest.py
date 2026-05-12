"""Smoke tests for the v0.7.0 RunManifest dataclass + build/write helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval_toolkit.artifacts import PredictionArtifactRef, PredictionColumns
from eval_toolkit.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    SourceRoleRecord,
    build_manifest,
    validate_source_roles,
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
def test_build_manifest_captures_source_roles_and_guardrails() -> None:
    m = build_manifest(
        run_id="demo",
        config={},
        source_roles=[
            SourceRoleRecord(source="train_pool", role="train", n_rows=10),
            {"source": "diagnostic", "role": "external_diagnostic", "notes": "held out"},
        ],
        required_source_roles=("train", "external_diagnostic"),
        guardrails=["do not tune on diagnostics"],
    )
    assert m.source_roles[0]["source"] == "train_pool"
    assert m.source_roles[1]["role"] == "external_diagnostic"
    assert m.guardrails == ["do not tune on diagnostics"]


@pytest.mark.unit
def test_build_manifest_captures_prediction_artifact_refs() -> None:
    ref = PredictionArtifactRef(
        uri="predictions.csv",
        media_type="text/csv",
        n_rows=2,
        columns=PredictionColumns(row_id="id", label="label", score="score"),
    )

    m = build_manifest(run_id="demo", config={}, prediction_artifacts=[ref])

    assert m.prediction_artifacts[0]["uri"] == "predictions.csv"
    assert m.prediction_artifacts[0]["columns"]["score"] == "score"  # type: ignore[index]


@pytest.mark.unit
def test_validate_source_roles_reports_missing_required_role() -> None:
    errors = validate_source_roles(
        [SourceRoleRecord(source="train_pool", role="train")],
        required_roles=("train", "locked_final_holdout"),
    )
    assert "missing required source role" in errors[-1]


@pytest.mark.unit
def test_build_manifest_rejects_invalid_source_roles() -> None:
    with pytest.raises(ValueError, match="invalid source_roles"):
        build_manifest(
            run_id="demo",
            config={},
            source_roles=[{"source": "", "role": "train"}],
        )


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
