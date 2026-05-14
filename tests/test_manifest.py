"""Smoke tests for the RunManifest dataclass + build/write helpers (v2)."""

from __future__ import annotations

import json
import re
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
def test_build_manifest_auto_populates_captured_at() -> None:
    """v2 — captured_at is auto-populated as ISO-8601 UTC at build time."""
    m = build_manifest(run_id="demo", config={"k": 5})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m.captured_at), m.captured_at


@pytest.mark.unit
def test_build_manifest_captures_data_revisions_and_metadata() -> None:
    """v2 — caller-supplied data_revisions and metadata land on RunManifest."""
    m = build_manifest(
        run_id="demo",
        config={},
        data_revisions={"hf_dataset:foo": "abc123", "hf_model:bar": "deadbeef"},
        metadata={"meta:cli_args": '["--profile", "fixtures"]'},
    )
    assert m.data_revisions == {"hf_dataset:foo": "abc123", "hf_model:bar": "deadbeef"}
    assert m.metadata == {"meta:cli_args": '["--profile", "fixtures"]'}


@pytest.mark.unit
def test_build_manifest_data_revisions_metadata_default_empty() -> None:
    m = build_manifest(run_id="demo", config={})
    assert m.data_revisions == {}
    assert m.metadata == {}


@pytest.mark.unit
def test_manifest_schema_version_is_v2() -> None:
    """v0.14.0 bumps the default schema_version from v1 to v2."""
    assert MANIFEST_SCHEMA_VERSION == "v2"
    m = build_manifest(run_id="demo", config={})
    assert m.schema_version == "v2"


@pytest.mark.unit
def test_build_manifest_accepts_explicit_git_sha() -> None:
    """v0.14.1 — explicit git_sha kwarg overrides capture_git_sha.

    Use case: pods / CI runners that rsync the source tree without
    ``.git/`` and capture the SHA out-of-band via an environment variable.
    """
    explicit = "deadbeef" * 5
    m = build_manifest(run_id="demo", config={}, git_sha=explicit)
    assert m.git_sha == explicit


@pytest.mark.unit
def test_build_manifest_falls_back_to_capture_when_git_sha_none() -> None:
    """Default behavior unchanged: git_sha=None invokes capture_git_sha."""
    m = build_manifest(run_id="demo", config={})
    # capture_git_sha returns either a 40-char SHA or None depending on
    # the test environment; we only assert the override didn't activate.
    assert m.git_sha is None or len(m.git_sha) >= 7


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
        # v2 — captured_at survives the JSON round-trip
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", loaded["captured_at"])
        # v2 — data_revisions / metadata are present (empty when not passed)
        assert loaded["data_revisions"] == {}
        assert loaded["metadata"] == {}


@pytest.mark.unit
def test_config_hash_is_deterministic() -> None:
    """Same config -> same hash; different config -> different hash."""
    m1 = build_manifest(run_id="r1", config={"a": 1, "b": 2})
    m2 = build_manifest(run_id="r2", config={"b": 2, "a": 1})  # key order swapped
    m3 = build_manifest(run_id="r3", config={"a": 1, "b": 3})
    assert m1.config_hash == m2.config_hash
    assert m1.config_hash != m3.config_hash
