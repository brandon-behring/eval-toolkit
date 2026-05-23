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
    make_manifest,
    validate_source_roles,
    write_manifest,
)


@pytest.mark.unit
def test_make_manifest_defaults() -> None:
    m = make_manifest(run_id="demo", config={"k": 5})
    assert isinstance(m, RunManifest)
    assert m.run_id == "demo"
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.config_hash.startswith("sha256:")


@pytest.mark.unit
def test_make_manifest_auto_populates_captured_at() -> None:
    """v2 — captured_at is auto-populated as ISO-8601 UTC at build time."""
    m = make_manifest(run_id="demo", config={"k": 5})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m.captured_at), m.captured_at


@pytest.mark.unit
def test_make_manifest_captures_data_revisions_and_metadata() -> None:
    """v2 — caller-supplied data_revisions and metadata land on RunManifest."""
    m = make_manifest(
        run_id="demo",
        config={},
        data_revisions={"hf_dataset:foo": "abc123", "hf_model:bar": "deadbeef"},
        metadata={"meta:cli_args": '["--profile", "fixtures"]'},
    )
    assert m.data_revisions == {"hf_dataset:foo": "abc123", "hf_model:bar": "deadbeef"}
    assert m.metadata == {"meta:cli_args": '["--profile", "fixtures"]'}


@pytest.mark.unit
def test_make_manifest_data_revisions_metadata_default_empty() -> None:
    m = make_manifest(run_id="demo", config={})
    assert m.data_revisions == {}
    assert m.metadata == {}


@pytest.mark.unit
def test_manifest_schema_version_default() -> None:
    """v0.23.0 bumps the default schema_version from v2 to v3 (contamination_flags).

    v0.14.0 was the v1 → v2 bump (captured_at + data_revisions + metadata).
    v0.23.0 is the v2 → v3 bump (contamination_flags).
    """
    assert MANIFEST_SCHEMA_VERSION == "v3"
    m = make_manifest(run_id="demo", config={})
    assert m.schema_version == "v3"
    # v3 default: contamination_flags is present but empty when not supplied.
    assert m.contamination_flags == {}


@pytest.mark.unit
def test_manifest_contamination_flags_accepts_valid_values() -> None:
    """v0.23.0 — contamination_flags accepts the 4 enum values."""
    m = make_manifest(
        run_id="demo",
        config={},
        contamination_flags={
            "scorer_a": "verified_disjoint",
            "scorer_b": "suspected_contamination",
            "scorer_c": "vendor_black_box",
            "scorer_d": "unknown",
        },
    )
    assert m.contamination_flags == {
        "scorer_a": "verified_disjoint",
        "scorer_b": "suspected_contamination",
        "scorer_c": "vendor_black_box",
        "scorer_d": "unknown",
    }


@pytest.mark.unit
def test_manifest_contamination_flags_rejects_invalid_values() -> None:
    """v0.23.0 — invalid enum values raise at build time."""
    with pytest.raises(ValueError, match="invalid contamination_flags"):
        make_manifest(
            run_id="demo",
            config={},
            contamination_flags={"scorer_x": "totally_clean"},
        )


@pytest.mark.unit
def test_manifest_guardrails_accepts_strings_and_objects() -> None:
    """v0.23.0 — guardrails permits non-empty strings or non-empty dicts."""
    m = make_manifest(
        run_id="demo",
        config={},
        guardrails=[
            "no-leakage",
            {"source_freshness_check": {"timestamp": "2026-05-14T00:00:00Z"}},
        ],
    )
    assert len(m.guardrails) == 2
    assert m.guardrails[0] == "no-leakage"
    assert isinstance(m.guardrails[1], dict)


@pytest.mark.unit
def test_manifest_guardrails_rejects_empty_entries() -> None:
    """v0.23.0 — empty strings AND empty dicts both fail validation."""
    with pytest.raises(ValueError, match="guardrails must be non-empty"):
        make_manifest(run_id="demo", config={}, guardrails=["valid", ""])
    with pytest.raises(ValueError, match="guardrails must be non-empty"):
        make_manifest(run_id="demo", config={}, guardrails=[{}])


@pytest.mark.unit
def test_make_manifest_accepts_explicit_git_sha() -> None:
    """v0.14.1 — explicit git_sha kwarg overrides capture_git_sha.

    Use case: pods / CI runners that rsync the source tree without
    ``.git/`` and capture the SHA out-of-band via an environment variable.
    """
    explicit = "deadbeef" * 5
    m = make_manifest(run_id="demo", config={}, git_sha=explicit)
    assert m.git_sha == explicit


@pytest.mark.unit
def test_make_manifest_falls_back_to_capture_when_git_sha_none() -> None:
    """Default behavior unchanged: git_sha=None invokes capture_git_sha."""
    m = make_manifest(run_id="demo", config={})
    # capture_git_sha returns either a 40-char SHA or None depending on
    # the test environment; we only assert the override didn't activate.
    assert m.git_sha is None or len(m.git_sha) >= 7


@pytest.mark.unit
def test_make_manifest_captures_env() -> None:
    m = make_manifest(run_id="demo", config={})
    assert "python" in m.env
    assert "platform" in m.env
    assert "eval_toolkit" in m.env
    assert "eval_toolkit" in m.code_versions


@pytest.mark.unit
def test_make_manifest_hashes_data_files(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    p.write_bytes(b"some bytes")
    m = make_manifest(run_id="demo", config={}, data_files={"data": p})
    assert "data" in m.data_hashes
    assert m.data_hashes["data"].startswith("sha256:")


@pytest.mark.unit
def test_make_manifest_config_path_hashes_file_bytes(tmp_path: Path) -> None:
    """v0.34.0 (#10): when config_path supplied, config_hash captures file bytes.

    The file-bytes hash differs from the canonical-JSON hash because YAML
    comments + key ordering + whitespace are stripped during parse.
    """
    import hashlib

    config_file = tmp_path / "config.yaml"
    config_file.write_bytes(b"# top-line comment\nk: 5\nn: 10\n")
    m = make_manifest(
        run_id="demo",
        config={"k": 5, "n": 10},  # parsed equivalent
        config_path=config_file,
    )
    expected_hex = hashlib.sha256(config_file.read_bytes()).hexdigest()
    assert m.config_hash == f"sha256:{expected_hex}"


@pytest.mark.unit
def test_make_manifest_default_path_preserves_canonical_json_hash(tmp_path: Path) -> None:
    """Without config_path, config_hash remains the canonical-JSON hash (existing behavior)."""
    m_path = make_manifest(
        run_id="demo",
        config={"k": 5, "n": 10},
        config_path=None,  # explicit None = existing behavior
    )
    m_default = make_manifest(run_id="demo", config={"k": 5, "n": 10})
    # Both should match (no config_path → canonical-JSON path on both)
    assert m_path.config_hash == m_default.config_hash


@pytest.mark.unit
def test_make_manifest_config_path_vs_canonical_diverge(tmp_path: Path) -> None:
    """File-bytes hash and canonical-JSON hash differ on the same logical config.

    Proves that config_path captures information (whitespace, comments,
    YAML formatting) that the canonical-JSON path strips.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_bytes(b"# comment\nk: 5\n")
    m_filepath = make_manifest(run_id="demo", config={"k": 5}, config_path=config_file)
    m_canonical = make_manifest(run_id="demo", config={"k": 5})
    assert m_filepath.config_hash != m_canonical.config_hash, (
        "file-bytes hash and canonical-JSON hash should differ on the same logical config "
        "(file includes comment + trailing newline)"
    )


@pytest.mark.unit
def test_make_manifest_collects_versioned_objects() -> None:
    class WithVersion:
        version = "1.2.3"

    class WithoutVersion:
        pass

    m = make_manifest(
        run_id="demo",
        config={},
        versioned={"my_scorer": WithVersion(), "no_v": WithoutVersion()},
    )
    assert m.versioned_objects == {"my_scorer": "1.2.3"}


@pytest.mark.unit
def test_make_manifest_captures_source_roles_and_guardrails() -> None:
    m = make_manifest(
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
def test_make_manifest_captures_prediction_artifact_refs() -> None:
    ref = PredictionArtifactRef(
        uri="predictions.csv",
        media_type="text/csv",
        n_rows=2,
        columns=PredictionColumns(row_id="id", label="label", score="score"),
    )

    m = make_manifest(run_id="demo", config={}, prediction_artifacts=[ref])

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
def test_make_manifest_rejects_invalid_source_roles() -> None:
    with pytest.raises(ValueError, match="invalid source_roles"):
        make_manifest(
            run_id="demo",
            config={},
            source_roles=[{"source": "", "role": "train"}],
        )


@pytest.mark.unit
def test_write_manifest_roundtrip() -> None:
    m = make_manifest(run_id="demo", config={"k": 5})
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
    m1 = make_manifest(run_id="r1", config={"a": 1, "b": 2})
    m2 = make_manifest(run_id="r2", config={"b": 2, "a": 1})  # key order swapped
    m3 = make_manifest(run_id="r3", config={"a": 1, "b": 3})
    assert m1.config_hash == m2.config_hash
    assert m1.config_hash != m3.config_hash
