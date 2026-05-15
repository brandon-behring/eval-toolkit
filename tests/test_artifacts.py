"""Tests for strict artifact helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval_toolkit.artifacts import (
    PredictionArtifactRef,
    PredictionColumns,
    error_metric,
    sanitize_for_json,
    skipped_metric,
    validate_manifest,
    validate_payload,
    validate_prediction_artifact_ref,
    validate_results,
    write_json_strict,
)


@pytest.mark.unit
def test_sanitize_for_json_replaces_non_finite_numbers() -> None:
    payload = {
        "ok": 1.25,
        "bad": float("nan"),
        "arr": np.array([0.0, np.inf, -np.inf]),
    }

    sanitized = sanitize_for_json(payload)

    assert isinstance(sanitized, dict)
    assert sanitized["ok"] == 1.25
    assert sanitized["bad"]["status"] == "skipped"  # type: ignore[index]
    assert sanitized["arr"][1]["status"] == "skipped"  # type: ignore[index]


# --- v0.13: nan_strategy on sanitize_for_json ---


@pytest.mark.unit
def test_sanitize_for_json_nan_strategy_null_replaces_with_json_null() -> None:
    """nan_strategy='null' returns None for non-finite values (vs default skipped dict)."""
    payload = {"metric_a": float("nan"), "metric_b": float("inf"), "metric_c": 0.5}
    out = sanitize_for_json(payload, nan_strategy="null")
    assert isinstance(out, dict)
    assert out["metric_a"] is None
    assert out["metric_b"] is None
    assert out["metric_c"] == 0.5


@pytest.mark.unit
def test_sanitize_for_json_nan_strategy_raise_surfaces_silent_bugs() -> None:
    """nan_strategy='raise' raises ValueError on first non-finite value."""
    with pytest.raises(ValueError, match="non-finite"):
        sanitize_for_json(float("nan"), nan_strategy="raise")
    with pytest.raises(ValueError, match="non-finite"):
        sanitize_for_json({"nested": [1.0, float("inf"), 3.0]}, nan_strategy="raise")


@pytest.mark.unit
def test_sanitize_for_json_nan_strategy_validates_unknown() -> None:
    with pytest.raises(ValueError, match="nan_strategy"):
        sanitize_for_json(float("nan"), nan_strategy="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_sanitize_for_json_nan_strategy_propagates_through_recursion() -> None:
    """nan_strategy is threaded through nested mappings, sequences, np arrays."""
    nested = {
        "list": [1.0, float("nan"), 3.0],
        "dict": {"sub": float("inf")},
        "array": np.array([0.0, np.nan, 2.0]),
    }
    out = sanitize_for_json(nested, nan_strategy="null")
    assert isinstance(out, dict)
    assert out["list"][1] is None  # type: ignore[index]
    assert out["dict"]["sub"] is None  # type: ignore[index]
    assert out["array"][1] is None  # type: ignore[index]


@pytest.mark.unit
def test_write_json_strict_never_emits_nan_or_infinity(tmp_path) -> None:
    path = write_json_strict(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "skipped": skipped_metric("not enough data"),
            "error": error_metric("bootstrap failed", exc_type="ValueError"),
        },
        tmp_path / "artifact.json",
    )

    raw = path.read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    loaded = json.loads(raw)
    assert loaded["nan"]["status"] == "skipped"
    assert loaded["skipped"] == {"value": None, "status": "skipped", "reason": "not enough data"}
    assert loaded["error"]["details"]["exc_type"] == "ValueError"


@pytest.mark.unit
def test_prediction_artifact_ref_serializes_column_mapping() -> None:
    ref = PredictionArtifactRef(
        uri="predictions.csv",
        media_type="text/csv",
        sha256="sha256:abc",
        n_rows=3,
        columns=PredictionColumns(
            row_id="id",
            content_hash="hash",
            label="label",
            score="score",
            scorer="scorer",
            slice="slice",
            provenance={"source": "source"},
        ),
    )

    out = ref.to_dict()

    assert out["uri"] == "predictions.csv"
    assert out["n_rows"] == 3
    assert out["columns"]["row_id"] == "id"  # type: ignore[index]
    assert out["columns"]["provenance"] == {"source": "source"}  # type: ignore[index]


@pytest.mark.unit
def test_validate_payload_accepts_strict_result_shape() -> None:
    payload = {
        "schema_version": "v1",
        "run_id": "r",
        "config": {},
        "by_slice": {},
        "aggregate_evidence": {"status": "diagnostic"},
    }

    validate_payload(payload, "results.v1.json")


@pytest.mark.unit
def test_prediction_columns_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label must be non-empty"):
        PredictionColumns(label="", score="score")


@pytest.mark.unit
def test_prediction_columns_rejects_empty_score() -> None:
    with pytest.raises(ValueError, match="score must be non-empty"):
        PredictionColumns(label="label", score="")


@pytest.mark.unit
def test_prediction_columns_to_dict_omits_absent_optional_columns() -> None:
    cols = PredictionColumns(label="label", score="score")
    out = cols.to_dict()
    assert out == {"label": "label", "score": "score"}
    assert "row_id" not in out
    assert "provenance" not in out


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_empty_uri() -> None:
    with pytest.raises(ValueError, match="uri must be non-empty"):
        PredictionArtifactRef(
            uri="",
            media_type="text/csv",
            columns=PredictionColumns(label="label", score="score"),
        )


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_empty_media_type() -> None:
    with pytest.raises(ValueError, match="media_type must be non-empty"):
        PredictionArtifactRef(
            uri="uri",
            media_type="",
            columns=PredictionColumns(label="label", score="score"),
        )


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_negative_n_rows() -> None:
    with pytest.raises(ValueError, match="n_rows must be non-negative"):
        PredictionArtifactRef(
            uri="uri",
            media_type="text/csv",
            n_rows=-1,
            columns=PredictionColumns(label="label", score="score"),
        )


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_bool_n_rows() -> None:
    """bool is a subclass of int — guard against True/False sneaking through."""
    with pytest.raises(ValueError, match="n_rows must be non-negative"):
        PredictionArtifactRef(
            uri="uri",
            media_type="text/csv",
            n_rows=True,  # type: ignore[arg-type]
            columns=PredictionColumns(label="label", score="score"),
        )


@pytest.mark.unit
def test_prediction_artifact_ref_with_mapping_columns_validates_label() -> None:
    with pytest.raises(ValueError, match="columns must include a label column"):
        PredictionArtifactRef(
            uri="u",
            media_type="text/csv",
            columns={"score": "score"},
        )


@pytest.mark.unit
def test_prediction_artifact_ref_with_mapping_columns_validates_score() -> None:
    with pytest.raises(ValueError, match="columns must include a score column"):
        PredictionArtifactRef(
            uri="u",
            media_type="text/csv",
            columns={"label": "label"},
        )


@pytest.mark.unit
def test_prediction_artifact_ref_with_mapping_columns_serializes() -> None:
    ref = PredictionArtifactRef(
        uri="u",
        media_type="text/csv",
        columns={"label": "y", "score": "p"},
    )
    out = ref.to_dict()
    assert out["columns"] == {"label": "y", "score": "p"}
    assert "n_rows" not in out
    assert "metadata" not in out
    assert "sha256" not in out


@pytest.mark.unit
def test_sanitize_for_json_handles_numpy_generic() -> None:
    """np.int64, np.float32, etc. are converted via .item()."""
    assert sanitize_for_json(np.int64(7)) == 7
    assert sanitize_for_json(np.float64(0.5)) == 0.5


@pytest.mark.unit
def test_sanitize_for_json_routes_to_dict_capable_object() -> None:
    """Anything with a callable to_dict() is normalized via that method."""

    class _Container:
        def to_dict(self) -> dict[str, object]:
            return {"k": 1.5}

    sanitized = sanitize_for_json(_Container())
    assert sanitized == {"k": 1.5}


@pytest.mark.unit
def test_sanitize_for_json_routes_plain_dataclass() -> None:
    """A dataclass instance without to_dict() is normalized via asdict()."""
    import dataclasses

    @dataclasses.dataclass
    class _D:
        x: int
        y: float

    sanitized = sanitize_for_json(_D(1, 2.5))
    assert sanitized == {"x": 1, "y": 2.5}


@pytest.mark.unit
def test_sanitize_for_json_fallback_stringifies_unknown_types() -> None:
    """Unrecognized objects (no Mapping/Sequence/dataclass/to_dict) become str()."""

    class _Opaque:
        def __str__(self) -> str:
            return "opaque-token"

    assert sanitize_for_json(_Opaque()) == "opaque-token"


# --- v0.14: typed validate_* helpers (F9.2) ---


@pytest.mark.unit
def test_validate_manifest_dispatches_on_schema_version_v2() -> None:
    """validate_manifest dispatches on payload['schema_version']."""
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v2",
        "run_id": "demo",
        "captured_at": "2026-05-13T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.14.0"},
        "env": {"python": "3.11"},
    }
    validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_dispatches_on_schema_version_v1() -> None:
    """validate_manifest still accepts legacy v1 payloads for V4.2-era reruns."""
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v1",
        "run_id": "legacy",
        "code_versions": {"eval_toolkit": "0.13.0"},
        "env": {"python": "3.11"},
    }
    validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_rejects_unknown_schema_version() -> None:
    """An unrecognized schema_version is an error, not a silent fall-through."""
    payload = {
        "schema_version": "v99",
        "run_id": "demo",
        "code_versions": {},
        "env": {},
    }
    with pytest.raises(ValueError, match="unknown manifest schema_version"):
        validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_defaults_to_current_when_field_missing() -> None:
    """If schema_version is absent, fall back to the current default (v2)."""
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v2",
        "run_id": "demo",
        "captured_at": "2026-05-13T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.14.0"},
        "env": {},
    }
    validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_v3_accepts_well_formed_payload() -> None:
    """v3 manifest validates cleanly when all v3-required fields are present.

    v3 (v0.23.0+) added the required ``contamination_flags`` object. The
    dispatcher must route ``schema_version: v3`` to the v3 schema, NOT
    fall back to v2 — otherwise the new required field would be silently
    unchecked.
    """
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v3",
        "run_id": "demo-v3",
        "captured_at": "2026-05-15T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.27.2"},
        "env": {"python": "3.13"},
        "contamination_flags": {
            "vendor_model": "vendor_black_box",
            "in_house_scorer": "verified_disjoint",
        },
    }
    validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_v3_rejects_missing_contamination_flags() -> None:
    """v3 manifest missing ``contamination_flags`` is rejected.

    This is the load-bearing distinction between v2 and v3: a v3 manifest
    without ``contamination_flags`` should fail validation rather than
    silently passing. Catches dispatcher misrouting (e.g., a v3 payload
    being validated against v2 schema by accident).
    """
    pytest.importorskip("jsonschema")
    jsonschema = pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v3",
        "run_id": "demo-v3-bad",
        "captured_at": "2026-05-15T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.27.2"},
        "env": {"python": "3.13"},
        # Missing the required `contamination_flags` field
    }
    with pytest.raises(jsonschema.ValidationError, match="contamination_flags"):
        validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_v3_rejects_unknown_contamination_flag_value() -> None:
    """v3 manifest rejects contamination-flag values outside the enum.

    The schema enumerates the allowed values; an unrecognized value
    (e.g., typo) must be caught. Catches contract drift where someone
    adds a new flag value to the producer without updating the schema.
    """
    pytest.importorskip("jsonschema")
    jsonschema = pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v3",
        "run_id": "demo-v3-bad-enum",
        "captured_at": "2026-05-15T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.27.2"},
        "env": {"python": "3.13"},
        "contamination_flags": {
            "vendor_model": "totally_made_up_value",
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(payload)


@pytest.mark.unit
def test_validate_manifest_v2_payload_validated_under_v2_schema_not_v3() -> None:
    """A v2 payload (no ``contamination_flags``) validates cleanly under v2.

    Asserts that the dispatcher does NOT eagerly route v2 payloads through
    v3 (which would reject them for missing ``contamination_flags``). Reads
    legacy v0.22-era manifests must keep working.
    """
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v2",
        "run_id": "legacy-v2",
        "captured_at": "2026-05-10T12:00:00Z",
        "code_versions": {"eval_toolkit": "0.22.0"},
        "env": {"python": "3.13"},
        # No contamination_flags — that's v3-only.
    }
    validate_manifest(payload)  # should NOT raise


@pytest.mark.unit
def test_validate_results_passes_for_well_formed_payload() -> None:
    """validate_results wraps validate_payload with the results.v1 schema name."""
    pytest.importorskip("jsonschema")
    payload = {
        "schema_version": "v1",
        "run_id": "demo",
        "config": {},
        "by_slice": {},
    }
    validate_results(payload)


@pytest.mark.unit
def test_validate_prediction_artifact_ref_passes_for_well_formed_payload() -> None:
    """validate_prediction_artifact_ref validates a single ref payload."""
    pytest.importorskip("jsonschema")
    payload = {
        "uri": "predictions.csv",
        "media_type": "text/csv",
        "columns": {"label": "label", "score": "score"},
    }
    validate_prediction_artifact_ref(payload)


@pytest.mark.unit
def test_validate_prediction_artifact_ref_rejects_missing_columns() -> None:
    """columns.label and columns.score are required."""
    pytest.importorskip("jsonschema")
    from jsonschema.exceptions import ValidationError

    payload = {
        "uri": "predictions.csv",
        "media_type": "text/csv",
        "columns": {"score": "score"},  # missing 'label'
    }
    with pytest.raises(ValidationError):
        validate_prediction_artifact_ref(payload)


# --- v0.15.0: PredictionArtifactRef.role accepts str | list[str] (F5.2) ---


@pytest.mark.unit
def test_prediction_artifact_ref_accepts_role_list() -> None:
    """v0.15.0 — role can be a list of slice / fold names."""
    ref = PredictionArtifactRef(
        uri="predictions.parquet",
        media_type="application/vnd.apache.parquet",
        role=["fold_test", "ood_tensortrust", "ood_indirect"],
        columns=PredictionColumns(label="y_true", score="y_score"),
    )
    out = ref.to_dict()
    assert out["role"] == ["fold_test", "ood_tensortrust", "ood_indirect"]


@pytest.mark.unit
def test_prediction_artifact_ref_accepts_role_string_back_compat() -> None:
    """Pre-v0.15 string-role callers still work; output stays string-shaped."""
    ref = PredictionArtifactRef(
        uri="predictions.parquet",
        media_type="application/vnd.apache.parquet",
        role="predictions",
        columns=PredictionColumns(label="y_true", score="y_score"),
    )
    out = ref.to_dict()
    assert out["role"] == "predictions"


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_empty_role_list() -> None:
    with pytest.raises(ValueError, match="role list must be non-empty"):
        PredictionArtifactRef(
            uri="predictions.parquet",
            media_type="application/vnd.apache.parquet",
            role=[],
            columns=PredictionColumns(label="y_true", score="y_score"),
        )


@pytest.mark.unit
def test_prediction_artifact_ref_rejects_role_list_with_empty_entries() -> None:
    with pytest.raises(ValueError, match="role list must be non-empty"):
        PredictionArtifactRef(
            uri="predictions.parquet",
            media_type="application/vnd.apache.parquet",
            role=["valid", ""],
            columns=PredictionColumns(label="y_true", score="y_score"),
        )


@pytest.mark.unit
def test_validate_prediction_artifact_ref_accepts_role_list() -> None:
    """v0.15.0 — inline schema accepts role as array of strings."""
    pytest.importorskip("jsonschema")
    payload = {
        "uri": "predictions.parquet",
        "media_type": "application/vnd.apache.parquet",
        "role": ["fold_test", "ood_tensortrust"],
        "columns": {"label": "y_true", "score": "y_score"},
    }
    validate_prediction_artifact_ref(payload)
