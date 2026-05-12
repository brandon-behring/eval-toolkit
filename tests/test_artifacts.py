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
    validate_payload,
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
