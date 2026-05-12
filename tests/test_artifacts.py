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
