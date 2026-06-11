"""Tests for eval_toolkit.analysis — prediction-artifact readers and bootstrap wrappers.

Complements the smoke-level coverage from `test_v09_contracts.py` with
targeted edge cases for each public helper and validation path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.analysis import (
    CsvPredictionReader,
    JsonlPredictionReader,
    PredictionArrays,
    bootstrap_metric_from_predictions,
    load_prediction_arrays,
    paired_diff_from_prediction_refs,
)


def _csv_ref(path: Path, *, media_type: str = "text/csv") -> dict[str, object]:
    return {
        "uri": str(path),
        "media_type": media_type,
        "columns": {
            "label": "label",
            "score": "score",
            "row_id": "row_id",
            "content_hash": "content_hash",
        },
    }


def _jsonl_ref(path: Path, *, media_type: str = "application/jsonl") -> dict[str, object]:
    return {
        "uri": str(path),
        "media_type": media_type,
        "columns": {
            "label": "label",
            "score": "score",
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.mark.unit
def test_prediction_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        PredictionArrays(labels=np.array([0, 1]), scores=np.array([0.1, 0.5, 0.9]))


@pytest.mark.unit
def test_jsonl_reader_round_trip(tmp_path: Path) -> None:
    rows = [
        {"label": 0, "score": 0.1, "extra": "ignored"},
        {"label": 1, "score": 0.8},
    ]
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, rows)
    reader = JsonlPredictionReader()
    table = reader.read_predictions(str(path), columns={"label": "label", "score": "score"})
    assert list(table["label"]) == [0, 1]
    assert list(table["score"]) == [0.1, 0.8]


@pytest.mark.unit
def test_jsonl_reader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"label": 0, "score": 0.2})
        + "\n"
        + "\n"  # blank line
        + json.dumps({"label": 1, "score": 0.7})
        + "\n"
    )
    reader = JsonlPredictionReader()
    table = reader.read_predictions(str(path), columns={"label": "label", "score": "score"})
    assert len(table["label"]) == 2


@pytest.mark.unit
def test_load_prediction_arrays_routes_to_csv_via_media_type(tmp_path: Path) -> None:
    rows = [
        {"label": 0, "score": 0.1, "row_id": "r0", "content_hash": "h0"},
        {"label": 1, "score": 0.8, "row_id": "r1", "content_hash": "h1"},
    ]
    path = tmp_path / "preds.csv"
    _write_csv(path, rows)
    arrays = load_prediction_arrays(_csv_ref(path))
    assert arrays.labels.tolist() == [0, 1]
    assert arrays.scores.tolist() == [0.1, 0.8]
    assert arrays.row_ids == ("r0", "r1")
    assert arrays.content_hashes == ("h0", "h1")


@pytest.mark.unit
def test_load_prediction_arrays_routes_to_jsonl_via_extension(tmp_path: Path) -> None:
    """A .jsonl extension picks the JsonlPredictionReader when media_type is generic."""
    rows = [{"label": 0, "score": 0.3}, {"label": 1, "score": 0.6}]
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, rows)
    ref = {"uri": str(path), "media_type": "", "columns": {"label": "label", "score": "score"}}
    arrays = load_prediction_arrays(ref)
    assert arrays.labels.tolist() == [0, 1]


@pytest.mark.unit
def test_load_prediction_arrays_rejects_missing_columns_mapping(tmp_path: Path) -> None:
    path = tmp_path / "preds.csv"
    path.write_text("label,score\n0,0.1\n")
    bad_ref = {"uri": str(path), "media_type": "text/csv"}  # no columns
    with pytest.raises(ValueError, match="columns mapping"):
        load_prediction_arrays(bad_ref)


@pytest.mark.unit
def test_load_prediction_arrays_rejects_missing_uri() -> None:
    bad_ref = {
        "media_type": "text/csv",
        "columns": {"label": "label", "score": "score"},
    }
    with pytest.raises(ValueError, match="non-empty uri"):
        load_prediction_arrays(bad_ref)


@pytest.mark.unit
def test_load_prediction_arrays_rejects_missing_label_column(tmp_path: Path) -> None:
    path = tmp_path / "preds.csv"
    path.write_text("score\n0.1\n")
    bad_ref = {
        "uri": str(path),
        "media_type": "text/csv",
        "columns": {"score": "score"},  # no label
    }
    with pytest.raises(ValueError, match="label"):
        load_prediction_arrays(bad_ref)


@pytest.mark.unit
def test_reader_routing_rejects_unsupported_media_type(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    path.write_bytes(b"\x00\x01")  # nonsense bytes; we never actually read
    ref = {
        "uri": str(path),
        "media_type": "application/parquet",
        "columns": {"label": "label", "score": "score"},
    }
    with pytest.raises(ValueError, match="no built-in prediction reader"):
        load_prediction_arrays(ref)


@pytest.mark.unit
def test_bootstrap_metric_from_predictions_returns_ci(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 60
    labels = np.array([0, 1] * (n // 2), dtype=int)
    scores = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.15, size=n), 0, 1)
    path = tmp_path / "preds.csv"
    rows = [
        {"label": int(lab), "score": float(s), "row_id": f"r{i}", "content_hash": f"h{i}"}
        for i, (lab, s) in enumerate(zip(labels, scores, strict=True))
    ]
    _write_csv(path, rows)
    out = bootstrap_metric_from_predictions(_csv_ref(path), n_resamples=20, rng=7)
    assert out["n_resamples"] == 20
    # v0.48 §5B: schema rewritten from {ci_95: [l, h]} to {low: l, high: h}
    assert out["low"] <= out["point"] <= out["high"]


@pytest.mark.unit
def test_paired_diff_rejects_shape_mismatch(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_csv(p1, [{"label": 0, "score": 0.1, "row_id": "r0", "content_hash": "h0"}])
    _write_csv(
        p2,
        [
            {"label": 0, "score": 0.1, "row_id": "r0", "content_hash": "h0"},
            {"label": 1, "score": 0.8, "row_id": "r1", "content_hash": "h1"},
        ],
    )
    with pytest.raises(ValueError, match="same number of rows"):
        paired_diff_from_prediction_refs(_csv_ref(p1), _csv_ref(p2), n_resamples=5, rng=1)


@pytest.mark.unit
def test_paired_diff_rejects_label_mismatch(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_csv(p1, [{"label": 0, "score": 0.1, "row_id": "r0", "content_hash": "h0"}])
    _write_csv(p2, [{"label": 1, "score": 0.1, "row_id": "r0", "content_hash": "h0"}])
    with pytest.raises(ValueError, match="identical labels"):
        paired_diff_from_prediction_refs(_csv_ref(p1), _csv_ref(p2), n_resamples=5, rng=1)


@pytest.mark.unit
def test_paired_diff_rejects_row_id_mismatch(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_csv(p1, [{"label": 0, "score": 0.1, "row_id": "r0", "content_hash": "h0"}])
    _write_csv(p2, [{"label": 0, "score": 0.2, "row_id": "rZ", "content_hash": "h0"}])
    with pytest.raises(ValueError, match="identical row_ids"):
        paired_diff_from_prediction_refs(_csv_ref(p1), _csv_ref(p2), n_resamples=5, rng=1)


@pytest.mark.unit
def test_paired_diff_rejects_content_hash_mismatch(tmp_path: Path) -> None:
    """row_ids match (both empty), but content hashes differ → reject."""
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_csv(p1, [{"label": 0, "score": 0.1, "content_hash": "h0"}])
    _write_csv(p2, [{"label": 0, "score": 0.2, "content_hash": "hZ"}])
    ref1 = {
        "uri": str(p1),
        "media_type": "text/csv",
        "columns": {"label": "label", "score": "score", "content_hash": "content_hash"},
    }
    ref2 = {
        "uri": str(p2),
        "media_type": "text/csv",
        "columns": {"label": "label", "score": "score", "content_hash": "content_hash"},
    }
    with pytest.raises(ValueError, match="identical content_hashes"):
        paired_diff_from_prediction_refs(ref1, ref2, n_resamples=5, rng=1)


@pytest.mark.unit
def test_csv_reader_used_directly(tmp_path: Path) -> None:
    """Exercise CsvPredictionReader's public read_predictions API directly."""
    rows = [{"label": 0, "score": 0.4}, {"label": 1, "score": 0.7}]
    path = tmp_path / "p.csv"
    _write_csv(path, rows)
    reader = CsvPredictionReader()
    table = reader.read_predictions(str(path), columns={"label": "label", "score": "score"})
    assert table["label"] == ["0", "1"]
    assert table["score"] == ["0.4", "0.7"]


@pytest.mark.unit
def test_csv_reader_raises_actionable_error_on_missing_column(tmp_path: Path) -> None:
    """R8-F3 regression: missing CSV column → actionable ValueError, not cryptic dtype error.

    Pre-v0.51 the reader filled missing columns with empty strings,
    causing the downstream ``np.asarray(..., dtype=int)`` in
    load_prediction_arrays to fail with
    "invalid literal for int() with base 10: ''". Root cause was
    obscured. v0.51 detects missing columns at the read boundary and
    raises with the file path + missing column name.
    """
    # CSV has only "label" column, but caller requests "score" too.
    rows = [{"label": 0}, {"label": 1}]
    path = tmp_path / "p.csv"
    _write_csv(path, rows)
    reader = CsvPredictionReader()
    with pytest.raises(ValueError, match=r"missing required column"):
        reader.read_predictions(str(path), columns={"label": "label", "score": "score"})


@pytest.mark.unit
def test_jsonl_reader_missing_key_raises(tmp_path: Path) -> None:
    """#96: a row missing a declared key fails fast with uri + row context (R8-F3 pattern).

    Pre-#96 the missing score loaded as None, coerced to NaN inside
    ``np.asarray(dtype=float)``, and surfaced (if at all) deep in the metric
    computation with no file/row context.
    """
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, [{"label": 0, "score": 0.2}, {"label": 1}])
    reader = JsonlPredictionReader()
    with pytest.raises(ValueError, match=r"row 2 is missing required key\(s\) \['score'\]"):
        reader.read_predictions(str(path), columns={"label": "label", "score": "score"})


# --- per-row CSV validation + reader parity on sparse identity columns (#98) ---


@pytest.mark.unit
def test_csv_reader_empty_cell_raises(tmp_path: Path) -> None:
    """#98: an empty cell in a declared column fails fast with file + row context."""
    path = tmp_path / "p.csv"
    path.write_text("label,score\n0,0.4\n1,\n", encoding="utf-8")
    reader = CsvPredictionReader()
    with pytest.raises(ValueError, match=r"row 3 is missing value\(s\).*\['score'\]"):
        reader.read_predictions(str(path), columns={"label": "label", "score": "score"})


@pytest.mark.unit
def test_csv_reader_sparse_row_id_raises(tmp_path: Path) -> None:
    """#98: a declared-but-sparse identity column raises instead of loading ''.

    Pre-#98 the empty cell loaded a ``""`` placeholder that silently weakened
    paired-diff row alignment (two rows aligning on ``""``).
    """
    path = tmp_path / "p.csv"
    path.write_text("label,score,row_id\n0,0.4,r0\n1,0.7,\n", encoding="utf-8")
    reader = CsvPredictionReader()
    with pytest.raises(ValueError, match=r"row 3.*\['row_id'\]"):
        reader.read_predictions(
            str(path), columns={"label": "label", "score": "score", "row_id": "row_id"}
        )


@pytest.mark.unit
def test_csv_reader_short_row_raises(tmp_path: Path) -> None:
    """A physically short row (DictReader fills None) raises the same error."""
    path = tmp_path / "p.csv"
    path.write_text("label,score,content_hash\n0,0.4\n", encoding="utf-8")
    reader = CsvPredictionReader()
    with pytest.raises(ValueError, match=r"row 2.*\['content_hash'\]"):
        reader.read_predictions(
            str(path),
            columns={"label": "label", "score": "score", "content_hash": "content_hash"},
        )


@pytest.mark.unit
def test_jsonl_reader_sparse_row_id_raises(tmp_path: Path) -> None:
    """#98 parity pin: JSONL raises on a declared-but-sparse identity column."""
    path = tmp_path / "p.jsonl"
    _write_jsonl(path, [{"label": 0, "score": 0.2, "row_id": "r0"}, {"label": 1, "score": 0.9}])
    reader = JsonlPredictionReader()
    with pytest.raises(ValueError, match=r"row 2 is missing required key\(s\) \['row_id'\]"):
        reader.read_predictions(
            str(path), columns={"label": "label", "score": "score", "row_id": "row_id"}
        )


@pytest.mark.unit
def test_jsonl_reader_null_content_hash_raises(tmp_path: Path) -> None:
    """#98 parity pin: an explicit ``null`` identity value counts as missing."""
    path = tmp_path / "p.jsonl"
    _write_jsonl(path, [{"label": 0, "score": 0.2, "content_hash": None}])
    reader = JsonlPredictionReader()
    with pytest.raises(ValueError, match=r"row 1.*\['content_hash'\]"):
        reader.read_predictions(
            str(path),
            columns={"label": "label", "score": "score", "content_hash": "content_hash"},
        )


@pytest.mark.unit
def test_jsonl_reader_null_value_raises(tmp_path: Path) -> None:
    """A JSON null for a declared key is treated the same as a missing key."""
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, [{"label": None, "score": 0.2}])
    reader = JsonlPredictionReader()
    with pytest.raises(ValueError, match=r"row 1 is missing required key\(s\) \['label'\]"):
        reader.read_predictions(str(path), columns={"label": "label", "score": "score"})


@pytest.mark.unit
def test_jsonl_reader_invalid_json_row_raises(tmp_path: Path) -> None:
    """#96: a malformed row reports the actual file row (json.loads always says 'line 1')."""
    path = tmp_path / "preds.jsonl"
    path.write_text(json.dumps({"label": 0, "score": 0.2}) + "\nnot json\n")
    reader = JsonlPredictionReader()
    with pytest.raises(ValueError, match="row 2 is not valid JSON"):
        reader.read_predictions(str(path), columns={"label": "label", "score": "score"})


@pytest.mark.unit
def test_load_prediction_arrays_rejects_non_finite_scores(tmp_path: Path) -> None:
    """#96: a bare JSON NaN token passes the per-row key check but must not reach metrics."""
    path = tmp_path / "preds.jsonl"
    path.write_text('{"label": 0, "score": NaN}\n{"label": 1, "score": 0.9}\n')
    with pytest.raises(ValueError, match="non-finite score"):
        load_prediction_arrays(_jsonl_ref(path))


@pytest.mark.unit
def test_jsonl_reader_utf8_content(tmp_path: Path) -> None:
    """#97: prediction artifacts with non-ASCII text columns read as UTF-8
    regardless of the platform locale codec (Windows cp1252)."""
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"label": 1, "score": 0.9, "row_id": "Δ§é-1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    reader = JsonlPredictionReader()
    table = reader.read_predictions(
        str(path), columns={"label": "label", "score": "score", "row_id": "row_id"}
    )
    assert list(table["row_id"]) == ["Δ§é-1"]


@pytest.mark.unit
def test_load_prediction_arrays_rejects_non_binary_labels(tmp_path: Path) -> None:
    """v1.9.0 pre-tag review (S4): float labels must not silently truncate (0.7 → 0)."""
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, [{"label": 0.7, "score": 0.2}, {"label": 1, "score": 0.9}])
    with pytest.raises(ValueError, match="non-binary label"):
        load_prediction_arrays(_jsonl_ref(path))
