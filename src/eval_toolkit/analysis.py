"""Post-run analysis over retained prediction artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval_toolkit.bootstrap import bootstrap_ci, paired_bootstrap_diff
from eval_toolkit.metrics import pr_auc
from eval_toolkit.protocols import PredictionReader

__all__ = [
    "CsvPredictionReader",
    "JsonlPredictionReader",
    "PredictionArrays",
    "bootstrap_metric_from_predictions",
    "load_prediction_arrays",
    "paired_diff_from_prediction_refs",
]


@dataclass(frozen=True, slots=True)
class PredictionArrays:
    """Numeric arrays loaded from a prediction artifact."""

    labels: np.ndarray
    scores: np.ndarray
    row_ids: tuple[str, ...] = ()
    content_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate array shape."""
        if self.labels.shape != self.scores.shape:
            raise ValueError("labels and scores must have identical shape")


class CsvPredictionReader:
    """Read CSV prediction files into a column-oriented mapping."""

    def read_predictions(
        self,
        uri: str,
        *,
        columns: Mapping[str, str],
    ) -> Mapping[str, Sequence[object]]:
        """Read a local CSV file."""
        wanted = set(columns.values())
        out: dict[str, list[object]] = {col: [] for col in wanted}
        with Path(uri).open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for col in wanted:
                    out[col].append(row.get(col, ""))
        return out


class JsonlPredictionReader:
    """Read JSON Lines prediction files into a column-oriented mapping."""

    def read_predictions(
        self,
        uri: str,
        *,
        columns: Mapping[str, str],
    ) -> Mapping[str, Sequence[object]]:
        """Read a local JSONL file."""
        wanted = set(columns.values())
        out: dict[str, list[object]] = {col: [] for col in wanted}
        with Path(uri).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for col in wanted:
                    out[col].append(row.get(col))
        return out


def load_prediction_arrays(
    ref: Mapping[str, Any],
    *,
    reader: PredictionReader | None = None,
) -> PredictionArrays:
    """Load labels and scores from a prediction artifact reference."""
    columns = ref.get("columns")
    if not isinstance(columns, Mapping):
        raise ValueError("prediction ref must include a columns mapping")
    label_col = _required_column(columns, "label")
    score_col = _required_column(columns, "score")
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError("prediction ref must include a non-empty uri")
    selected_reader = reader or _reader_for_ref(ref)
    reader_columns = {str(k): str(v) for k, v in columns.items() if isinstance(v, str)}
    table = selected_reader.read_predictions(uri, columns=reader_columns)
    labels = np.asarray(table[label_col], dtype=int)
    scores = np.asarray(table[score_col], dtype=float)
    row_id_col = columns.get("row_id")
    hash_col = columns.get("content_hash")
    row_ids = tuple(str(v) for v in table.get(str(row_id_col), ())) if row_id_col else ()
    hashes = tuple(str(v) for v in table.get(str(hash_col), ())) if hash_col else ()
    return PredictionArrays(labels=labels, scores=scores, row_ids=row_ids, content_hashes=hashes)


def bootstrap_metric_from_predictions(
    ref: Mapping[str, Any],
    *,
    reader: PredictionReader | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Compute a PR-AUC bootstrap CI from one prediction ref."""
    arrays = load_prediction_arrays(ref, reader=reader)
    return bootstrap_ci(
        arrays.labels,
        arrays.scores,
        pr_auc,
        n_resamples=n_resamples,
        seed=seed,
    ).to_dict()


def paired_diff_from_prediction_refs(
    baseline_ref: Mapping[str, Any],
    candidate_ref: Mapping[str, Any],
    *,
    baseline_reader: PredictionReader | None = None,
    candidate_reader: PredictionReader | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Compute paired PR-AUC delta from two prediction refs."""
    baseline = load_prediction_arrays(baseline_ref, reader=baseline_reader)
    candidate = load_prediction_arrays(candidate_ref, reader=candidate_reader)
    if baseline.labels.shape != candidate.labels.shape:
        raise ValueError("prediction refs must have the same number of rows")
    if not np.array_equal(baseline.labels, candidate.labels):
        raise ValueError("prediction refs must have identical labels for paired comparison")
    if baseline.row_ids and candidate.row_ids and baseline.row_ids != candidate.row_ids:
        raise ValueError("prediction refs must have identical row_ids for paired comparison")
    if (
        baseline.content_hashes
        and candidate.content_hashes
        and baseline.content_hashes != candidate.content_hashes
    ):
        raise ValueError("prediction refs must have identical content_hashes for paired comparison")
    return paired_bootstrap_diff(
        baseline.labels,
        baseline.scores,
        candidate.scores,
        pr_auc,
        n_resamples=n_resamples,
        seed=seed,
    ).to_dict()


def _reader_for_ref(ref: Mapping[str, Any]) -> PredictionReader:
    media_type = str(ref.get("media_type", ""))
    uri = str(ref.get("uri", ""))
    if media_type in {"text/csv", "application/csv"} or uri.endswith(".csv"):
        return CsvPredictionReader()
    if media_type in {"application/jsonl", "application/x-ndjson"} or uri.endswith(".jsonl"):
        return JsonlPredictionReader()
    raise ValueError(
        "no built-in prediction reader for this artifact; pass a PredictionReader "
        "or use CSV/JSONL"
    )


def _required_column(columns: Mapping[str, object], key: str) -> str:
    value = columns.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"prediction columns must include {key!r}")
    return value
