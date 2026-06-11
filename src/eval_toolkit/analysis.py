"""Post-run analysis over retained prediction artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval_toolkit._rng import RNGLike, SeedLike
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
        """Read a local CSV file.

        Raises
        ------
        ValueError
            If the CSV file is missing any of the requested columns (R8-F3
            audit fix). Pre-v0.51 the reader silently filled missing
            columns with empty strings, deferring failure to a cryptic
            ``ValueError: invalid literal for int() with base 10: ''``
            during ``np.asarray(..., dtype=int)`` in
            :func:`load_prediction_arrays`. v0.51 detects the missing
            columns at read time and raises with the file path + column
            name so the root cause is obvious.

            Also raised, with the file path + row number, when any row has
            a missing or empty-string cell in a declared column (#98 — the
            per-row mirror of the JSONL reader's check). Pre-#98 a
            declared-but-sparse identity column such as ``row_id`` loaded
            ``""`` placeholders that silently weakened
            :func:`paired_diff_from_prediction_refs` row alignment, and a
            sparse ``score`` died downstream without row context.
        """
        wanted = set(columns.values())
        out: dict[str, list[object]] = {col: [] for col in wanted}
        with Path(uri).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # R8-F3: validate the header up-front so missing columns
            # surface as a clear ValueError rather than as a cryptic
            # dtype-conversion failure downstream.
            header = set(reader.fieldnames or [])
            missing = sorted(wanted - header)
            if missing:
                raise ValueError(
                    f"CSV file at {uri!r} is missing required column(s) "
                    f"{missing}; available columns: {sorted(header)}"
                )
            for row_no, row in enumerate(reader, start=2):  # header is file row 1
                empty = sorted(col for col in wanted if row.get(col) is None or row.get(col) == "")
                if empty:
                    raise ValueError(
                        f"CSV file at {uri!r} row {row_no} is missing value(s) for "
                        f"declared column(s) {empty} (empty cell or short row); "
                        f"declared columns must be populated on every row"
                    )
                for col in wanted:
                    out[col].append(row[col])
        return out


class JsonlPredictionReader:
    """Read JSON Lines prediction files into a column-oriented mapping."""

    def read_predictions(
        self,
        uri: str,
        *,
        columns: Mapping[str, str],
    ) -> Mapping[str, Sequence[object]]:
        """Read a local JSONL file.

        Raises
        ------
        ValueError
            If any non-blank row is not valid JSON, or is missing (or has
            ``null`` for) a key declared in the ``columns`` mapping.
            Validated at read time — the R8-F3 pattern already applied to
            CSV headers — so a missing ``score`` key surfaces with the file
            path + row number instead of being coerced to NaN deep inside
            the metric computation (or, for ``label``, dying as a
            context-free ``TypeError``).
        """
        wanted = set(columns.values())
        out: dict[str, list[object]] = {col: [] for col in wanted}
        with Path(uri).open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    # json.loads on a single line always reports "line 1",
                    # actively misdirecting on which file row is broken.
                    raise ValueError(
                        f"JSONL file at {uri!r} row {line_no} is not valid JSON: {exc}"
                    ) from exc
                missing = sorted(col for col in wanted if row.get(col) is None)
                if missing:
                    raise ValueError(
                        f"JSONL file at {uri!r} row {line_no} is missing required "
                        f"key(s) {missing} (or they are null); "
                        f"available keys: {sorted(row)}"
                    )
                for col in wanted:
                    out[col].append(row.get(col))
        return out


def load_prediction_arrays(
    ref: Mapping[str, Any],
    *,
    reader: PredictionReader | None = None,
) -> PredictionArrays:
    """Load labels and scores from a prediction artifact reference.

    Raises
    ------
    ValueError
        If ``ref`` lacks a ``columns`` mapping, lacks a non-empty ``uri``,
        its ``columns`` mapping is missing the ``label`` / ``score`` keys
        (re-raised from :func:`_required_column`), the loaded scores
        contain non-finite values (a bare ``NaN`` token in JSONL or a
        ``"nan"`` cell in CSV passes the readers' per-row key checks but
        must not flow into metrics as a silent NaN), or the loaded labels
        are not all in ``{0, 1}`` (an int cast would silently truncate
        ``0.7 → 0``, flipping ground truth).
    """
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
    # Load labels as float first: np.asarray(..., dtype=int) silently
    # TRUNCATES numeric non-integers (0.7 → 0), flipping ground truth with
    # in-domain values no downstream gate can catch (v1.9.0 pre-tag review).
    labels_raw = np.asarray(table[label_col], dtype=float)
    bad_labels = ~np.isin(labels_raw, (0.0, 1.0))
    if bad_labels.any():
        first_bad = int(np.flatnonzero(bad_labels)[0])
        raise ValueError(
            f"prediction artifact at {uri!r} column {label_col!r} contains "
            f"non-binary label(s); first bad value {labels_raw[first_bad]!r} "
            f"at data row index {first_bad}"
        )
    labels = labels_raw.astype(int)
    scores = np.asarray(table[score_col], dtype=float)
    if not np.isfinite(scores).all():
        first_bad = int(np.flatnonzero(~np.isfinite(scores))[0])
        raise ValueError(
            f"prediction artifact at {uri!r} column {score_col!r} contains "
            f"non-finite score(s) (NaN/inf); first at data row index {first_bad}"
        )
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
    rng: RNGLike | SeedLike | None = 42,
) -> dict[str, object]:
    """Compute a PR-AUC bootstrap CI from one prediction ref."""
    arrays = load_prediction_arrays(ref, reader=reader)
    return bootstrap_ci(
        arrays.labels,
        arrays.scores,
        pr_auc,
        n_resamples=n_resamples,
        rng=rng,
    ).to_dict()


def paired_diff_from_prediction_refs(
    baseline_ref: Mapping[str, Any],
    candidate_ref: Mapping[str, Any],
    *,
    baseline_reader: PredictionReader | None = None,
    candidate_reader: PredictionReader | None = None,
    n_resamples: int = 1000,
    rng: RNGLike | SeedLike | None = 42,
) -> dict[str, object]:
    """Compute paired PR-AUC delta from two prediction refs.

    Raises
    ------
    ValueError
        If the two refs disagree on row count, label values, ``row_ids``,
        or ``content_hashes``; or if either ref is malformed (re-raised
        from :func:`load_prediction_arrays`).
    """
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
        rng=rng,
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
