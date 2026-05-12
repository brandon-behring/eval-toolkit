"""Strict artifact helpers for JSON outputs and prediction references."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "MetricState",
    "PredictionArtifactRef",
    "PredictionColumns",
    "error_metric",
    "sanitize_for_json",
    "skipped_metric",
    "validate_payload",
    "write_json_strict",
]

MetricStatus = Literal["ok", "skipped", "error"]


@dataclass(frozen=True, slots=True)
class MetricState:
    """Structured state for metrics that may be unavailable or invalid."""

    value: object | None
    status: MetricStatus
    reason: str = ""
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        out: dict[str, object] = {
            "value": sanitize_for_json(self.value),
            "status": self.status,
            "reason": self.reason,
        }
        if self.details:
            out["details"] = sanitize_for_json(self.details)
        return out


def skipped_metric(reason: str, **details: object) -> dict[str, object]:
    """Return a structured skipped-metric payload."""
    return MetricState(value=None, status="skipped", reason=reason, details=details).to_dict()


def error_metric(reason: str, **details: object) -> dict[str, object]:
    """Return a structured errored-metric payload."""
    return MetricState(value=None, status="error", reason=reason, details=details).to_dict()


@dataclass(frozen=True, slots=True)
class PredictionColumns:
    """Column mapping for a retained prediction artifact."""

    label: str
    score: str
    row_id: str | None = None
    content_hash: str | None = None
    scorer: str | None = None
    slice: str | None = None
    text: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required prediction columns."""
        if not self.label:
            raise ValueError("PredictionColumns.label must be non-empty")
        if not self.score:
            raise ValueError("PredictionColumns.score must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation with absent optional columns omitted."""
        out: dict[str, object] = {"label": self.label, "score": self.score}
        for key in ("row_id", "content_hash", "scorer", "slice", "text"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.provenance:
            out["provenance"] = dict(self.provenance)
        return out


@dataclass(frozen=True, slots=True)
class PredictionArtifactRef:
    """Manifest reference to a retained prediction artifact."""

    uri: str
    media_type: str
    columns: PredictionColumns | Mapping[str, object]
    sha256: str = ""
    n_rows: int | None = None
    role: str = "predictions"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the stable prediction-reference shape."""
        if not self.uri:
            raise ValueError("PredictionArtifactRef.uri must be non-empty")
        if not self.media_type:
            raise ValueError("PredictionArtifactRef.media_type must be non-empty")
        if self.n_rows is not None and (isinstance(self.n_rows, bool) or self.n_rows < 0):
            raise ValueError("PredictionArtifactRef.n_rows must be non-negative when present")
        columns = self.columns
        if isinstance(columns, Mapping):
            if not isinstance(columns.get("label"), str) or not columns.get("label"):
                raise ValueError("PredictionArtifactRef.columns must include a label column")
            if not isinstance(columns.get("score"), str) or not columns.get("score"):
                raise ValueError("PredictionArtifactRef.columns must include a score column")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        if isinstance(self.columns, PredictionColumns):
            columns: object = self.columns.to_dict()
        else:
            columns = dict(self.columns)
        out: dict[str, object] = {
            "uri": self.uri,
            "media_type": self.media_type,
            "role": self.role,
            "columns": sanitize_for_json(columns),
        }
        if self.sha256:
            out["sha256"] = self.sha256
        if self.n_rows is not None:
            out["n_rows"] = self.n_rows
        if self.metadata:
            out["metadata"] = sanitize_for_json(self.metadata)
        return out


def sanitize_for_json(payload: object) -> object:
    """Return a strict-JSON-safe copy of ``payload``.

    Non-finite floats are converted to structured skipped metric objects. This
    keeps artifact JSON RFC 8259-compliant while preserving why a numeric value
    could not be represented.
    """
    if payload is None or isinstance(payload, (str, bool, int)):
        return payload
    if isinstance(payload, float):
        if math.isfinite(payload):
            return payload
        return skipped_metric(f"non-finite numeric value: {payload!r}")
    if isinstance(payload, np.generic):
        return sanitize_for_json(payload.item())
    if isinstance(payload, np.ndarray):
        return sanitize_for_json(payload.tolist())
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return sanitize_for_json(payload.to_dict())
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return sanitize_for_json(asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [sanitize_for_json(value) for value in payload]
    return str(payload)


def write_json_strict(
    payload: object,
    path: Path | str,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Write strict RFC 8259 JSON after sanitizing non-finite values."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_for_json(payload)
    out_path.write_text(json.dumps(sanitized, indent=indent, sort_keys=sort_keys, allow_nan=False))
    return out_path


def validate_payload(payload: object, schema_name: str) -> None:
    """Validate a payload against a bundled schema.

    ``jsonschema`` is intentionally optional at runtime. Install
    ``eval-toolkit[validation]`` or the dev extra to use this helper.
    """
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "validate_payload requires the optional validation extra: "
            "install 'eval-toolkit[validation]'"
        ) from exc

    schema_path = resources.files("eval_toolkit") / "schemas" / schema_name
    schema = json.loads(schema_path.read_text())
    Draft202012Validator(schema).validate(sanitize_for_json(payload))
