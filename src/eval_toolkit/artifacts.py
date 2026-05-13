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
    "validate_manifest",
    "validate_payload",
    "validate_prediction_artifact_ref",
    "validate_results",
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
    """Manifest reference to a retained prediction artifact.

    ``role`` accepts ``str`` or ``list[str]`` since v0.15.0 (F5.2): a
    single artifact that covers multiple slices / fold-roles can name them
    explicitly instead of carrying a synthetic single-string role plus a
    ``metadata["slices"]`` list. The schema accepts both shapes.
    """

    uri: str
    media_type: str
    columns: PredictionColumns | Mapping[str, object]
    sha256: str = ""
    n_rows: int | None = None
    role: str | list[str] = "predictions"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the stable prediction-reference shape."""
        if not self.uri:
            raise ValueError("PredictionArtifactRef.uri must be non-empty")
        if not self.media_type:
            raise ValueError("PredictionArtifactRef.media_type must be non-empty")
        if self.n_rows is not None and (isinstance(self.n_rows, bool) or self.n_rows < 0):
            raise ValueError("PredictionArtifactRef.n_rows must be non-negative when present")
        if isinstance(self.role, list):
            if not self.role or not all(
                isinstance(r, str) and r.strip() for r in self.role
            ):
                raise ValueError(
                    "PredictionArtifactRef.role list must be non-empty and contain "
                    "only non-empty strings"
                )
        elif isinstance(self.role, str):
            if not self.role.strip():
                raise ValueError("PredictionArtifactRef.role must be non-empty")
        else:
            raise TypeError(
                f"PredictionArtifactRef.role must be str or list[str], "
                f"got {type(self.role).__name__}"
            )
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
        # Preserve role as a list when caller passed a list; otherwise a string.
        role: object = list(self.role) if isinstance(self.role, list) else self.role
        out: dict[str, object] = {
            "uri": self.uri,
            "media_type": self.media_type,
            "role": role,
            "columns": sanitize_for_json(columns),
        }
        if self.sha256:
            out["sha256"] = self.sha256
        if self.n_rows is not None:
            out["n_rows"] = self.n_rows
        if self.metadata:
            out["metadata"] = sanitize_for_json(self.metadata)
        return out


NanStrategy = Literal["skipped", "null", "raise"]


def sanitize_for_json(
    payload: object,
    *,
    nan_strategy: NanStrategy = "skipped",
) -> object:
    """Return a strict-JSON-safe copy of ``payload``.

    Parameters
    ----------
    payload : object
        Arbitrary nested structure of primitives, mappings, sequences,
        dataclasses, numpy scalars / arrays, or objects with ``to_dict``.
    nan_strategy : {"skipped", "null", "raise"}, optional
        How to handle non-finite floats (NaN / ±Inf). Default ``"skipped"``
        replaces them with a structured
        :func:`skipped_metric` dict (keeps the reason auditable; preserves
        pre-v0.13 behavior). ``"null"`` replaces with JSON ``null`` (use when
        downstream consumers expect numeric-or-null without the structured
        sentinel). ``"raise"`` raises ``ValueError`` on first non-finite
        value, surfacing scoring bugs that would otherwise pass silently.
        Closes F4.1 from the V4 consumer feedback log.

    Returns
    -------
    object
        A nested structure that ``json.dumps(..., allow_nan=False)`` will
        accept. RFC 8259 compliant.

    Raises
    ------
    ValueError
        If ``nan_strategy="raise"`` and ``payload`` contains a non-finite
        float anywhere in the structure.
    """
    if payload is None or isinstance(payload, (str, bool, int)):
        return payload
    if isinstance(payload, float):
        if math.isfinite(payload):
            return payload
        if nan_strategy == "skipped":
            return skipped_metric(f"non-finite numeric value: {payload!r}")
        if nan_strategy == "null":
            return None
        if nan_strategy == "raise":
            raise ValueError(f"non-finite numeric value: {payload!r}")
        raise ValueError(
            f"nan_strategy must be one of 'skipped', 'null', 'raise'; "
            f"got {nan_strategy!r}"
        )
    if isinstance(payload, np.generic):
        return sanitize_for_json(payload.item(), nan_strategy=nan_strategy)
    if isinstance(payload, np.ndarray):
        return sanitize_for_json(payload.tolist(), nan_strategy=nan_strategy)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return sanitize_for_json(payload.to_dict(), nan_strategy=nan_strategy)
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return sanitize_for_json(asdict(payload), nan_strategy=nan_strategy)
    if isinstance(payload, Mapping):
        return {
            str(key): sanitize_for_json(value, nan_strategy=nan_strategy)
            for key, value in payload.items()
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [sanitize_for_json(value, nan_strategy=nan_strategy) for value in payload]
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


_KNOWN_MANIFEST_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})


def validate_manifest(payload: Mapping[str, object]) -> None:
    """Validate a serialized ``RunManifest`` payload.

    Dispatches on ``payload["schema_version"]`` (``"v1"`` or ``"v2"``); falls
    back to the current default (``"v2"``) when the field is absent. Closes
    F9.2: callers no longer pass magic schema-name strings to
    :func:`validate_payload` and risk drift when the version bumps.

    Parameters
    ----------
    payload : Mapping[str, object]
        Serialized manifest dict (typically ``RunManifest.to_dict()``).

    Raises
    ------
    ImportError
        If the optional ``validation`` extra is not installed.
    jsonschema.ValidationError
        If the payload does not conform to the schema for its declared
        ``schema_version``.
    ValueError
        If ``schema_version`` is set but unrecognized.
    """
    raw_version = payload.get("schema_version", "v2")
    version = raw_version if isinstance(raw_version, str) else "v2"
    if version not in _KNOWN_MANIFEST_VERSIONS:
        raise ValueError(
            f"unknown manifest schema_version {version!r}; "
            f"expected one of {sorted(_KNOWN_MANIFEST_VERSIONS)}"
        )
    validate_payload(payload, f"manifest.{version}.json")


def validate_results(payload: Mapping[str, object]) -> None:
    """Validate a serialized ``RunResult`` payload against ``results.v1.json``.

    Thin wrapper over :func:`validate_payload` so callers do not pass magic
    schema-name strings (F9.2).
    """
    validate_payload(payload, "results.v1.json")


_PREDICTION_ARTIFACT_REF_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["uri", "media_type", "columns"],
    "properties": {
        "uri": {"type": "string", "minLength": 1},
        "media_type": {"type": "string", "minLength": 1},
        "sha256": {"type": "string"},
        "n_rows": {"type": "integer", "minimum": 0},
        # v0.15.0 (F5.2): role accepts a string or an array of strings.
        "role": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            ],
        },
        "metadata": {"type": "object"},
        "columns": {
            "type": "object",
            "required": ["label", "score"],
            "properties": {
                "row_id": {"type": "string", "minLength": 1},
                "content_hash": {"type": "string", "minLength": 1},
                "label": {"type": "string", "minLength": 1},
                "score": {"type": "string", "minLength": 1},
                "scorer": {"type": "string", "minLength": 1},
                "slice": {"type": "string", "minLength": 1},
                "text": {"type": "string", "minLength": 1},
                "provenance": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


def validate_prediction_artifact_ref(payload: Mapping[str, object]) -> None:
    """Validate a single serialized :class:`PredictionArtifactRef` payload.

    Mirrors the inline schema embedded in ``manifest.v2.json`` for
    ``prediction_artifacts`` items, so callers can validate refs
    independently of the surrounding manifest. Closes F9.2.

    Parameters
    ----------
    payload : Mapping[str, object]
        Serialized prediction artifact reference (typically
        ``PredictionArtifactRef.to_dict()``).

    Raises
    ------
    ImportError
        If the optional ``validation`` extra is not installed.
    jsonschema.ValidationError
        If the payload does not conform.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "validate_prediction_artifact_ref requires the optional validation extra: "
            "install 'eval-toolkit[validation]'"
        ) from exc
    Draft202012Validator(_PREDICTION_ARTIFACT_REF_SCHEMA).validate(sanitize_for_json(payload))
