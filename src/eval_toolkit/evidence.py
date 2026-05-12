"""Generic evidence metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "AggregateEvidence",
    "EvidenceAxis",
    "PairingMetadata",
    "RECOMMENDED_SOURCE_ROLES",
]

AggregateStatus = Literal["inferential", "descriptive", "diagnostic", "unsupported"]

RECOMMENDED_SOURCE_ROLES: tuple[str, ...] = (
    "train",
    "calibration",
    "locked_eval",
    "external_diagnostic",
    "excluded",
)


@dataclass(frozen=True, slots=True)
class EvidenceAxis:
    """Named evidence axis such as fold, seed, source_out, view, or slice."""

    name: str
    value: str

    def __post_init__(self) -> None:
        """Validate a non-empty axis."""
        if not self.name:
            raise ValueError("EvidenceAxis.name must be non-empty")
        if not self.value:
            raise ValueError("EvidenceAxis.value must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class PairingMetadata:
    """Machine-readable description of comparison pairedness."""

    paired: bool
    unit: str = "row"
    valid_scope: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "paired": self.paired,
            "unit": self.unit,
            "valid_scope": self.valid_scope,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class AggregateEvidence:
    """Typed status for aggregate rows or summaries."""

    status: AggregateStatus
    method: str
    axes: tuple[EvidenceAxis, ...] = ()
    notes: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize axes to a tuple."""
        object.__setattr__(self, "axes", tuple(self.axes))
        if not self.method:
            raise ValueError("AggregateEvidence.method must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        out: dict[str, object] = {
            "status": self.status,
            "method": self.method,
            "axes": [axis.to_dict() for axis in self.axes],
            "notes": self.notes,
        }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out
