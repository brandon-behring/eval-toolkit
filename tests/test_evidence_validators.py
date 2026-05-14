"""Error-path coverage for ``eval_toolkit.evidence`` dataclass validators (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``evidence.py`` has 3 ``__post_init__`` validators that ``raise
ValueError`` on empty inputs, with zero dedicated tests asserting
they fire. These dataclasses underpin manifest serialization, so
silent regression in their validation would corrupt evidence
metadata downstream.

Coverage matrix (raise-site → test):

| File:line in evidence.py | Test |
|---|---|
| L36 (EvidenceAxis.name empty) | test_evidence_axis_empty_name_raises |
| L38 (EvidenceAxis.value empty) | test_evidence_axis_empty_value_raises |
| L78 (AggregateEvidence.method empty) | test_aggregate_evidence_empty_method_raises |
"""

from __future__ import annotations

import pytest

from eval_toolkit.evidence import AggregateEvidence, EvidenceAxis


def test_evidence_axis_empty_name_raises() -> None:
    """``EvidenceAxis(name="", value="x")`` raises ValueError. Covers ``evidence.py:36``."""
    with pytest.raises(ValueError, match="EvidenceAxis.name must be non-empty"):
        EvidenceAxis(name="", value="x")


def test_evidence_axis_empty_value_raises() -> None:
    """``EvidenceAxis(name="x", value="")`` raises ValueError. Covers ``evidence.py:38``."""
    with pytest.raises(ValueError, match="EvidenceAxis.value must be non-empty"):
        EvidenceAxis(name="x", value="")


def test_aggregate_evidence_empty_method_raises() -> None:
    """``AggregateEvidence(method="")`` raises ValueError. Covers ``evidence.py:78``."""
    with pytest.raises(ValueError, match="AggregateEvidence.method must be non-empty"):
        AggregateEvidence(status="inferential", method="")


def test_evidence_axis_constructs_with_valid_inputs() -> None:
    """Sanity: a fully-specified ``EvidenceAxis`` constructs without error.

    Provides a positive control so a future regression that breaks
    construction of valid inputs would also fail this suite (not just
    the negative tests above).
    """
    axis = EvidenceAxis(name="fold", value="3")
    assert axis.name == "fold"
    assert axis.value == "3"


def test_aggregate_evidence_constructs_with_valid_inputs() -> None:
    """Sanity: a fully-specified ``AggregateEvidence`` constructs without error."""
    ev = AggregateEvidence(
        status="inferential",
        method="bootstrap_paired",
        axes=(EvidenceAxis(name="fold", value="0"),),
    )
    assert ev.status == "inferential"
    assert ev.method == "bootstrap_paired"
    assert len(ev.axes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
