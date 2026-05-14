"""Round-trip stability test for ``contamination_flags`` (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``RunManifest.contamination_flags`` (added in v0.24.0 with manifest.v3)
has 4 valid enum values — ``verified_disjoint``,
``suspected_contamination``, ``vendor_black_box``, ``unknown`` — but
no test exercises the full ``Manifest → JSON → validate → load →
Manifest`` cycle for each value. A schema drift (e.g. an enum value
silently dropped from ``schemas/manifest.v3.json``) would not be
caught by the existing per-value construction tests.

This test pins the cycle for each enum value so future schema /
serialization changes that break the round-trip would fail this
suite immediately.
"""

from __future__ import annotations

import json

import pytest

from eval_toolkit.artifacts import validate_manifest
from eval_toolkit.manifest import build_manifest

VALID_FLAGS: tuple[str, ...] = (
    "verified_disjoint",
    "suspected_contamination",
    "vendor_black_box",
    "unknown",
)


@pytest.mark.parametrize("flag_value", VALID_FLAGS)
def test_contamination_flag_round_trip_via_json_validation(flag_value: str) -> None:
    """Every valid contamination_flags enum survives JSON serialize → validate → reload.

    For each enum value:
    1. Build a RunManifest with ``contamination_flags={"scorer_A": flag}``.
    2. Serialize to JSON via ``to_dict() → json.dumps``.
    3. Re-parse via ``json.loads`` and call ``validate_manifest`` (which
       checks against ``schemas/manifest.v3.json``).
    4. Assert the deserialized payload preserves the flag value verbatim.

    Catches schema drift on the ``contamination_flags`` enum (e.g., a
    value silently removed from the schema or remapped) that would
    otherwise sneak through if all per-value tests only construct in-
    memory manifests without round-tripping.
    """
    manifest = build_manifest(
        run_id=f"test_run_{flag_value}",
        config={"k": 5, "test_label": "round_trip"},
        contamination_flags={"scorer_A": flag_value},
    )
    payload = manifest.to_dict()
    serialized = json.dumps(payload, default=str)
    reparsed = json.loads(serialized)
    # Validates against schemas/manifest.v3.json — raises if the enum is broken.
    validate_manifest(reparsed)
    # The flag must round-trip verbatim through JSON + validation.
    assert reparsed["contamination_flags"]["scorer_A"] == flag_value, (
        f"Flag {flag_value!r} did not survive round-trip; "
        f"got {reparsed['contamination_flags']['scorer_A']!r}"
    )


def test_contamination_flag_invalid_value_rejected_at_build_time() -> None:
    """Invalid contamination_flags values are rejected at ``build_manifest`` time.

    Schema validation only fires after construction; the
    ``build_manifest`` validator should reject invalid values before
    serialization to give early, clear error messages.
    """
    with pytest.raises(ValueError, match="invalid contamination_flags values"):
        build_manifest(
            run_id="test_invalid",
            config={"k": 5},
            contamination_flags={"scorer_A": "not_a_real_enum_value"},
        )


def test_contamination_flag_multi_scorer_round_trip() -> None:
    """Multiple scorers with different contamination_flags all round-trip.

    Companion to the parametrized single-scorer test — verifies the
    schema accepts a non-trivial mapping shape, not just one-key
    dicts.
    """
    flags = {
        "scorer_A": "verified_disjoint",
        "scorer_B": "suspected_contamination",
        "scorer_C": "vendor_black_box",
        "scorer_D": "unknown",
    }
    manifest = build_manifest(
        run_id="test_multi_scorer",
        config={"k": 5},
        contamination_flags=flags,
    )
    payload = manifest.to_dict()
    serialized = json.dumps(payload, default=str)
    reparsed = json.loads(serialized)
    validate_manifest(reparsed)
    assert reparsed["contamination_flags"] == flags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
