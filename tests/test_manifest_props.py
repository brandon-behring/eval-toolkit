"""Hypothesis property tests for v0.7.0 RunManifest.

Restores coverage on `src/eval_toolkit/manifest.py` toward the 90 % gate.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# config_hash determinism
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    keys=st.lists(
        st.text(
            min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        ),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=8),
)
@pytest.mark.slow
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_config_hash_invariant_to_key_order(keys: list[str], values: list[int]) -> None:
    """SHA-256 over canonical-JSON is invariant to dict key order."""
    n = min(len(keys), len(values))
    if n == 0:
        return
    items = list(zip(keys[:n], values[:n], strict=True))
    config1 = dict(items)
    config2 = dict(reversed(items))  # different insertion order, same mapping
    m1 = build_manifest(run_id="r1", config=config1)
    m2 = build_manifest(run_id="r2", config=config2)
    assert m1.config_hash == m2.config_hash


@pytest.mark.property
@given(
    config=st.dictionaries(
        keys=st.text(
            min_size=1, max_size=8, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        ),
        values=st.integers(-1000, 1000),
        min_size=1,
        max_size=5,
    ),
)
@pytest.mark.slow
@settings(deadline=None, max_examples=15)
def test_config_hash_changes_when_config_changes(config: dict[str, int]) -> None:
    """If we mutate ANY value in the config, the hash changes."""
    m1 = build_manifest(run_id="r1", config=config)
    mutated = dict(config)
    first_key = next(iter(mutated))
    mutated[first_key] = mutated[first_key] + 1
    m2 = build_manifest(run_id="r2", config=mutated)
    assert m1.config_hash != m2.config_hash


# ---------------------------------------------------------------------------
# Schema version invariants
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(run_id=st.text(min_size=1, max_size=30))
@settings(deadline=None, max_examples=10)
def test_schema_version_always_current(run_id: str) -> None:
    """Every build_manifest result has the current MANIFEST_SCHEMA_VERSION."""
    m = build_manifest(run_id=run_id, config={"k": 5})
    assert m.schema_version == MANIFEST_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# data_hashes prefix invariant
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n_files=st.integers(1, 5),
    payload_size=st.integers(1, 1000),
)
@settings(deadline=None, max_examples=10)
def test_data_hashes_always_sha256_prefixed(n_files: int, payload_size: int) -> None:
    """Every entry in data_hashes is prefixed 'sha256:' (or empty for missing files)."""
    with tempfile.TemporaryDirectory() as d:
        files = {}
        for i in range(n_files):
            p = Path(d) / f"data_{i}.bin"
            p.write_bytes(b"x" * payload_size)
            files[f"data_{i}"] = p
        m = build_manifest(run_id="r", config={}, data_files=files)
        for digest in m.data_hashes.values():
            assert digest == "" or digest.startswith("sha256:")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    run_id=st.text(min_size=1, max_size=30),
    seeds=st.dictionaries(
        keys=st.sampled_from(["global", "bootstrap", "torch", "dataloader"]),
        values=st.integers(0, 99999),
        min_size=0,
        max_size=4,
    ),
)
@settings(deadline=None, max_examples=10)
def test_manifest_json_round_trip(run_id: str, seeds: dict[str, int]) -> None:
    """write_manifest → read JSON preserves run_id, schema_version, seeds."""
    m = build_manifest(run_id=run_id, config={"k": 5}, seeds=seeds)
    with tempfile.TemporaryDirectory() as d:
        path = write_manifest(m, d)
        loaded = json.loads(path.read_text())
    assert loaded["run_id"] == run_id
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert loaded["seeds"] == seeds


# ---------------------------------------------------------------------------
# Versioned object collection
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(version=st.text(min_size=1, max_size=20))
@settings(deadline=None, max_examples=10)
def test_versioned_objects_collected_when_version_present(version: str) -> None:
    """Objects with a `version` attribute land in versioned_objects."""

    class _V:
        def __init__(self, v: str) -> None:
            self.version = v

    m = build_manifest(
        run_id="r",
        config={},
        versioned={"my_obj": _V(version)},
    )
    assert m.versioned_objects == {"my_obj": version}


@pytest.mark.property
def test_versioned_objects_skip_when_version_absent() -> None:
    """Objects without a `version` attribute are silently skipped."""

    class _NoVersion:
        pass

    m = build_manifest(
        run_id="r",
        config={},
        versioned={"obj": _NoVersion()},
    )
    assert m.versioned_objects == {}
