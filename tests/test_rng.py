"""Unit tests for `eval_toolkit._rng.spawn_seed_sequences`.

Round 8 audit (C4(b)) confirmed that the v0.50.0 implementation extracted
entropy from the generator's seed-source rather than drawing from current
state — advancing a generator by N draws had zero effect on subsequent
spawns. v0.51.0 fixes this by drawing fresh entropy from the generator on
every call. These tests pin the state-respecting contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit._rng import spawn_seed_sequences


@pytest.mark.unit
def test_spawn_advanced_generator_differs_from_fresh() -> None:
    """A generator advanced past its construction state must spawn different children."""
    fresh = np.random.default_rng(7)
    advanced = np.random.default_rng(7)
    # Advance `advanced` by drawing 1000 ints.
    _ = advanced.integers(0, 1_000_000, size=1000)

    children_fresh = spawn_seed_sequences(fresh, 3)
    children_advanced = spawn_seed_sequences(advanced, 3)

    # At least one child seed differs between fresh and advanced.
    fresh_entropy = [ss.entropy for ss in children_fresh]
    advanced_entropy = [ss.entropy for ss in children_advanced]
    assert fresh_entropy != advanced_entropy, (
        "spawn_seed_sequences must respect generator state — advanced "
        "generator should produce different children than a fresh one. "
        f"fresh={fresh_entropy}, advanced={advanced_entropy}"
    )


@pytest.mark.unit
def test_spawn_same_state_yields_same_children() -> None:
    """Two generators at the same state must spawn identical children (determinism)."""
    a = np.random.default_rng(42)
    b = np.random.default_rng(42)
    children_a = spawn_seed_sequences(a, 4)
    children_b = spawn_seed_sequences(b, 4)
    assert [ss.entropy for ss in children_a] == [ss.entropy for ss in children_b]


@pytest.mark.unit
def test_spawn_repeated_calls_advance_state() -> None:
    """Sequential calls on the same Generator must yield different children each time."""
    rng = np.random.default_rng(123)
    first = spawn_seed_sequences(rng, 3)
    second = spawn_seed_sequences(rng, 3)
    first_entropy = [ss.entropy for ss in first]
    second_entropy = [ss.entropy for ss in second]
    assert first_entropy != second_entropy, (
        "Sequential calls on the same Generator must advance state and "
        "produce different children. This was the root cause of bootstrap "
        "non-independence across (slice, scorer) pairs pre-v0.51."
    )


@pytest.mark.unit
def test_spawn_children_are_independent_seedsequences() -> None:
    """Each child must be a usable SeedSequence that seeds an independent Generator."""
    children = spawn_seed_sequences(np.random.default_rng(0), 3)
    assert all(isinstance(ss, np.random.SeedSequence) for ss in children)
    # Confirm each child seeds an independent Generator (different draws).
    streams = [np.random.default_rng(ss).integers(0, 1_000_000, size=5).tolist() for ss in children]
    # All three streams should differ (independence between SeedSequences).
    assert streams[0] != streams[1]
    assert streams[1] != streams[2]
    assert streams[0] != streams[2]


@pytest.mark.unit
def test_spawn_with_int_seed_input() -> None:
    """Integer-seed input also works (per SPEC 7 SeedLike contract)."""
    children = spawn_seed_sequences(42, 2)
    assert len(children) == 2
    assert all(isinstance(ss, np.random.SeedSequence) for ss in children)


@pytest.mark.unit
def test_spawn_with_none_input() -> None:
    """None input draws fresh OS entropy; result is non-deterministic but valid."""
    children = spawn_seed_sequences(None, 2)
    assert len(children) == 2
    assert all(isinstance(ss, np.random.SeedSequence) for ss in children)


@pytest.mark.unit
def test_spawn_n_zero_returns_empty_list() -> None:
    """Spawning 0 children is a degenerate but valid case."""
    children = spawn_seed_sequences(np.random.default_rng(7), 0)
    assert children == []
