"""Unit tests for seeds module."""

from __future__ import annotations

import random

import numpy as np
import pytest

from eval_toolkit.seeds import set_global_seeds


@pytest.mark.unit
def test_set_global_seeds_deterministic() -> None:
    """Same seed → identical RNG output."""
    set_global_seeds(42)
    a1 = np.random.rand(5)
    r1 = random.random()
    set_global_seeds(42)
    a2 = np.random.rand(5)
    r2 = random.random()
    assert np.array_equal(a1, a2)
    assert r1 == r2


@pytest.mark.unit
def test_set_global_seeds_different_seeds() -> None:
    set_global_seeds(1)
    a1 = np.random.rand(5)
    set_global_seeds(2)
    a2 = np.random.rand(5)
    assert not np.array_equal(a1, a2)


@pytest.mark.unit
def test_set_global_seeds_validates_type() -> None:
    with pytest.raises(TypeError):
        set_global_seeds("42")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        set_global_seeds(42.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_set_global_seeds_validates_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seeds(-1)


@pytest.mark.unit
def test_set_global_seeds_works_without_torch() -> None:
    """If torch is not installed, set_global_seeds should still work for numpy/random."""
    # Even if torch IS installed, this confirms no exception is raised.
    set_global_seeds(0)
    assert True  # smoke check: no exception
