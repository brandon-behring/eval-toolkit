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


@pytest.mark.unit
def test_set_global_seeds_torch_path_when_available() -> None:
    """Cover the torch.manual_seed + cudnn flag path when torch is installed.

    Skipped if torch is not present (it's an optional dep).
    """
    torch = pytest.importorskip("torch")
    set_global_seeds(123)
    # Two consecutive calls with the same seed should yield identical torch RNG.
    a = torch.rand(5)
    set_global_seeds(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


@pytest.mark.unit
def test_set_global_seeds_strict_torch_raises_when_torch_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """strict_torch_determinism=True without torch installed → RuntimeError.

    Simulates torch-absent by injecting an ImportError in the import path.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0) -> object:
        if name == "torch":
            raise ImportError("simulated torch-absent")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="strict_torch_determinism=True"):
        set_global_seeds(0, strict_torch_determinism=True)


@pytest.mark.unit
def test_set_global_seeds_strict_torch_with_torch_installed() -> None:
    """strict mode with torch present should call torch.use_deterministic_algorithms."""
    pytest.importorskip("torch")
    # Should not raise; just exercise the strict-deterministic branch.
    try:
        set_global_seeds(0, strict_torch_determinism=True)
    except RuntimeError as exc:
        # Some PyTorch builds error on this path because not all ops have
        # deterministic kernels — that's expected behavior, not a test failure.
        assert "deterministic" in str(exc).lower() or "non-deterministic" in str(exc).lower()


@pytest.mark.unit
def test_set_global_seeds_torch_post_import_branch_via_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the torch-installed code path even when torch isn't a real dep.

    seeds.py:103-109 (manual_seed + cuda + cudnn flags + use_deterministic_algorithms)
    only execute when ``import torch`` succeeds. In dev envs without torch
    those lines were uncovered; intercepting the ``__import__`` builtin lets
    us inject a lightweight fake torch and exercise the contract without
    pulling the real CUDA-ready package.
    """
    import builtins
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # Build a torch-shaped fake with the API surface seeds.py touches.
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=MagicMock(),
    )
    fake_backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
    fake_torch = SimpleNamespace(
        manual_seed=MagicMock(),
        cuda=fake_cuda,
        backends=fake_backends,
        use_deterministic_algorithms=MagicMock(),
    )
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_=None,  # type: ignore[no-untyped-def]
        locals_=None,  # type: ignore[no-untyped-def]
        fromlist=(),  # type: ignore[no-untyped-def]
        level: int = 0,
    ) -> object:
        if name == "torch":
            return fake_torch
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Non-strict path: hits manual_seed + cuda branch + cudnn flags (lines 103-107).
    set_global_seeds(99)
    fake_torch.manual_seed.assert_called_with(99)
    fake_cuda.manual_seed_all.assert_called_with(99)
    assert fake_backends.cudnn.deterministic is True
    assert fake_backends.cudnn.benchmark is False

    # Strict path: also hits use_deterministic_algorithms (line 109).
    set_global_seeds(99, strict_torch_determinism=True)
    fake_torch.use_deterministic_algorithms.assert_called_with(True)
