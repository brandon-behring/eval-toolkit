"""Unit tests for config: frozen_config decorator, from_yaml loader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from eval_toolkit.config import from_yaml, frozen_config


@pytest.mark.unit
def test_frozen_config_makes_frozen_dataclass() -> None:
    @frozen_config
    class C:
        x: int
        y: str = "default"

        def __post_init__(self) -> None:
            if self.x < 0:
                raise ValueError("x must be >= 0")

    c = C(x=5)
    assert c.x == 5
    assert c.y == "default"

    # Frozen — assignment raises
    with pytest.raises(FrozenInstanceError):
        c.x = 10  # type: ignore[misc]


@pytest.mark.unit
def test_frozen_config_validates_in_post_init() -> None:
    @frozen_config
    class C:
        lr: float

        def __post_init__(self) -> None:
            if self.lr <= 0:
                raise ValueError(f"lr must be > 0, got {self.lr}")

    with pytest.raises(ValueError, match="lr must be"):
        C(lr=-0.1)


@pytest.mark.unit
def test_frozen_config_uses_slots() -> None:
    """slots=True means no __dict__."""

    @frozen_config
    class C:
        x: int

    c = C(x=1)
    assert not hasattr(c, "__dict__")


@pytest.mark.unit
def test_from_yaml_round_trip(tmp_path: Path) -> None:
    @frozen_config
    class C:
        lr: float
        batch_size: int = 16

        def __post_init__(self) -> None:
            if self.lr <= 0:
                raise ValueError("lr must be > 0")

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("lr: 0.001\nbatch_size: 32\n")
    cfg = from_yaml(yaml_path, C)
    assert cfg.lr == 0.001
    assert cfg.batch_size == 32


@pytest.mark.unit
def test_from_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    @frozen_config
    class C:
        x: int

    p = tmp_path / "cfg.yaml"
    p.write_text("x: 5\nunknown_key: 99\n")
    with pytest.raises(KeyError, match="unknown_key"):
        from_yaml(p, C)


@pytest.mark.unit
def test_from_yaml_missing_file() -> None:
    @frozen_config
    class C:
        x: int

    with pytest.raises(FileNotFoundError):
        from_yaml("/no/such/cfg.yaml", C)


@pytest.mark.unit
def test_from_yaml_rejects_non_dataclass() -> None:
    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="dataclass"):
        from_yaml("/tmp/whatever.yaml", NotADataclass)  # type: ignore[arg-type]
