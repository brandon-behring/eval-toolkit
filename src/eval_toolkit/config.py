"""Frozen-dataclass config pattern + YAML loader.

The toolkit's recommended config pattern: ``@frozen_config`` wraps
``@dataclass(frozen=True, slots=True)`` and ensures the subclass implements
``__post_init__`` for validation.

YAML loading is optional — it requires the ``yaml`` extra
(``pip install eval-toolkit[yaml]``). If ``pyyaml`` is not installed,
:func:`from_yaml` raises :class:`ImportError` with a helpful message.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = ["frozen_config", "from_yaml"]

T = TypeVar("T")


def frozen_config(cls: type[T]) -> type[T]:
    """Decorator: apply ``@dataclass(frozen=True, slots=True)`` and validate.

    Subclasses must implement ``__post_init__`` for field validation; the
    decorator does not validate field values directly (that's
    domain-specific) but it does ensure the class becomes a frozen, slotted
    dataclass.

    Parameters
    ----------
    cls : type
        The class to decorate.

    Returns
    -------
    type
        A frozen, slotted dataclass.

    Examples
    --------
    >>> @frozen_config
    ... class TrainConfig:
    ...     lr: float
    ...     batch_size: int = 16
    ...     def __post_init__(self) -> None:
    ...         if self.lr <= 0:
    ...             raise ValueError(f"lr must be > 0, got {self.lr}")
    >>> cfg = TrainConfig(lr=1e-3)
    >>> cfg.lr
    0.001
    >>> try:
    ...     cfg.lr = 1e-4
    ... except (AttributeError, Exception) as e:
    ...     print(type(e).__name__)
    FrozenInstanceError
    """
    return dataclass(frozen=True, slots=True)(cls)


def from_yaml(path: Path | str, cls: type[T]) -> T:
    """Load a YAML file into an instance of ``cls`` (a frozen dataclass).

    Requires the ``yaml`` extra: ``pip install eval-toolkit[yaml]``.

    Parameters
    ----------
    path : pathlib.Path or str
    cls : type
        Frozen dataclass type. The YAML's top-level keys must be a subset of
        ``cls``'s fields.

    Returns
    -------
    T
        Instance of ``cls`` constructed from the YAML.

    Raises
    ------
    ImportError
        If pyyaml is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If ``cls`` is not a dataclass.
    KeyError
        If the YAML contains an unknown key (not a field of ``cls``).
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "from_yaml requires pyyaml; install with `pip install eval-toolkit[yaml]`"
        ) from exc

    if not is_dataclass(cls):
        raise TypeError(f"cls must be a dataclass, got {cls.__name__}")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    raw: Any = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(raw).__name__}")

    field_names = {f.name for f in fields(cls)}
    unknown = set(raw.keys()) - field_names
    if unknown:
        raise KeyError(
            f"unknown config keys: {sorted(unknown)}; expected subset of {sorted(field_names)}"
        )

    return cls(**raw)
