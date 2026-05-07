"""Path normalization helpers for repo-relative path display in config dicts.

Pure helpers — no filesystem touch beyond ``Path.resolve()`` for relative-path
checks. For SHA-256 file hashes and run-dir creation, see :mod:`eval_toolkit.provenance`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "path_for_config",
    "resolve_repo_path",
    "split_provenance_config",
]


def resolve_repo_path(path: Path | str, repo_root: Path | str | None = None) -> Path:
    """Resolve a possibly repo-relative path without requiring it to exist.

    Parameters
    ----------
    path : pathlib.Path or str
    repo_root : pathlib.Path or str or None, optional
        Repository root. If ``None``, ``Path.cwd()`` is used.

    Returns
    -------
    pathlib.Path
        Absolute path. If ``path`` is already absolute, returns ``path.resolve()``.
        Otherwise, returns ``(repo_root / path).resolve()``.
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    return (root / p).resolve()


def path_for_config(
    path: Path | str | None,
    repo_root: Path | str | None = None,
) -> str | None:
    """Return a stable repo-relative path string when possible, otherwise absolute.

    Useful for serializing paths into config JSON: keeps repo-internal paths
    portable across machines while still exposing absolute paths for
    repo-external locations.

    Parameters
    ----------
    path : pathlib.Path or str or None
        ``None`` returns ``None`` (so this is null-safe over optional config fields).
    repo_root : pathlib.Path or str or None, optional
        Repository root. If ``None``, ``Path.cwd()`` is used.

    Returns
    -------
    str or None
    """
    if path is None:
        return None
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    resolved = resolve_repo_path(path, root)
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def split_provenance_config(
    config: Mapping[str, Any],
    repo_root: Path | str | None = None,
    *,
    path_keys: tuple[str, ...] = ("path", "dir", "file", "splits_dir", "model_path"),
) -> dict[str, Any]:
    """Walk a config dict and normalize Path-typed values to repo-relative strings.

    Walks ``config`` recursively. Any value that is a :class:`pathlib.Path` is
    converted via :func:`path_for_config`. Values whose key contains any of
    ``path_keys`` (case-insensitive substring match) and is a string are also
    normalized. Non-path values pass through unchanged.

    Parameters
    ----------
    config : Mapping[str, Any]
    repo_root : pathlib.Path or str or None, optional
        If ``None``, ``Path.cwd()`` is used.
    path_keys : tuple of str, optional
        Substrings (case-insensitive) that mark a string value as a path
        worth normalizing.

    Returns
    -------
    dict[str, Any]
        New dict with normalized path values; original is unmodified.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     cfg = {"model_path": Path(tmp) / "m.pkl", "n_resamples": 1000}
    ...     out = split_provenance_config(cfg, repo_root=tmp)
    >>> isinstance(out["model_path"], str)
    True
    >>> out["n_resamples"]
    1000
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()

    def _normalize(key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: _normalize(k, v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(_normalize(key, v) for v in value)
        if isinstance(value, Path):
            return path_for_config(value, root)
        if isinstance(value, str) and any(pk.lower() in key.lower() for pk in path_keys):
            return path_for_config(value, root)
        return value

    return {k: _normalize(k, v) for k, v in config.items()}
