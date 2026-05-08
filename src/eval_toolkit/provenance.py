"""Provenance helpers: file hashing, run-directory layout, optional git SHA, figure metadata.

All filesystem-touching functions are isolated here so library callers can
import :mod:`eval_toolkit.metrics` and :mod:`eval_toolkit.bootstrap` without
pulling in any IO surface.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl

__all__ = [
    "capture_git_sha",
    "figure_metadata",
    "file_sha256",
    "make_run_dir",
]


def file_sha256(path: Path | str, *, strict: bool = False) -> str | None:
    """SHA-256 hex digest of an existing file.

    Default behavior is permissive: returns ``None`` if the path does not
    exist or is not a regular file. Pass ``strict=True`` to raise
    :class:`FileNotFoundError` instead — useful when the caller's invariants
    require the digest to exist.

    Parameters
    ----------
    path : pathlib.Path or str
    strict : bool, optional
        If ``True``, raise :class:`FileNotFoundError` when ``path`` is missing
        or not a regular file. Default ``False`` (return ``None``).

    Returns
    -------
    str or None
        64-character hex digest. ``None`` only when ``strict=False`` and the
        path is absent / not a regular file.

    Raises
    ------
    FileNotFoundError
        Only when ``strict=True`` and ``path`` is missing or not a regular file.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    ...     _ = f.write("hello\\n")
    ...     p = Path(f.name)
    >>> digest = file_sha256(p)
    >>> len(digest)
    64
    >>> file_sha256("/no/such/file") is None
    True
    >>> try:
    ...     file_sha256("/no/such/file", strict=True)
    ... except FileNotFoundError:
    ...     print("raised")
    raised
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        if strict:
            raise FileNotFoundError(f"file_sha256: path missing or not a regular file: {p}")
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_dir(base: Path | str, prefix: str = "run") -> Path:
    """Create and return a timestamped run directory.

    Format: ``{base}/{prefix}_{YYYY-MM-DD_HHMMSS}``.

    Parameters
    ----------
    base : pathlib.Path or str
        Parent directory for run dirs (e.g., ``evals/``).
    prefix : str, optional
        Prefix before the timestamp. Default ``"run"``.

    Returns
    -------
    pathlib.Path
        The newly-created (or pre-existing) run directory.
    """
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def capture_git_sha(repo_root: Path | str | None = None) -> str | None:
    """Current HEAD full SHA, or ``None`` if not in a git repo.

    Optional and injectable: callers that want a specific SHA (e.g., from CI
    metadata) should pass it directly to whichever function consumes it
    rather than calling this helper.

    Parameters
    ----------
    repo_root : pathlib.Path or str or None, optional
        Working directory for ``git``. ``None`` uses the current process cwd.

    Returns
    -------
    str or None
        Full SHA hex string, or ``None`` if ``git`` is unavailable or the
        directory is not inside a repository.
    """
    cwd = Path(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def figure_metadata(provenance: dict[str, str], dpi: int = 300) -> dict[str, str]:
    """Build a JSON-serializable provenance dict for ``save_figure``.

    Combines caller-supplied keys with auto-fields ``timestamp_utc``
    (ISO-8601), ``matplotlib_version``, and ``figure_dpi``. Matches the
    schema produced inside :func:`eval_toolkit.plotting.save_figure`.

    Parameters
    ----------
    provenance : dict[str, str]
        Caller-supplied provenance keys (e.g., ``git_sha``, ``run_id``).
    dpi : int, optional
        DPI to record. Default 300.

    Returns
    -------
    dict[str, str]
        Combined provenance dict.

    Examples
    --------
    >>> meta = figure_metadata({"run_id": "test-001"}, dpi=150)
    >>> sorted(meta.keys())
    ['figure_dpi', 'matplotlib_version', 'run_id', 'timestamp_utc']
    >>> meta["figure_dpi"]
    '150'
    """
    return {
        **provenance,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "matplotlib_version": str(mpl.__version__),
        "figure_dpi": str(dpi),
    }
