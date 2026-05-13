"""Provenance helpers: file hashing, run-directory layout, optional git SHA, figure metadata.

All filesystem-touching functions are isolated here so library callers can
import :mod:`eval_toolkit.metrics` and :mod:`eval_toolkit.bootstrap` without
pulling in any IO surface.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "FileHash",
    "FileHashMissing",
    "capture_git_sha",
    "compute_file_hash",
    "figure_metadata",
    "file_sha256",
    "make_run_dir",
]


@dataclass(frozen=True, slots=True)
class FileHash:
    """Sentinel-style hit for a successful file hash.

    Returned (alongside :class:`FileHashMissing`) from
    :func:`compute_file_hash` so callers can pattern-match on the outcome
    rather than testing for a magic ``None`` (F5.1).
    """

    sha256: str

    def __post_init__(self) -> None:
        """Validate the hex-digest shape."""
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError(
                f"FileHash.sha256 must be a 64-character hex digest, got {self.sha256!r}"
            )


@dataclass(frozen=True, slots=True)
class FileHashMissing:
    """Sentinel-style miss for a file hash that could not be computed.

    Carries the reason ``"missing"`` (path does not exist) or
    ``"not_a_file"`` (path exists but is not a regular file). Pair with
    :class:`FileHash` for exhaustive pattern matching.
    """

    reason: str
    path: str


def compute_file_hash(path: Path | str) -> FileHash | FileHashMissing:
    """SHA-256 hex digest of an existing file (sentinel-typed).

    Replaces the v1 :func:`file_sha256` ``str | None`` shape with an
    explicit union (F5.1). Callers can match on the union members
    instead of testing the return for ``None``:

    .. code-block:: python

        match compute_file_hash(path):
            case FileHash(sha256=h):
                record["digest"] = f"sha256:{h}"
            case FileHashMissing(reason="missing"):
                ...

    :func:`file_sha256` remains available as a backward-compatible thin
    wrapper that flattens to ``str | None``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    ...     _ = f.write("hello\\n")
    ...     p = Path(f.name)
    >>> result = compute_file_hash(p)
    >>> isinstance(result, FileHash) and len(result.sha256) == 64
    True
    >>> compute_file_hash("/no/such/file")
    FileHashMissing(reason='missing', path='/no/such/file')
    """
    p = Path(path)
    if not p.exists():
        return FileHashMissing(reason="missing", path=str(path))
    if not p.is_file():
        return FileHashMissing(reason="not_a_file", path=str(path))
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return FileHash(sha256=h.hexdigest())


def file_sha256(path: Path | str, *, strict: bool = False) -> str | None:
    """SHA-256 hex digest of an existing file.

    Default behavior is permissive: returns ``None`` if the path does not
    exist or is not a regular file. Pass ``strict=True`` to raise
    :class:`FileNotFoundError` instead — useful when the caller's invariants
    require the digest to exist.

    .. note::
       New code should prefer :func:`compute_file_hash`, which returns a
       :class:`FileHash` / :class:`FileHashMissing` union and avoids the
       ``None`` ambiguity (F5.1). This helper is retained for backward
       compatibility.

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
    result = compute_file_hash(path)
    if isinstance(result, FileHash):
        return result.sha256
    if strict:
        raise FileNotFoundError(
            f"file_sha256: path missing or not a regular file: {result.path}"
        )
    return None


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
    try:
        import matplotlib as mpl  # noqa: PLC0415

        matplotlib_version = str(mpl.__version__)
    except ImportError:
        matplotlib_version = "unavailable"

    return {
        **provenance,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "matplotlib_version": matplotlib_version,
        "figure_dpi": str(dpi),
    }
