"""Unit tests for provenance: file_sha256, make_run_dir, capture_git_sha, figure_metadata."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from eval_toolkit.provenance import (
    FileHash,
    FileHashMissing,
    capture_git_sha,
    compute_file_hash,
    figure_metadata,
    file_sha256,
    make_run_dir,
)


@pytest.mark.unit
def test_file_sha256_matches_stdlib(tmp_path: Path) -> None:
    p = tmp_path / "data.txt"
    p.write_text("hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_sha256(p) == expected


@pytest.mark.unit
def test_file_sha256_returns_none_for_missing() -> None:
    assert file_sha256("/no/such/file/exists") is None


@pytest.mark.unit
def test_file_sha256_returns_none_for_directory(tmp_path: Path) -> None:
    assert file_sha256(tmp_path) is None


@pytest.mark.unit
def test_file_sha256_strict_raises_on_missing() -> None:
    """``strict=True`` raises ``FileNotFoundError`` instead of returning ``None``."""
    with pytest.raises(FileNotFoundError, match="path missing or not a regular file"):
        file_sha256("/no/such/file/exists", strict=True)


@pytest.mark.unit
def test_file_sha256_strict_raises_on_directory(tmp_path: Path) -> None:
    """``strict=True`` rejects directories (not regular files)."""
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path, strict=True)


@pytest.mark.unit
def test_compute_file_hash_returns_filehash_on_success(tmp_path: Path) -> None:
    """v0.15.0 sentinel idiom — hit returns a FileHash with the hex digest."""
    p = tmp_path / "data.txt"
    p.write_text("hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    result = compute_file_hash(p)
    assert isinstance(result, FileHash)
    assert result.sha256 == expected


@pytest.mark.unit
def test_compute_file_hash_returns_missing_when_absent() -> None:
    """v0.15.0 sentinel idiom — miss carries reason='missing' and the path."""
    result = compute_file_hash("/no/such/file/exists")
    assert isinstance(result, FileHashMissing)
    assert result.reason == "missing"
    assert result.path == "/no/such/file/exists"


@pytest.mark.unit
def test_compute_file_hash_returns_missing_for_directory(tmp_path: Path) -> None:
    """Directories return reason='not_a_file' so callers can distinguish."""
    result = compute_file_hash(tmp_path)
    assert isinstance(result, FileHashMissing)
    assert result.reason == "not_a_file"


@pytest.mark.unit
def test_filehash_rejects_short_digest() -> None:
    """FileHash post-init validates the hex-digest shape."""
    with pytest.raises(ValueError, match="64-character"):
        FileHash(sha256="too-short")


@pytest.mark.unit
def test_file_sha256_remains_backward_compatible(tmp_path: Path) -> None:
    """v0.15.0 — file_sha256 still returns str|None and is consistent with compute_file_hash."""
    p = tmp_path / "data.txt"
    p.write_text("hello world")
    sentinel = compute_file_hash(p)
    legacy = file_sha256(p)
    assert isinstance(sentinel, FileHash)
    assert legacy == sentinel.sha256


@pytest.mark.unit
def test_file_sha256_strict_returns_digest_when_present(tmp_path: Path) -> None:
    """``strict=True`` returns the digest unchanged when the file exists."""
    p = tmp_path / "hello.txt"
    p.write_text("hello\n")
    digest_strict = file_sha256(p, strict=True)
    digest_default = file_sha256(p)
    assert digest_strict == digest_default
    assert digest_strict is not None
    assert len(digest_strict) == 64


@pytest.mark.unit
def test_make_run_dir_creates_path(tmp_path: Path) -> None:
    out = make_run_dir(tmp_path, prefix="run")
    assert out.exists()
    assert out.is_dir()
    assert re.match(r"run_\d{4}-\d{2}-\d{2}_\d{6}", out.name)


@pytest.mark.unit
def test_make_run_dir_custom_prefix(tmp_path: Path) -> None:
    out = make_run_dir(tmp_path, prefix="eval")
    assert out.name.startswith("eval_")


@pytest.mark.unit
def test_capture_git_sha_returns_none_outside_repo(tmp_path: Path) -> None:
    """Outside a git repo, returns None (no exception)."""
    # tmp_path is not in any git repo
    assert capture_git_sha(tmp_path) is None


@pytest.mark.unit
def test_capture_git_sha_in_git_repo(tmp_path: Path) -> None:
    """Inside a git repo, returns the SHA."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=T",
            "commit",
            "--allow-empty",
            "-m",
            "init",
            "--no-gpg-sign",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    sha = capture_git_sha(tmp_path)
    assert sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


@pytest.mark.unit
def test_figure_metadata_combines_keys() -> None:
    user = {"git_sha": "abc", "run_id": "r1"}
    out = figure_metadata(user, dpi=150)
    assert out["git_sha"] == "abc"
    assert out["run_id"] == "r1"
    assert out["figure_dpi"] == "150"
    assert "timestamp_utc" in out
    assert "matplotlib_version" in out


@pytest.mark.unit
def test_figure_metadata_does_not_mutate_input() -> None:
    user = {"x": "y"}
    figure_metadata(user)
    assert user == {"x": "y"}
