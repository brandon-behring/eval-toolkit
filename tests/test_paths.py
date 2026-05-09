"""Unit tests for paths module."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_toolkit.paths import (
    path_for_config,
    resolve_repo_path,
    split_provenance_config,
)


@pytest.mark.unit
def test_resolve_repo_path_absolute_passthrough(tmp_path: Path) -> None:
    """Absolute paths pass through (resolved)."""
    out = resolve_repo_path(tmp_path)
    assert out == tmp_path.resolve()
    assert out.is_absolute()


@pytest.mark.unit
def test_resolve_repo_path_relative(tmp_path: Path) -> None:
    """Relative paths are resolved against repo_root."""
    out = resolve_repo_path("data/file.txt", repo_root=tmp_path)
    assert out == (tmp_path / "data/file.txt").resolve()


@pytest.mark.unit
def test_path_for_config_inside_repo(tmp_path: Path) -> None:
    """Paths inside repo_root return repo-relative."""
    p = tmp_path / "data" / "x.parquet"
    out = path_for_config(p, tmp_path)
    assert out == "data/x.parquet"


@pytest.mark.unit
def test_path_for_config_outside_repo(tmp_path: Path) -> None:
    """Paths outside repo_root pass through unchanged (absolute)."""
    # tmp_path.parent is guaranteed to exist and be outside tmp_path itself,
    # so we exercise the "outside-repo" branch without depending on /tmp
    # specifically (v0.8.3: tighten from `out is not None` to value-equality).
    elsewhere = str(tmp_path.parent / "outside.txt")
    out = path_for_config(elsewhere, tmp_path)
    assert out == elsewhere
    assert Path(out).is_absolute()


@pytest.mark.unit
def test_path_for_config_none_passthrough() -> None:
    assert path_for_config(None) is None


@pytest.mark.unit
def test_split_provenance_config_normalizes_paths(tmp_path: Path) -> None:
    cfg = {
        "model_path": tmp_path / "m.pkl",
        "n_resamples": 1000,
        "seed": 42,
    }
    out = split_provenance_config(cfg, repo_root=tmp_path)
    assert out["model_path"] == "m.pkl"
    assert out["n_resamples"] == 1000
    assert out["seed"] == 42


@pytest.mark.unit
def test_split_provenance_config_walks_nested(tmp_path: Path) -> None:
    cfg = {
        "outer": {
            "inner_path": tmp_path / "a" / "b.txt",
            "value": 100,
        },
    }
    out = split_provenance_config(cfg, repo_root=tmp_path)
    assert out["outer"]["inner_path"] == "a/b.txt"
    assert out["outer"]["value"] == 100


@pytest.mark.unit
def test_split_provenance_config_does_not_mutate(tmp_path: Path) -> None:
    p = tmp_path / "x.pkl"
    original = {"model_path": p, "seed": 1}
    _ = split_provenance_config(original, repo_root=tmp_path)
    assert original["model_path"] == p  # unchanged
