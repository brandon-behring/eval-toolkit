"""Validation-path coverage for eval_toolkit.manifest.

Pairs with `test_manifest.py` (which covers the happy path). These
tests target the error branches of `validate_source_roles`,
`build_manifest`'s guardrail / prediction-artifact validation, and the
mapping-shape normalizers `_object_to_dict` /
`_prediction_artifact_to_dict`.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from eval_toolkit.manifest import (
    SourceRoleRecord,
    _is_git_dirty,
    _object_to_dict,
    _prediction_artifact_to_dict,
    build_manifest,
    gpu_info,
    validate_source_roles,
)


@pytest.mark.unit
def test_validate_source_roles_flags_duplicate_source() -> None:
    errors = validate_source_roles(
        [
            SourceRoleRecord(source="train_pool", role="train"),
            SourceRoleRecord(source="train_pool", role="extra"),
        ],
    )
    assert any("duplicate source" in err for err in errors)


@pytest.mark.unit
def test_validate_source_roles_flags_empty_role() -> None:
    errors = validate_source_roles([{"source": "data", "role": ""}])
    assert any("role must be a non-empty string" in err for err in errors)


@pytest.mark.unit
def test_validate_source_roles_flags_negative_n_rows() -> None:
    errors = validate_source_roles(
        [{"source": "data", "role": "train", "n_rows": -1}],
    )
    assert any("n_rows must be a non-negative integer" in err for err in errors)


@pytest.mark.unit
def test_validate_source_roles_flags_bool_n_rows() -> None:
    """bool is a subclass of int — reject True/False explicitly."""
    errors = validate_source_roles(
        [{"source": "data", "role": "train", "n_rows": True}],
    )
    assert any("n_rows must be a non-negative integer" in err for err in errors)


@pytest.mark.unit
def test_validate_source_roles_flags_non_string_notes() -> None:
    errors = validate_source_roles(
        [{"source": "data", "role": "train", "notes": 123}],
    )
    assert any("notes must be a string" in err for err in errors)


@pytest.mark.unit
def test_validate_source_roles_flags_non_mapping_metadata() -> None:
    errors = validate_source_roles(
        [{"source": "data", "role": "train", "metadata": "not-a-dict"}],
    )
    assert any("metadata must be an object" in err for err in errors)


@pytest.mark.unit
def test_build_manifest_rejects_empty_guardrail() -> None:
    with pytest.raises(ValueError, match="guardrails must be non-empty strings"):
        build_manifest(run_id="r", config={}, guardrails=["valid", ""])


@pytest.mark.unit
def test_object_to_dict_accepts_plain_mapping() -> None:
    out = _object_to_dict({"a": 1, "b": "x"})
    assert out == {"a": 1, "b": "x"}


@pytest.mark.unit
def test_object_to_dict_accepts_to_dict_object() -> None:
    class _R:
        def to_dict(self) -> dict[str, object]:
            return {"k": "v"}

    out = _object_to_dict(_R())
    assert out == {"k": "v"}


@pytest.mark.unit
def test_object_to_dict_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="expected mapping or object with to_dict"):
        _object_to_dict(42)


@pytest.mark.unit
def test_prediction_artifact_to_dict_from_mapping() -> None:
    """Mapping-shaped artifact refs pass through normalization."""
    out = _prediction_artifact_to_dict(
        {
            "uri": "p.csv",
            "media_type": "text/csv",
            "columns": {"label": "y", "score": "s"},
        }
    )
    assert out["uri"] == "p.csv"
    assert out["columns"]["label"] == "y"


@pytest.mark.unit
def test_prediction_artifact_from_mapping_rejects_missing_uri() -> None:
    with pytest.raises(ValueError, match="non-empty uri"):
        _prediction_artifact_to_dict(
            {"media_type": "text/csv", "columns": {"label": "y", "score": "s"}}
        )


@pytest.mark.unit
def test_prediction_artifact_from_mapping_rejects_missing_media_type() -> None:
    with pytest.raises(ValueError, match="non-empty media_type"):
        _prediction_artifact_to_dict({"uri": "p", "columns": {"label": "y", "score": "s"}})


@pytest.mark.unit
def test_prediction_artifact_from_mapping_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="columns object"):
        _prediction_artifact_to_dict({"uri": "p", "media_type": "text/csv"})


@pytest.mark.unit
def test_prediction_artifact_from_mapping_rejects_missing_label() -> None:
    with pytest.raises(ValueError, match="columns must include label"):
        _prediction_artifact_to_dict(
            {"uri": "p", "media_type": "text/csv", "columns": {"score": "s"}}
        )


@pytest.mark.unit
def test_prediction_artifact_from_mapping_rejects_missing_score() -> None:
    with pytest.raises(ValueError, match="columns must include score"):
        _prediction_artifact_to_dict(
            {"uri": "p", "media_type": "text/csv", "columns": {"label": "y"}}
        )


@pytest.mark.unit
def test_gpu_info_handles_missing_nvidia_smi() -> None:
    """When nvidia-smi is unavailable, gpu_info returns ({}, None) silently."""
    with patch(
        "eval_toolkit.manifest.subprocess.run",
        side_effect=FileNotFoundError("nvidia-smi missing"),
    ):
        info, cuda = gpu_info()
    assert info == {}
    assert cuda is None


@pytest.mark.unit
def test_gpu_info_handles_timeout() -> None:
    with patch(
        "eval_toolkit.manifest.subprocess.run",
        side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
    ):
        info, cuda = gpu_info()
    assert info == {}
    assert cuda is None


@pytest.mark.unit
def test_gpu_info_handles_nonzero_return_code() -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
    with patch("eval_toolkit.manifest.subprocess.run", return_value=completed):
        info, cuda = gpu_info()
    assert info == {}
    assert cuda is None


@pytest.mark.unit
def test_gpu_info_handles_short_csv_row() -> None:
    """nvidia-smi outputting fewer than 3 csv columns returns empty info."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="A100,40960\n", stderr="")
    with patch("eval_toolkit.manifest.subprocess.run", return_value=completed):
        info, cuda = gpu_info()
    assert info == {}
    assert cuda is None


@pytest.mark.unit
def test_gpu_info_handles_non_integer_memory() -> None:
    """nvidia-smi with a non-integer memory column still parses but memory_gb is empty."""
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="A100,not-a-number,535.86\n", stderr=""
    )
    with patch("eval_toolkit.manifest.subprocess.run", return_value=completed):
        info, cuda = gpu_info()
    assert info["name"] == "A100"
    assert info["memory_gb"] == ""
    assert cuda == "535.86"


@pytest.mark.unit
def test_gpu_info_parses_well_formed_output() -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="A100,40960,535.86.10\n", stderr=""
    )
    with patch("eval_toolkit.manifest.subprocess.run", return_value=completed):
        info, cuda = gpu_info()
    assert info["name"] == "A100"
    assert info["count"] == "1"
    assert info["memory_gb"] == "40.0"
    assert cuda == "535.86.10"


@pytest.mark.unit
def test_is_git_dirty_returns_false_when_git_missing() -> None:
    with patch(
        "eval_toolkit.manifest.subprocess.run",
        side_effect=FileNotFoundError("git missing"),
    ):
        assert _is_git_dirty() is False


@pytest.mark.unit
def test_is_git_dirty_returns_false_on_timeout() -> None:
    with patch(
        "eval_toolkit.manifest.subprocess.run",
        side_effect=subprocess.TimeoutExpired("git", 5),
    ):
        assert _is_git_dirty() is False


@pytest.mark.unit
def test_is_git_dirty_returns_true_when_porcelain_nonempty() -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=" M file.py\n", stderr="")
    with patch("eval_toolkit.manifest.subprocess.run", return_value=completed):
        assert _is_git_dirty() is True
