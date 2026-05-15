"""Tests for the ``eval-toolkit`` CLI (``python -m eval_toolkit ...``).

Covers all four subcommands (``schemas list``, ``schemas show``,
``schemas check``, ``validate``) plus the missing-optional-dependency
degrade path for ``validate``. Mix of subprocess-based smoke tests
(for end-to-end happy paths) and in-process ``main([...])`` calls
(for finer-grained error-mocking).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from eval_toolkit.__main__ import main


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m eval_toolkit ...`` and capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, "-m", "eval_toolkit", *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# schemas list
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schemas_list_outputs_all_bundled_schemas() -> None:
    result = _run_cli("schemas", "list")
    assert result.returncode == 0
    names = result.stdout.strip().split("\n")
    assert "results.v1" in names
    assert "results_full.v1" in names
    assert "manifest.v1" in names


# ---------------------------------------------------------------------------
# schemas show
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schemas_show_returns_valid_json() -> None:
    result = _run_cli("schemas", "show", "results.v1")
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed["$schema"].startswith("https://json-schema.org")
    assert parsed["title"].startswith("eval-toolkit")


@pytest.mark.unit
def test_schemas_show_tolerates_full_filename() -> None:
    """Passing 'results.v1.json' (full filename) works like 'results.v1'."""
    result = _run_cli("schemas", "show", "results.v1.json")
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert "title" in parsed


@pytest.mark.unit
def test_schemas_show_rejects_unknown_schema() -> None:
    result = _run_cli("schemas", "show", "definitely_not_a_real_schema")
    assert result.returncode == 2
    assert "unknown schema" in result.stderr


@pytest.mark.unit
def test_schemas_without_subcommand_prints_help_and_exits_nonzero() -> None:
    """argparse requires a sub-subcommand under 'schemas'."""
    result = _run_cli("schemas")
    assert result.returncode != 0
    # argparse writes 'required' / 'invalid' messages to stderr.
    assert result.stderr  # non-empty


# ---------------------------------------------------------------------------
# schemas check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schemas_check_bundled_pass() -> None:
    """All bundled schemas meta-validate against Draft 2020-12."""
    result = _run_cli("schemas", "check")
    assert result.returncode == 0
    # One ``  <name>.json: OK`` line per bundled schema.
    assert "results.v1.json: OK" in result.stdout
    assert "manifest.v1.json: OK" in result.stdout


@pytest.mark.unit
def test_main_in_process_schemas_check_happy_path(capsys) -> None:
    rc = main(["schemas", "check"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "results.v1.json: OK" in captured.out


@pytest.mark.unit
def test_schemas_check_empty_directory_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """Empty schemas directory is a sanity-failure (e.g., accidental deletion)."""
    from eval_toolkit import __main__ as cli

    monkeypatch.setattr(cli, "_schemas_dir", lambda: tmp_path)
    rc = main(["schemas", "check"])
    assert rc == 2
    assert "no schemas found" in capsys.readouterr().err


@pytest.mark.unit
def test_schemas_check_malformed_schema_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """A bundled schema that fails Draft 2020-12 meta-validation exits 1."""
    from eval_toolkit import __main__ as cli

    bad = tmp_path / "broken.v1.json"
    # ``type: "notARealType"`` is rejected by the Draft 2020-12 meta-schema.
    bad.write_text(
        json.dumps(
            {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "notARealType"}
        )
    )
    monkeypatch.setattr(cli, "_schemas_dir", lambda: tmp_path)
    rc = main(["schemas", "check"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Schema validation failures" in err
    assert "broken.v1.json" in err


# ---------------------------------------------------------------------------
# validate (happy path + error branches)
# ---------------------------------------------------------------------------


def _well_formed_results_payload() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "run_id": "demo",
        "config": {"n_resamples": 100},
        "by_slice": {
            "dev": {
                "n": 100,
                "n_positive": 50,
                "by_scorer": {"model": {"pr_auc": 0.82}},
            }
        },
    }


@pytest.mark.unit
def test_validate_well_formed_results_payload(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _well_formed_results_payload()
    path = tmp_path / "good.json"
    path.write_text(json.dumps(payload))
    result = _run_cli("validate", str(path), "results.v1")
    assert result.returncode == 0
    assert "OK against results.v1" in result.stdout


@pytest.mark.unit
def test_validate_bad_payload_returns_validation_error(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    # Missing every required field.
    path = tmp_path / "bad.json"
    path.write_text("{}")
    result = _run_cli("validate", str(path), "results.v1")
    assert result.returncode == 1
    assert "VALIDATION ERROR" in result.stderr


@pytest.mark.unit
def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    result = _run_cli("validate", str(tmp_path / "nope.json"), "results.v1")
    assert result.returncode == 2
    assert "file not found" in result.stderr


@pytest.mark.unit
def test_validate_rejects_unknown_schema(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    path = tmp_path / "x.json"
    path.write_text("{}")
    result = _run_cli("validate", str(path), "definitely_not_a_schema")
    assert result.returncode == 2
    assert "unknown schema" in result.stderr


@pytest.mark.unit
def test_validate_degrades_when_jsonschema_missing(monkeypatch, tmp_path: Path) -> None:
    """In-process call with jsonschema-import-blocked exits 3 with a helpful message."""
    # Force import of jsonschema to fail inside _cmd_validate.
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    path = tmp_path / "x.json"
    path.write_text("{}")
    captured: list[str] = []
    original_stderr_write = sys.stderr.write

    def _capture(msg: str) -> int:
        captured.append(msg)
        return original_stderr_write(msg)

    monkeypatch.setattr(sys.stderr, "write", _capture)
    rc = main(["validate", str(path), "results.v1"])
    assert rc == 3
    joined = "".join(captured)
    assert "[validation]" in joined or "validate requires" in joined


# ---------------------------------------------------------------------------
# In-process direct-call smoke (cheaper than subprocess; complements above)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_in_process_list_round_trip(capsys) -> None:
    rc = main(["schemas", "list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "results.v1" in captured.out


@pytest.mark.unit
def test_main_in_process_show_rejects_unknown(capsys) -> None:
    rc = main(["schemas", "show", "nope"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown schema" in captured.err


@pytest.mark.unit
def test_main_in_process_show_happy_path(capsys) -> None:
    rc = main(["schemas", "show", "results.v1"])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["title"].startswith("eval-toolkit")


@pytest.mark.unit
def test_main_in_process_show_full_filename(capsys) -> None:
    """`results.v1.json` (full filename) is accepted."""
    rc = main(["schemas", "show", "results.v1.json"])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["title"]


@pytest.mark.unit
def test_main_in_process_validate_happy_path(tmp_path: Path, capsys) -> None:
    pytest.importorskip("jsonschema")
    payload = _well_formed_results_payload()
    path = tmp_path / "good.json"
    path.write_text(json.dumps(payload))
    rc = main(["validate", str(path), "results.v1"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "OK against results.v1" in captured.out


@pytest.mark.unit
def test_main_in_process_validate_full_filename_schema(tmp_path: Path, capsys) -> None:
    """validate accepts the full schema filename like 'results.v1.json'."""
    pytest.importorskip("jsonschema")
    path = tmp_path / "good.json"
    path.write_text(json.dumps(_well_formed_results_payload()))
    rc = main(["validate", str(path), "results.v1.json"])
    assert rc == 0


@pytest.mark.unit
def test_main_in_process_validate_bad_payload(tmp_path: Path, capsys) -> None:
    pytest.importorskip("jsonschema")
    path = tmp_path / "bad.json"
    path.write_text("{}")
    rc = main(["validate", str(path), "results.v1"])
    assert rc == 1
    assert "VALIDATION ERROR" in capsys.readouterr().err


@pytest.mark.unit
def test_main_in_process_validate_missing_file(tmp_path: Path, capsys) -> None:
    pytest.importorskip("jsonschema")
    rc = main(["validate", str(tmp_path / "nope.json"), "results.v1"])
    assert rc == 2
    assert "file not found" in capsys.readouterr().err


@pytest.mark.unit
def test_main_in_process_validate_unknown_schema(tmp_path: Path, capsys) -> None:
    pytest.importorskip("jsonschema")
    path = tmp_path / "x.json"
    path.write_text("{}")
    rc = main(["validate", str(path), "definitely_not_a_schema"])
    assert rc == 2
    assert "unknown schema" in capsys.readouterr().err
