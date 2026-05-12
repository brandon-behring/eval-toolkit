"""Import-boundary tests for the lazy package root."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


@pytest.mark.unit
def test_import_eval_toolkit_does_not_import_optional_heavy_dependencies() -> None:
    code = """
import json
import sys
import eval_toolkit

heavy = ["pandas", "matplotlib", "pyarrow", "datasets", "torch"]
print(json.dumps({name: name in sys.modules for name in heavy}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=True,
    )

    imported = json.loads(result.stdout)
    assert imported == {
        "pandas": False,
        "matplotlib": False,
        "pyarrow": False,
        "datasets": False,
        "torch": False,
    }


@pytest.mark.unit
def test_lazy_top_level_export_still_resolves() -> None:
    code = """
import json
import sys
import eval_toolkit

_ = eval_toolkit.Scorer
print(json.dumps({
    "has_scorer": hasattr(eval_toolkit.Scorer, "__instancecheck__"),
    "pandas": "pandas" in sys.modules,
    "matplotlib": "matplotlib" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=True,
    )

    observed = json.loads(result.stdout)
    assert observed == {"has_scorer": True, "pandas": False, "matplotlib": False}
