"""Root-level pytest configuration: Sybil doc-block testing.

Registers Sybil here (rather than under ``tests/``) so its
``pytest_collect_file`` hook applies project-wide and picks up
``README.md`` and ``docs/`` — which sit outside ``[tool.pytest.ini_options]
testpaths``. Doc files are added to ``testpaths`` in ``pyproject.toml`` so
pytest descends into them; the glob filter below restricts Sybil to files
that actually exist (so partial v0.7.0 commits stay collectible).

Test-tree-only configuration (matplotlib backend etc.) lives in
``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

_REPO_ROOT = Path(__file__).resolve().parent


def _existing(*relative_globs: str) -> list[str]:
    """Return only the relative globs whose matched files actually exist.

    Sybil's ``patterns`` argument errors out if any pattern matches zero
    files; this filter lets the doc tree grow incrementally without
    breaking collection on early commits.
    """
    keep: list[str] = []
    for pattern in relative_globs:
        if list(_REPO_ROOT.glob(pattern)):
            keep.append(pattern)
    return keep


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    patterns=_existing(
        "README.md",
        "docs/extending.md",
        "docs/methodology/*.md",
        "docs/examples/*.md",
    ),
).pytest()
