"""Local-runnable nox session config for eval-toolkit.

Mirrors the tox.ini matrix using nox's Python-API style. Contributors who
prefer nox over tox can run `nox` to execute the same multi-Python test +
lint + type matrix that tox.ini defines.

Sessions
--------
- ``tests``: pytest + coverage gate on each Python interpreter available
- ``lint``: ruff + black --check
- ``type``: mypy strict
- ``doctest``: pytest --doctest-modules on math kernels
- ``visual``: pytest-mpl baseline validation
"""

from __future__ import annotations

from pathlib import Path

import nox

nox.options.sessions = ["tests", "lint", "type", "doctest"]
nox.options.reuse_existing_virtualenvs = True

PY_VERSIONS = ["3.11", "3.12", "3.13"]

# Canonical doctest module list shared with Makefile, tox.ini, and ci.yml.
DOCTEST_MODULES = Path(".doctest-modules").read_text().strip().split("\n")


@nox.session(python=PY_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the full pytest suite with coverage gate."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--cov=eval_toolkit", "--cov-fail-under=92", *session.posargs)


@nox.session(python=False)
def fast(session: nox.Session) -> None:
    """Run the fast subset (skip @pytest.mark.slow) for the local iteration loop.

    CI runs the full suite; this session is for developer-side velocity.
    """
    session.run("uv", "run", "pytest", "-m", "not slow", "-q", external=True)


@nox.session
def lint(session: nox.Session) -> None:
    """ruff + black --check."""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "tests")
    session.run("black", "--check", "src", "tests")


@nox.session
def type(session: nox.Session) -> None:
    """mypy strict on src/."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src")


@nox.session
def doctest(session: nox.Session) -> None:
    """Run doctests on the math-kernel + utility modules."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--doctest-modules", *DOCTEST_MODULES)


@nox.session
def visual(session: nox.Session) -> None:
    """pytest-mpl visual-regression check against baseline PNGs."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--mpl", "tests/test_plotting_visual.py")
