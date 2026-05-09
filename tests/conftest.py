"""Shared pytest configuration for eval-toolkit (test-tree only).

Sets ``matplotlib.use("Agg")`` before any plotting import so headless CI runs do
not try to open a GUI backend (mirrors the prompt_injection_detector and
temporalcv conftest pattern).

Sybil registration for project docs lives in the root-level ``conftest.py``
so it applies to ``README.md`` and ``docs/`` (which sit outside ``tests/``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow Agg-backend selection)
import pytest


@pytest.fixture(autouse=True)
def _close_figures_after_each_test() -> None:
    """Close all matplotlib figures after each test.

    Hoisted from test_plotting_smoke.py / test_plotting_visual.py in v0.8.3
    to dedupe the autouse fixture and prevent figure leaks across the whole
    test tree (matters once Sybil-rendered plots also hit pyplot state).
    """
    yield
    plt.close("all")
