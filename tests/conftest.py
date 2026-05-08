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
