"""Shared pytest configuration for eval-toolkit.

Sets ``matplotlib.use("Agg")`` before any plotting import so headless CI runs do
not try to open a GUI backend (mirrors the prompt_injection_detector and
temporalcv conftest pattern).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
