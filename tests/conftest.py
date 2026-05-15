"""Shared pytest configuration for eval-toolkit (test-tree only).

Sets ``matplotlib.use("Agg")`` before any plotting import so headless CI runs do
not try to open a GUI backend (mirrors the prompt_injection_detector and
temporalcv conftest pattern).

Sybil registration for project docs lives in the root-level ``conftest.py``
so it applies to ``README.md`` and ``docs/`` (which sit outside ``tests/``).

The ``balanced_binary_inputs`` / ``imbalanced_binary_inputs`` /
``single_class_inputs`` / ``constant_score_inputs`` fixtures (v0.25.1)
are defined here as a shared scaffold; rollout-to-existing-tests is
deferred to v0.26.0 to keep this release docs-only.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow Agg-backend selection)
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures_after_each_test() -> None:
    """Close all matplotlib figures after each test.

    Hoisted from test_plotting_smoke.py / test_plotting_visual.py in v0.8.3
    to dedupe the autouse fixture and prevent figure leaks across the whole
    test tree (matters once Sybil-rendered plots also hit pyplot state).
    """
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Shared input fixtures (v0.25.1).
#
# Reduce copy-paste of ``np.array([0]*5 + [1]*5)``-style scaffolds that
# currently live in 15+ test files. These fixtures are defined here but
# not yet adopted by existing tests — usage rollout is v0.26.0 scope
# to keep v0.25.1 a pure docs/scaffolding release.
# ---------------------------------------------------------------------------


@pytest.fixture
def balanced_binary_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Balanced y_true (n=200, 50/50) + discriminative y_score.

    Returns
    -------
    (y_true, y_score) where:
        y_true ∈ {0, 1}, exactly 100 of each.
        y_score ≈ 0.5 + 0.3 * (y_true - 0.5) + Gaussian noise, clipped to [0, 1].
        Discriminative but not separable — typical baseline test fixture.
    """
    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    rng.shuffle(y_true)
    y_score = (0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.1, size=200)).clip(0.0, 1.0)
    return y_true, y_score


@pytest.fixture
def imbalanced_binary_inputs() -> tuple[np.ndarray, np.ndarray]:
    """5% positive prevalence (n=500) + discriminative y_score.

    Returns
    -------
    (y_true, y_score) where:
        y_true ∈ {0, 1}, 25 positives / 475 negatives.
        y_score is discriminative; intended for tests that exercise
        rare-positive paths (PR-AUC, lift-at-low-FPR, calibration on
        imbalance).
    """
    rng = np.random.default_rng(42)
    y_true = np.zeros(500, dtype=int)
    y_true[:25] = 1
    rng.shuffle(y_true)
    y_score = (0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.1, size=500)).clip(0.0, 1.0)
    return y_true, y_score


@pytest.fixture
def single_class_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Single-class y_true (all-positive, n=50) + scores in [0, 1]."""
    rng = np.random.default_rng(42)
    y_true = np.ones(50, dtype=int)
    y_score = rng.uniform(0.0, 1.0, size=50)
    return y_true, y_score


@pytest.fixture
def constant_score_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Mixed-class y_true (n=100) with constant y_score = 0.5.

    Useful for testing rank-information-degenerate paths (max-F1 selectors,
    Youden J, PR-AUC tie-handling).
    """
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_score = np.full(100, 0.5, dtype=float)
    return y_true, y_score


# ---------------------------------------------------------------------------
# Shared Scorer test doubles (v0.30.0 refactor #4 — DRY)
#
# The Scorer Protocol is `predict_proba(X) -> np.ndarray`. Three minimal
# implementations cover the common test patterns. Specialized stubs
# (slice-content-sensitive, deterministic-text-score, etc.) stay in their
# own test files because their behavior is the point of the test.
# ---------------------------------------------------------------------------


class StubScorer:
    """Returns a precomputed scores array on every call.

    Use when the test needs deterministic, known outputs (golden tests,
    schema-validation round-trips, etc.).
    """

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict_proba(self, X: object) -> np.ndarray:
        return self._scores


class UniformScorer:
    """Returns ``rng.uniform(0, 1, size=len(X))`` on every call.

    Use when test only needs a valid scorer (input shape correct, output
    in [0, 1]) but doesn't care about specific values.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, X: object) -> np.ndarray:
        return self._rng.uniform(0, 1, size=len(X))  # type: ignore[arg-type]


class ErrorScorer:
    """Raises a configurable exception on every call.

    Use to exercise the harness's ``on_scorer_error`` paths.
    """

    def __init__(
        self,
        exc_type: type[BaseException] = RuntimeError,
        message: str = "intentional failure for tests",
    ) -> None:
        self._exc_type = exc_type
        self._message = message

    def predict_proba(self, X: object) -> np.ndarray:
        raise self._exc_type(self._message)
