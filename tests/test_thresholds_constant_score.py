"""Constant-score behavior coverage for ``eval_toolkit.thresholds`` (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``thresholds.py:108`` and ``:125`` raise RuntimeError when sklearn's
``precision_recall_curve`` / ``roc_curve`` returns zero thresholds —
a defensive path documented as triggering on constant ``y_score``.
Existing property tests in ``test_thresholds_props.py:38,80`` use
``try / except RuntimeError: return`` to silently skip the case.

Empirical investigation during v0.26.0 found that sklearn returns
**at least one threshold** (the constant value itself) even on a
constant ``y_score`` — so the defensive `len(thresholds) == 0`
guard is unreachable through normal API use. This test module
therefore documents the actual behavior in two parts:

1. **Positive coverage**: ``MaxF1Selector`` and ``YoudenJSelector``
   handle constant ``y_score`` without raising — they return a
   valid threshold at the constant value.
2. **Defensive-path coverage via monkeypatch**: ``_pr_curve_trim``
   and ``_roc_curve_trim`` raise RuntimeError if sklearn ever
   returns zero thresholds (mocked via monkeypatch). This pins the
   error contract so a future refactor that drops the defensive
   check would fail the suite.

Together these replace the ``try/except: return`` skip pattern with
explicit assertions about the actual behavior — both the normal
constant-score path and the defensive guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit import thresholds as thr_mod
from eval_toolkit.thresholds import MaxF1Selector, YoudenJSelector

# ---------------------------------------------------------------------------
# Part 1: positive coverage — constant scores produce a valid result.
# ---------------------------------------------------------------------------


def _constant_score_inputs(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Mixed-class y_true with constant y_score = 0.5."""
    y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int)
    s = np.full(n, 0.5)
    return y, s


def test_max_f1_selector_handles_constant_score_without_raising() -> None:
    """``MaxF1Selector.select(y, constant_s)`` returns a valid result.

    Pins the actual sklearn 1.x behavior: constant ``y_score`` produces
    a single threshold at the constant value, so MaxF1Selector returns
    a degenerate-but-valid ``ThresholdResult`` (all rows predicted
    positive, F1 = 2·prevalence/(1+prevalence) = 0.667 at 50%
    prevalence).
    """
    y, s = _constant_score_inputs(n=8)
    result = MaxF1Selector().select(y, s)
    assert result.threshold == pytest.approx(0.5), (
        f"Constant-score MaxF1 should return threshold at the constant value (0.5); "
        f"got {result.threshold}"
    )
    # Recall is 1.0 (all predicted positive); precision is the prevalence.
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(0.5)


def test_youden_j_selector_handles_constant_score_without_raising() -> None:
    """``YoudenJSelector.select(y, constant_s)`` returns a valid result.

    Same as the MaxF1 test: sklearn produces a usable curve from
    constant scores, so the selector does not raise.
    """
    y, s = _constant_score_inputs(n=8)
    result = YoudenJSelector().select(y, s)
    assert np.isfinite(result.threshold)
    assert result.recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Part 2: defensive-path coverage — monkeypatch to force zero thresholds.
# ---------------------------------------------------------------------------


def test_pr_curve_trim_raises_runtimeerror_on_zero_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_pr_curve_trim`` raises RuntimeError if sklearn returns 0 thresholds.

    Covers ``thresholds.py:108``. The defensive guard fires only if
    ``precision_recall_curve`` returns an empty thresholds array —
    unreachable through normal API use on sklearn 1.x but pinned here
    so a future refactor that drops the guard would fail the suite.
    """

    def _fake_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, ...]:
        # Return shape that triggers the `len(thresholds) == 0` branch.
        return np.array([1.0]), np.array([0.0]), np.array([])

    monkeypatch.setattr(thr_mod, "precision_recall_curve", _fake_pr_curve)
    y, s = _constant_score_inputs(n=8)
    with pytest.raises(RuntimeError, match="PR curve has no thresholds.*y_score may be constant"):
        thr_mod._pr_curve_trim(y, s)


def test_roc_curve_trim_raises_runtimeerror_on_zero_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_roc_curve_trim`` raises RuntimeError if sklearn returns 0 thresholds.

    Covers ``thresholds.py:125``. Same monkeypatch pattern as the PR
    case above; pins the defensive guard's error contract.
    """

    def _fake_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, ...]:
        return np.array([0.0]), np.array([0.0]), np.array([])

    monkeypatch.setattr(thr_mod, "roc_curve", _fake_roc_curve)
    y, s = _constant_score_inputs(n=8)
    with pytest.raises(RuntimeError, match="ROC curve has no thresholds.*y_score may be constant"):
        thr_mod._roc_curve_trim(y, s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
