"""Tests for private helpers extracted from ``harness.py`` in v0.27.0.

Per the /exploring-options test-scope decision (Option 3 — minimal,
behavioral-invariants only): this file covers invariants that are
*easier to assert at the helper level than via the public API*. Happy
paths for the helpers remain covered by the existing public-API tests
(``test_harness_v22.py``, ``test_harness_v07.py``,
``test_harness_smoke.py``) — duplicating them here would couple the
test suite to private signatures without information gain.

Scope:

- Skip-condition branching in ``_compute_paired_diffs`` — hard to
  trigger via the public ``evaluate()`` API because each branch needs
  a tailored slice + scores arrangement; trivial at the helper level.
- ``_resolve_y_score`` regression-guards for the v0.27.0 carve-outs
  (``MemoryError`` / ``AssertionError`` propagate even in
  ``on_scorer_error='record'`` mode).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice, _compute_paired_diffs, _resolve_y_score

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _UniformScorer:
    """Deterministic scorer for tests — returns shuffled uniforms keyed on seed=42."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=len(X))


class _MemoryErrorScorer:
    """Raises MemoryError on every call (simulates OOM / resource exhaustion)."""

    def predict_proba(self, X: list[str]) -> np.ndarray:  # noqa: ARG002
        raise MemoryError("simulated OOM during predict_proba")


class _AssertionErrorScorer:
    """Raises AssertionError on every call (simulates internal-invariant violation)."""

    def predict_proba(self, X: list[str]) -> np.ndarray:  # noqa: ARG002
        raise AssertionError("simulated invariant violation")


def _balanced_slice(n: int = 60, name: str = "test") -> EvalSlice:
    """n rows, balanced 0/1 labels (label column = i % 2)."""
    df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": [i % 2 for i in range(n)]})
    return EvalSlice(name=name, df=df)


def _single_class_slice(n: int = 60, name: str = "test") -> EvalSlice:
    """n rows, all positive (label=1)."""
    df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": [1] * n})
    return EvalSlice(name=name, df=df)


# ---------------------------------------------------------------------------
# _compute_paired_diffs — skip-condition branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "case,slice_factory,scores_in_cache,expected_reason",
    [
        # Scorer present in `scorers` but missing scores_by_scorer (skipped/errored
        # for this slice). Constructed by populating only one of the two scorers'
        # scores in the cache.
        (
            "scorer_skipped_this_slice",
            _balanced_slice,
            ("a_only",),
            "one or both scorers skipped this slice",
        ),
        # Single-class slice: PR-AUC Δ is degenerate.
        (
            "single_class",
            _single_class_slice,
            ("both",),
            "single-class slice; PR-AUC Δ degenerate",
        ),
    ],
)
def test_compute_paired_diffs_skip_reasons(
    case: str,
    slice_factory,
    scores_in_cache: tuple[str, ...],
    expected_reason: str,
) -> None:
    """Each skip branch records a single ``{'skipped': <reason>}`` entry."""
    slice_ = slice_factory()
    scorer_a = _UniformScorer()
    scorer_b = _UniformScorer()
    scorers = {"a": scorer_a, "b": scorer_b}

    # Build the scores_by_scorer dict per the test case.
    if scores_in_cache == ("a_only",):
        scores_by_scorer = {"a": np.zeros(len(slice_.df), dtype=np.float64)}
    else:
        scores_by_scorer = {
            "a": np.linspace(0.0, 1.0, len(slice_.df), dtype=np.float64),
            "b": np.linspace(1.0, 0.0, len(slice_.df), dtype=np.float64),
        }

    diffs = _compute_paired_diffs(
        slice_,
        scores_by_scorer,
        scorers,
        [("a", "b")],
        n_resamples=10,
        seed=0,
    )
    assert "b_minus_a" in diffs, f"case={case}: missing entry"
    assert diffs["b_minus_a"] == {"skipped": expected_reason}, f"case={case}: wrong reason"


@pytest.mark.unit
def test_compute_paired_diffs_skip_small_n() -> None:
    """n < 30 produces a ``{'skipped': 'n=N < 30'}`` entry with the actual N."""
    slice_ = _balanced_slice(n=10)  # below the 30-row threshold
    scorers = {"a": _UniformScorer(), "b": _UniformScorer()}
    scores_by_scorer = {
        "a": np.linspace(0.0, 1.0, 10, dtype=np.float64),
        "b": np.linspace(1.0, 0.0, 10, dtype=np.float64),
    }
    diffs = _compute_paired_diffs(
        slice_,
        scores_by_scorer,
        scorers,
        [("a", "b")],
        n_resamples=10,
        seed=0,
    )
    assert diffs["b_minus_a"] == {"skipped": "n=10 < 30"}


# ---------------------------------------------------------------------------
# _resolve_y_score — v0.27.0 exception carve-outs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_y_score_propagates_memory_error_in_record_mode() -> None:
    """``MemoryError`` propagates even with ``on_scorer_error='record'``.

    OOM signals an environment failure, not a scorer bug — recording it
    as a per-scorer error would mask the actual cause and silently
    continue a run that should abort.
    """
    slice_ = _balanced_slice()
    with pytest.raises(MemoryError, match="simulated OOM"):
        _resolve_y_score(
            _MemoryErrorScorer(),
            slice_,
            precomputed_scores=None,
            on_scorer_error="record",
            attack_style=None,
        )


@pytest.mark.unit
def test_resolve_y_score_propagates_assertion_error_in_record_mode() -> None:
    """``AssertionError`` propagates even with ``on_scorer_error='record'``.

    Internal-invariant violations should surface loudly; recording them
    would silently continue past a broken invariant.
    """
    slice_ = _balanced_slice()
    with pytest.raises(AssertionError, match="simulated invariant violation"):
        _resolve_y_score(
            _AssertionErrorScorer(),
            slice_,
            precomputed_scores=None,
            on_scorer_error="record",
            attack_style=None,
        )
