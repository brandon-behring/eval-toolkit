"""Coverage-targeted tests for harness + seeds error paths and defensive code.

Extracted from the v0.27.x-era ``test_coverage_gap.py`` during the
v0.30.1 hygiene split — every assertion preserved verbatim; only the
file boundary changed.

Pairs with the happy-path coverage in ``test_harness_smoke.py`` and
the edge cases in ``test_harness_edge_cases.py`` /
``test_harness_fault_injection.py``. Targets:

- Scorer Protocol duck-typing acceptance,
- EvalSlice.strata empty / array branches,
- ``seeds.set_global_seeds`` validation paths.

Seeds tests fold in here because there are only 2 of them — too few for
a standalone ``test_coverage_seeds.py`` to be worth its own file.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.harness import EvalSlice, Scorer

# ---------------------------------------------------------------------------
# harness: edge cases + Scorer protocol exercise
# ---------------------------------------------------------------------------


class _ConstantScorer:
    """Scorer Protocol implementation returning a constant score."""

    def __init__(self, value: float = 0.5) -> None:
        self._value = value

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return np.full(len(X), self._value, dtype=float)


@pytest.mark.unit
def test_scorer_protocol_accepts_duck_typed() -> None:
    scorer: Scorer = _ConstantScorer(0.7)
    assert isinstance(scorer.predict_proba(["a", "b", "c"]), np.ndarray)


@pytest.mark.unit
def test_eval_slice_strata_returns_none_when_unset() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    sl = EvalSlice(name="t", df=df)
    assert sl.strata is None


@pytest.mark.unit
def test_eval_slice_strata_returns_array() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1], "stratum": ["x", "y"]})
    sl = EvalSlice(name="t", df=df, strata_col="stratum")
    arr = sl.strata
    assert arr is not None and arr.shape == (2,)


# ---------------------------------------------------------------------------
# seeds: exercise the public branch; torch branch optional
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_global_seeds_makes_numpy_deterministic() -> None:
    from eval_toolkit.seeds import set_global_seeds

    set_global_seeds(42)
    a = np.random.rand(10)
    set_global_seeds(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_set_global_seeds_rejects_invalid() -> None:
    from eval_toolkit.seeds import set_global_seeds

    with pytest.raises(TypeError, match="int"):
        set_global_seeds("42")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seeds(-1)
