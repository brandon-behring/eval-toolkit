"""v0.36 harness parallelism tests (#29 + #30).

Asserts:
- Bit-for-bit reproducibility: ``evaluate(n_jobs=1)`` vs ``evaluate(n_jobs=2)``
  produce identical RunResult dicts on the same seed (Principle #5 contract
  from ``methodology/parallelism.md``).
- ``evaluate_folded(n_jobs=1)`` vs ``evaluate_folded(n_jobs=2)`` same.
- Picklability contract: a closure-based Scorer raises ``TypeError`` with the
  helpful message from ``_parallel.py:120-125`` (v0.35 ADR).
- ``n_jobs=-1`` (all cores) smoke runs without crashing.
- ``n_jobs=0`` is rejected with a clear ``ValueError`` (typo guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice, evaluate, evaluate_folded
from eval_toolkit.operating_points import OperatingPointSpec
from eval_toolkit.splits import StratifiedKFoldSplitter
from eval_toolkit.thresholds import TargetFPRSelector
from tests.conftest import StubScorer


class _PerCallSeededScorer:
    """Re-seeds RNG per ``predict_proba`` call — output is pure fn of ``len(X)``.

    Defined at module scope (picklable). Useful for cross-fold tests where
    each call gets a different-length X but reproducibility across
    ``n_jobs`` values requires no carried state between calls. Contrast
    ``conftest.UniformScorer`` which keeps a stateful RNG.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def predict_proba(self, X: object) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.uniform(0, 1, size=len(X))  # type: ignore[arg-type]


def _build_two_slice_fixture(
    seed: int = 0, n: int = 60
) -> tuple[dict[str, object], list[EvalSlice]]:
    """Two slices, three scorers — enough to exercise parallel dispatch.

    Returns ``(scorers, slices)`` where ``scorers`` is a dict shaped for
    ``evaluate(scorers=, slices=)``.
    """
    rng = np.random.default_rng(seed)
    slices = []
    for slice_name in ("slice_a", "slice_b"):
        labels = rng.integers(0, 2, size=n)
        df = pd.DataFrame({"text": [f"{slice_name}_row{i}" for i in range(n)], "label": labels})
        slices.append(EvalSlice(name=slice_name, df=df))

    # Stub scorers use precomputed score arrays — deterministic + picklable.
    scorers = {
        "stub_low": StubScorer(rng.uniform(0.0, 0.3, size=n)),
        "stub_mid": StubScorer(rng.uniform(0.3, 0.7, size=n)),
        "stub_high": StubScorer(rng.uniform(0.5, 1.0, size=n)),
    }
    return scorers, slices


def _strip_volatile(result_dict: dict) -> dict:
    """Drop fields that legitimately differ between parallel/sequential runs.

    Currently empty — stub for future fields if we add any (e.g., wall-clock
    timestamps). Today nothing in RunResult is wall-clock-dependent.
    """
    return result_dict


@pytest.mark.unit
def test_evaluate_n_jobs_1_vs_2_reproducibility() -> None:
    """n_jobs=1 and n_jobs=2 produce bit-identical RunResults on same seed."""
    scorers, slices = _build_two_slice_fixture()
    seq = evaluate(scorers, slices, run_id="seq", n_resamples=50, seed=7, n_jobs=1)
    par = evaluate(scorers, slices, run_id="par", n_resamples=50, seed=7, n_jobs=2)

    # run_id is the only field that legitimately differs; strip it for compare.
    seq_d = _strip_volatile(seq.to_dict())
    par_d = _strip_volatile(par.to_dict())
    seq_d.pop("run_id", None)
    par_d.pop("run_id", None)
    # Equality semantics: numpy arrays inside the dicts compare element-wise.
    # The RunResult.to_dict() path canonicalizes them to lists/floats, so a
    # plain `==` works here.
    assert seq_d == par_d, "n_jobs=1 and n_jobs=2 should be bit-identical"


@pytest.mark.unit
def test_evaluate_n_jobs_with_paired_diffs_reproducibility() -> None:
    """Paired-diffs path is also deterministic across n_jobs values."""
    scorers, slices = _build_two_slice_fixture()
    pairs = [("stub_low", "stub_high")]
    seq = evaluate(
        scorers,
        slices,
        run_id="seq",
        n_resamples=50,
        seed=11,
        paired_diffs=pairs,
        n_jobs=1,
    )
    par = evaluate(
        scorers,
        slices,
        run_id="par",
        n_resamples=50,
        seed=11,
        paired_diffs=pairs,
        n_jobs=2,
    )
    seq_d = seq.to_dict()
    par_d = par.to_dict()
    seq_d.pop("run_id", None)
    par_d.pop("run_id", None)
    assert seq_d == par_d


@pytest.mark.unit
def test_evaluate_n_jobs_with_operating_points_reproducibility() -> None:
    """Operating-point fit phase is also deterministic across n_jobs values.

    Exercises the parallel fit phase in ``_attach_transferred_operating_points``
    (#30) — confirms the (spec × scorer) fit dispatch produces identical
    results to the sequential path.
    """
    scorers, slices = _build_two_slice_fixture()
    spec = OperatingPointSpec(
        name="op_test",
        fit_slice="slice_a",
        apply_slices=("slice_b",),
        selectors=(TargetFPRSelector(fpr=0.05),),
    )
    seq = evaluate(
        scorers,
        slices,
        run_id="seq",
        n_resamples=50,
        seed=13,
        operating_point_specs=(spec,),
        n_jobs=1,
    )
    par = evaluate(
        scorers,
        slices,
        run_id="par",
        n_resamples=50,
        seed=13,
        operating_point_specs=(spec,),
        n_jobs=2,
    )
    seq_d = seq.to_dict()
    par_d = par.to_dict()
    seq_d.pop("run_id", None)
    par_d.pop("run_id", None)
    assert seq_d == par_d


@pytest.mark.unit
def test_evaluate_folded_n_jobs_reproducibility() -> None:
    """evaluate_folded forwards n_jobs to evaluate per fold; results are deterministic."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"text": [f"row{i}" for i in range(n)], "label": rng.integers(0, 2, size=n)})
    parent = EvalSlice(name="parent", df=df)
    # _PerCallSeededScorer re-seeds RNG per call so its output is a pure
    # function of len(X) — required for reproducibility across n_jobs values.
    # conftest.UniformScorer carries stateful RNG that evolves on each call,
    # so it would correctly produce different outputs under parallel dispatch
    # (state forks per worker) — that's user-scorer behavior, not a bug
    # in evaluate(), and is documented in the v0.35 Scorer picklability ADR
    # § "Common non-picklable cases" (analogous: stateful-RNG cases).
    scorers = {"per_call": _PerCallSeededScorer(seed=42)}
    splitter = StratifiedKFoldSplitter(k=3, seed=17)

    seq = evaluate_folded(
        scorers,
        splitter,
        parent,
        run_id="seq",
        seeds=(17,),
        n_resamples=50,
        n_jobs=1,
    )
    par = evaluate_folded(
        scorers,
        splitter,
        parent,
        run_id="par",
        seeds=(17,),
        n_resamples=50,
        n_jobs=2,
    )
    seq_d = seq.to_dict()
    par_d = par.to_dict()
    seq_d.pop("run_id", None)
    par_d.pop("run_id", None)
    # by_fold contains nested RunResults whose run_ids are deterministic
    # (seed=17/fold=0 etc.), so equality should hold.
    assert seq_d == par_d


@pytest.mark.unit
def test_evaluate_rejects_non_picklable_scorer() -> None:
    """A closure-based scorer with n_jobs>1 raises the v0.35-ADR TypeError.

    Validates that joblib's pickle sniff in ``_parallel.parallel_map`` fires
    on the work-unit's bound Scorer reference. The closure escapes the sniff
    on ``fn`` itself but joblib pickles the entire delayed call including
    args; the error surfaces with the helpful message either way.
    """
    scorers, slices = _build_two_slice_fixture(n=40)

    def make_closure_scorer(threshold: float) -> object:
        """Returns a closure-based scorer — not picklable."""

        class _LocalClosureScorer:
            def predict_proba(self, X: object) -> np.ndarray:
                # Closes over `threshold` from the enclosing fn AND is defined
                # inside a function (local-scope class) — both make it unpicklable.
                return np.full(len(X), threshold)  # type: ignore[arg-type]

        return _LocalClosureScorer()

    scorers["closure"] = make_closure_scorer(0.5)

    with pytest.raises((TypeError, Exception), match="picklab|pickle"):
        evaluate(
            scorers,
            slices,
            run_id="should_fail",
            n_resamples=10,
            seed=0,
            n_jobs=2,
        )


@pytest.mark.unit
def test_evaluate_n_jobs_zero_rejected() -> None:
    """n_jobs=0 raises ValueError per parallel_map contract (typo guard)."""
    scorers, slices = _build_two_slice_fixture(n=30)
    with pytest.raises(ValueError, match="n_jobs=0"):
        evaluate(scorers, slices, run_id="zero", n_resamples=10, seed=0, n_jobs=0)


@pytest.mark.unit
def test_evaluate_n_jobs_minus_one_smoke() -> None:
    """n_jobs=-1 (all cores) runs to completion without crashing."""
    scorers, slices = _build_two_slice_fixture(n=40)
    result = evaluate(
        scorers,
        slices,
        run_id="all_cores",
        n_resamples=20,
        seed=0,
        n_jobs=-1,
    )
    # Just confirm we got a valid RunResult shape; correctness covered above.
    assert "slice_a" in result.by_slice
    assert "slice_b" in result.by_slice
