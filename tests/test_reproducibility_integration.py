"""End-to-end reproducibility integration test (v0.25.0).

Validates the dossier claim from `eval-ecosystem/_dossier/02_reproducibility_and_metadata.md`
§ B1 (Pineau et al. 2021) at the integration level: a seeded harness
run produces bit-identical metric output when re-run with the same
seed, and a different seed produces a different output.

Existing tests in `tests/test_seeds.py` verify RNG-level determinism
(numpy, random, torch); this file lifts that guarantee to the
*harness output* level — the metric dict that downstream consumers
actually serialize and gate on.

Per round-4 Q3 of the planning rounds: assertions use ``np.array_equal``
(strict bit-identity). If a real BLAS-determinism issue surfaces, the
test should fail loudly — that's a signal worth investigating, not
papering over with a tolerance bump.

References
----------
Pineau, J. et al. "Improving reproducibility in machine learning
research (a report from the NeurIPS 2019 reproducibility program)."
JMLR 22, 2021. arXiv:2003.12206.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice, evaluate_scorer_on_slice


class _DeterministicScorer:
    """Returns a fixed score array — no internal randomness."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores.astype(float).copy()

    def predict_proba(self, X: object) -> np.ndarray:
        return self._scores.copy()


def _build_fixture(n: int = 200, seed: int = 0) -> tuple[EvalSlice, _DeterministicScorer]:
    """Synthetic mixed-class slice + a deterministic scorer.

    The scorer returns a fixed score vector (not a function of X) so
    any non-determinism in the resulting metric dict comes from the
    bootstrap RNG, not from the scorer itself.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Discriminative scores — y=1 rows skewed high, y=0 skewed low
    base = rng.uniform(0.0, 1.0, size=n)
    scores = (0.4 * base + 0.6 * y).clip(0.0, 1.0)
    df = pd.DataFrame({"text": [f"row{i}" for i in range(n)], "label": y})
    return EvalSlice(name="repro_test_slice", df=df), _DeterministicScorer(scores)


def _walk_floats(obj: object) -> list[tuple[str, object]]:
    """Recursively flatten a metric dict into (path, value) pairs.

    Yields any leaf that is a float, int, np.floating, np.integer, np.ndarray,
    or list of numerics. Strings, None, dicts, and other types are walked
    or skipped. The path string is for diagnostic clarity in assertion
    messages.
    """
    out: list[tuple[str, object]] = []

    def _walk(prefix: str, val: object) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                _walk(f"{prefix}[{i}]", v)
        elif isinstance(val, (int | float | np.floating | np.integer | np.ndarray)):
            out.append((prefix, val))

    _walk("", obj)
    return out


def _assert_bit_identical(run1: dict[str, object], run2: dict[str, object]) -> None:
    """Strict bit-identity check on every numeric leaf via np.array_equal.

    Uses ``equal_nan=True`` so that two runs producing identical NaN
    values (e.g., a degenerate bootstrap CI returning NaN on both
    seeded re-runs) count as bit-identical. This preserves strict
    equality for finite floats — NaN handling is the only relaxation.
    """
    leaves1 = dict(_walk_floats(run1))
    leaves2 = dict(_walk_floats(run2))
    assert leaves1.keys() == leaves2.keys(), (
        f"Result dicts have different numeric-leaf keys: "
        f"only-in-run1={leaves1.keys() - leaves2.keys()}, "
        f"only-in-run2={leaves2.keys() - leaves1.keys()}"
    )
    mismatches: list[str] = []
    for key in sorted(leaves1):
        v1 = np.asarray(leaves1[key])
        v2 = np.asarray(leaves2[key])
        if not np.array_equal(v1, v2, equal_nan=True):
            mismatches.append(f"  {key}: run1={v1!r}, run2={v2!r}")
    if mismatches:
        raise AssertionError(
            "Bit-identity violated on the following numeric leaves "
            "(np.array_equal mismatch):\n" + "\n".join(mismatches[:20])
        )


def test_evaluate_scorer_on_slice_bit_identical_under_replay() -> None:
    """Seeded re-run produces bit-identical metric dict output (Pineau 2021).

    Two invocations of evaluate_scorer_on_slice with the same scorer,
    the same slice, the same seed, and the same kwargs must produce a
    metric dict whose every numeric leaf passes np.array_equal.

    A real failure here would indicate non-deterministic state somewhere
    in the harness path (BLAS reductions, scipy.optimize local minima,
    multi-threaded numpy reductions). Surface such failures rather than
    tolerating them.
    """
    slice_, scorer = _build_fixture(n=200, seed=0)

    # Two independent runs at the same seed
    result1 = evaluate_scorer_on_slice(
        scorer,
        slice_,
        rng=42,
        n_resamples=200,
        bootstrap_roc_auc=True,
        compute_mce=True,
        compute_brier=True,
    )
    result2 = evaluate_scorer_on_slice(
        scorer,
        slice_,
        rng=42,
        n_resamples=200,
        bootstrap_roc_auc=True,
        compute_mce=True,
        compute_brier=True,
    )

    _assert_bit_identical(result1, result2)


def test_evaluate_scorer_on_slice_different_seed_changes_output() -> None:
    """Negative control: a different seed produces measurably different output.

    Without this control, a constant-output bug would silently pass the
    bit-identity test above. We require at least one numeric leaf to
    differ between rng=42 and rng=43.
    """
    slice_, scorer = _build_fixture(n=200, seed=0)

    result_a = evaluate_scorer_on_slice(
        scorer, slice_, rng=42, n_resamples=200, bootstrap_roc_auc=True
    )
    result_b = evaluate_scorer_on_slice(
        scorer, slice_, rng=43, n_resamples=200, bootstrap_roc_auc=True
    )

    leaves_a = dict(_walk_floats(result_a))
    leaves_b = dict(_walk_floats(result_b))
    assert leaves_a.keys() == leaves_b.keys()
    # At least one CI bound or bootstrap-derived stat must differ
    differing = [
        k for k in leaves_a if not np.array_equal(np.asarray(leaves_a[k]), np.asarray(leaves_b[k]))
    ]
    assert differing, (
        "Expected at least one numeric leaf to differ between rng=42 and rng=43; "
        "either the harness ignores its seed parameter (bug) or the slice has zero "
        "bootstrap variance."
    )


def test_evaluate_scorer_on_slice_replay_idempotent_after_other_calls() -> None:
    """Bit-identity holds even when other harness calls happen in between.

    Guards against accidental global-state leakage (e.g., a module-level
    RNG that gets mutated by sibling calls). The same fixture+seed
    should produce the same output regardless of what else has been
    called.
    """
    slice_, scorer = _build_fixture(n=150, seed=0)

    result_first = evaluate_scorer_on_slice(scorer, slice_, rng=42, n_resamples=100)
    # Inject sibling calls that touch numpy RNG / harness internals
    other_slice, other_scorer = _build_fixture(n=80, seed=999)
    _ = evaluate_scorer_on_slice(other_scorer, other_slice, rng=7, n_resamples=50)
    _ = evaluate_scorer_on_slice(other_scorer, other_slice, rng=11, n_resamples=50)

    result_after = evaluate_scorer_on_slice(scorer, slice_, rng=42, n_resamples=100)

    _assert_bit_identical(result_first, result_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
