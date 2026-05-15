"""Multi-slice fault-injection tests for the harness's on_scorer_error="record" path.

Existing tests in ``test_harness_v07.py`` cover the single-slice case (one
broken scorer, one slice). These tests target a subtler scenario: a scorer
that succeeds on some slices and fails on others. The failure mode they
guard against is **error-state bleed**: a scorer error on slice A
contaminating the metrics block for slice B.

The harness's ``evaluate_scorer_on_slice`` is supposed to compute per-(slice,
scorer) independently; if a refactor accidentally shared a cache or used a
module-level mutable state, slice B's metrics would carry forward the slice
A error. These tests assert independence.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice, evaluate

FAIL_MARKER = "TRIGGER_FAILURE"


class _SliceContentSensitiveScorer:
    """Raises ``RuntimeError`` on any slice containing a FAIL_MARKER feature.

    Implements the :class:`~eval_toolkit.protocols.Scorer` protocol. Decides
    behavior by inspecting the features of the slice it was called with —
    so it can succeed on some slices and fail on others in the same
    ``evaluate()`` call.
    """

    def predict_proba(self, X: object) -> np.ndarray:
        items: Iterable[object] = X  # type: ignore[assignment]
        for x in items:
            if isinstance(x, str) and FAIL_MARKER in x:
                raise RuntimeError(f"Encountered {FAIL_MARKER!r} feature; refusing to score")
        return np.full(len(list(X)) if not hasattr(X, "__len__") else len(X), 0.5)  # type: ignore[arg-type]


def _make_slice(name: str, n: int, *, with_fail_marker: bool, seed: int = 42) -> EvalSlice:
    """Build a synthetic EvalSlice with ``n`` rows, optionally tainted with FAIL_MARKER."""
    rng = np.random.default_rng(seed)
    features = [f"slice_{name}_row_{i}" for i in range(n)]
    if with_fail_marker:
        # Sprinkle the marker so the scorer raises
        features[n // 2] = FAIL_MARKER + "_" + features[n // 2]
    df = pd.DataFrame({"text": features, "label": rng.integers(0, 2, size=n)})
    return EvalSlice(name=name, df=df)


@pytest.mark.unit
def test_multi_slice_fault_injection_records_errors_per_slice() -> None:
    """Scorer fails on slices A and C, succeeds on B. All three are recorded with their own state.

    Verifies the harness's per-(slice, scorer) independence: failures on A
    and C must NOT contaminate B's by_scorer block. If a refactor shared a
    cache or wrote per-scorer state across slices, B's metrics would carry
    forward the A error or vice versa.
    """
    slice_a = _make_slice("A", n=80, with_fail_marker=True)
    slice_b = _make_slice("B", n=80, with_fail_marker=False)
    slice_c = _make_slice("C", n=80, with_fail_marker=True)

    result = evaluate(
        scorers={"trip": _SliceContentSensitiveScorer()},
        slices=[slice_a, slice_b, slice_c],
        run_id="fault-injection",
        n_resamples=20,
        on_scorer_error="record",
    )

    # All three slices present in output
    assert set(result.by_slice.keys()) == {"A", "B", "C"}

    # Slices A and C must have error markers under the "trip" scorer
    for failing in ("A", "C"):
        entry = result.by_slice[failing]["by_scorer"]["trip"]
        assert "error" in entry, f"Slice {failing} should have an error marker; got {entry!r}"
        assert (
            FAIL_MARKER in entry["error"]
        ), f"Slice {failing} error message should mention {FAIL_MARKER}; got {entry['error']!r}"
        assert entry["exc_type"] == "RuntimeError"
        assert "traceback" in entry
        assert entry["scores"] == []

    # Slice B must have normal metrics — no error marker, no leakage from A or C
    entry_b = result.by_slice["B"]["by_scorer"]["trip"]
    assert (
        "error" not in entry_b
    ), f"Slice B should be clean; got error entry: {entry_b.get('error')}"
    assert "exc_type" not in entry_b
    # Constant-score (0.5) scorer: pr_auc should still be a finite float; B was not erroring
    pr_auc_value = entry_b.get("pr_auc")
    assert isinstance(
        pr_auc_value, float
    ), f"Slice B should have a numeric pr_auc; got {pr_auc_value!r} (type={type(pr_auc_value).__name__})"


@pytest.mark.unit
def test_multi_slice_fault_injection_healthy_scorer_unaffected_by_failing_scorer() -> None:
    """Two scorers on three slices; one scorer fails on A+C but the healthy scorer's
    B-slice metrics must be identical to a no-fail-injection control.

    Catches a cache- or state-bleed between scorers AND slices: even when
    one scorer fails on some slices, every other scorer's per-slice output
    must be unchanged.
    """
    slice_a = _make_slice("A", n=80, with_fail_marker=True)
    slice_b = _make_slice("B", n=80, with_fail_marker=False)
    slice_c = _make_slice("C", n=80, with_fail_marker=True)

    class _ConstantScorer:
        def predict_proba(self, X: object) -> np.ndarray:
            return np.full(len(X), 0.7)  # type: ignore[arg-type]

    # Run with fault injection
    result_with_fault = evaluate(
        scorers={
            "trip": _SliceContentSensitiveScorer(),
            "healthy": _ConstantScorer(),
        },
        slices=[slice_a, slice_b, slice_c],
        run_id="fault-injection-mixed",
        n_resamples=20,
        on_scorer_error="record",
        seed=42,
    )

    # Run a control without any failing scorer
    result_control = evaluate(
        scorers={"healthy": _ConstantScorer()},
        slices=[slice_a, slice_b, slice_c],
        run_id="control",
        n_resamples=20,
        seed=42,
    )

    # The healthy scorer's per-slice block must be byte-identical between
    # the two runs for every slice — including A and C, where the OTHER
    # scorer was failing
    for slice_name in ("A", "B", "C"):
        h_fault = result_with_fault.by_slice[slice_name]["by_scorer"]["healthy"]
        h_control = result_control.by_slice[slice_name]["by_scorer"]["healthy"]
        # Compare metric values (drop only meta fields that might differ in run_id context)
        for k in ("pr_auc", "roc_auc", "brier", "n", "n_positive"):
            if k in h_fault or k in h_control:
                assert h_fault.get(k) == h_control.get(k), (
                    f"Slice {slice_name}.healthy.{k}: differs between fault and control "
                    f"(fault={h_fault.get(k)} control={h_control.get(k)}) — possible state bleed"
                )


@pytest.mark.unit
def test_multi_slice_fault_injection_n_and_n_positive_recorded_on_error() -> None:
    """Even when a scorer errors, slice-level metadata (n, n_positive) must be recorded.

    The slice metadata is independent of scorer execution; an error on the
    scorer must not blank out n / n_positive on the slice block.
    """
    slice_a = _make_slice("A", n=100, with_fail_marker=True)

    result = evaluate(
        scorers={"trip": _SliceContentSensitiveScorer()},
        slices=[slice_a],
        run_id="meta-on-error",
        n_resamples=20,
        on_scorer_error="record",
    )

    a_block = result.by_slice["A"]
    assert a_block["n"] == 100
    assert isinstance(a_block["n_positive"], int)
    # Scorer block has the error
    assert "error" in a_block["by_scorer"]["trip"]
