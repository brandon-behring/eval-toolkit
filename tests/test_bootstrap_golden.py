"""Golden tests for ``bootstrap_ci`` — pin exact BCa/percentile output on canonical seeds.

Property tests in ``test_bootstrap_props.py`` cover invariants (CI width sane,
contains-true-value-on-average, etc.). These golden tests are surgical:
they pin EXACT numerical output (rounded to ±1e-9) on six canonical stress
points so a numerical regression from scipy/numpy version drift, an
off-by-one in BCa quantile indexing, or a sign-flip in the bias-correction
factor fails immediately.

Cases (one per known stress point in the BCa+percentile math):

1. ``pr_auc_balanced_n200_BCa_rng42``           — typical balanced baseline
2. ``roc_auc_balanced_n200_BCa_rng42``          — different metric, same shape
3. ``pr_auc_balanced_n200_percentile_rng42``    — non-BCa code path
4. ``pr_auc_imbalanced_p05_n200_BCa_rng42``     — BCa skew correction (a-hat) path
5. ``pr_auc_n10_percentile_rng42``              — smallest valid n; quantile edge
6. ``roc_auc_tied_n40_BCa_rng42``               — rank-resolution under ties

Regenerating
------------
Goldens live at ``tests/golden/bootstrap_ci/cases.json``. Regenerate with::

    REGEN_BOOTSTRAP_GOLDEN=1 uv run python -m pytest tests/test_bootstrap_golden.py

then review the diff and commit. Regen ONLY when scipy/numpy bumps cause
intentional output drift, NOT to silence a regression — the whole point of
the golden is to FAIL when the math changes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.bootstrap import bootstrap_ci
from eval_toolkit.metrics import pr_auc, roc_auc

GOLDEN_DIR = Path(__file__).parent / "golden" / "bootstrap_ci"
GOLDEN_PATH = GOLDEN_DIR / "cases.json"
REGEN = os.environ.get("REGEN_BOOTSTRAP_GOLDEN") == "1"
TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Data recipes — each produces a deterministic (y_true, y_score) pair.
# Recipes are seeded so the inputs reconstruct identically across runs.
# ---------------------------------------------------------------------------


def _data_balanced_n200() -> tuple[np.ndarray, np.ndarray]:
    """Balanced n=200, discriminative scores. Baseline case."""
    rng = np.random.default_rng(42)
    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    rng.shuffle(y)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=200), 0.0, 1.0)
    return y, s


def _data_imbalanced_p05_n200() -> tuple[np.ndarray, np.ndarray]:
    """5% positive prevalence (n=200), 10 positives / 190 negatives. BCa skew path."""
    rng = np.random.default_rng(42)
    y = np.zeros(200, dtype=int)
    y[:10] = 1
    rng.shuffle(y)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=200), 0.0, 1.0)
    return y, s


def _data_small_n10() -> tuple[np.ndarray, np.ndarray]:
    """n=10 (smallest valid for bootstrap_ci). Exercises quantile edge."""
    rng = np.random.default_rng(42)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, size=10), 0.0, 1.0)
    return y, s


def _data_tied_n40() -> tuple[np.ndarray, np.ndarray]:
    """n=40 with only 3 unique score values (heavy ties). Rank-resolution path.

    20 positives, 20 negatives; scores drawn from {0.2, 0.5, 0.8} with
    discriminative bias toward labels.
    """
    rng = np.random.default_rng(42)
    y = np.concatenate([np.zeros(20), np.ones(20)]).astype(int)
    rng.shuffle(y)
    # Discriminative tied scores: pos → high tier preferred, neg → low tier
    high_p = 0.6  # positives lean to 0.8
    low_p = 0.6  # negatives lean to 0.2
    s = np.empty(40)
    for i, yi in enumerate(y):
        u = rng.uniform()
        if yi == 1:
            s[i] = 0.8 if u < high_p else (0.5 if u < 0.85 else 0.2)
        else:
            s[i] = 0.2 if u < low_p else (0.5 if u < 0.85 else 0.8)
    return y, s


DATA_RECIPES: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "balanced_n200": _data_balanced_n200,
    "imbalanced_p05_n200": _data_imbalanced_p05_n200,
    "small_n10": _data_small_n10,
    "tied_n40": _data_tied_n40,
}

METRICS: dict[str, Callable[..., float]] = {
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
}

# ---------------------------------------------------------------------------
# Case definitions — frozen list. To change inputs, bump the case id and
# regenerate the golden so the diff is reviewable.
# ---------------------------------------------------------------------------

CASES: list[dict[str, object]] = [
    {
        "id": "pr_auc_balanced_n200_BCa_rng42",
        "data": "balanced_n200",
        "metric": "pr_auc",
        "kwargs": {"n_resamples": 500, "method": "BCa", "rng": 42, "confidence": 0.95},
    },
    {
        "id": "roc_auc_balanced_n200_BCa_rng42",
        "data": "balanced_n200",
        "metric": "roc_auc",
        "kwargs": {"n_resamples": 500, "method": "BCa", "rng": 42, "confidence": 0.95},
    },
    {
        "id": "pr_auc_balanced_n200_percentile_rng42",
        "data": "balanced_n200",
        "metric": "pr_auc",
        "kwargs": {"n_resamples": 500, "method": "percentile", "rng": 42, "confidence": 0.95},
    },
    {
        "id": "pr_auc_imbalanced_p05_n200_BCa_rng42",
        "data": "imbalanced_p05_n200",
        "metric": "pr_auc",
        "kwargs": {"n_resamples": 500, "method": "BCa", "rng": 42, "confidence": 0.95},
    },
    {
        "id": "pr_auc_n10_percentile_rng42",
        "data": "small_n10",
        "metric": "pr_auc",
        "kwargs": {"n_resamples": 500, "method": "percentile", "rng": 42, "confidence": 0.95},
    },
    {
        "id": "roc_auc_tied_n40_BCa_rng42",
        "data": "tied_n40",
        "metric": "roc_auc",
        "kwargs": {"n_resamples": 500, "method": "BCa", "rng": 42, "confidence": 0.95},
    },
]


def _run_case(case: dict[str, object]) -> dict[str, float]:
    """Run bootstrap_ci for one case; return the comparable scalars rounded to 1e-9."""
    y, s = DATA_RECIPES[str(case["data"])]()
    metric = METRICS[str(case["metric"])]
    kwargs = dict(case["kwargs"])  # type: ignore[arg-type]
    result = bootstrap_ci(y, s, metric=metric, **kwargs)
    return {
        "point_estimate": round(float(result.point_estimate), 9),
        "ci_low": round(float(result.ci_low), 9),
        "ci_high": round(float(result.ci_high), 9),
        "method": result.method,
        "n_resamples": result.n_resamples,
        "confidence": round(float(result.confidence), 9),
    }


@pytest.mark.golden
def test_bootstrap_ci_canonical_golden() -> None:
    """Bootstrap CI on 6 canonical cases matches the pinned golden values.

    Updates the golden file via ``REGEN_BOOTSTRAP_GOLDEN=1`` (see module
    docstring for the workflow).
    """
    actual = {str(case["id"]): _run_case(case) for case in CASES}

    if REGEN or not GOLDEN_PATH.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        if REGEN:
            pytest.skip(
                f"REGEN: wrote {GOLDEN_PATH}; re-run without REGEN_BOOTSTRAP_GOLDEN to validate."
            )
        # First-time bootstrap of the golden — write + skip with a hint
        pytest.skip(f"Initial golden written to {GOLDEN_PATH}; re-run to validate.")

    expected = json.loads(GOLDEN_PATH.read_text())

    # Compare per-field with tolerance so we get a useful diff on failure
    for case_id, actual_block in actual.items():
        assert case_id in expected, f"Case {case_id!r} missing from golden — regenerate?"
        exp = expected[case_id]
        for key in ("point_estimate", "ci_low", "ci_high"):
            assert actual_block[key] == pytest.approx(exp[key], abs=TOLERANCE), (
                f"{case_id}.{key}: actual={actual_block[key]} vs expected={exp[key]} "
                f"(diff={actual_block[key] - exp[key]:.2e})"
            )
        # Non-numeric pins: method, n_resamples, confidence — exact match
        for key in ("method", "n_resamples", "confidence"):
            assert (
                actual_block[key] == exp[key]
            ), f"{case_id}.{key}: actual={actual_block[key]} vs expected={exp[key]}"

    # Detect newly added cases that haven't been pinned yet
    extra_in_golden = set(expected) - set(actual)
    assert not extra_in_golden, f"Golden contains unused cases: {extra_in_golden} — regen?"
