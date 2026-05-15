"""End-to-end pipeline tests: loader → evaluate → write_run_result → JSON schema validation.

Each unit-level layer (loader, harness, artifacts) has its own tests, but the
contract *between* them is what breaks silently on schema bumps or signature
drift. These tests exercise the full pipe so a renamed schema field or a
loader-returns-the-wrong-shape bug fails the suite at the boundary.

Mirrors the v0.27.2 base-install bug class: 95% line coverage missed that
`from eval_toolkit import evaluate` was broken because the test that asks
"does the public path actually work end-to-end" didn't exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.artifacts import validate_results
from eval_toolkit.harness import EvalSlice, evaluate, write_run_result
from eval_toolkit.loaders import DataFrameLoader, SingleSliceLoader


class _DiscriminativeScorer:
    """Stub that returns precomputed scores. Mirrors `_StubScorer` in test_harness_smoke.py."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict_proba(self, X: object) -> np.ndarray:
        return self._scores


def _make_synthetic_dataframe(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic labeled DataFrame with a discriminative signal + split column.

    Returns balanced labels (~50/50) split 70/30 train/test so each slice has
    both classes for sklearn metric computation. Score column is unused at
    DataFrame-load time; scorers operate on the produced EvalSlice.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # 70/30 train/test split
    n_train = int(n * 0.7)
    splits = np.array(["train"] * n_train + ["test"] * (n - n_train))
    rng.shuffle(splits)
    return pd.DataFrame(
        {
            "text": [f"row_{i}" for i in range(n)],
            "label": labels,
            "split": splits,
        }
    )


@pytest.mark.smoke
def test_e2e_dataframeloader_to_schema_validated_json(tmp_path: Path) -> None:
    """Full pipeline: DataFrame → DataFrameLoader → evaluate → write_run_result → schema-validate.

    Asserts the produced JSON conforms to schemas/results.v1.json. A schema
    field rename or a removed-required-key drift fails here.
    """
    pytest.importorskip("jsonschema")

    df = _make_synthetic_dataframe(n=200, seed=42)
    loader = DataFrameLoader(
        df=df,
        split_col="split",
        feature_col="text",
        label_col="label",
        name="synthetic_e2e",
    )
    splits = loader.load_splits()

    assert set(splits.keys()) == {"train", "test"}
    test_slice = splits["test"]
    assert isinstance(test_slice, EvalSlice)
    # Sanity: the slice has both classes (so metrics can compute)
    assert 0 in test_slice.y_true and 1 in test_slice.y_true

    # Score the test slice with a discriminative-but-noisy stub
    rng = np.random.default_rng(123)
    scores = np.clip(
        0.5 + 0.3 * (test_slice.y_true - 0.5) + rng.normal(0, 0.1, size=len(test_slice.df)),
        0.0,
        1.0,
    )
    scorer = _DiscriminativeScorer(scores=scores)

    result = evaluate(
        scorers={"stub": scorer},
        slices=[test_slice],
        run_id="e2e-dataframeloader",
        n_resamples=50,
        seed=42,
    )

    run_dir = tmp_path / "e2e_run"
    compact_path, full_path = write_run_result(result, run_dir)
    assert compact_path.exists() and full_path.exists()

    # The compact JSON is what consumers see; it must validate against the schema
    compact_payload = json.loads(compact_path.read_text())
    validate_results(compact_payload)  # raises jsonschema.ValidationError on contract drift

    # Schema-required behavior checks
    assert compact_payload["schema_version"] == "v1"
    assert compact_payload["run_id"] == "e2e-dataframeloader"
    assert "test" in compact_payload["by_slice"]
    test_block = compact_payload["by_slice"]["test"]
    assert "stub" in test_block["by_scorer"]
    assert "n" in test_block
    assert "n_positive" in test_block


@pytest.mark.smoke
def test_e2e_singleslice_loader_to_schema_validated_json(tmp_path: Path) -> None:
    """SingleSliceLoader path: pre-built EvalSlice → "all" key → evaluate → schema-validate."""
    pytest.importorskip("jsonschema")

    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame(
        {
            "text": [f"row_{i}" for i in range(n)],
            "label": rng.integers(0, 2, size=n),
        }
    )
    parent_slice = EvalSlice(name="source", df=df)
    loader = SingleSliceLoader(slice_=parent_slice)
    splits = loader.load_splits()

    assert set(splits.keys()) == {"all"}
    all_slice = splits["all"]

    # Score with discriminative stub
    scores = np.clip(0.5 + 0.3 * (all_slice.y_true - 0.5) + rng.normal(0, 0.1, size=n), 0.0, 1.0)
    scorer = _DiscriminativeScorer(scores=scores)

    result = evaluate(
        scorers={"stub": scorer},
        slices=[all_slice],
        run_id="e2e-singleslice",
        n_resamples=50,
        seed=42,
    )

    run_dir = tmp_path / "e2e_single"
    compact_path, _ = write_run_result(result, run_dir)
    payload = json.loads(compact_path.read_text())

    validate_results(payload)
    assert payload["schema_version"] == "v1"
    assert "all" in payload["by_slice"]


@pytest.mark.smoke
def test_e2e_paired_diffs_preserved_through_schema(tmp_path: Path) -> None:
    """Paired-diffs in evaluate output survive the JSON round-trip and schema validation.

    Catches the case where a paired-diff field gets renamed or its shape
    changes but unit tests on bootstrap_ci pass because they don't traverse
    the full RunResult.to_dict() → JSON → schema path.
    """
    pytest.importorskip("jsonschema")

    df = _make_synthetic_dataframe(n=200, seed=42)
    loader = DataFrameLoader(df=df, split_col="split", feature_col="text", label_col="label")
    test_slice = loader.load_splits()["test"]
    n = len(test_slice.df)

    rng = np.random.default_rng(42)
    scores_a = rng.uniform(0, 1, size=n)
    scores_b = np.clip(scores_a + 0.1 * test_slice.y_true, 0.0, 1.0)

    result = evaluate(
        scorers={"a": _DiscriminativeScorer(scores_a), "b": _DiscriminativeScorer(scores_b)},
        slices=[test_slice],
        run_id="e2e-paired",
        n_resamples=50,
        paired_diffs=[("a", "b")],
        seed=42,
    )

    run_dir = tmp_path / "e2e_paired"
    compact_path, _ = write_run_result(result, run_dir)
    payload = json.loads(compact_path.read_text())

    validate_results(payload)
    diffs = payload["by_slice"]["test"]["paired_diffs"]
    assert "b_minus_a" in diffs
