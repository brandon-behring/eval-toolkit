"""Golden test: SimilarityStrategy calibration on the 50-pair adversarial holdout.

Migrated from `prompt-injection-detection-submission/data/dedup_holdout.jsonl`
(see `tests/golden/data/dedup_holdout_provenance.md` for the consumer git
SHA + license trail).

Per /exploring-options Q7a:
- 3 deterministic strategies (TfidfCosineStrategy, ExactNormalizedHashStrategy,
  JaccardNgramStrategy) → strict snapshot at thresholds {0.75, 0.80, 0.85}
- EmbeddingCosineStrategy → soft bounds (FPR < 0.5, FNR < 0.5) at threshold
  0.80; not snapshotted (survives sentence-transformers version drift)

Regenerate the snapshot with::

    REGEN_DEDUP_HOLDOUT_GOLDEN=1 pytest tests/golden/test_dedup_holdout_calibration.py -q

Or use ``scripts/refresh_dedup_holdout.py`` which also re-syncs the
fixture from the consumer repo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.text_dedup import (
    ExactNormalizedHashStrategy,
    JaccardNgramStrategy,
    TfidfCosineStrategy,
)

_GOLDEN_DIR = Path(__file__).parent / "data"
_HOLDOUT_PATH = _GOLDEN_DIR / "dedup_holdout.jsonl"
_EXPECTED_PATH = _GOLDEN_DIR / "dedup_holdout_expected.json"
_THRESHOLDS = (0.75, 0.80, 0.85)
_DETERMINISTIC_STRATEGY_NAMES = (
    "TfidfCosineStrategy",
    "ExactNormalizedHashStrategy",
    "JaccardNgramStrategy",
)


def _load_holdout() -> tuple[list[str], list[str], list[bool]]:
    """Load (text_a, text_b, true_duplicate) triples from the holdout JSONL.

    Skips the metadata header line (``_metadata: true``). Uses explicit
    UTF-8 because the corpus contains non-ASCII characters (German umlauts
    etc.) and Windows defaults to cp1252 (charmap) which can't decode them.
    """
    text_as: list[str] = []
    text_bs: list[str] = []
    truths: list[bool] = []
    with _HOLDOUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_metadata"):
                continue
            text_as.append(rec["text_a"])
            text_bs.append(rec["text_b"])
            truths.append(bool(rec["true_duplicate"]))
    return text_as, text_bs, truths


def _strategy_from_name(name: str):
    """Construct a fresh strategy instance from its class name."""
    return {
        "TfidfCosineStrategy": TfidfCosineStrategy(),
        "ExactNormalizedHashStrategy": ExactNormalizedHashStrategy(),
        "JaccardNgramStrategy": JaccardNgramStrategy(),
    }[name]


def _pairwise_sims(strategy, text_as: list[str], text_bs: list[str]) -> np.ndarray:
    """Return a (n,) array of pair-wise similarities for the i-th (text_a, text_b).

    Runs ``strategy.pairs_across(text_as, text_bs, k=n)`` once for the whole
    50-pair corpus (TF-IDF and others need >2 documents to build vocabulary;
    single-pair calls fail with "empty vocabulary"). Then extracts the
    similarity at the diagonal i↔i position by finding where ``indices[i]``
    contains ``i``.
    """
    n = len(text_as)
    sims_matrix, indices_matrix = strategy.pairs_across(text_as, text_bs, k=n)
    diag_sims = np.zeros(n, dtype=np.float64)
    for i in range(n):
        match = np.where(indices_matrix[i] == i)[0]
        if len(match) > 0:
            diag_sims[i] = float(sims_matrix[i, match[0]])
        # else: i not in top-n (impossible when k=n), leave 0.0
    return diag_sims


def _fpr_fnr_for_strategy_at_threshold(
    strategy_name: str,
    threshold: float,
    text_as: list[str],
    text_bs: list[str],
    truths: list[bool],
    diag_sims: np.ndarray,
) -> tuple[float, float]:
    """Compute (FPR, FNR) from precomputed diagonal sims at the given threshold."""
    predicted = [sim >= threshold for sim in diag_sims]
    fp = sum(1 for p, t in zip(predicted, truths, strict=True) if p and not t)
    tn = sum(1 for p, t in zip(predicted, truths, strict=True) if not p and not t)
    fn = sum(1 for p, t in zip(predicted, truths, strict=True) if not p and t)
    tp = sum(1 for p, t in zip(predicted, truths, strict=True) if p and t)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr


def _compute_full_grid(
    text_as: list[str], text_bs: list[str], truths: list[bool]
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute FPR/FNR grid for all 3 deterministic strategies × 3 thresholds.

    Computes the diagonal sims once per strategy (full-corpus pairs_across
    is needed for TF-IDF vocab + Jaccard n-gram set construction); then
    iterates thresholds in O(n).
    """
    grid: dict[str, dict[str, dict[str, float]]] = {}
    for name in _DETERMINISTIC_STRATEGY_NAMES:
        strategy = _strategy_from_name(name)
        diag_sims = _pairwise_sims(strategy, text_as, text_bs)
        per_threshold: dict[str, dict[str, float]] = {}
        for t in _THRESHOLDS:
            fpr, fnr = _fpr_fnr_for_strategy_at_threshold(
                name, t, text_as, text_bs, truths, diag_sims
            )
            per_threshold[f"{t:.2f}"] = {"fpr": fpr, "fnr": fnr}
        grid[name] = per_threshold
    return grid


@pytest.mark.golden
def test_dedup_holdout_calibration_deterministic_strategies() -> None:
    """Strict snapshot check on the 3 deterministic strategies (#18).

    Snapshot is regenerated via ``REGEN_DEDUP_HOLDOUT_GOLDEN=1``. The
    snapshot file is checked in at ``tests/golden/data/dedup_holdout_expected.json``.
    """
    text_as, text_bs, truths = _load_holdout()
    grid = _compute_full_grid(text_as, text_bs, truths)

    if os.environ.get("REGEN_DEDUP_HOLDOUT_GOLDEN") == "1":
        _EXPECTED_PATH.write_text(
            json.dumps(grid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip(f"REGEN_DEDUP_HOLDOUT_GOLDEN=1; wrote snapshot to {_EXPECTED_PATH}")

    assert _EXPECTED_PATH.exists(), (
        f"Snapshot missing: {_EXPECTED_PATH}. Regenerate with: "
        "REGEN_DEDUP_HOLDOUT_GOLDEN=1 pytest tests/golden/test_dedup_holdout_calibration.py"
    )
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))

    for name in _DETERMINISTIC_STRATEGY_NAMES:
        for t_key in (f"{t:.2f}" for t in _THRESHOLDS):
            exp = expected[name][t_key]
            obs = grid[name][t_key]
            assert obs["fpr"] == pytest.approx(exp["fpr"], rel=1e-9), (
                f"{name} @ threshold={t_key}: snapshot FPR={exp['fpr']}, "
                f"observed FPR={obs['fpr']}. "
                "Refresh: scripts/refresh_dedup_holdout.py"
            )
            assert obs["fnr"] == pytest.approx(exp["fnr"], rel=1e-9), (
                f"{name} @ threshold={t_key}: snapshot FNR={exp['fnr']}, "
                f"observed FNR={obs['fnr']}. "
                "Refresh: scripts/refresh_dedup_holdout.py"
            )


@pytest.mark.golden
@pytest.mark.slow
def test_dedup_holdout_calibration_embedding_strategy_soft_bound() -> None:
    """Soft-bound assertion on EmbeddingCosineStrategy + make_minilm_embedder (#18).

    Per Q7a: not in the strict snapshot because sentence-transformers
    model weights can shift across upstream releases; assertion is the
    loose "doesn't completely fall over" bound (``FPR < 0.5 and FNR < 0.5``
    at the canonical threshold 0.80 per ADR-027).

    Gated by ``pytest.importorskip("sentence_transformers")`` because
    ``[embeddings]`` is optional (sentence-transformers + torch transitive
    ~700MB, not in ``[all]`` per v0.33.1 design).
    """
    pytest.importorskip("sentence_transformers")

    from eval_toolkit import make_minilm_embedder
    from eval_toolkit.text_dedup import EmbeddingCosineStrategy

    text_as, text_bs, truths = _load_holdout()
    embedder = make_minilm_embedder()
    strategy = EmbeddingCosineStrategy(embedder=embedder)

    # Classify all 50 pairs at the canonical threshold 0.80 (ADR-027) using
    # the same full-corpus diagonal-sim pattern as the deterministic test.
    diag_sims = _pairwise_sims(strategy, text_as, text_bs)
    predicted = [sim >= 0.80 for sim in diag_sims]
    fp = sum(1 for p, t in zip(predicted, truths, strict=True) if p and not t)
    tn = sum(1 for p, t in zip(predicted, truths, strict=True) if not p and not t)
    fn = sum(1 for p, t in zip(predicted, truths, strict=True) if not p and t)
    tp = sum(1 for p, t in zip(predicted, truths, strict=True) if p and t)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    assert fpr < 0.5, f"EmbeddingCosineStrategy@0.80 FPR={fpr:.3f}; expected < 0.5"
    assert fnr < 0.5, f"EmbeddingCosineStrategy@0.80 FNR={fnr:.3f}; expected < 0.5"
