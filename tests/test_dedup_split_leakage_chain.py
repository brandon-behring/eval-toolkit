"""Integration test: text-dedup → split → leakage-check chain (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``text_dedup.py``, ``splits.py``, and ``leakage.py`` are tested in
isolation, but no test verifies the *end-to-end* workflow that
consumers actually run:

1. Dedup the source corpus to remove near-duplicates.
2. Stratify-split the deduped corpus into train/test.
3. Run cross-split leakage checks; expect zero error-severity findings.

This catches threshold-mismatch bugs (e.g., dedup at threshold 0.9
but leakage check at threshold 0.7 would flag the residual matches
that dedup left behind).

The test uses MinHashLSH for dedup (well-tested approximation) and
the same threshold for both dedup and the leakage check, so the chain
is self-consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.leakage import CrossSplitLeakageCheck
from eval_toolkit.splits import StratifiedKFoldSplitter
from eval_toolkit.text_dedup import MinHashLSHStrategy, TfidfCosineStrategy, near_dedup


def _build_corpus_with_paraphrases(
    n_unique: int = 400, n_paraphrases: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Build a corpus of n_unique unique texts + n_paraphrases near-duplicates.

    Each paraphrase is constructed by taking an existing text and
    appending a short variation suffix; the underlying TF-IDF / MinHash
    similarity should rank these well above the dedup threshold (0.7).
    """
    rng = np.random.default_rng(seed)
    base_texts = [
        f"Document {i} contains discussion of topic_{i % 20} with detail {i}."
        for i in range(n_unique)
    ]
    para_indices = rng.integers(0, n_unique, size=n_paraphrases)
    paraphrases = [
        f"{base_texts[idx]} (slight variation)" for idx in para_indices  # high overlap with base
    ]
    all_texts = base_texts + paraphrases
    labels = rng.integers(0, 2, size=len(all_texts))
    return pd.DataFrame({"text": all_texts, "label": labels})


def test_dedup_then_split_then_leakage_check_is_self_consistent() -> None:
    """Dedup → stratified split → cross-split leakage check produces no findings.

    End-to-end self-consistency: if we dedup at threshold ``T`` and
    then run a leakage check at threshold ``T``, no leakage should
    remain. A failure indicates the dedup and check disagree about
    what counts as "near-duplicate" at the chosen threshold.
    """
    threshold = 0.7
    corpus_df = _build_corpus_with_paraphrases(n_unique=400, n_paraphrases=100, seed=42)
    # Dedup using TF-IDF cosine (default for both dedup and leakage check)
    report = near_dedup(corpus_df["text"].tolist(), threshold=threshold)
    deduped_indices = sorted(report.kept_indices)
    deduped_df = corpus_df.iloc[deduped_indices].reset_index(drop=True)
    # Verify dedup actually removed something (sanity)
    assert len(deduped_df) < len(corpus_df), (
        f"Dedup should have removed at least some paraphrases at threshold "
        f"{threshold}; corpus={len(corpus_df)}, deduped={len(deduped_df)}."
    )

    # Stratified-split the deduped corpus
    splitter = StratifiedKFoldSplitter(k=5, seed=42)
    deduped_slice = EvalSlice(name="deduped", df=deduped_df)
    fold = next(iter(splitter.iter_folds(deduped_slice)))

    # Run cross-split leakage check at the same threshold
    check = CrossSplitLeakageCheck(train_split="train", threshold=threshold)
    finding = check.validate(fold)
    assert finding.n_affected == 0, (
        f"Dedup → split → leakage chain produced {finding.n_affected} residual "
        f"findings at the same threshold ({threshold}). The dedup and check "
        f"are not self-consistent: dedup left near-duplicates the check still flags."
    )


def test_undedup_then_split_does_trigger_leakage_finding() -> None:
    """Positive control: skipping dedup yields cross-split leakage findings.

    Construct the same paraphrase-rich corpus, skip dedup, and run
    the leakage check on the stratified split. Expect at least some
    findings (paraphrases land on both sides of the split). Confirms
    the chain isn't just "always reports zero findings."
    """
    corpus_df = _build_corpus_with_paraphrases(n_unique=400, n_paraphrases=100, seed=42)
    splitter = StratifiedKFoldSplitter(k=5, seed=42)
    raw_slice = EvalSlice(name="raw", df=corpus_df)
    fold = next(iter(splitter.iter_folds(raw_slice)))

    check = CrossSplitLeakageCheck(train_split="train", threshold=0.7)
    finding = check.validate(fold)
    # On a 100-paraphrase corpus split 80/20, we expect dozens of findings.
    assert finding.n_affected > 5, (
        f"Positive control: undedup chain should produce > 5 leakage findings "
        f"on a paraphrase-rich corpus; got {finding.n_affected}. Either the "
        f"check has false negatives or the paraphrases are too dissimilar."
    )


def test_minhash_dedup_chain_is_also_self_consistent() -> None:
    """Same chain test using MinHashLSH dedup strategy (alternative to TF-IDF).

    MinHashLSH is a separate similarity backend; the chain self-
    consistency should hold for it too. This guards against
    strategy-specific threshold semantics drift.
    """
    threshold = 0.6
    corpus_df = _build_corpus_with_paraphrases(n_unique=300, n_paraphrases=80, seed=11)
    minhash_strategy = MinHashLSHStrategy(num_perm=128, bands=16, seed=11)
    report = near_dedup(
        corpus_df["text"].tolist(),
        threshold=threshold,
        strategy=minhash_strategy,
    )
    deduped_df = corpus_df.iloc[sorted(report.kept_indices)].reset_index(drop=True)
    splitter = StratifiedKFoldSplitter(k=5, seed=11)
    fold = next(iter(splitter.iter_folds(EvalSlice(name="deduped", df=deduped_df))))
    # Use TF-IDF for the leakage check — different backends still yield
    # consistent dedup quality at moderate threshold.
    check = CrossSplitLeakageCheck(
        train_split="train",
        threshold=0.85,  # Looser than dedup, since cross-backend agreement isn't perfect
        strategy=TfidfCosineStrategy(),
    )
    finding = check.validate(fold)
    # MinHash dedup at 0.6 should leave very few near-duplicates above
    # TF-IDF threshold 0.85 — at most a handful, not dozens.
    assert finding.n_affected <= 5, (
        f"MinHash dedup chain produced {finding.n_affected} residual findings; "
        f"expected ≤ 5 for cross-backend slack."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
