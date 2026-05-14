"""Integration test: stratified K-fold splits → leakage checks (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``splits.py`` and ``leakage.py`` are tested in isolation, but no
test verifies the *interaction* — does a clean stratified split
produce zero leakage findings, or do the two modules disagree about
what counts as "duplicate"?

This test exercises the cross-module chain:

1. Build a clean balanced corpus with no duplicates and no
   group/source/temporal structure.
2. Run ``StratifiedKFoldSplitter(k=5)`` to produce 5 (train, test)
   fold dicts.
3. For one fold, run a representative leakage check
   (:class:`ExactDuplicateCheck`) on the resulting train/test split.
4. Assert the check produces zero error-severity findings.

A real failure here would indicate either (a) the stratified
splitter accidentally duplicates rows across folds, or (b) the
leakage check has false positives on legitimate splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.leakage import (
    CrossSplitLeakageCheck,
    ExactDuplicateCheck,
    NormalizedFormLeakageCheck,
)
from eval_toolkit.splits import StratifiedKFoldSplitter


def _build_clean_corpus(n: int = 500, seed: int = 42) -> EvalSlice:
    """Build a corpus of n distinct texts with balanced labels.

    Each text is unique (no duplicates), labels are balanced 50/50,
    no group/source/temporal columns. This is the most leakage-safe
    fixture possible; any leakage finding would indicate a bug.
    """
    rng = np.random.default_rng(seed)
    # Distinct integers as text content, alternating with seed-derived suffixes
    # to guarantee uniqueness across n=500 rows.
    texts = [f"row_{i}_unique_content_{int(rng.integers(0, 1_000_000))}" for i in range(n)]
    labels = np.zeros(n, dtype=int)
    labels[: n // 2] = 1
    rng.shuffle(labels)
    df = pd.DataFrame({"text": texts, "label": labels})
    return EvalSlice(name="clean_corpus", df=df)


def test_stratified_split_produces_no_exact_duplicate_leakage() -> None:
    """``StratifiedKFoldSplitter`` output passes ``ExactDuplicateCheck`` cleanly.

    Catches any false positives from the leakage detector on
    legitimately clean splits (e.g., the within-split duplicate
    check accidentally flagging the train and test slices for
    sharing the parent text column).
    """
    corpus = _build_clean_corpus(n=500, seed=42)
    splitter = StratifiedKFoldSplitter(k=5, seed=42)
    folds = list(splitter.iter_folds(corpus))
    assert len(folds) == 5
    # Test the first fold's leakage profile; all folds are equivalent up to
    # which rows go to train vs test.
    fold = folds[0]
    check = ExactDuplicateCheck()
    finding = check.validate(fold)
    assert finding.severity == "warning" or not finding.drop_indices, (
        f"ExactDuplicateCheck produced findings on a clean stratified split: "
        f"severity={finding.severity}, drop_indices={finding.drop_indices}, "
        f"n_affected={finding.n_affected}. This is a false positive."
    )
    assert finding.n_affected == 0, (
        f"ExactDuplicateCheck found {finding.n_affected} duplicate rows in clean "
        f"corpus — every text was constructed unique. Likely bug in either "
        f"the splitter (duplicating rows) or the check (false positive)."
    )


def test_stratified_split_produces_no_cross_split_leakage() -> None:
    """``StratifiedKFoldSplitter`` output passes ``CrossSplitLeakageCheck`` cleanly.

    The cross-split check is more sensitive than the within-split
    duplicate check — it looks for near-duplicates between train and
    test using a similarity strategy. On a corpus with all-unique
    texts and a stratified split, it should find no findings at
    error severity.
    """
    corpus = _build_clean_corpus(n=500, seed=42)
    splitter = StratifiedKFoldSplitter(k=5, seed=42)
    folds = list(splitter.iter_folds(corpus))
    fold = folds[0]
    check = CrossSplitLeakageCheck(train_split="train", threshold=0.9)
    finding = check.validate(fold)
    assert finding.n_affected == 0, (
        f"CrossSplitLeakageCheck flagged {finding.n_affected} train↔test "
        f"near-duplicates on a clean unique-text corpus; severity={finding.severity}. "
        f"Either the splitter is duplicating rows or the check has false positives."
    )


def test_stratified_split_produces_no_normalized_form_leakage() -> None:
    """``StratifiedKFoldSplitter`` output passes ``NormalizedFormLeakageCheck`` cleanly.

    The normalized-form check (NFKC + lowercase + whitespace collapse)
    catches variants of the same text. On a corpus where every text
    has an explicit unique suffix, normalization cannot create
    cross-text matches.
    """
    corpus = _build_clean_corpus(n=500, seed=42)
    splitter = StratifiedKFoldSplitter(k=5, seed=42)
    folds = list(splitter.iter_folds(corpus))
    fold = folds[0]
    check = NormalizedFormLeakageCheck()
    finding = check.validate(fold)
    assert finding.n_affected == 0, (
        f"NormalizedFormLeakageCheck flagged {finding.n_affected} normalized-form "
        f"matches on a corpus with explicit unique suffixes; severity={finding.severity}."
    )


def test_intentionally_duplicated_corpus_does_trigger_leakage_finding() -> None:
    """Positive control: when we INSERT a deliberate duplicate, the check fires.

    Guards against the cleanness tests above passing because the check
    is broken (always returns 0 findings). Construct a corpus with one
    intentional row duplicated across train and test, verify
    ``CrossSplitLeakageCheck`` catches it.
    """
    rng = np.random.default_rng(42)
    n = 100
    texts = [f"row_{i}" for i in range(n)]
    labels = np.zeros(n, dtype=int)
    labels[: n // 2] = 1
    rng.shuffle(labels)
    df = pd.DataFrame({"text": texts, "label": labels})

    # Manually split: first 50 → train, last 50 → test, plus a deliberate
    # duplicate text in test.
    train_df = df.iloc[:50].copy()
    test_df = df.iloc[50:].copy()
    test_df.loc[test_df.index[0], "text"] = train_df.iloc[0]["text"]  # leak

    splits = {
        "train": EvalSlice(name="train", df=train_df.reset_index(drop=True)),
        "test": EvalSlice(name="test", df=test_df.reset_index(drop=True)),
    }
    check = CrossSplitLeakageCheck(train_split="train", threshold=0.9)
    finding = check.validate(splits)
    assert finding.n_affected >= 1, (
        f"Positive control failed: CrossSplitLeakageCheck did not catch a deliberate "
        f"duplicate. n_affected={finding.n_affected}; check is broken or threshold "
        f"too lax."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
