# Worked example: nested seed-split composition over `SourceDisjointKFoldSplitter`

> **What this shows.** The canonical OOD-CV protocol per Bayle 2020 +
> matched-budget per HELM lineage: LODO k-fold splits × multiple seeds ×
> within-fold stratified train/val. Each toolkit primitive does one thing;
> the composition is project glue.
>
> **Runtime:** ~1 s. Pure-numpy/sklearn core; no optional deps.

## Why this composition

`SourceDisjointKFoldSplitter` is *single-seed* — its `seed=` only shuffles
the source-bucket order, not the per-row split. For multi-seed protocols
(standard for variance estimation across re-runs), loop seeds *outside* the
splitter and apply stratified train/val per seed. Each fold's train
partition then gets sub-split into train+val with the per-seed
`sklearn.model_selection.train_test_split`.

This was a v0.34.0-era cookbook addition (closes #19) — the pattern is
**not** invented by the toolkit; the splitter API is intentionally minimal
and composes with stdlib + sklearn for the loops that wrap it.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from eval_toolkit import EvalSlice, SourceDisjointKFoldSplitter, set_global_seeds
set_global_seeds(42)
```

## Synthetic data: 4 sources, balanced labels

```python
rng = np.random.default_rng(42)
n_per_source = 50
df = pd.DataFrame({
    "text": [f"row_{i}" for i in range(4 * n_per_source)],
    "label": rng.integers(0, 2, size=4 * n_per_source),
    "source": np.repeat(["src_a", "src_b", "src_c", "src_d"], n_per_source),
})
parent = EvalSlice(df=df, name="full_corpus")
```

## LODO k-fold × multi-seed × stratified train/val

```python
splitter = SourceDisjointKFoldSplitter(source_col="source", k=4, seed=42)
seeds = (42, 43, 44)

split_count = 0
for fold_idx, fold_dict in enumerate(splitter.iter_folds(parent)):
    train_full = fold_dict["train"]   # 3-source train partition (EvalSlice)
    test_slice = fold_dict["test"]    # held-out 1-source test partition
    for seed in seeds:
        # Project glue: stratified train/val split per seed.
        # The splitter doesn't own this — sklearn does, and that's intentional.
        train_idx, val_idx = train_test_split(
            range(len(train_full.df)),
            stratify=train_full.df["label"],
            test_size=0.20,
            random_state=seed,
        )
        # 4 folds × 3 seeds × {train, val, test} = 36 splits
        split_count += 3
assert split_count == 4 * 3 * 3  # 36 splits total
```

## Common pitfalls

- **Don't pass `seed=` to the splitter for variance estimation.** The
  splitter seed only shuffles source-bucket order — it does NOT produce
  different per-row partitions. Each fold's row membership is deterministic
  given `(source_col, k)`. Variance comes from the per-seed *train/val*
  split, not from the fold partition.
- **Stratify on label, not source.** Source is already partitioned at
  fold level; stratifying on source within the train partition would
  bias against the LODO design intent.

## See also

- `SourceDisjointKFoldSplitter` ([API](../api/splits.md))
- `PurgedKFoldSplitter` for time-aware LODO with embargo
- [methodology/splits.md](../methodology/splits.md) for the fold-design
  rationale + when to use each splitter
