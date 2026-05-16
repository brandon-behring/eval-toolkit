# Splits

> **Background** *(skip if you've internalized this)*. A *split* partitions
> your dataset into train and one or more eval sets. The choice of how to
> split is methodological: random partitioning measures interpolation
> across your sample; group-disjoint partitioning measures generalization
> across users / patients / sources; time-aware partitioning measures
> forecasting. The wrong split silently inflates metrics — a model can
> ace random K-fold and still flop in production because the production
> distribution sees groups / time periods / sources the training set
> never saw.

This chapter covers when to use each [`Splitter`
](../api/splits.md) reference impl, when CV alone is
insufficient, and how to compose splits with leakage checks.

## Setup

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice
```

A 40-row mixed fixture used throughout:

```python
df = pd.DataFrame({
    "text":   [f"row_{i}" for i in range(40)],
    "label":  [i % 2 for i in range(40)],
    "group":  [i // 4 for i in range(40)],   # 10 groups of 4 rows each
    "source": [f"src_{i % 5}" for i in range(40)],  # 5 sources
    "t":      np.arange(40),                  # monotone timestamps
})
parent = EvalSlice(name="all", df=df)
```

(holdout-vs-kfold)=
## Holdout vs K-fold
**Holdout.** A single train/test partition. Cheap, single number, easy to
report. Variance is high — a different random seed gives a different
metric. Suitable for a *final* evaluation against a frozen test set.

**K-fold cross-validation.** K train/test partitions; report mean ± CI of
the per-fold metric. Lower variance, better use of small data. Suitable
for *development* (model selection, hyperparameter tuning,
methodology choices).

**Don't conflate the two.** Use K-fold to *develop*, then a separately
held-out test set to *report*. Reporting a K-fold mean as if it were a
holdout is a subtle leakage of model-selection information into the
metric.

```python
from eval_toolkit import HoldoutSplitter, StratifiedKFoldSplitter

holdout = HoldoutSplitter(test_size=0.25, seed=42)
folds = list(holdout.iter_folds(parent))
print(f"holdout: {len(folds)} fold(s); train={len(folds[0]['train'].df)} test={len(folds[0]['test'].df)}")

cv = StratifiedKFoldSplitter(k=5, seed=42)
print(f"5-fold: {cv.get_n_splits(parent)} folds")
```

Both Splitters yield the same shape — `dict[str, EvalSlice]` keyed by
``"train"`` / ``"test"``. Holdout is just K=1, so consumer code can treat
both identically:

```python
for splitter in [HoldoutSplitter(seed=42), StratifiedKFoldSplitter(k=3, seed=42)]:
    for i, fold in enumerate(splitter.iter_folds(parent)):
        print(f"  {type(splitter).__name__} fold {i}: "
              f"train={len(fold['train'].df)} test={len(fold['test'].df)}")
```

(stratified-kfold)=
## Stratified K-fold
**When to use.** Default for binary classification with class imbalance.
Keeps the positive/negative ratio stable across folds, which keeps PR-AUC
estimates comparable across folds.

**Primitive.**
[`StratifiedKFoldSplitter`](../api/splits.md) wraps
`sklearn.model_selection.StratifiedKFold`.

```python
imbalanced = pd.DataFrame({
    "text":  [f"row_{i}" for i in range(50)],
    "label": [1 if i < 5 else 0 for i in range(50)],  # 10 % positive
})
imbal_slice = EvalSlice(name="imbal", df=imbalanced)

cv = StratifiedKFoldSplitter(k=5, seed=42)
for fold in cv.iter_folds(imbal_slice):
    n_pos = int(fold["test"].y_true.sum())
    print(f"  test fold: n={len(fold['test'].df)} pos={n_pos}")
```

> **What NOT to do.** Don't pass an unstratified `KFold` for binary
> classification with rare positives. A fold with zero positives breaks
> PR-AUC entirely (single-class slices return `np.nan`).

(group-kfold)=
## Group-disjoint K-fold
**When to use.** Whenever rows cluster by an identifier with strong
within-cluster correlation: same patient, same user, same document
across paragraphs, same author, same source. Random K-fold leaks across
groups; group-disjoint K-fold doesn't.

**Primitive.**
[`GroupKFoldSplitter`](../api/splits.md) wraps
`sklearn.model_selection.GroupKFold`. Pass `group_col=` (column name in
the slice's dataframe) or `groups=` (numpy array at `iter_folds` call).

```python
from eval_toolkit import GroupKFoldSplitter

splitter = GroupKFoldSplitter(k=5, group_col="group")
for fold in splitter.iter_folds(parent):
    train_groups = set(fold["train"].df["group"].tolist())
    test_groups  = set(fold["test"].df["group"].tolist())
    assert not (train_groups & test_groups)
print("group-disjoint K-fold OK")
```

Pair with [`GroupLeakageCheck`](../api/leakage.md) (see
[leakage.md §6](leakage.md#group-leakage)) to catch the case where the
underlying dataset already has the same group ID in pre-existing splits.

(source-disjoint-kfold)=
## Source-disjoint K-fold (NEW in v0.7.0)
**When to use.** Stronger than group-disjoint when you want the test
fold's sources to **never appear in any training fold across the whole
CV procedure** — not just within a single fold. The pattern that
``prompt-injection-sdd`` hand-rolled (3-fold, 3 seeds, 9 runs total).

**Primitive.**
[`SourceDisjointKFoldSplitter`](../api/splits.md)
generalizes the pattern. Distinct sources are sorted, shuffled with the
seed, then round-robin assigned to K folds. Fold *i*'s test set = rows
whose source falls in bucket *i*.

**Difference from `GroupKFoldSplitter`.** GroupKFold only enforces
disjointness *within* a fold. Source-disjoint round-robin guarantees
that the *union* of test folds covers all sources, with each source
appearing in exactly one test fold across the procedure.

```python
from eval_toolkit import SourceDisjointKFoldSplitter

splitter = SourceDisjointKFoldSplitter(source_col="source", k=5, seed=42)
for i, fold in enumerate(splitter.iter_folds(parent)):
    test_sources = sorted(set(fold["test"].df["source"].tolist()))
    print(f"  fold {i} test sources: {test_sources}")
```

For multi-seed × CV (the prompt-injection-sdd 9-run pattern), wrap the
splitter in a seed loop in your harness call:

```python
from eval_toolkit.harness import evaluate_folded
# evaluate_folded(scorers, splitter, slice_, run_id="r", seeds=(1, 2, 3), ...)
```

(time-series-splits)=
## Time-aware splits
**When to use.** Whenever the data has a temporal dimension and the
production deployment will see *future* timestamps not in the training
set. Random K-fold mixes time periods and lets the model interpolate
across them; time-aware splits force the model to extrapolate.

**Primitive.**
[`TimeSeriesSplitter`](../api/splits.md) wraps
`sklearn.model_selection.TimeSeriesSplit`. Each fold's train set is
everything ≤ a moving boundary; the test set is the next chunk after.

```python
from eval_toolkit import TimeSeriesSplitter

splitter = TimeSeriesSplitter(k=4, time_col="t")
for i, fold in enumerate(splitter.iter_folds(parent)):
    max_train = fold["train"].df["t"].max()
    min_test  = fold["test"].df["t"].min()
    assert max_train < min_test
    print(f"  fold {i}: train≤{max_train}  test≥{min_test}")
```

Pair with
[`TemporalLeakageCheck`](../api/leakage.md) (see
[leakage.md §7](leakage.md#temporal-leakage)) to verify the invariant
end-to-end.

(cv-and-ood)=
## When CV alone is insufficient
K-fold CV with any random partitioning estimates *interpolation across
your sample*, not generalization to a new population. For
"out-of-distribution" claims (the model works on data shaped like the
production stream, not just like dev) you need **two** layers:

1. **Source-disjoint CV during development** — measures sensitivity to
   domain shift across your *known* axes (source, time, group).
2. **A separate locked final-holdout test set**, *never used* in
   development, drawn from a distribution as close to production as you
   can get.

The CV mean is your "expected dev-time performance under domain shift";
the holdout is your "honest single-shot release metric". Confusing the
two is what the literature calls *underspecification* — see Recht et al.
2019 and the
[Hidden Leaks in Time Series](https://arxiv.org/html/2512.06932v1) work
on validation-strategy leakage.

(pitfalls)=
## Pitfalls / Common mistakes
- **Tuning hyperparameters on the test fold.** If your fold loop is
  outermost and your inner loop tunes hyperparameters using the test
  fold, your "K-fold mean" is a test-fold-leaky number. Use a *nested*
  CV (inner CV on the train fold for HP search, outer CV for reporting)
  or a separate validation slice carved off the train fold.
- **Reporting K-fold mean without CI.** Use
  [`cv_clt_ci`](../api/bootstrap.md) (or call
  `evaluate_folded(...)` which auto-computes it). A single mean number
  hides the cross-fold variance and is impossible to compare across
  papers.
- **Using `GroupKFold` when you actually want
  `SourceDisjointKFoldSplitter`.** The two have meaningfully different
  guarantees — see §source-disjoint above. If your concern is "test
  sources never seen in train", you want source-disjoint.
- **`shuffle=False` in `StratifiedKFold` with sorted-by-time data.**
  Without shuffling, the first folds get the earliest rows — accidentally
  recreating a time-aware split (probably not what you intended).
- **Splitting before deduplication.** If your dataset has duplicates and
  you split first, the duplicates might end up on opposite sides of the
  split — instant cross-split leakage. Run
  [`ExactDuplicateCheck`](../api/leakage.md) +
  [`NearDuplicateCheck`](../api/leakage.md) on the *full*
  dataset before splitting.

## Further reading

- Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical
  Learning.* §7.10 (cross-validation methodology).
- Yan, X. et al. *Hidden Leaks in Time Series Forecasting.* arXiv 2025.
  [arXiv:2512.06932](https://arxiv.org/html/2512.06932v1) — formalizes
  validation-strategy leakage.
- Recht, B. et al. *Do ImageNet classifiers generalize to ImageNet?*
  ICML 2019. — empirical CV-vs-final-holdout divergence on a public
  benchmark.
- sklearn docs:
  [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html),
  [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html),
  [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

See also: [leakage.md](leakage.md),
[comparison.md](comparison.md) (CV CI computation).
