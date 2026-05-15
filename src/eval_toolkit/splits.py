"""Data splitting: pluggable :class:`Splitter` Protocol + reference impls.

A :class:`Splitter` takes a single :class:`~eval_toolkit.harness.EvalSlice`
and yields fold-dicts ready for :func:`eval_toolkit.harness.evaluate`. K=1
splitters (:class:`HoldoutSplitter`) yield one item; K=5 splitters yield five
fold-dicts each shaped like ``{"train": EvalSlice, "test": EvalSlice}``.

Reference impls wrap sklearn splitters except for
:class:`SourceDisjointKFoldSplitter`, which generalizes the source-disjoint
pattern that ``prompt-injection-sdd`` hand-rolls today.

Composes naturally with :class:`~eval_toolkit.loaders.DatasetLoader`:
loaders return a single ``{"all": slice}`` dict for un-split data; pipe
``splits["all"]`` into the Splitter for cross-validated evaluation. Splits
returned by the loader (e.g. HF ``DatasetDict``-style ``{"train", "test"}``)
can be fed into the harness directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from eval_toolkit.harness import EvalSlice

__all__ = [
    "GroupKFoldSplitter",
    "HoldoutSplitter",
    "PoolBuilder",
    "SourceDisjointKFoldSplitter",
    "Splitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
    "iter_folds_with_pool",
]


@runtime_checkable
class Splitter(Protocol):
    """Iterates folds, each as a named-splits dict ready for ``evaluate(...)``.

    :meth:`iter_folds` yields ``Iterator[dict[str, EvalSlice]]``. K=1 yields a
    single holdout-shaped item; K=5 yields five fold dicts. The caller composes
    with a seed loop for multi-seed × CV.

    All reference impls preserve the parent slice's feature_col / label_col /
    strata_col so each fold's :class:`EvalSlice` is interchangeable with the
    parent — pass straight to :func:`eval_toolkit.harness.evaluate` without
    further wiring.
    """

    def iter_folds(  # pragma: no cover
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield one fold-dict per iteration."""
        ...

    def get_n_splits(self, slice_: EvalSlice) -> int:  # pragma: no cover
        """Return the number of folds this splitter will produce on ``slice_``."""
        ...


def _slice_subset(parent: EvalSlice, mask_or_idx: np.ndarray, name: str) -> EvalSlice:
    """Build a child :class:`EvalSlice` carrying the parent's column metadata."""
    sub_df = parent.df.iloc[mask_or_idx].reset_index(drop=True)
    return EvalSlice(
        name=name,
        df=sub_df,
        description=f"{parent.description} [{name}]" if parent.description else name,
        feature_col=parent.feature_col,
        label_col=parent.label_col,
        strata_col=parent.strata_col,
    )


@dataclass(frozen=True, slots=True)
class HoldoutSplitter:
    """Single-iteration (k=1) holdout split via sklearn ``train_test_split``.

    Unifies holdout into the same iterator shape as K-fold so callers can treat
    holdout and CV identically (one ``for fold in splitter.iter_folds(slice_)``
    loop covers both).

    Parameters
    ----------
    test_size : float, optional
        Fraction in (0, 1). Default 0.2.
    stratify : bool, optional
        If True, stratify on labels. Default True (binary class imbalance is
        the dominant case for this toolkit).
    seed : int, optional
        RNG seed. Default 42.
    """

    test_size: float = 0.2
    stratify: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the holdout fraction."""
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield exactly one ``{"train", "test"}`` fold dict."""
        n = len(slice_.df)
        idx = np.arange(n)
        stratify_arr = slice_.y_true if self.stratify else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            stratify=stratify_arr,
            random_state=self.seed,
        )
        yield {
            "train": _slice_subset(slice_, train_idx, "train"),
            "test": _slice_subset(slice_, test_idx, "test"),
        }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Always 1 — holdout is k=1."""
        return 1


@dataclass(frozen=True, slots=True)
class StratifiedKFoldSplitter:
    """K-fold cross-validation with class-stratification.

    Wraps :class:`sklearn.model_selection.StratifiedKFold`. Default for
    binary class imbalance — keeps positive/negative ratios stable across
    folds.

    Parameters
    ----------
    k : int, optional
        Number of folds. Default 5.
    shuffle : bool, optional
        Shuffle indices before splitting. Default True.
    seed : int, optional
        RNG seed. Default 42 (only used when ``shuffle=True``).
    """

    k: int = 5
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate k."""
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield k fold dicts, each ``{"train", "test"}``."""
        skf = StratifiedKFold(
            n_splits=self.k,
            shuffle=self.shuffle,
            random_state=self.seed if self.shuffle else None,
        )
        y = slice_.y_true
        x_dummy = np.arange(len(y)).reshape(-1, 1)
        for train_idx, test_idx in skf.split(x_dummy, y):
            yield {
                "train": _slice_subset(slice_, train_idx, "train"),
                "test": _slice_subset(slice_, test_idx, "test"),
            }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``self.k``."""
        return self.k


@dataclass(frozen=True, slots=True)
class GroupKFoldSplitter:
    """K-fold CV with group-disjoint test partitions.

    Wraps :class:`sklearn.model_selection.GroupKFold`. Required when rows
    cluster by user / patient / source / document and within-cluster
    correlation would inflate eval metrics if any group spans train↔test.

    Parameters
    ----------
    k : int, optional
        Number of folds. Default 5.
    group_col : str, optional
        Column name in the parent slice's dataframe carrying group ids.
        ``None`` means callers must pass ``groups`` explicitly to ``iter_folds``.
        Default ``None``.
    """

    k: int = 5
    group_col: str | None = None

    def __post_init__(self) -> None:
        """Validate k."""
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")

    def _resolve_groups(self, slice_: EvalSlice, groups: np.ndarray | None) -> np.ndarray:
        """Pick the groups array from explicit param or column, with diagnostics."""
        if groups is not None:
            return np.asarray(groups)
        if self.group_col is None:
            raise ValueError(
                "GroupKFoldSplitter needs either `group_col` set or `groups=...` "
                "passed to iter_folds; neither was provided."
            )
        if self.group_col not in slice_.df.columns:
            raise KeyError(
                f"group_col {self.group_col!r} not in slice columns {list(slice_.df.columns)}"
            )
        arr: np.ndarray = slice_.df[self.group_col].to_numpy()
        return arr

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield k fold dicts with group-disjoint test partitions."""
        gkf = GroupKFold(n_splits=self.k)
        g = self._resolve_groups(slice_, groups)
        x_dummy = np.arange(len(g)).reshape(-1, 1)
        for train_idx, test_idx in gkf.split(x_dummy, slice_.y_true, groups=g):
            yield {
                "train": _slice_subset(slice_, train_idx, "train"),
                "test": _slice_subset(slice_, test_idx, "test"),
            }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``self.k``."""
        return self.k


@dataclass(frozen=True, slots=True)
class SourceDisjointKFoldSplitter:
    """K-fold CV partitioning sources into disjoint groups (round-robin).

    Generalizes the source-disjoint split pattern from ``prompt-injection-sdd``:
    given a ``source_col`` of categorical values, sort distinct sources by a
    deterministic key and round-robin assign to k folds. Fold ``i``'s test set
    = rows whose source is in bucket ``i``.

    Stronger than :class:`GroupKFoldSplitter` for the OOD-claim case: the test
    fold's sources never appear anywhere in the model's training set across
    the whole CV procedure (whereas GroupKFold only enforces train↔test
    disjointness within each fold).

    Parameters
    ----------
    k : int, optional
        Number of folds. Default 3 (matches prompt-injection-sdd convention).
    source_col : str
        Column in the parent slice's dataframe carrying source labels.
    seed : int, optional
        RNG seed for source-order shuffling. Default 42.
    """

    source_col: str
    k: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate k."""
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield k fold dicts; each fold's test sources are disjoint from train sources.

        Raises
        ------
        KeyError
            If ``self.source_col`` is not a column in ``slice_.df``.
        """
        if self.source_col not in slice_.df.columns:
            raise KeyError(
                f"source_col {self.source_col!r} not in slice columns " f"{list(slice_.df.columns)}"
            )
        sources = slice_.df[self.source_col].to_numpy()
        unique_sources = np.array(sorted(set(sources.tolist())))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_sources)
        # Round-robin: bucket i = sources at positions [i, i+k, i+2k, ...].
        for fold_idx in range(self.k):
            test_sources = set(unique_sources[fold_idx :: self.k].tolist())
            test_mask = np.array([s in test_sources for s in sources])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield {
                "train": _slice_subset(slice_, train_idx, "train"),
                "test": _slice_subset(slice_, test_idx, "test"),
            }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``self.k`` (capped at the number of distinct sources)."""
        if self.source_col not in slice_.df.columns:
            return self.k  # caller will hit the KeyError on iter_folds
        n_sources = int(slice_.df[self.source_col].nunique())
        return min(self.k, n_sources)


@dataclass(frozen=True, slots=True)
class TimeSeriesSplitter:
    """Time-aware K-fold via :class:`sklearn.model_selection.TimeSeriesSplit`.

    Each fold's train set is everything ≤ a moving boundary; the test set is
    the next chunk after the boundary. Required for honest time-series eval —
    use with :class:`~eval_toolkit.leakage.TemporalLeakageCheck` to verify the
    invariant.

    Parameters
    ----------
    k : int, optional
        Number of folds. Default 5.
    time_col : str, optional
        Column name carrying a sortable timestamp. If set, the parent slice
        is sorted by this column before splitting. ``None`` assumes the
        slice is already in temporal order.
    """

    k: int = 5
    time_col: str | None = None

    def __post_init__(self) -> None:
        """Validate k."""
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield k fold dicts respecting the temporal ordering.

        Raises
        ------
        KeyError
            If ``self.time_col`` is set but not present in ``slice_.df``.
        """
        if self.time_col is not None:
            if self.time_col not in slice_.df.columns:
                raise KeyError(
                    f"time_col {self.time_col!r} not in slice columns " f"{list(slice_.df.columns)}"
                )
            sorted_df = slice_.df.sort_values(self.time_col).reset_index(drop=True)
            sorted_slice = EvalSlice(
                name=slice_.name,
                df=sorted_df,
                description=slice_.description,
                feature_col=slice_.feature_col,
                label_col=slice_.label_col,
                strata_col=slice_.strata_col,
            )
        else:
            sorted_slice = slice_
        tss = TimeSeriesSplit(n_splits=self.k)
        x_dummy = np.arange(len(sorted_slice.df)).reshape(-1, 1)
        for train_idx, test_idx in tss.split(x_dummy):
            yield {
                "train": _slice_subset(sorted_slice, train_idx, "train"),
                "test": _slice_subset(sorted_slice, test_idx, "test"),
            }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``self.k``."""
        return self.k


# ---------------------------------------------------------------------------
# PoolBuilder Protocol + iter_folds_with_pool composition (v0.19.0)
# ---------------------------------------------------------------------------


@runtime_checkable
class PoolBuilder(Protocol):
    """Augment a fold's train slice with an external pool and split off val.

    Composes with :class:`Splitter`: the Splitter produces a fold's
    ``{"train", "test"}``; the PoolBuilder enriches ``train`` (typically
    by injecting an external negative pool or a domain-specific corpus)
    and carves a ``val`` slice off the augmented training set.

    Implementations carry their pool state in instance attributes so the
    composition helper :func:`iter_folds_with_pool` can be configured
    once outside the fold loop.
    """

    def build(  # pragma: no cover
        self,
        train: EvalSlice,
        *,
        fold_idx: int,
    ) -> dict[str, EvalSlice]:
        """Return at least ``{"train": ..., "val": ...}`` for this fold.

        The original ``test`` slice from the Splitter is reattached by
        :func:`iter_folds_with_pool`. Implementations may return additional
        keys (e.g. ``"calibration"``) which the helper forwards verbatim.
        """
        ...


def iter_folds_with_pool(
    splitter: Splitter,
    slice_: EvalSlice,
    *,
    pool_builder: PoolBuilder,
    groups: np.ndarray | None = None,
) -> Iterator[dict[str, EvalSlice]]:
    """Compose a :class:`Splitter` with a :class:`PoolBuilder`.

    Each yielded fold-dict contains the PoolBuilder's ``train``/``val``
    plus the Splitter's original ``test``. Additional keys returned by
    the PoolBuilder are forwarded unchanged.

    Parameters
    ----------
    splitter : Splitter
        Yields per-fold ``{"train", "test"}`` (or richer) dicts.
    slice_ : EvalSlice
        The parent slice whose rows the splitter partitions.
    pool_builder : PoolBuilder
        Carved-train/val builder; sees one fold's ``train`` at a time.
    groups : np.ndarray or None, optional
        Forwarded to ``splitter.iter_folds`` (e.g. for
        :class:`GroupKFoldSplitter`).

    Yields
    ------
    dict[str, EvalSlice]
        Per-fold dict with at minimum ``train``, ``val``, ``test``.

    Raises
    ------
    ValueError
        If the ``pool_builder.build(...)`` return dict does not contain
        both ``"train"`` and ``"val"`` keys.

    Examples
    --------
    >>> import pandas as pd
    >>> from eval_toolkit.harness import EvalSlice
    >>> from eval_toolkit.splits import (
    ...     StratifiedKFoldSplitter,
    ...     PoolBuilder,
    ...     iter_folds_with_pool,
    ... )
    >>> df = pd.DataFrame({
    ...     "text": [f"t{i}" for i in range(20)],
    ...     "label": [i % 2 for i in range(20)],
    ... })
    >>> parent = EvalSlice(name="all", df=df)
    >>> class TrivialPool:
    ...     def build(self, train, *, fold_idx):
    ...         # Identity pool builder: train passes through; val carved 50/50.
    ...         n = len(train.df)
    ...         half = n // 2
    ...         val_df = train.df.iloc[:half].copy()
    ...         tr_df = train.df.iloc[half:].copy()
    ...         return {
    ...             "train": EvalSlice(name=f"fold{fold_idx}_train", df=tr_df),
    ...             "val": EvalSlice(name=f"fold{fold_idx}_val", df=val_df),
    ...         }
    >>> folds = list(iter_folds_with_pool(
    ...     StratifiedKFoldSplitter(k=2, seed=0),
    ...     parent,
    ...     pool_builder=TrivialPool(),
    ... ))
    >>> len(folds)
    2
    >>> sorted(folds[0].keys())
    ['test', 'train', 'val']
    """
    for fold_idx, fold in enumerate(splitter.iter_folds(slice_, groups=groups)):
        train = fold["train"]
        test = fold["test"]
        built = pool_builder.build(train, fold_idx=fold_idx)
        if "train" not in built or "val" not in built:
            raise ValueError(
                "PoolBuilder.build must return at least {'train', 'val'}; "
                f"got keys {sorted(built.keys())}"
            )
        # PoolBuilder's keys (train, val, possibly more) take precedence;
        # test is reattached from the Splitter.
        yield {**built, "test": test}
