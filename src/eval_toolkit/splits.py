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
    "PurgedKFoldSplitter",
    "SourceDisjointKFoldSplitter",
    "Splitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
    "compute_label_overlap",
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
        """Yield up to ``min(k, n_sources)`` fold dicts; each test set is source-disjoint.

        When ``self.k > n_sources`` the fold count is CAPPED at
        ``n_sources`` (matching :meth:`get_n_splits`) and a
        :class:`UserWarning` is emitted. The pre-v0.51 implementation
        always looped ``range(self.k)`` and yielded empty test sets for
        the surplus folds — silently violating the implicit
        ``len(list(iter_folds())) == get_n_splits()`` invariant. R8-C2
        audit fix.

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
        n_sources = len(unique_sources)
        # R8-C2 fix: cap fold count at n_sources. Without this cap,
        # range(self.k) would yield empty test partitions when k > n_sources
        # because unique_sources[fold_idx :: self.k] is empty once
        # fold_idx >= n_sources. The cap matches get_n_splits()'s
        # already-correct min(self.k, n_sources) return.
        effective_k = min(self.k, n_sources)
        if self.k > n_sources:
            import warnings as _warnings

            _warnings.warn(
                f"SourceDisjointKFoldSplitter k={self.k} > n_sources={n_sources}; "
                f"iter_folds capped to {effective_k} folds. Set k explicitly to "
                "match your data, or accept that fewer folds will be produced "
                "than requested. R8-C2 audit fix.",
                UserWarning,
                stacklevel=2,
            )
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_sources)
        # Round-robin: bucket i = sources at positions [i, i+effective_k, ...].
        for fold_idx in range(effective_k):
            test_sources = set(unique_sources[fold_idx::effective_k].tolist())
            test_mask = np.array([s in test_sources for s in sources])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield {
                "train": _slice_subset(slice_, train_idx, "train"),
                "test": _slice_subset(slice_, test_idx, "test"),
            }

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``min(self.k, n_sources)`` — the actual fold count from :meth:`iter_folds`.

        Pre-v0.51 this value DIFFERED from ``len(list(iter_folds()))``
        when ``k > n_sources`` (the iter_folds loop ran ``range(self.k)``
        and yielded empty test partitions). v0.51 capped both call sites
        at ``min(self.k, n_sources)``, restoring the contract that these
        two methods agree on fold count. R8-C2 audit fix.
        """
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


# ---------------------------------------------------------------------------
# Purged K-fold for label-overlap protection (v0.28.0)
#
# Adapted from temporalcv (Behring 2026) for the financial / forecasting
# label-overlap case: when labels use future data (e.g., H-day forward
# returns), train and test folds can overlap in their LABEL windows even
# when their FEATURE windows don't. Purging drops a band of training
# samples within ``purge_gap`` of each test fold; embargo drops an
# additional fraction of n samples bordering each test fold.
# ---------------------------------------------------------------------------


def compute_label_overlap(
    t_train: np.ndarray,
    t_test: np.ndarray,
    horizon: int,
) -> np.ndarray:
    r"""Boolean ``(n_train, n_test)`` matrix: True where label windows overlap.

    For h-step forward labels, the label at time ``t`` depends on the data
    at times ``[t, t+h]``. Two samples ``t_train[i]`` and ``t_test[j]``
    have label-window overlap if their windows share at least one time
    point — equivalently, if ``|t_train[i] - t_test[j]| < horizon``.

    Use this to audit whether a given train/test split has any label
    leakage. Standalone helper; does NOT require a particular splitter.

    Parameters
    ----------
    t_train : np.ndarray, shape (n_train,)
        Time indices of the training set (any sortable numeric type).
    t_test : np.ndarray, shape (n_test,)
        Time indices of the test set.
    horizon : int
        Label horizon (e.g., ``5`` for 5-step forward returns). Must be
        non-negative; ``horizon=0`` means no overlap is possible.

    Returns
    -------
    np.ndarray, shape (n_train, n_test), dtype bool
        Entry ``(i, j)`` is ``True`` iff
        ``|t_train[i] - t_test[j]| < horizon``.

    Raises
    ------
    ValueError
        If ``horizon`` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> t_train = np.array([0, 1, 5, 6])
    >>> t_test = np.array([3, 4])
    >>> overlap = compute_label_overlap(t_train, t_test, horizon=3)
    >>> overlap
    array([[False, False],
           [ True, False],
           [ True,  True],
           [False,  True]])
    >>> # Sample 0 (t=0): no overlap with test (|0-3|=3, |0-4|=4 ≥ horizon)
    >>> # Sample 1 (t=1): overlaps test[0]=3 (|1-3|=2 < 3)
    >>> # Sample 2 (t=5): overlaps both (|5-3|=2, |5-4|=1)
    >>> # Sample 3 (t=6): overlaps test[1]=4 (|6-4|=2 < 3)

    Notes
    -----
    The check is **symmetric in time**: ``|t_train - t_test| < horizon``
    treats overlap in either temporal direction equally. For strictly
    forward-only label overlap (train-before-test), filter the result
    with ``(t_test[None, :] - t_train[:, None]) > 0``.

    For h-step forward labels: label at time t covers ``[t, t+h)``, so
    two labels at times ``t1, t2`` share data iff their intervals
    overlap, which holds iff ``|t1 - t2| < h``.

    References
    ----------
    .. [1] López de Prado, M. (2018). "Advances in Financial Machine
           Learning." Wiley. Chapter 7: Cross-Validation in Finance.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if horizon == 0:
        return np.zeros((len(t_train), len(t_test)), dtype=bool)
    t_train_arr = np.asarray(t_train)
    t_test_arr = np.asarray(t_test)
    # Outer absolute difference: (n_train, n_test)
    dist = np.abs(t_train_arr[:, None] - t_test_arr[None, :])
    overlap: np.ndarray = dist < horizon
    return overlap


def _apply_purge_embargo(
    test_idx: np.ndarray,
    n_samples: int,
    purge_gap: int,
    embargo_pct: float,
) -> np.ndarray:
    """Build a training-index array excluding the test fold + purge + embargo.

    The test fold's indices are contiguous (TimeSeriesSplit-style); purging
    drops `[test_min - purge_gap, test_max + purge_gap]` from training;
    embargo drops an additional `floor(embargo_pct * n_samples)` indices
    after the test fold (one-sided: protects the post-test region from
    label-window leakage when labels are forward-looking).

    Adapted from temporalcv's ``_apply_purge_and_embargo`` but vectorized
    (no Python-level set/loop) and asymmetric-by-default (embargo only on
    the post-test side, matching López de Prado's original definition).
    """
    test_min = int(np.min(test_idx))
    test_max = int(np.max(test_idx))
    purge_start = max(0, test_min - purge_gap)
    purge_end = min(n_samples, test_max + 1 + purge_gap)
    n_embargo = int(embargo_pct * n_samples)
    embargo_end = min(n_samples, test_max + 1 + n_embargo)

    full_idx = np.arange(n_samples)
    # Mask out: the test fold itself + purge band on both sides + post-test embargo
    keep = np.ones(n_samples, dtype=bool)
    keep[purge_start:purge_end] = False  # zeroes out test + purge band
    keep[test_max + 1 : embargo_end] = False  # post-test embargo
    return full_idx[keep]


@dataclass(frozen=True, slots=True)
class PurgedKFoldSplitter:
    r"""Time-aware k-fold with explicit purge gap + post-test embargo.

    Pattern from López de Prado (2018) Ch. 7: when labels have a forward
    lookahead (e.g., H-step returns), train and test folds can overlap in
    their **label windows** even when their **feature windows** don't.
    Standard k-fold leaks information through this overlap. PurgedKFold
    drops a ``purge_gap``-sample band straddling each test fold's boundary
    plus a post-test ``embargo_pct * n`` window — preventing both
    backward and forward label-overlap leakage.

    Implements the :class:`Splitter` Protocol, yielding
    ``{"train": EvalSlice, "test": EvalSlice}`` dicts.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds. Default 5. Must be ≥ 2.
    purge_gap : int, optional
        Samples to drop on each side of every test fold's boundary.
        Default 0 (no purging — equivalent to vanilla TimeSeriesSplit).
        For h-step forward labels, ``purge_gap=h`` is the canonical choice.
    embargo_pct : float, optional
        Additional embargo as a fraction of total ``n``, applied **after**
        each test fold (one-sided, López de Prado convention). Default
        0.0. Typical: 0.01 (1%).
    time_col : str or None, optional
        Column carrying a sortable timestamp. If set, the parent slice is
        sorted by this column before splitting. ``None`` assumes the slice
        is already in temporal order. Default ``"timestamp"``.

    Raises
    ------
    ValueError
        At construction time if ``n_splits < 2`` or ``purge_gap < 0`` or
        ``embargo_pct ∉ [0, 1)``.
    KeyError
        At ``iter_folds`` time if ``time_col`` is set but not present in
        the slice DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from eval_toolkit.harness import EvalSlice
    >>> from eval_toolkit.splits import PurgedKFoldSplitter
    >>> df = pd.DataFrame({
    ...     "text": [f"row{i}" for i in range(50)],
    ...     "label": [i % 2 for i in range(50)],
    ...     "t": list(range(50)),
    ... })
    >>> parent = EvalSlice(name="all", df=df)
    >>> spl = PurgedKFoldSplitter(n_splits=5, purge_gap=2, embargo_pct=0.02, time_col="t")
    >>> folds = list(spl.iter_folds(parent))
    >>> len(folds)
    5
    >>> sorted(folds[0].keys())
    ['test', 'train']

    Notes
    -----
    **Two units in one signature**: ``purge_gap`` is an absolute count of
    samples (int) while ``embargo_pct`` is a fraction (float). This
    mirrors López de Prado / temporalcv conventions verbatim — users
    moving between libraries see the same parameter names. Use the
    standalone helper :func:`compute_label_overlap` to size ``purge_gap``
    for a known label horizon.

    See Also
    --------
    eval_toolkit.splits.compute_label_overlap :
        Audit label-window overlap between arbitrary train/test sets.
    eval_toolkit.splits.TimeSeriesSplitter :
        Time-aware k-fold without purging — use when labels have no
        lookahead horizon.

    References
    ----------
    .. [1] López de Prado, M. (2018). "Advances in Financial Machine
           Learning." Wiley. Chapter 7.
    """

    n_splits: int = 5
    purge_gap: int = 0
    embargo_pct: float = 0.0
    time_col: str | None = "timestamp"

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {self.purge_gap}")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {self.embargo_pct}")

    def iter_folds(
        self,
        slice_: EvalSlice,
        *,
        groups: np.ndarray | None = None,
    ) -> Iterator[dict[str, EvalSlice]]:
        """Yield ``n_splits`` fold dicts with purge + embargo applied.

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

        n_samples = len(sorted_slice.df)
        if self.n_splits >= n_samples:
            raise ValueError(f"n_splits ({self.n_splits}) must be < n_samples ({n_samples})")

        # Fold sizes (mirrors TimeSeriesSplit / temporalcv: trailing folds
        # absorb the remainder)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_idx = np.arange(current, current + fold_size)
            train_idx = _apply_purge_embargo(
                test_idx,
                n_samples=n_samples,
                purge_gap=self.purge_gap,
                embargo_pct=self.embargo_pct,
            )
            yield {
                "train": _slice_subset(sorted_slice, train_idx, "train"),
                "test": _slice_subset(sorted_slice, test_idx, "test"),
            }
            current += fold_size

    def get_n_splits(self, slice_: EvalSlice) -> int:
        """Return ``self.n_splits``."""
        return self.n_splits
