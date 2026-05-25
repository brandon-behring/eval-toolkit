"""Generic conformance tests for all 8 ``@runtime_checkable`` Protocols.

Covers (v0.7+ unless noted): ``ThresholdSelector``, ``LeakageCheck``,
``Splitter``, ``DatasetLoader``, ``Versioned``, ``Scorer``,
``SliceAwareScorer``, ``SimilarityStrategy`` (text_dedup, v0.2+).

Doubles as:
  1. A self-test on every reference implementation in the toolkit.
  2. A copy-paste template a downstream consumer can adapt to validate
     their own custom impls before plugging them into the harness.

Each Protocol section asserts the *contract*: the runtime ``isinstance``
check passes, required attributes / methods are present and behave as
documented, and the impl is deterministic on the same input.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval_toolkit import (
    CostSensitiveSelector,
    CrossSplitLeakageCheck,
    DataFrameLoader,
    DatasetLoader,
    EvalSlice,
    ExactDuplicateCheck,
    ExactNormalizedHashStrategy,
    GroupKFoldSplitter,
    GroupLeakageCheck,
    HoldoutSplitter,
    JaccardNgramStrategy,
    LabelConflictCheck,
    LeakageCheck,
    LeakageFinding,
    MaxF1Selector,
    NearDuplicateCheck,
    NormalizedFormLeakageCheck,
    SingleSliceLoader,
    SourceDisjointKFoldSplitter,
    Splitter,
    StratifiedKFoldSplitter,
    TargetFPRSelector,
    TargetPrecisionSelector,
    TargetRecallSelector,
    TemporalLeakageCheck,
    TfidfCosineStrategy,
    ThresholdSelector,
    TimeSeriesSplitter,
    Versioned,
    YoudenJSelector,
)
from eval_toolkit.harness import Scorer, SliceAwareScorer
from eval_toolkit.metrics import ThresholdResult
from eval_toolkit.text_dedup import SimilarityStrategy
from eval_toolkit.thresholds import CostMatrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_binary() -> tuple[np.ndarray, np.ndarray]:
    """200-row binary fixture with a real signal."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, size=200).astype(int)
    s = np.clip(0.6 * y + rng.normal(0, 0.25, size=200), 0, 1)
    return y, s


@pytest.fixture
def synth_slice(synth_binary: tuple[np.ndarray, np.ndarray]) -> EvalSlice:
    """200-row EvalSlice (with `text`, `label`, `group`, `source`, `timestamp`)."""
    y, s = synth_binary
    n = len(y)
    rng = np.random.default_rng(0)
    return EvalSlice(
        name="synth",
        df=pd.DataFrame(
            {
                "text": [f"sample {i}" for i in range(n)],
                "label": y,
                "group": rng.integers(0, 10, size=n),
                "source": rng.choice(["a", "b", "c"], size=n),
                "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
                "score": s,
            }
        ),
        feature_col="text",
        label_col="label",
    )


# ---------------------------------------------------------------------------
# ThresholdSelector
# ---------------------------------------------------------------------------


THRESHOLD_SELECTORS: list[ThresholdSelector] = [
    MaxF1Selector(),
    TargetRecallSelector(recall=0.8),
    TargetPrecisionSelector(precision=0.5),
    TargetFPRSelector(fpr=0.1),
    YoudenJSelector(),
    CostSensitiveSelector(cost_matrix=CostMatrix(prior=0.3, fp_cost=1.0, fn_cost=5.0)),
]


@pytest.mark.unit
@pytest.mark.parametrize("selector", THRESHOLD_SELECTORS, ids=lambda s: type(s).__name__)
def test_threshold_selector_conformance(
    selector: ThresholdSelector,
    synth_binary: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = synth_binary
    # 1. Runtime structural check.
    assert isinstance(selector, ThresholdSelector)
    # 2. criterion attribute is non-empty str.
    assert isinstance(selector.criterion, str) and selector.criterion
    # 3. select() returns a ThresholdResult.
    result = selector.select(y, s)
    assert isinstance(result, ThresholdResult)
    # 4. Threshold lies inside (or just outside by ε) the score range.
    eps = 1e-9
    assert s.min() - eps <= result.threshold <= s.max() + eps
    # 5. Idempotent on identical input.
    again = selector.select(y, s)
    assert result.threshold == again.threshold
    assert result.f1 == again.f1
    # 6. The result's criterion field matches the selector's criterion.
    assert result.criterion == selector.criterion


@pytest.mark.unit
def test_threshold_selector_negative_isinstance() -> None:
    """An object without `select` + `criterion` is NOT a ThresholdSelector."""

    class _NotASelector:
        pass

    assert not isinstance(_NotASelector(), ThresholdSelector)


# ---------------------------------------------------------------------------
# LeakageCheck
# ---------------------------------------------------------------------------


def _leakage_checks() -> list[LeakageCheck]:
    return [
        ExactDuplicateCheck(target_splits=("train", "test")),
        NearDuplicateCheck(target_splits=("train", "test"), threshold=0.9),
        NormalizedFormLeakageCheck(target_splits=("train", "test")),
        CrossSplitLeakageCheck(train_split="train", eval_splits=("test",)),
        GroupLeakageCheck(group_col="group", target_splits=("train", "test")),
        LabelConflictCheck(target_splits=("train", "test")),
        TemporalLeakageCheck(time_col="timestamp", split_order=("train", "test")),
    ]


@pytest.fixture
def two_split_fixture(synth_slice: EvalSlice) -> dict[str, EvalSlice]:
    n = len(synth_slice.df)
    half = n // 2
    train_df = synth_slice.df.iloc[:half].reset_index(drop=True)
    test_df = synth_slice.df.iloc[half:].reset_index(drop=True)
    return {
        "train": EvalSlice(name="train", df=train_df, feature_col="text", label_col="label"),
        "test": EvalSlice(name="test", df=test_df, feature_col="text", label_col="label"),
    }


@pytest.mark.unit
@pytest.mark.parametrize("check", _leakage_checks(), ids=lambda c: type(c).__name__)
def test_leakage_check_conformance(
    check: LeakageCheck,
    two_split_fixture: dict[str, EvalSlice],
) -> None:
    # 1. Runtime structural check.
    assert isinstance(check, LeakageCheck)
    # 2. name attribute is non-empty str.
    assert isinstance(check.name, str) and check.name
    # 3. validate() returns a LeakageFinding.
    finding = check.validate(two_split_fixture)
    assert isinstance(finding, LeakageFinding)
    # 4. Severity is one of the documented values.
    assert finding.severity in {"error", "warning", "info"}
    # 5. n_affected is non-negative.
    assert finding.n_affected >= 0
    # 6. Deterministic — same input → same n_affected.
    again = check.validate(two_split_fixture)
    assert finding.n_affected == again.n_affected


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def _splitters() -> list[tuple[Splitter, dict[str, object]]]:
    """(splitter, kwargs) pairs — kwargs passed through to iter_folds()."""
    return [
        (HoldoutSplitter(test_size=0.2, seed=42), {}),
        (StratifiedKFoldSplitter(k=3, seed=42), {}),
        (GroupKFoldSplitter(k=3, group_col="group"), {}),
        (SourceDisjointKFoldSplitter(source_col="source", k=3, seed=42), {}),
        (TimeSeriesSplitter(k=3, time_col="timestamp"), {}),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "splitter,kwargs",
    _splitters(),
    ids=lambda x: type(x).__name__ if hasattr(x, "iter_folds") else "kwargs",
)
def test_splitter_conformance(
    splitter: Splitter,
    kwargs: dict[str, object],
    synth_slice: EvalSlice,
) -> None:
    # 1. Runtime structural check.
    assert isinstance(splitter, Splitter)
    # 2. get_n_splits returns a positive int.
    n_splits = splitter.get_n_splits(synth_slice)
    assert isinstance(n_splits, int) and n_splits >= 1
    # 3. iter_folds yields exactly that many dicts.
    folds = list(splitter.iter_folds(synth_slice, **kwargs))  # type: ignore[arg-type]
    assert len(folds) == n_splits
    # 4. Each fold dict contains at least a "test" split with a non-empty EvalSlice.
    for fold in folds:
        assert isinstance(fold, dict) and "test" in fold
        assert isinstance(fold["test"], EvalSlice)
        assert len(fold["test"].df) > 0
    # 5. For K > 1: test rows across folds are disjoint (using the unique
    # `text` column as the row identifier — DataFrame indices may be reset
    # to 0..n-1 per fold so .index is unreliable).
    if n_splits > 1:
        seen_texts: list[str] = []
        for fold in folds:
            seen_texts.extend(fold["test"].df["text"].tolist())
        assert len(seen_texts) == len(set(seen_texts)), (
            f"{type(splitter).__name__}: test rows duplicated across folds "
            f"({len(seen_texts) - len(set(seen_texts))} duplicates)"
        )


@pytest.mark.unit
def test_source_disjoint_kfold_caps_at_n_sources_when_k_exceeds() -> None:
    """R8-C2 regression: SourceDisjointKFoldSplitter with k > n_sources caps loop.

    Pre-v0.51: ``iter_folds`` looped ``range(self.k)`` and yielded empty
    test partitions for the surplus folds when k > n_sources, while
    ``get_n_splits`` returned ``min(k, n_sources)``. The two methods
    disagreed on fold count — a silent contract violation. v0.51 caps
    ``iter_folds`` at the same ``min(k, n_sources)`` value AND emits a
    UserWarning so the caller knows the cap was applied.
    """
    import warnings as _warnings

    import pandas as pd

    df = pd.DataFrame(
        {
            "text": [f"row{i}" for i in range(30)],
            "label": [i % 2 for i in range(30)],
            "source": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        }
    )
    parent = EvalSlice(name="parent", df=df)
    splitter = SourceDisjointKFoldSplitter(k=5, source_col="source", seed=42)

    # The R8-C2 fix triggers a UserWarning when k > n_sources.
    with _warnings.catch_warnings(record=True) as ws:
        _warnings.simplefilter("always")
        folds = list(splitter.iter_folds(parent))
    n_splits = splitter.get_n_splits(parent)

    # Contract: len(iter_folds()) == get_n_splits()
    assert len(folds) == n_splits == 3, (
        f"iter_folds yielded {len(folds)} but get_n_splits returned "
        f"{n_splits}; pre-v0.51 this was 5 vs 3 (contract violation)"
    )
    # No empty test partitions
    for f in folds:
        assert len(f["test"].df) > 0, "every fold must have a non-empty test partition"
    # UserWarning was emitted
    user_warnings = [w for w in ws if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1, "UserWarning must be emitted when k > n_sources"
    assert "k=5" in str(user_warnings[0].message)
    assert "n_sources=3" in str(user_warnings[0].message)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


def _dataset_loaders(synth_slice: EvalSlice) -> list[DatasetLoader]:
    df = synth_slice.df.copy()
    df["__split__"] = ["train"] * (len(df) // 2) + ["test"] * (len(df) - len(df) // 2)
    return [
        DataFrameLoader(
            df=df,
            split_col="__split__",
            feature_col="text",
            label_col="label",
            name="synth-df",
            description="synth fixture",
        ),
        SingleSliceLoader(
            slice_=synth_slice,
            name="synth-single",
            description="synth fixture",
        ),
    ]


@pytest.mark.unit
def test_dataset_loader_conformance(synth_slice: EvalSlice) -> None:
    for loader in _dataset_loaders(synth_slice):
        # 1. Runtime structural check.
        assert isinstance(loader, DatasetLoader), f"{type(loader).__name__} fails isinstance"
        # 2. load_splits returns a dict[str, EvalSlice], non-empty.
        splits = loader.load_splits()
        assert isinstance(splits, dict) and len(splits) >= 1
        for k, v in splits.items():
            assert isinstance(k, str) and k
            assert isinstance(v, EvalSlice)
        # 3. describe() returns a Croissant-subset dict.
        desc = loader.describe()
        assert isinstance(desc, dict)
        for required in ("name", "description", "distribution"):
            assert required in desc, f"{type(loader).__name__}.describe() missing {required!r}"


@pytest.mark.unit
def test_parquet_glob_loader_roundtrip(synth_slice: EvalSlice) -> None:
    """ParquetGlobLoader needs filesystem fixtures — kept separate from the
    in-memory loaders above."""
    pyarrow = pytest.importorskip("pyarrow")
    del pyarrow  # imported just to gate the test
    from eval_toolkit import ParquetGlobLoader

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synth.parquet"
        synth_slice.df.to_parquet(path)
        loader = ParquetGlobLoader(
            splits={"all": str(path)},
            feature_col="text",
            label_col="label",
            name="synth-parquet",
        )
        assert isinstance(loader, DatasetLoader)
        splits = loader.load_splits()
        assert "all" in splits and len(splits["all"].df) == len(synth_slice.df)
        desc = loader.describe()
        assert "distribution" in desc


# ---------------------------------------------------------------------------
# Versioned (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_versioned_protocol_positive() -> None:
    """An object exposing ``version: str`` IS a Versioned."""

    class _MyScorer:
        version = "1.2.3"

    obj = _MyScorer()
    assert isinstance(obj, Versioned)
    assert obj.version == "1.2.3"
    # Stable across calls.
    assert obj.version == "1.2.3"


@pytest.mark.unit
def test_versioned_protocol_negative() -> None:
    """An object WITHOUT a ``version`` attribute is NOT a Versioned."""

    class _Untagged:
        pass

    assert not isinstance(_Untagged(), Versioned)


# ---------------------------------------------------------------------------
# Scorer + SliceAwareScorer (the central Protocols every consumer extends)
# ---------------------------------------------------------------------------


class _ConstantScorer:
    """Returns a fixed score per row (smallest valid Scorer impl)."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def predict_proba(self, X: list[str] | pd.Series | np.ndarray) -> np.ndarray:
        return np.full(len(X), self._score, dtype=float)


class _SliceSkippingScorer(_ConstantScorer):
    """Adds the should_score_slice() opt-in for cost-controlled slice skipping."""

    def __init__(self, score: float = 0.5, skip_slice: str = "skip_me") -> None:
        super().__init__(score)
        self._skip = skip_slice

    def should_score_slice(self, slice_name: str) -> bool:
        return slice_name != self._skip


@pytest.mark.unit
def test_scorer_protocol_conformance() -> None:
    scorer = _ConstantScorer(score=0.7)
    # 1. Runtime structural check.
    assert isinstance(scorer, Scorer)
    # 2. predict_proba returns 1-D float ndarray of correct length.
    out = scorer.predict_proba(["a", "b", "c"])
    assert isinstance(out, np.ndarray) and out.shape == (3,) and out.dtype == np.float64
    # 3. Idempotent on identical input.
    out2 = scorer.predict_proba(["a", "b", "c"])
    assert np.array_equal(out, out2)


@pytest.mark.unit
def test_scorer_protocol_negative() -> None:
    """An object without predict_proba is NOT a Scorer."""

    class _NotAScorer:
        pass

    assert not isinstance(_NotAScorer(), Scorer)


@pytest.mark.unit
def test_slice_aware_scorer_protocol_conformance() -> None:
    scorer = _SliceSkippingScorer(score=0.3, skip_slice="ood")
    # SliceAwareScorer extends Scorer — both isinstance checks pass.
    assert isinstance(scorer, Scorer)
    assert isinstance(scorer, SliceAwareScorer)
    # should_score_slice returns bool with the documented contract.
    assert scorer.should_score_slice("dev_test") is True
    assert scorer.should_score_slice("ood") is False


@pytest.mark.unit
def test_slice_aware_scorer_negative() -> None:
    """A plain Scorer (no should_score_slice) is NOT a SliceAwareScorer."""
    plain = _ConstantScorer()
    assert isinstance(plain, Scorer)
    assert not isinstance(plain, SliceAwareScorer)


# ---------------------------------------------------------------------------
# SimilarityStrategy (text_dedup backbone — drives all near-leakage checks)
# ---------------------------------------------------------------------------


def _fixed_embedder(texts) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Tiny deterministic 'embedding' for EmbeddingCosineStrategy tests."""
    out = np.zeros((len(texts), 4), dtype=np.float64)
    for i, t in enumerate(texts):
        out[i, 0] = float(len(t))
        out[i, 1] = float(sum(map(ord, t)) % 100)
        out[i, 2] = float(t.count(" "))
        out[i, 3] = 1.0  # bias dim
    return out


def _similarity_strategies() -> list[SimilarityStrategy]:
    from eval_toolkit import EmbeddingCosineStrategy

    strategies: list[SimilarityStrategy] = [
        TfidfCosineStrategy(ngram_range=(1, 2)),
        ExactNormalizedHashStrategy(),
        JaccardNgramStrategy(n=3, analyzer="char"),
        EmbeddingCosineStrategy(embedder=_fixed_embedder),
    ]
    # MinHashLSH needs the optional `datasketch` package.
    try:
        import datasketch  # noqa: F401

        from eval_toolkit import MinHashLSHStrategy

        strategies.append(MinHashLSHStrategy(n=3, num_perm=64, bands=8, seed=42))
    except ImportError:
        pass
    return strategies


@pytest.fixture(scope="module")
def synth_texts() -> list[str]:
    return [
        "the quick brown fox jumps",
        "the quick brown fox jumps!",  # near-dupe
        "the slow blue cat naps",
        "lorem ipsum dolor sit amet",
        "lorem ipsum dolor sit amet.",  # near-dupe
        "completely unrelated content here",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("strategy", _similarity_strategies(), ids=lambda s: type(s).__name__)
def test_similarity_strategy_conformance(
    strategy: SimilarityStrategy,
    synth_texts: list[str],
) -> None:
    # 1. Runtime structural check.
    assert isinstance(strategy, SimilarityStrategy)
    # 2. pairs_within: aligned (similarities, indices) of shape (n, k_eff).
    n = len(synth_texts)
    k = 3
    sims, idx = strategy.pairs_within(synth_texts, k=k)
    assert isinstance(sims, np.ndarray) and isinstance(idx, np.ndarray)
    assert sims.shape == idx.shape
    assert sims.shape[0] == n
    assert sims.shape[1] <= k  # k_eff = min(k, n_reference)
    # 3. pairs_across: shape (n_query, k_eff) over reference list.
    query = synth_texts[:3]
    ref = synth_texts[3:]
    sims_x, idx_x = strategy.pairs_across(query, ref, k=2)
    assert sims_x.shape == idx_x.shape
    assert sims_x.shape[0] == len(query)
    assert sims_x.shape[1] <= 2


@pytest.mark.unit
def test_similarity_strategy_negative() -> None:
    """An object missing pairs_within / pairs_across is NOT a SimilarityStrategy."""

    class _NotAStrategy:
        pass

    assert not isinstance(_NotAStrategy(), SimilarityStrategy)
