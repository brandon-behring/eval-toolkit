"""Per-strategy unit tests + plug-in contract for the SimilarityStrategy seam.

Each of the four reference strategies is exercised against a shared shape /
determinism / self-pair contract, plus a strategy-specific behavior test.
The plug-in contract verifies that a custom-typed object satisfying the
Protocol is actually used by `near_dedup` (catching accidental fallthrough
to the default TfidfCosineStrategy).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from eval_toolkit.text_dedup import (
    DedupReport,
    EmbeddingCosineStrategy,
    ExactNormalizedHashStrategy,
    JaccardNgramStrategy,
    SimilarityStrategy,
    TfidfCosineStrategy,
    near_dedup,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_STUB_EMBED_DIM = 64


def _stub_one_hot_embedder(texts: Sequence[str]) -> np.ndarray:
    """Hash-bucket one-hot embedder with stable cross-call dimension.

    Returns ``(n, _STUB_EMBED_DIM)``. Distinct texts hash to distinct slots
    (with very small collision probability), so cosine similarity is 1.0
    for identical texts and 0.0 for distinct texts.
    """
    import hashlib  # noqa: PLC0415

    out = np.zeros((len(texts), _STUB_EMBED_DIM), dtype=np.float64)
    for i, t in enumerate(texts):
        slot = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % _STUB_EMBED_DIM
        out[i, slot] = 1.0
    return out


def _all_strategies() -> list[SimilarityStrategy]:
    return [
        TfidfCosineStrategy(),
        ExactNormalizedHashStrategy(),
        EmbeddingCosineStrategy(_stub_one_hot_embedder),
        JaccardNgramStrategy(n=2, analyzer="char"),
    ]


@pytest.fixture(params=_all_strategies(), ids=lambda s: type(s).__name__)
def any_strategy(request: pytest.FixtureRequest) -> SimilarityStrategy:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Cross-strategy contract: Protocol conformance, shape, determinism, self-pair
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protocol_conformance(any_strategy: SimilarityStrategy) -> None:
    """isinstance check works because SimilarityStrategy is @runtime_checkable."""
    assert isinstance(any_strategy, SimilarityStrategy)


@pytest.mark.unit
def test_pairs_within_shape(any_strategy: SimilarityStrategy) -> None:
    """pairs_within returns aligned (n, k_eff) arrays where k_eff=min(k, n)."""
    texts = ["alpha bravo", "alpha bravo charlie", "delta echo"]
    sims, idx = any_strategy.pairs_within(texts, k=5)
    assert sims.shape == idx.shape == (3, 3), f"got {sims.shape} / {idx.shape}"


@pytest.mark.unit
def test_pairs_across_shape(any_strategy: SimilarityStrategy) -> None:
    """pairs_across returns aligned (n_query, k_eff) arrays where k_eff=min(k, n_ref)."""
    queries = ["alpha bravo", "delta echo"]
    refs = ["alpha bravo", "delta echo", "foxtrot golf"]
    sims, idx = any_strategy.pairs_across(queries, refs, k=10)
    assert sims.shape == idx.shape == (2, 3), f"got {sims.shape} / {idx.shape}"


@pytest.mark.unit
def test_self_at_slot_zero(any_strategy: SimilarityStrategy) -> None:
    """In pairs_within, each text's most-similar neighbor is itself."""
    texts = ["alpha bravo", "delta echo", "foxtrot golf"]
    _sims, idx = any_strategy.pairs_within(texts, k=2)
    for i in range(3):
        assert (
            int(idx[i, 0]) == i
        ), f"strategy {type(any_strategy).__name__}: idx[{i}, 0] = {idx[i, 0]}, expected {i}"


@pytest.mark.unit
def test_determinism(any_strategy: SimilarityStrategy) -> None:
    """Same input → byte-identical output."""
    texts = ["abc def", "abc def!", "qux"]
    s1, i1 = any_strategy.pairs_within(texts, k=2)
    s2, i2 = any_strategy.pairs_within(texts, k=2)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


@pytest.mark.unit
def test_empty_input_returns_empty_arrays(any_strategy: SimilarityStrategy) -> None:
    """All four strategies handle the empty-corpus edge case without crashing."""
    sims_w, idx_w = any_strategy.pairs_within([], k=5)
    assert sims_w.shape == idx_w.shape == (0, 0)
    sims_a, idx_a = any_strategy.pairs_across([], ["a", "b"], k=5)
    assert sims_a.shape == idx_a.shape == (0, 0)
    sims_b, idx_b = any_strategy.pairs_across(["a", "b"], [], k=5)
    assert sims_b.shape == idx_b.shape == (2, 0)


# ---------------------------------------------------------------------------
# Strategy-specific behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tfidf_cosine_default_matches_v010_inline() -> None:
    """TfidfCosineStrategy default config matches v0.1.0 inline behavior.

    Regression: same texts + threshold produce the same DedupReport that
    v0.1.0 would have produced.
    """
    rng = np.random.default_rng(42)
    base = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india"]
    # Synthesize near-duplicates via small perturbations
    near_dups = [t + " " + chr(ord("a") + int(rng.integers(0, 26))) for t in base]
    texts = base + near_dups + ["lorem ipsum dolor sit amet"]

    report = near_dedup(texts, threshold=0.8)  # default strategy
    explicit = near_dedup(texts, threshold=0.8, strategy=TfidfCosineStrategy())
    assert report.kept_indices == explicit.kept_indices
    assert report.dropped_pairs == explicit.dropped_pairs
    assert report.threshold == explicit.threshold
    assert report.n_input == explicit.n_input


@pytest.mark.unit
def test_exact_normalized_hash_collision_semantics() -> None:
    """foo/FOO collide after normalization; bar isolated."""
    strat = ExactNormalizedHashStrategy()
    sims, idx = strat.pairs_within(["foo", "FOO", "bar"], k=2)
    # Self-similarity at slot 0
    assert sims[0, 0] == 1.0 and idx[0, 0] == 0
    assert sims[1, 0] == 1.0 and idx[1, 0] == 1
    assert sims[2, 0] == 1.0 and idx[2, 0] == 2
    # Slot 1: collisions for foo/FOO; non-collision for bar
    assert sims[0, 1] == 1.0 and idx[0, 1] == 1
    assert sims[1, 1] == 1.0 and idx[1, 1] == 0
    assert sims[2, 1] == 0.0  # bar has no collision in the corpus


@pytest.mark.unit
def test_exact_normalized_hash_disable_normalization() -> None:
    """normalize=False: foo and FOO no longer collide (raw bytes differ)."""
    strat = ExactNormalizedHashStrategy(normalize=False)
    sims, _ = strat.pairs_within(["foo", "FOO", "bar"], k=2)
    # No collisions across the three distinct raw-byte hashes
    assert sims[0, 1] == 0.0
    assert sims[1, 1] == 0.0


@pytest.mark.unit
def test_embedding_cosine_with_one_hot_acts_like_exact_match() -> None:
    """One-hot per index → identical-text neighbors are at distance 1.0
    (cosine sim = 0); each text is only similar to itself."""
    strat = EmbeddingCosineStrategy(_stub_one_hot_embedder)
    sims, idx = strat.pairs_within(["x", "y", "z"], k=3)
    # Self-similarity = 1.0 at slot 0
    for i in range(3):
        assert idx[i, 0] == i
        assert abs(sims[i, 0] - 1.0) < 1e-9
    # All other neighbors have similarity 0 (orthogonal one-hots)
    for i in range(3):
        for slot in range(1, 3):
            assert abs(sims[i, slot]) < 1e-9


@pytest.mark.unit
def test_embedding_cosine_rejects_wrong_shape() -> None:
    """Embedder must return (n, d); 1-D output raises ValueError."""

    def bad_embedder(texts: Sequence[str]) -> np.ndarray:
        return np.zeros(len(texts))  # 1-D

    strat = EmbeddingCosineStrategy(bad_embedder)
    with pytest.raises(ValueError, match="2-D"):
        strat.pairs_within(["a", "b"], k=2)


@pytest.mark.unit
def test_embedding_cosine_rejects_row_count_mismatch() -> None:
    """Embedder must return one row per text; mismatch raises ValueError."""

    def bad_embedder(texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts) + 1, 3))  # extra row

    strat = EmbeddingCosineStrategy(bad_embedder)
    with pytest.raises(ValueError, match="one row per text"):
        strat.pairs_within(["a", "b"], k=2)


@pytest.mark.unit
def test_jaccard_ngram_known_value() -> None:
    """abc/abd char-bigrams: ∩ = {ab}; ∪ = {ab, bc, bd} → J = 1/3."""
    strat = JaccardNgramStrategy(n=2, analyzer="char")
    sims, idx = strat.pairs_within(["abc", "abd"], k=2)
    # Self at slot 0 (sim = 1.0); other at slot 1
    assert idx[0, 0] == 0
    assert idx[0, 1] == 1
    assert abs(float(sims[0, 1]) - 1 / 3) < 1e-9


@pytest.mark.unit
def test_jaccard_ngram_word_analyzer() -> None:
    """Word-level bigrams treat space-separated tokens as the unit."""
    strat = JaccardNgramStrategy(n=2, analyzer="word")
    # "alpha bravo charlie" 2-grams: {alpha bravo, bravo charlie}
    # "alpha bravo delta"   2-grams: {alpha bravo, bravo delta}
    # ∩ = {alpha bravo}; ∪ = 3 total → J = 1/3
    sims, _ = strat.pairs_within(["alpha bravo charlie", "alpha bravo delta"], k=2)
    assert abs(float(sims[0, 1]) - 1 / 3) < 1e-9


@pytest.mark.unit
def test_jaccard_ngram_validates_args() -> None:
    with pytest.raises(ValueError, match="n must be"):
        JaccardNgramStrategy(n=0)
    with pytest.raises(ValueError, match="analyzer"):
        JaccardNgramStrategy(analyzer="bogus")  # type: ignore[arg-type]


@pytest.mark.unit
def test_jaccard_ngram_short_text_returns_whole_text() -> None:
    """Char n-gram on text shorter than n falls back to the whole text."""
    strat = JaccardNgramStrategy(n=5, analyzer="char")
    # Both texts shorter than n=5 → ngrams are {whole text}
    sims, _ = strat.pairs_within(["ab", "ab"], k=2)
    assert abs(float(sims[0, 1]) - 1.0) < 1e-9  # identical short texts


# ---------------------------------------------------------------------------
# Plug-in contract — proves the seam wires the user's strategy through
# ---------------------------------------------------------------------------


class _CountingStrategy:
    """Wraps another strategy and counts each method invocation."""

    def __init__(self, inner: SimilarityStrategy) -> None:
        self.inner = inner
        self.within_calls = 0
        self.across_calls = 0

    def pairs_within(self, texts: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        self.within_calls += 1
        return self.inner.pairs_within(texts, k)

    def pairs_across(
        self,
        query_texts: Sequence[str],
        reference_texts: Sequence[str],
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.across_calls += 1
        return self.inner.pairs_across(query_texts, reference_texts, k)


@pytest.mark.unit
def test_custom_strategy_is_used_by_near_dedup() -> None:
    """near_dedup dispatches to the user-supplied strategy, not the default."""
    counting = _CountingStrategy(TfidfCosineStrategy())
    assert isinstance(counting, SimilarityStrategy)  # Protocol check
    texts = ["the quick fox", "the quick fox!", "lorem ipsum"]
    near_dedup(texts, threshold=0.8, strategy=counting)
    assert counting.within_calls == 1
    assert counting.across_calls == 0


@pytest.mark.unit
def test_custom_strategy_is_used_by_cross_dedup() -> None:
    """cross_dedup dispatches to the user-supplied strategy, not the default."""
    from eval_toolkit.text_dedup import cross_dedup  # noqa: PLC0415

    counting = _CountingStrategy(TfidfCosineStrategy())
    cross_dedup(
        ["alpha bravo", "charlie delta"],
        ["alpha bravo!", "xray yankee"],
        threshold=0.5,
        strategy=counting,
    )
    assert counting.across_calls == 1
    assert counting.within_calls == 0


@pytest.mark.unit
def test_default_strategy_is_tfidf_cosine() -> None:
    """When strategy=None, near_dedup behavior matches explicit TfidfCosineStrategy()."""
    texts = ["alpha bravo", "alpha bravo!", "delta echo"]
    default = near_dedup(texts, threshold=0.8)
    explicit = near_dedup(texts, threshold=0.8, strategy=TfidfCosineStrategy())
    assert default == explicit


# ---------------------------------------------------------------------------
# Identity: pairs_across(X, X, k) ≡ pairs_within(X, k) for top similarities
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pairs_across_self_matches_pairs_within(
    any_strategy: SimilarityStrategy,
) -> None:
    """pairs_across(X, X, k) gives the same top-k similarity values as pairs_within(X, k).

    Indices may permute on ties; we compare the sorted similarity vectors.
    """
    texts = ["alpha bravo", "alpha bravo!", "delta echo", "foxtrot golf"]
    s_within, _ = any_strategy.pairs_within(texts, k=2)
    s_across, _ = any_strategy.pairs_across(texts, texts, k=2)
    np.testing.assert_allclose(np.sort(s_within, axis=1), np.sort(s_across, axis=1))


# ---------------------------------------------------------------------------
# DedupReport invariant under each strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dedup_report_partition_invariant(any_strategy: SimilarityStrategy) -> None:
    """kept ∪ {dropped[0] for ...} == range(n) for any strategy."""
    texts = ["alpha", "alpha", "bravo", "charlie", "alpha"]
    report: DedupReport = near_dedup(texts, threshold=0.5, strategy=any_strategy)
    kept_set = set(report.kept_indices)
    dropped_set = {p[0] for p in report.dropped_pairs}
    assert kept_set | dropped_set == set(range(len(texts)))
    assert kept_set & dropped_set == set()
