"""Hypothesis property tests for the SimilarityStrategy seam.

Each property is parameterized across all four reference strategies so any
strategy-specific deviation surfaces as a per-strategy failure. Properties
that can't hold for every strategy (e.g., threshold monotonicity assumes
similarity scores are stable across calls — true for all four shipped
strategies) are scoped explicitly to the strategies where they hold.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hst

from eval_toolkit.text_dedup import (
    EmbeddingCosineStrategy,
    ExactNormalizedHashStrategy,
    JaccardNgramStrategy,
    SimilarityStrategy,
    TfidfCosineStrategy,
    cross_dedup,
    near_dedup,
)

_STUB_EMBED_DIM = 64


def _stub_one_hot_embedder(texts: Sequence[str]) -> np.ndarray:
    """Hash-bucket one-hot embedder; stable feature dimension across calls."""
    import hashlib  # noqa: PLC0415

    out = np.zeros((len(texts), _STUB_EMBED_DIM), dtype=np.float64)
    for i, t in enumerate(texts):
        slot = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % _STUB_EMBED_DIM
        out[i, slot] = 1.0
    return out


def _strategies() -> list[SimilarityStrategy]:
    return [
        TfidfCosineStrategy(),
        ExactNormalizedHashStrategy(),
        EmbeddingCosineStrategy(_stub_one_hot_embedder),
        JaccardNgramStrategy(n=2, analyzer="char"),
    ]


# Hypothesis text strategy: alphabetic-only, length 3-12. Avoids
# stopword-only inputs that crash TfidfVectorizer.
_alpha_text = hst.text(
    alphabet=hst.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=3,
    max_size=12,
)
_text_lists = hst.lists(_alpha_text, min_size=1, max_size=15, unique=True)


@pytest.fixture(params=_strategies(), ids=lambda s: type(s).__name__)
def any_strategy(request: pytest.FixtureRequest) -> SimilarityStrategy:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Partition invariant: every input row ends up in kept or dropped, never both
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(texts=_text_lists, threshold=hst.floats(0.1, 0.99))
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_dedup_partition_holds_under_strategy(
    any_strategy: SimilarityStrategy, texts: list[str], threshold: float
) -> None:
    """For any strategy: kept ∪ {dropped[0]} = range(n); kept ∩ {dropped[0]} = ∅."""
    report = near_dedup(texts, threshold=threshold, strategy=any_strategy)
    kept = set(report.kept_indices)
    dropped = {p[0] for p in report.dropped_pairs}
    assert kept | dropped == set(range(len(texts)))
    assert kept & dropped == set()


# ---------------------------------------------------------------------------
# Idempotence: running dedup on the kept subset is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(texts=_text_lists, threshold=hst.floats(0.1, 0.99))
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_dedup_idempotent_on_kept_subset(
    any_strategy: SimilarityStrategy, texts: list[str], threshold: float
) -> None:
    """near_dedup(near_dedup(X).kept_texts) drops nothing further."""
    report = near_dedup(texts, threshold=threshold, strategy=any_strategy)
    kept_texts = [texts[i] for i in report.kept_indices]
    if len(kept_texts) <= 1:
        return  # trivially idempotent
    second = near_dedup(kept_texts, threshold=threshold, strategy=any_strategy)
    assert second.n_dropped == 0


# ---------------------------------------------------------------------------
# Threshold monotonicity: lower threshold → at most as many kept rows
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(texts=_text_lists)
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_threshold_monotonicity(any_strategy: SimilarityStrategy, texts: list[str]) -> None:
    """Lower threshold drops more aggressively → n_kept(0.5) ≤ n_kept(0.9)."""
    low = near_dedup(texts, threshold=0.5, strategy=any_strategy)
    high = near_dedup(texts, threshold=0.9, strategy=any_strategy)
    assert low.n_kept <= high.n_kept


# ---------------------------------------------------------------------------
# Cross-vs-self: cross_dedup(X, X, t) drops everything (every eval row
# matches itself in the train set at sim=1.0 ≥ t for any t in (0, 1))
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(texts=_text_lists, threshold=hst.floats(0.1, 0.99))
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_cross_dedup_self_drops_everything(
    any_strategy: SimilarityStrategy, texts: list[str], threshold: float
) -> None:
    """cross_dedup(X, X, t) keeps no eval rows (every row matches itself in train)."""
    kept = cross_dedup(texts, texts, threshold=threshold, strategy=any_strategy)
    assert kept == []


# ---------------------------------------------------------------------------
# Empty + singleton corner cases (deterministic, not Hypothesis-driven)
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_empty_and_singleton_corner_cases(any_strategy: SimilarityStrategy) -> None:
    """near_dedup([]) and near_dedup([x]) return valid reports."""
    empty = near_dedup([], threshold=0.5, strategy=any_strategy)
    assert empty.n_input == 0 and empty.n_kept == 0 and empty.n_dropped == 0

    single = near_dedup(["solo"], threshold=0.5, strategy=any_strategy)
    assert single.n_input == 1 and single.n_kept == 1 and single.n_dropped == 0


# ---------------------------------------------------------------------------
# kept_indices is sorted (downstream callers may assume order)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(texts=_text_lists, threshold=hst.floats(0.1, 0.99))
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_kept_indices_are_sorted(
    any_strategy: SimilarityStrategy, texts: list[str], threshold: float
) -> None:
    """report.kept_indices is monotonically increasing for all strategies."""
    report = near_dedup(texts, threshold=threshold, strategy=any_strategy)
    assert report.kept_indices == sorted(report.kept_indices)
