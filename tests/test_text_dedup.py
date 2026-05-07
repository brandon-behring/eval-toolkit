"""Unit + property tests for text_dedup."""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eval_toolkit.text_dedup import (
    DedupReport,
    cross_dedup,
    near_dedup,
    normalize_text_for_dedup,
    sha256_text,
)


@pytest.mark.unit
def test_normalize_text_lowers_collapses_whitespace() -> None:
    assert normalize_text_for_dedup("Hello   World") == "hello world"
    assert normalize_text_for_dedup("\tFOO\n bar  ") == "foo bar"


@pytest.mark.unit
def test_normalize_text_idempotent() -> None:
    for text in ["abc", "Hello World", "  foo  "]:
        once = normalize_text_for_dedup(text)
        twice = normalize_text_for_dedup(once)
        assert once == twice


@pytest.mark.unit
def test_normalize_text_validates() -> None:
    with pytest.raises(TypeError):
        normalize_text_for_dedup(123)  # type: ignore[arg-type]


@pytest.mark.unit
def test_sha256_text_matches_stdlib() -> None:
    text = "hello world"
    expected = hashlib.sha256(normalize_text_for_dedup(text).encode("utf-8")).hexdigest()
    assert sha256_text(text) == expected


@pytest.mark.unit
def test_sha256_text_normalize_kwarg() -> None:
    """normalize=False bypasses normalization."""
    raw = sha256_text("Hello World", normalize=False)
    norm = sha256_text("hello world", normalize=False)
    assert raw != norm


@pytest.mark.unit
def test_sha256_text_length() -> None:
    assert len(sha256_text("anything")) == 64


@pytest.mark.unit
def test_near_dedup_empty() -> None:
    rep = near_dedup([], threshold=0.9)
    assert rep.n_kept == 0
    assert rep.n_dropped == 0


@pytest.mark.unit
def test_near_dedup_single_item() -> None:
    rep = near_dedup(["foo"], threshold=0.9)
    assert rep.kept_indices == [0]
    assert rep.n_dropped == 0


@pytest.mark.unit
def test_near_dedup_drops_exact_duplicates() -> None:
    rep = near_dedup(
        ["the quick brown fox jumps", "the quick brown fox jumps", "lorem ipsum dolor"]
    )
    assert rep.n_kept == 2  # one of the dupes plus the unique


@pytest.mark.unit
def test_near_dedup_validates() -> None:
    with pytest.raises(TypeError):
        near_dedup("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        near_dedup(["a", "b"], threshold=0.0)
    with pytest.raises(ValueError):
        near_dedup(["a", "b"], threshold=1.0)


@pytest.mark.unit
def test_cross_dedup_keeps_when_no_train() -> None:
    eval_texts = ["foo", "bar", "baz"]
    keep = cross_dedup(train_texts=[], eval_texts=eval_texts)
    assert keep == [0, 1, 2]


@pytest.mark.unit
def test_cross_dedup_drops_train_duplicates() -> None:
    train = ["the cat sat on the mat", "another sentence here"]
    eval_texts = ["the cat sat on the mat", "totally different content right"]
    keep = cross_dedup(train, eval_texts, threshold=0.9)
    assert 0 not in keep  # eval[0] duplicates train[0]
    assert 1 in keep  # eval[1] is independent


@pytest.mark.unit
def test_dedup_report_props() -> None:
    rep = DedupReport(kept_indices=[0, 2], dropped_pairs=[(1, 0, 0.95)], threshold=0.9, n_input=3)
    assert rep.n_kept == 2
    assert rep.n_dropped == 1


# ---------------------------------------------------------------------- properties

# TF-IDF requires non-empty vocabulary; restrict text to alphabetic-mostly content
# so the vectorizer doesn't fail with "empty vocabulary" on stopword-only inputs.
_word_text = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=3,
    max_size=20,
)


@pytest.mark.property
@given(st.text(min_size=0, max_size=100))
def test_normalize_text_idempotence(text: str) -> None:
    once = normalize_text_for_dedup(text)
    twice = normalize_text_for_dedup(once)
    assert once == twice


@pytest.mark.property
@given(st.text(min_size=1, max_size=50))
def test_sha256_text_deterministic(text: str) -> None:
    assert sha256_text(text) == sha256_text(text)


@pytest.mark.property
@given(st.lists(_word_text, min_size=2, max_size=20))
def test_near_dedup_idempotent(texts: list[str]) -> None:
    """near_dedup of dedup result equals dedup result."""
    rep1 = near_dedup(texts, threshold=0.9)
    kept_texts = [texts[i] for i in rep1.kept_indices]
    if not kept_texts:
        return  # nothing to re-dedup
    rep2 = near_dedup(kept_texts, threshold=0.9)
    assert rep2.n_kept == rep1.n_kept


@pytest.mark.property
@given(st.lists(_word_text, min_size=1, max_size=10))
def test_near_dedup_kept_subset_of_input(texts: list[str]) -> None:
    rep = near_dedup(texts, threshold=0.9)
    assert all(0 <= i < len(texts) for i in rep.kept_indices)
    assert rep.n_kept + rep.n_dropped == len(texts)


@pytest.mark.property
@given(
    st.lists(_word_text, min_size=1, max_size=10),
    st.lists(_word_text, min_size=1, max_size=10),
)
def test_cross_dedup_returns_subset_of_eval_indices(
    train: list[str], eval_texts: list[str]
) -> None:
    kept = cross_dedup(train, eval_texts, threshold=0.9)
    assert all(0 <= i < len(eval_texts) for i in kept)
    assert kept == sorted(kept)
