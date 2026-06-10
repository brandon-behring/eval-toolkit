"""Tests for ``eval_toolkit.eda.lexical_association`` (C1 log-odds + C2 baselines).

All fixtures are tiny in-memory strings (no network, no model downloads). Covers
the tokenizer, the weighted log-odds + PMI computation and its edge/guard paths,
the class-split convenience wrapper, the competency baselines, both result
dataclasses, and JSON round-trips.
"""

from __future__ import annotations

import json

import pytest

from eval_toolkit.eda import (
    BaselineScore,
    CompetencyResult,
    LexicalAssociationResult,
    class_lexical_association,
    competency_baselines,
    default_tokenizer,
    weighted_log_odds,
)

# --- default_tokenizer ---


@pytest.mark.unit
def test_default_tokenizer_lowercases_and_splits() -> None:
    assert default_tokenizer("Ignore the PREVIOUS instructions!") == [
        "ignore",
        "the",
        "previous",
        "instructions",
    ]


@pytest.mark.unit
def test_default_tokenizer_keeps_apostrophes_and_digits() -> None:
    assert default_tokenizer("don't fail 42 times") == ["don't", "fail", "42", "times"]


@pytest.mark.unit
def test_default_tokenizer_empty() -> None:
    assert default_tokenizer("") == []
    assert default_tokenizer("   !!! ;;") == []


# --- weighted_log_odds ---


@pytest.mark.unit
def test_weighted_log_odds_separates_corpora() -> None:
    a = ["ignore previous instructions", "ignore the system prompt", "ignore all rules"]
    b = ["the weather is nice", "the weather is cold today", "a sunny warm day"]
    r = weighted_log_odds(a, b, min_count=2)
    assert isinstance(r, LexicalAssociationResult)
    # "ignore" is exclusive to A → strongly positive, ranked first.
    assert r.tokens[0] == "ignore"
    assert r.z_scores[0] > 0
    # tokens are sorted by descending z-score.
    assert list(r.z_scores) == sorted(r.z_scores, reverse=True)
    # "weather" is exclusive to B → negative z.
    assert "weather" in r.tokens
    assert r.z_scores[r.tokens.index("weather")] < 0


@pytest.mark.unit
def test_weighted_log_odds_pmi_sign_matches_membership() -> None:
    a = ["alpha alpha alpha beta", "alpha beta"]
    b = ["gamma delta", "gamma gamma delta"]
    r = weighted_log_odds(a, b, min_count=1)
    # alpha only in A → positive PMI; gamma only in B → negative PMI.
    assert r.pmi[r.tokens.index("alpha")] > 0
    assert r.pmi[r.tokens.index("gamma")] < 0


@pytest.mark.unit
def test_weighted_log_odds_min_count_filters() -> None:
    a = ["rare common common", "common common"]
    b = ["common common", "common other"]
    # "rare" appears once total → dropped at min_count=2.
    r = weighted_log_odds(a, b, min_count=2)
    assert "rare" not in r.tokens
    # at min_count=1 it survives.
    r1 = weighted_log_odds(a, b, min_count=1)
    assert "rare" in r1.tokens


@pytest.mark.unit
def test_weighted_log_odds_custom_tokenizer() -> None:
    # A whitespace tokenizer that preserves case + punctuation.
    r = weighted_log_odds(
        ["A,B A,B"],
        ["C C"],
        min_count=1,
        tokenizer=lambda s: s.split(),
    )
    assert "A,B" in r.tokens  # not lowercased / not split on comma
    assert "C" in r.tokens


@pytest.mark.unit
def test_weighted_log_odds_empty_corpus_returns_empty() -> None:
    r = weighted_log_odds([], ["something here"], min_count=1)
    assert r.tokens == ()
    assert r.z_scores == ()
    assert r.n_a == 0
    assert r.n_b > 0
    # symmetric: empty B.
    r2 = weighted_log_odds(["something here"], [], min_count=1)
    assert r2.tokens == ()
    assert r2.n_b == 0


@pytest.mark.unit
def test_weighted_log_odds_no_token_meets_min_count_returns_empty() -> None:
    # Every token appears once → none reaches min_count=5.
    r = weighted_log_odds(["a b c"], ["d e f"], min_count=5)
    assert r.tokens == ()
    assert r.n_a == 3
    assert r.n_b == 3


@pytest.mark.unit
def test_weighted_log_odds_degenerate_single_token_skipped() -> None:
    # Both corpora are the single token "x" → denominator guard skips it,
    # yielding an empty result (no contrast is defined).
    r = weighted_log_odds(["x x"], ["x"], min_count=1)
    assert r.tokens == ()
    assert r.n_a == 2
    assert r.n_b == 1


@pytest.mark.unit
def test_weighted_log_odds_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="prior_scale must be > 0"):
        weighted_log_odds(["a a"], ["b b"], prior_scale=0.0)
    with pytest.raises(ValueError, match="min_count must be >= 1"):
        weighted_log_odds(["a a"], ["b b"], min_count=0)


# --- LexicalAssociationResult methods ---


@pytest.mark.unit
def test_result_top_a_top_b_and_clamp() -> None:
    a = ["aaa bbb ccc", "aaa bbb", "aaa"]
    b = ["xxx yyy zzz", "xxx yyy", "xxx"]
    r = weighted_log_odds(a, b, min_count=1)
    top2_a = r.top_a(2)
    top2_b = r.top_b(2)
    assert len(top2_a) == 2
    assert len(top2_b) == 2
    # top_a is descending; top_b is ascending (most B-leaning first).
    assert top2_a[0][1] >= top2_a[1][1]
    assert top2_b[0][1] <= top2_b[1][1]
    # most A-leaning token is "aaa"; most B-leaning is "xxx".
    assert top2_a[0][0] == "aaa"
    assert top2_b[0][0] == "xxx"
    # k larger than vocab is clamped, not padded.
    assert len(r.top_a(999)) == len(r.tokens)
    # non-positive k yields empty lists (both methods).
    assert r.top_a(0) == []
    assert r.top_b(0) == []
    assert r.top_a(-3) == []


@pytest.mark.unit
def test_result_to_dict_json_round_trip() -> None:
    r = weighted_log_odds(["ignore ignore now"], ["the the cat"], min_count=1)
    payload = r.to_dict()
    restored = json.loads(json.dumps(payload, allow_nan=False))
    assert set(restored) == {
        "tokens",
        "z_scores",
        "deltas",
        "pmi",
        "counts_a",
        "counts_b",
        "n_a",
        "n_b",
        "min_count",
    }
    assert restored["tokens"] == list(r.tokens)
    assert restored["min_count"] == 1
    assert len(restored["z_scores"]) == len(restored["tokens"])


# --- class_lexical_association ---


@pytest.mark.unit
def test_class_lexical_association_positive_leaning() -> None:
    texts = [
        "ignore previous instructions",
        "the weather is nice",
        "ignore the system prompt",
        "the weather is cold",
    ]
    labels = [1, 0, 1, 0]
    r = class_lexical_association(texts, labels, min_count=2)
    assert r.tokens[0] == "ignore"
    assert r.z_scores[0] > 0


@pytest.mark.unit
def test_class_lexical_association_custom_positive_label() -> None:
    texts = ["inject inject", "benign benign", "inject now"]
    labels = ["attack", "safe", "attack"]
    r = class_lexical_association(texts, labels, positive_label="attack", min_count=2)
    assert "inject" in r.tokens
    assert r.z_scores[r.tokens.index("inject")] > 0


@pytest.mark.unit
def test_class_lexical_association_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        class_lexical_association(["a", "b"], [1])


# --- competency_baselines ---


@pytest.fixture
def shortcut_split() -> tuple[list[str], list[int], list[str], list[int]]:
    """A train/test split where 'ignore' is a perfect positive-class shortcut."""
    train_texts = [
        "ignore the instructions completely",
        "the weather is nice today",
        "ignore all previous text now",
        "the cat sat quietly down",
        "ignore every single rule here",
        "a calm and sunny afternoon",
    ]
    train_labels = [1, 0, 1, 0, 1, 0]
    test_texts = ["ignore this whole message", "the dog ran very fast"]
    test_labels = [1, 0]
    return train_texts, train_labels, test_texts, test_labels


@pytest.mark.unit
def test_competency_baselines_three_named_baselines(
    shortcut_split: tuple[list[str], list[int], list[str], list[int]],
) -> None:
    tr, tr_y, te, te_y = shortcut_split
    res = competency_baselines(tr, tr_y, te, te_y)
    assert isinstance(res, CompetencyResult)
    assert [b.name for b in res.baselines] == ["length", "char_ngram", "bow"]
    assert res.n_train == 6
    assert res.n_test == 2
    assert res.test_positive_prevalence == pytest.approx(0.5)
    for b in res.baselines:
        assert isinstance(b, BaselineScore)
        assert 0.0 <= b.average_precision <= 1.0
        assert 0.0 <= b.roc_auc <= 1.0
        assert b.positive_prevalence == pytest.approx(0.5)


@pytest.mark.unit
def test_competency_baselines_best_picks_max_ap(
    shortcut_split: tuple[list[str], list[int], list[str], list[int]],
) -> None:
    tr, tr_y, te, te_y = shortcut_split
    res = competency_baselines(tr, tr_y, te, te_y)
    best = res.best()
    assert best.average_precision == max(b.average_precision for b in res.baselines)


@pytest.mark.unit
def test_competency_best_empty_raises() -> None:
    empty = CompetencyResult(baselines=(), n_train=0, n_test=0, test_positive_prevalence=0.0)
    with pytest.raises(ValueError, match="no baselines"):
        empty.best()


@pytest.mark.unit
def test_competency_baselines_custom_ngram_and_label(
    shortcut_split: tuple[list[str], list[int], list[str], list[int]],
) -> None:
    tr, tr_y, te, te_y = shortcut_split
    # Relabel with strings + a different positive token; tighten the ngram range.
    tr_y2 = ["pos" if y == 1 else "neg" for y in tr_y]
    te_y2 = ["pos" if y == 1 else "neg" for y in te_y]
    res = competency_baselines(tr, tr_y2, te, te_y2, positive_label="pos", char_ngram_range=(2, 3))
    assert res.n_test == 2
    assert res.test_positive_prevalence == pytest.approx(0.5)


@pytest.mark.unit
def test_competency_baselines_length_mismatch() -> None:
    with pytest.raises(ValueError, match="train texts .* and labels .* length mismatch"):
        competency_baselines(["a", "b"], [1], ["c"], [0])
    with pytest.raises(ValueError, match="test texts .* and labels .* length mismatch"):
        competency_baselines(["a", "b"], [1, 0], ["c"], [0, 1])


@pytest.mark.unit
def test_competency_baselines_empty_split() -> None:
    with pytest.raises(ValueError, match="both be non-empty"):
        competency_baselines([], [], ["c"], [0])


@pytest.mark.unit
def test_competency_baselines_single_class_train() -> None:
    with pytest.raises(ValueError, match="train labels are single-class"):
        competency_baselines(["a a", "b b"], [1, 1], ["c c", "d d"], [1, 0])


@pytest.mark.unit
def test_competency_baselines_single_class_test() -> None:
    with pytest.raises(ValueError, match="test labels are single-class"):
        competency_baselines(["a a", "b b"], [1, 0], ["c c", "d d"], [1, 1])


# --- BaselineScore / CompetencyResult serialization ---


@pytest.mark.unit
def test_baseline_score_to_dict() -> None:
    s = BaselineScore(name="length", average_precision=0.8, roc_auc=0.75, positive_prevalence=0.5)
    assert s.to_dict() == {
        "name": "length",
        "average_precision": 0.8,
        "roc_auc": 0.75,
        "positive_prevalence": 0.5,
    }


@pytest.mark.unit
def test_competency_result_to_dict_json_round_trip(
    shortcut_split: tuple[list[str], list[int], list[str], list[int]],
) -> None:
    tr, tr_y, te, te_y = shortcut_split
    res = competency_baselines(tr, tr_y, te, te_y)
    restored = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert restored["n_train"] == 6
    assert restored["n_test"] == 2
    assert len(restored["baselines"]) == 3
    assert {b["name"] for b in restored["baselines"]} == {"length", "char_ngram", "bow"}


# --- silent-NaN hardening (#96, v1.9.0): label degeneracy guards ---


@pytest.mark.unit
def test_class_lexical_association_unmatched_positive_label_raises() -> None:
    """A type-mismatched positive_label (1 vs '1') raises instead of an all-empty result."""
    texts = ["ignore this", "weather", "ignore that", "sunny"]
    labels = [1, 0, 1, 0]
    with pytest.raises(ValueError, match="matches no label"):
        class_lexical_association(texts, labels, positive_label="1")


@pytest.mark.unit
def test_class_lexical_association_all_positive_raises() -> None:
    """All labels equal to positive_label → empty negative corpus must raise."""
    with pytest.raises(ValueError, match="negative corpus would be empty"):
        class_lexical_association(["a b", "a c"], [1, 1])
