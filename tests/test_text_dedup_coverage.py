"""Coverage-focused tests for eval_toolkit.text_dedup.

Targets the validation-error and edge-case branches not exercised by
the strategy/property/happy-path suites: degenerate inputs to each
similarity strategy (empty corpus, single doc, short-text fallbacks),
MinHash LSH structural branches, ``audit_source_label_similarity``
parameter validation + per-axis filter combinations,
``_similarity_relation`` mode table, and ``cross_dedup`` validation
and empty-similarities branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_toolkit.text_dedup import (
    EmbeddingCosineStrategy,
    JaccardNgramStrategy,
    MinHashLSHStrategy,
    SimilarityAuditFinding,
    SimilarityAuditReport,
    SimilarityStrategy,
    audit_source_label_similarity,
    cross_dedup,
    near_dedup,
    sha256_text,
)

# ---------------------------------------------------------------------------
# sha256_text: TypeError on non-str input (line 267)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sha256_text_rejects_non_str() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        sha256_text(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SimilarityAuditFinding.to_dict: all-optional-fields populated (lines 162-176)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_similarity_audit_finding_to_dict_all_fields() -> None:
    finding = SimilarityAuditFinding(
        left_index=0,
        right_index=1,
        similarity=0.95,
        relation="within_source_same_label",
        left_source="train",
        right_source="train",
        left_label="pos",
        right_label="pos",
    )
    out = finding.to_dict()
    assert out["left_source"] == "train"
    assert out["right_source"] == "train"
    assert out["left_label"] == "pos"
    assert out["right_label"] == "pos"


@pytest.mark.unit
def test_similarity_audit_report_to_dict_empty() -> None:
    report = SimilarityAuditReport(
        findings=[], threshold=0.9, n_input=0, strategy="none", k_neighbors=5
    )
    out = report.to_dict()
    assert out["n_findings"] == 0
    assert out["findings"] == []


# ---------------------------------------------------------------------------
# JaccardNgramStrategy: word-analyzer short-input branch (line 681)
# + _jaccard empty-set branches (lines 688, 691)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_jaccard_word_analyzer_short_tokens() -> None:
    """When token count < n, falls back to the joined-tokens singleton."""
    strat = JaccardNgramStrategy(n=3, analyzer="word")
    # Both texts have only 2 tokens (< n=3), so they reduce to a single
    # joined-tokens shingle each. "a b" == "a b" → similarity 1.0; "a b" vs
    # "c d" → similarity 0 (disjoint singleton sets).
    sims, _ = strat.pairs_within(["a b", "a b"], k=2)
    # Self-similarity is 1.0 on the same-text pair (whether self-inclusive or not).
    assert sims.max() == 1.0


@pytest.mark.unit
def test_jaccard_word_analyzer_empty_text() -> None:
    """An empty string has an empty token list → empty shingle set."""
    strat = JaccardNgramStrategy(n=3, analyzer="word")
    sims, _ = strat.pairs_within(["", ""], k=2)
    # Both shingle sets empty: _jaccard returns 1.0 (vacuous-equal).
    assert sims[0, 0] == 1.0


@pytest.mark.unit
def test_jaccard_rejects_invalid_analyzer() -> None:
    with pytest.raises(ValueError, match="analyzer must be"):
        JaccardNgramStrategy(n=3, analyzer="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MinHashLSHStrategy: empty text and short-text shingles + cross-corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minhash_handles_empty_text() -> None:
    """Empty strings → empty shingle set → all-max signature; no crash."""
    strat = MinHashLSHStrategy(n=3, num_perm=32, bands=8)
    sims, _ = strat.pairs_within(["", "", ""], k=2)
    # All signatures equal → all pairwise sims == 1.0 by the exact_jaccard
    # vacuous-equal contract.
    assert sims[0, 0] == 1.0


@pytest.mark.unit
def test_minhash_handles_short_text() -> None:
    """Texts shorter than n trigger the short-text branch (line 857)."""
    strat = MinHashLSHStrategy(n=10, num_perm=32, bands=8)
    sims, idx = strat.pairs_within(["abc", "xyz"], k=2)
    assert sims.shape == (2, 2)
    assert idx.shape == (2, 2)


@pytest.mark.unit
def test_minhash_pairs_across_empty_query() -> None:
    """n_q == 0 short-circuits (line 957-961)."""
    strat = MinHashLSHStrategy(n=3, num_perm=32, bands=8)
    sims, idx = strat.pairs_across([], ["a quick text"], k=2)
    assert sims.shape == (0, 0)
    assert idx.shape == (0, 0)


@pytest.mark.unit
def test_minhash_pairs_across_empty_ref() -> None:
    """n_r == 0 short-circuits."""
    strat = MinHashLSHStrategy(n=3, num_perm=32, bands=8)
    sims, idx = strat.pairs_across(["a quick text"], [], k=2)
    assert sims.shape == (1, 0)


@pytest.mark.unit
def test_minhash_pairs_across_returns_signal() -> None:
    """Successful cross-corpus path with LSH candidates and exact jaccard rerank."""
    strat = MinHashLSHStrategy(n=3, num_perm=64, bands=16)
    sims, idx = strat.pairs_across(
        ["the quick brown fox jumps over the lazy dog"],
        [
            "the quick brown fox jumps over the lazy dog",
            "lorem ipsum dolor sit amet",
        ],
        k=2,
    )
    assert sims.shape == (1, 2)
    # First column is the near-identical match.
    assert sims[0, 0] > 0.5


# ---------------------------------------------------------------------------
# audit_source_label_similarity: validation paths (lines 1155-1161)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        audit_source_label_similarity(["a", "b"], threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        audit_source_label_similarity(["a", "b"], threshold=1.5)


@pytest.mark.unit
def test_audit_rejects_bad_k_neighbors() -> None:
    with pytest.raises(ValueError, match="k_neighbors must be"):
        audit_source_label_similarity(["a", "b"], k_neighbors=0)


@pytest.mark.unit
def test_audit_rejects_mismatched_sources() -> None:
    with pytest.raises(ValueError, match="sources must have the same length"):
        audit_source_label_similarity(["a", "b"], sources=["x"])


@pytest.mark.unit
def test_audit_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="labels must have the same length"):
        audit_source_label_similarity(["a", "b"], labels=[0])


@pytest.mark.unit
def test_audit_empty_corpus_returns_empty_report() -> None:
    """n == 0 short-circuits to an empty report."""
    report = audit_source_label_similarity([])
    assert report.n_findings == 0
    assert report.strategy == "none"


@pytest.mark.unit
def test_audit_filter_excludes_within_source() -> None:
    """When include_within_source=False, same-source pairs are filtered out."""
    texts = ["the quick brown fox", "the quick brown fox", "lorem ipsum dolor"]
    sources = ["train", "train", "test"]
    report = audit_source_label_similarity(
        texts,
        sources=sources,
        threshold=0.5,
        include_within_source=False,
        include_cross_source=True,
    )
    # Only cross-source pairs should appear; the near-identical train pair is filtered.
    for finding in report.findings:
        assert sources[finding.left_index] != sources[finding.right_index]


@pytest.mark.unit
def test_audit_filter_excludes_cross_source() -> None:
    """When include_cross_source=False, cross-source pairs are filtered."""
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    sources = ["train", "test", "test"]
    report = audit_source_label_similarity(
        texts,
        sources=sources,
        threshold=0.5,
        include_within_source=True,
        include_cross_source=False,
    )
    for finding in report.findings:
        assert sources[finding.left_index] == sources[finding.right_index]


@pytest.mark.unit
def test_audit_filter_excludes_same_label() -> None:
    """include_same_label=False filters out same-label pairs."""
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    labels = [1, 1, 0]
    report = audit_source_label_similarity(
        texts,
        labels=labels,
        threshold=0.5,
        include_same_label=False,
        include_cross_label=True,
    )
    for finding in report.findings:
        assert labels[finding.left_index] != labels[finding.right_index]


@pytest.mark.unit
def test_audit_filter_excludes_cross_label() -> None:
    """include_cross_label=False filters out different-label pairs."""
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    labels = [1, 1, 0]
    report = audit_source_label_similarity(
        texts,
        labels=labels,
        threshold=0.5,
        include_same_label=True,
        include_cross_label=False,
    )
    for finding in report.findings:
        assert labels[finding.left_index] == labels[finding.right_index]


# ---------------------------------------------------------------------------
# _similarity_relation: every mode table branch (lines 1257-1271)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_relation_unspecified_when_no_metadata() -> None:
    texts = ["the quick brown fox", "the quick brown fox"]
    report = audit_source_label_similarity(texts, threshold=0.5)
    assert all(f.relation == "unspecified" for f in report.findings)


@pytest.mark.unit
def test_relation_labels_only() -> None:
    """sources=None, labels=specified → 'same_label' / 'cross_label'."""
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    labels = [1, 1, 0]
    report = audit_source_label_similarity(texts, labels=labels, threshold=0.5)
    rels = {f.relation for f in report.findings}
    assert rels <= {"same_label", "cross_label"}


@pytest.mark.unit
def test_relation_sources_only() -> None:
    """labels=None, sources=specified → 'within_source' / 'cross_source'."""
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    sources = ["train", "train", "eval"]
    report = audit_source_label_similarity(texts, sources=sources, threshold=0.5)
    rels = {f.relation for f in report.findings}
    assert rels <= {"within_source", "cross_source"}


@pytest.mark.unit
def test_relation_within_source_cross_label() -> None:
    texts = ["the quick brown fox", "the quick brown fox"]
    report = audit_source_label_similarity(
        texts,
        sources=["train", "train"],
        labels=[1, 0],
        threshold=0.5,
    )
    assert any(f.relation == "within_source_cross_label" for f in report.findings)


@pytest.mark.unit
def test_relation_cross_source_same_label() -> None:
    texts = ["the quick brown fox", "the quick brown fox"]
    report = audit_source_label_similarity(
        texts,
        sources=["train", "eval"],
        labels=[1, 1],
        threshold=0.5,
    )
    assert any(f.relation == "cross_source_same_label" for f in report.findings)


@pytest.mark.unit
def test_relation_cross_source_cross_label() -> None:
    texts = ["the quick brown fox", "the quick brown fox"]
    report = audit_source_label_similarity(
        texts,
        sources=["train", "eval"],
        labels=[1, 0],
        threshold=0.5,
    )
    assert any(f.relation == "cross_source_cross_label" for f in report.findings)


# ---------------------------------------------------------------------------
# cross_dedup: validation + empty branches (lines 1313, 1317, 1324)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cross_dedup_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        cross_dedup(["a"], ["b"], threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        cross_dedup(["a"], ["b"], threshold=1.0)


@pytest.mark.unit
def test_cross_dedup_empty_train_keeps_all_eval() -> None:
    kept = cross_dedup([], ["a", "b", "c"])
    assert kept == [0, 1, 2]


@pytest.mark.unit
def test_cross_dedup_empty_eval_returns_empty() -> None:
    kept = cross_dedup(["a", "b"], [])
    assert kept == []


# ---------------------------------------------------------------------------
# EmbeddingCosineStrategy: buggy embedder catches inconsistent feature dim
# (Migrated from test_coverage_gap.py during v0.30.1 hygiene split.)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_embedding_cosine_pairs_across_rejects_dim_mismatch() -> None:
    """EmbeddingCosineStrategy.pairs_across catches buggy embedders that return
    different feature dimensions for query vs reference."""
    call_count = {"n": 0}

    def buggy_embedder(texts: list[str]) -> np.ndarray:
        call_count["n"] += 1
        # First call (refs) returns d=4; second call (queries) returns d=8.
        d = 4 if call_count["n"] == 1 else 8
        return np.zeros((len(texts), d), dtype=np.float64)

    strat = EmbeddingCosineStrategy(buggy_embedder)
    with pytest.raises(ValueError, match="inconsistent feature dimensions"):
        strat.pairs_across(["q1", "q2"], ["r1", "r2", "r3"], k=2)


# ---------------------------------------------------------------------------
# MinHashLSHStrategy: protocol conformance + near-duplicate detection
# (Migrated from test_coverage_gap.py v0.4.0 C4 section during v0.30.1
# hygiene split — every assertion preserved verbatim.)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minhash_lsh_satisfies_protocol() -> None:
    """MinHashLSHStrategy is a runtime-checkable SimilarityStrategy."""
    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16)
    assert isinstance(strat, SimilarityStrategy)


@pytest.mark.unit
def test_minhash_lsh_pairs_within_shape() -> None:
    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16, seed=0)
    texts = ["alpha bravo", "alpha bravo charlie", "delta echo", "foxtrot golf"]
    sims, idx = strat.pairs_within(texts, k=3)
    assert sims.shape == idx.shape == (4, 3)


@pytest.mark.unit
def test_minhash_lsh_self_similarity_is_one() -> None:
    """For pairs_within, each text's most-similar neighbor is itself with sim=1."""
    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16, seed=0)
    texts = ["alpha bravo charlie", "delta echo foxtrot"]
    sims, idx = strat.pairs_within(texts, k=2)
    for i in range(2):
        assert int(idx[i, 0]) == i
        assert sims[i, 0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_minhash_lsh_finds_near_duplicate() -> None:
    """Near-duplicate pair should be discovered + scored ≥ 0.5 Jaccard."""
    strat = MinHashLSHStrategy(n=3, num_perm=128, bands=16, seed=0)
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy doggo!",  # near-dup
        "completely unrelated lorem ipsum text content",
    ]
    sims, idx = strat.pairs_within(texts, k=2)
    # Top-1 (other than self) for text 0 should be text 1
    assert int(idx[0, 1]) == 1
    assert sims[0, 1] > 0.5  # high Jaccard between the near-dups


@pytest.mark.unit
def test_minhash_lsh_validates_args() -> None:
    with pytest.raises(ValueError, match="n must be"):
        MinHashLSHStrategy(n=0)
    with pytest.raises(ValueError, match="num_perm"):
        MinHashLSHStrategy(num_perm=4)
    with pytest.raises(ValueError, match="bands"):
        MinHashLSHStrategy(num_perm=128, bands=0)
    with pytest.raises(ValueError, match="bands"):
        MinHashLSHStrategy(num_perm=128, bands=200)
    with pytest.raises(ValueError, match="divisible"):
        MinHashLSHStrategy(num_perm=128, bands=15)  # 128 not divisible by 15


@pytest.mark.unit
def test_minhash_lsh_handles_empty_input() -> None:
    strat = MinHashLSHStrategy(n=2, num_perm=64, bands=16)
    sims_w, idx_w = strat.pairs_within([], k=5)
    assert sims_w.shape == idx_w.shape == (0, 0)
    sims_a, idx_a = strat.pairs_across([], ["a"], k=5)
    assert sims_a.shape == idx_a.shape == (0, 0)


@pytest.mark.unit
def test_minhash_lsh_in_near_dedup_orchestrator() -> None:
    """Plug-in contract: near_dedup accepts MinHashLSHStrategy via strategy=."""
    texts = [
        "the quick brown fox",
        "the quick brown fox!",  # near-dup
        "lorem ipsum dolor sit amet",
    ]
    strat = MinHashLSHStrategy(n=3, num_perm=128, bands=16, seed=0)
    report = near_dedup(texts, threshold=0.5, strategy=strat)
    # The near-dup pair should collapse to 1 entry
    assert report.n_kept == 2
