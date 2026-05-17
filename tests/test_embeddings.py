"""Tests for `eval_toolkit.embeddings`.

The slow test exercises a real `sentence-transformers` model and is skipped
when the optional dep is not installed (matches consumer experience for the
`[embeddings]` extra). The unit tests exercise the helpful-ImportError
branch and the lru_cache identity guarantee.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.unit
def test_make_minilm_embedder_raises_helpful_import_error_when_dep_missing() -> None:
    """If sentence-transformers is absent, the factory raises ImportError with
    a copy-paste install hint pointing at the `[embeddings]` extra."""
    # Clear any cached entries so we hit the import branch on every call.
    from eval_toolkit.embeddings import make_minilm_embedder

    make_minilm_embedder.cache_clear()

    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(ImportError, match=r"\[embeddings\]"),
    ):
        make_minilm_embedder(model_id="any/model")

    # Be a good neighbor: restore the cache state for downstream tests by
    # clearing again (the failed call won't have populated it, but other
    # tests in the run may have).
    make_minilm_embedder.cache_clear()


@pytest.mark.slow
def test_make_minilm_embedder_real_model_round_trip() -> None:
    """End-to-end: factory returns a callable that produces (n, 384) float64
    embeddings consumable by EmbeddingCosineStrategy.

    Downloads model weights on first run (~80MB, cached at HF_HOME).
    """
    pytest.importorskip("sentence_transformers")

    from eval_toolkit import EmbeddingCosineStrategy, make_minilm_embedder

    # Ensure we exercise a real model-loading call this run (not a cache hit
    # from another test) so failures point at the factory rather than at a
    # previously memoised object.
    make_minilm_embedder.cache_clear()

    embedder = make_minilm_embedder()
    texts = ["the cat sat on the mat", "a feline rested on the rug", "unrelated string"]
    emb = embedder(texts)

    assert isinstance(emb, np.ndarray), f"expected ndarray, got {type(emb).__name__}"
    assert emb.shape == (3, 384), f"expected (3, 384) for all-MiniLM-L6-v2, got {emb.shape}"
    assert emb.dtype == np.float64, f"expected float64 for cosine stability, got {emb.dtype}"

    # The factory's output must be valid input to EmbeddingCosineStrategy.
    strat = EmbeddingCosineStrategy(embedder=embedder)
    sims, idx = strat.pairs_within(texts, k=3)
    assert sims.shape == (3, 3) and idx.shape == (3, 3)
    # Each row's self should be slot 0 with similarity ~1.
    assert np.allclose(sims[:, 0], 1.0, atol=1e-5)


@pytest.mark.slow
def test_make_minilm_embedder_is_memoised_per_args() -> None:
    """lru_cache returns the same embedder callable for identical args
    (avoiding redundant model loads in tight evaluation loops)."""
    pytest.importorskip("sentence_transformers")

    from eval_toolkit.embeddings import make_minilm_embedder

    make_minilm_embedder.cache_clear()

    first = make_minilm_embedder()
    second = make_minilm_embedder()
    assert first is second, "lru_cache should return the same callable for identical args"

    # Different args → different cached callable.
    third = make_minilm_embedder(batch_size=64)
    assert third is not first, "different args should produce a distinct cache entry"
