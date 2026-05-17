"""Embedder factories for use with :class:`eval_toolkit.text_dedup.EmbeddingCosineStrategy`.

The toolkit's :class:`~eval_toolkit.text_dedup.EmbeddingCosineStrategy` owns
cosine + k-NN and accepts any ``Callable[[Sequence[str]], np.ndarray]`` as the
embedder. This module ships factories for the canonical recipes so callers
don't reinvent the wrapping each time.

Optional dependency: install via ``pip install eval-toolkit[embeddings]``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["make_minilm_embedder"]

_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def make_minilm_embedder(
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Factory for a MiniLM-based embedder compatible with EmbeddingCosineStrategy.

    Returns a callable ``(Sequence[str]) -> np.ndarray`` of shape ``(n, 384)``
    when using the default ``all-MiniLM-L6-v2`` model. The returned callable
    wraps a :class:`sentence_transformers.SentenceTransformer` instance loaded
    once per ``(model_id, device, batch_size)`` tuple (memoised via
    :func:`functools.lru_cache`, ``maxsize=8``).

    The canonical semantic-dedup recipe (cosine threshold 0.80 on
    ``all-MiniLM-L6-v2``) is what consumers reach for whenever lexical TF-IDF
    signal is insufficient (paraphrase, multilingual, semantic-near-duplicate).
    This factory ships that recipe pre-wired for
    :class:`~eval_toolkit.text_dedup.EmbeddingCosineStrategy`.

    Parameters
    ----------
    model_id : str, optional
        HuggingFace model id. Default: ``"sentence-transformers/all-MiniLM-L6-v2"``.
    device : str, optional
        Torch device string passed to :class:`SentenceTransformer`. Default:
        ``"cpu"``. Use ``"cuda"`` for GPU.
    batch_size : int, optional
        Batch size for :meth:`SentenceTransformer.encode`. Default: 32.

    Returns
    -------
    Callable[[Sequence[str]], np.ndarray]
        Embedder callable. Output is 2-D ``float64``; row count matches input
        length.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed. Install via
        ``pip install eval-toolkit[embeddings]``.

    Notes
    -----
    First call downloads model weights (~80MB for ``all-MiniLM-L6-v2``) to the
    HuggingFace on-disk cache (``HF_HOME``, default
    ``~/.cache/huggingface/hub/``). Subsequent calls reuse the cached weights.

    The embedder coerces output to ``float64`` for stable cosine arithmetic in
    :meth:`~eval_toolkit.text_dedup.EmbeddingCosineStrategy._embed`.

    Examples
    --------
    >>> from eval_toolkit import make_minilm_embedder, EmbeddingCosineStrategy  # doctest: +SKIP
    >>> embedder = make_minilm_embedder()                                        # doctest: +SKIP
    >>> strategy = EmbeddingCosineStrategy(embedder=embedder)                    # doctest: +SKIP
    >>> sims, idx = strategy.pairs_within(["hello world", "hi there"], k=2)      # doctest: +SKIP
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "make_minilm_embedder requires sentence-transformers. "
            "Install via: pip install eval-toolkit[embeddings]"
        ) from e

    _logger.debug(
        "loading SentenceTransformer model_id=%s device=%s batch_size=%d",
        model_id,
        device,
        batch_size,
    )
    model = SentenceTransformer(model_id, device=device)

    def embedder(texts: Sequence[str]) -> np.ndarray:
        result = model.encode(
            list(texts),
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float64)

    return embedder
