# Worked example: callable-embedder pattern for `EmbeddingCosineStrategy`

> **What this shows.** Semantic dedup with a caller-owned embedder. The
> toolkit's `EmbeddingCosineStrategy` owns cosine + k-NN; the caller owns
> the embedder. This keeps the toolkit dep-free of any specific embedding
> library (sentence-transformers, OpenAI, local PyTorch, etc.) while
> still offering a turnkey strategy class.
>
> **Runtime:** the runnable example below uses a stub embedder (numpy
> one-hot vectors) — completes in <1 s. The MiniLM and OpenAI patterns
> are illustrative (skipped under Sybil).

## Pattern 1: stub embedder (testable, runnable in CI)

The first thing to know is that **any** callable `Callable[[Sequence[str]],
np.ndarray]` that returns a 2-D array of shape `(n, d)` works. Use this
for unit tests:

```python
import numpy as np
from eval_toolkit.text_dedup import EmbeddingCosineStrategy, near_dedup

def stub_embedder(texts):
    # One-hot per text index — behaves like exact-match on text identity.
    return np.eye(len(texts))

strategy = EmbeddingCosineStrategy(embedder=stub_embedder)
report = near_dedup(
    texts=["hello world", "goodbye", "hello world"],
    threshold=0.80,
    strategy=strategy,
)
# DedupReport carries kept_indices + dropped_pairs (n_kept / n_dropped
# are properties, not callables).
# In the stub-embedder one-hot setup, each text is its own basis vector,
# so cosine between distinct texts is 0 → no duplicates dropped at 0.80.
assert report.n_dropped == 0
assert report.n_kept == 3
```

## Pattern 2: MiniLM via v0.33.1's `make_minilm_embedder` (recommended)

For real semantic dedup, the canonical recipe (per
[`prompt-injection-detection-submission`'s ADR-027](https://github.com/brandon-behring/prompt-injection-detection-submission))
is `sentence-transformers/all-MiniLM-L6-v2` at cosine threshold 0.80.
v0.33.1 ships {func}`~eval_toolkit.embeddings.make_minilm_embedder` as the
pre-wired factory — no embedder boilerplate needed:

<!-- skip: next -->
```python
from eval_toolkit import make_minilm_embedder, EmbeddingCosineStrategy
from eval_toolkit.text_dedup import near_dedup

embedder = make_minilm_embedder()  # cached via lru_cache; one model load per process
strategy = EmbeddingCosineStrategy(embedder=embedder)

report = near_dedup(
    texts=corpus,
    threshold=0.80,
    strategy=strategy,
)
```

Install the optional dep: `pip install eval-toolkit[embeddings]`. This
pulls `sentence-transformers` (which transitively pulls `torch`); see the
[methodology/parallelism.md](../methodology/parallelism.md) doc for why
this extra is intentionally NOT in `[all]`.

## Pattern 3: custom embedder (OpenAI, in-house model)

The factory pattern generalises — wrap any embedder API in a `Callable`
that takes `Sequence[str]` and returns `np.ndarray` of shape `(n, d)`:

<!-- skip: next -->
```python
import numpy as np
from openai import OpenAI
from eval_toolkit.text_dedup import EmbeddingCosineStrategy, near_dedup

client = OpenAI()

def openai_embedder(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),
    )
    return np.array([e.embedding for e in response.data], dtype=np.float64)

strategy = EmbeddingCosineStrategy(embedder=openai_embedder)
report = near_dedup(texts=corpus, threshold=0.85, strategy=strategy)
```

Same shape contract; the toolkit doesn't care which model produced the
vectors — only that you return `(n, d)` floats with consistent
dimensionality across calls.

## Common pitfalls

- **Embedder consistency**: the same callable must return same-shape
  vectors across calls. If your embedder model_id changes mid-run, the
  cosine k-NN will silently produce wrong neighbors.
- **Dimension mismatch on `pairs_across`**: `EmbeddingCosineStrategy`
  raises `ValueError` if query and reference embeddings have different
  feature dimensions — the failure is loud, but only at the
  `pairs_across` call boundary.
- **Batching for cost / throughput**: the toolkit doesn't batch — your
  embedder owns that. `make_minilm_embedder(batch_size=64)` is the
  default sentence-transformers batch.

## See also

- {class}`~eval_toolkit.text_dedup.EmbeddingCosineStrategy` ([API](../api/text_dedup.md))
- {func}`~eval_toolkit.embeddings.make_minilm_embedder` ([API](../api/embeddings.md))
- [methodology/text_dedup.md](../methodology/text_dedup.md) for strategy
  selection guidance (TF-IDF vs MinHash-LSH vs Embedding vs Jaccard).
