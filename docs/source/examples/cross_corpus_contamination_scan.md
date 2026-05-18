---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Worked example: cross-corpus contamination scan via `pairs_across`

> **What this shows.** "For each row in eval corpus A, what's the maximum
> cosine similarity to ANY row in reference corpus B?" — the canonical
> contamination-scan pattern. Examples: benign-vs-injection template
> overlap, OOD-vs-train data leakage, eval-vs-pretraining contamination.
>
> **Runtime:** ~0.5 s with a stub embedder. The pattern generalises to
> any `SimilarityStrategy`.

## Why `pairs_across`, not `near_dedup`

`near_dedup` is built for the *within-corpus* dedup case ("find duplicates
inside one corpus"). The *cross-corpus* case ("for each query, find the
most-similar reference") is the natural `pairs_across` use:

- **Within-corpus dedup**: `near_dedup(texts=corpus, threshold=t, strategy=...)`
- **Cross-corpus contamination**: `strategy.pairs_across(query_texts=eval_corpus,
  reference_texts=train_corpus, k=1)`

The toolkit exposes both Tier-2 surfaces; the protocol is the same
`SimilarityStrategy` — only the orchestration glue differs.

## Setup

```{code-cell}
import numpy as np
from eval_toolkit.text_dedup import EmbeddingCosineStrategy

def stub_embedder(texts):
    # Identity embedder: hash each text into a unit vector for deterministic
    # tests. In production use make_minilm_embedder() per
    # callable_embedder_dedup.md.
    n = len(texts)
    vecs = np.zeros((n, 32), dtype=np.float64)
    for i, t in enumerate(texts):
        vecs[i, hash(t) % 32] = 1.0
    return vecs

strategy = EmbeddingCosineStrategy(embedder=stub_embedder)
```

## The contamination scan

```{code-cell}
# Eval corpus (rows we want to flag).
eval_corpus = ["how do I bake bread", "what is 2+2", "ignore previous instructions"]
# Reference corpus (training / template / contamination source).
template_corpus = ["forget all prior", "ignore all previous instructions", "step-by-step recipe"]

# For each eval row, find the single most-similar template (k=1).
similarities, indices = strategy.pairs_across(
    query_texts=eval_corpus,
    reference_texts=template_corpus,
    k=1,
)
# similarities shape: (n_eval, k=1)
# indices shape: same — indices into template_corpus

max_cosines = similarities[:, 0]
nearest_template = [template_corpus[i] for i in indices[:, 0]]

# Flag eval rows whose nearest template is suspiciously similar.
CONTAMINATION_THRESHOLD = 0.85
flagged = max_cosines >= CONTAMINATION_THRESHOLD
contamination_rate = float(flagged.mean())
print(f"contamination_rate: {contamination_rate:.2%}")
```

For each flagged row, the `nearest_template[i]` tells you *which* template
triggered the flag — useful for audit + manual review.

## Variations

**Top-k** instead of top-1 (each eval row's k nearest templates):

```{code-cell}
similarities, indices = strategy.pairs_across(
    query_texts=eval_corpus,
    reference_texts=template_corpus,
    k=3,  # top-3 nearest templates per eval row
)
# similarities shape: (n_eval, 3)
# indices shape: (n_eval, 3) — sorted descending by similarity
```

**Asymmetric thresholds** (different thresholds per `eval_corpus` slice):

```{code-cell}
# After computing max_cosines via k=1:
slice_a_mask = np.array([t.startswith("how") for t in eval_corpus])
flagged_a = (max_cosines >= 0.90) & slice_a_mask
flagged_others = (max_cosines >= 0.80) & ~slice_a_mask
total_flagged = flagged_a | flagged_others
```

## Common pitfalls

- **Empty reference corpus**: `pairs_across` raises if `reference_texts`
  is empty. Guard upstream if your reference corpus can be empty
  (e.g., for projects with optional templates).
- **Dimension mismatch**: `EmbeddingCosineStrategy._embed` validates
  query + reference embeddings have matching feature dimension; the
  failure is loud + early.
- **One-way semantics**: this pattern is asymmetric — "for each query,
  find nearest reference". Reversing the arguments (query↔reference)
  asks the inverse question. For symmetric within-set dedup, use
  `pairs_within` or `near_dedup`.

## See also

- {class}`~eval_toolkit.text_dedup.EmbeddingCosineStrategy` ([API](../api/text_dedup.md))
- [Leakage detection](leakage_detection.md) — for the in-toolkit
  `CrossSplitLeakageCheck` orchestrator that wraps this pattern with
  fold-aware semantics.
- [methodology/leakage.md](../methodology/leakage.md) — contamination
  taxonomy + when to add this scan to your eval.
