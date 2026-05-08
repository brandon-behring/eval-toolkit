# Text deduplication

> **Background** *(skip if you've internalized this)*. Deduplication is
> the engine behind every leakage-detection check (see
> [leakage.md](leakage.md)) and a critical preprocessing step in its
> own right. The choice of *similarity strategy* — exact-hash, TF-IDF
> cosine, embedding cosine, MinHash-LSH, n-gram Jaccard — determines
> what counts as "similar". Picking the wrong one either misses real
> dupes (false-negative leakage that inflates eval metrics) or flags
> spurious matches (false-positive that drops valid eval rows).

This chapter covers
[`SimilarityStrategy`](../../src/eval_toolkit/text_dedup.py) — the
toolkit's pluggable similarity Protocol — and the five reference
impls: when to use each, threshold tuning, and how dedup composes
with [`LeakageCheck`](leakage.md).

## Setup

```python
from eval_toolkit import (
    SimilarityStrategy,
    TfidfCosineStrategy, ExactNormalizedHashStrategy,
    JaccardNgramStrategy, MinHashLSHStrategy,
    near_dedup, cross_dedup,
)
```

A small fixture used throughout:

```python
texts = [
    "the quick brown fox jumps",
    "the quick brown fox jumps!",       # near-dupe (punctuation)
    "the slow blue cat naps",
    "lorem ipsum dolor sit amet",
    "lorem ipsum dolor sit amet.",      # near-dupe (period)
]
```

## The five strategies {#strategies}

| Strategy | Sense of similarity | Cost | Best for |
|---|---|---|---|
| `ExactNormalizedHashStrategy` | Whitespace-normalized SHA-256 | O(n) | Catching exact dupes after normalization; near-zero false positives |
| `TfidfCosineStrategy` | Sparse TF-IDF cosine + k-NN | O(n × k_neighbors) | **Default**; balanced precision/recall on natural text |
| `EmbeddingCosineStrategy` | Caller-supplied embedding cosine | O(n²) (or use ANN) | Semantic dedup ("paraphrase" detection) |
| `MinHashLSHStrategy` | MinHash + Locality-Sensitive Hashing | O(n × b × r) | Web-scale corpora where TF-IDF doesn't fit |
| `JaccardNgramStrategy` | Set Jaccard on n-grams (brute-force) | O(n²) | Short texts; small n; diagnostic only |

### `ExactNormalizedHashStrategy`

Catches exact duplicates after Unicode normalization (NFKC + casefold +
whitespace collapse). Returns similarity 1.0 on collisions and 0.0
otherwise.

```python
from eval_toolkit.text_dedup import near_dedup as _near_dedup
report = near_dedup(texts, threshold=0.5, strategy=ExactNormalizedHashStrategy())
print(f"kept: {len(report.kept_indices)}; dropped: {[p[0] for p in report.dropped_pairs]}")
```

The "punctuation-different" pair above (idx 0/1) won't be caught by
exact-hash because `"...fox jumps"` and `"...fox jumps!"` differ after
normalization. Use TF-IDF for that.

### `TfidfCosineStrategy` (default)

Sparse TF-IDF (sklearn's `TfidfVectorizer` with character n-grams),
cosine similarity, top-k nearest neighbors per query. Default
`threshold=0.9` is conservative; lower for paraphrase-y domains.

```python
report = near_dedup(texts, threshold=0.85, strategy=TfidfCosineStrategy())
print(f"kept: {len(report.kept_indices)}; dropped: {[p[0] for p in report.dropped_pairs]}")
```

The punctuation-different pair shows up as similarity ~0.95–0.99
under TF-IDF char-n-grams; the threshold determines whether it's
flagged.

### `EmbeddingCosineStrategy`

Catches semantic paraphrases ("ignore all previous instructions" vs
"please disregard the prior directives"). Requires the caller to
supply pre-computed embeddings; the strategy itself just does cosine.

<!-- skip: next -->
```python
# Pseudo-code: requires embeddings (e.g., sentence-transformers)
# from sentence_transformers import SentenceTransformer
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = embedder.encode(texts, normalize_embeddings=True)
# strategy = EmbeddingCosineStrategy(embeddings)
# report = near_dedup(texts, threshold=0.9, strategy=strategy)
```

The toolkit's strategy doesn't load a model — keeping `transformers`
out of core deps. Consumer side: pre-encode, pass the array.

### `MinHashLSHStrategy`

For corpora too large to fit in TF-IDF memory (typically n > 100 k
texts). Uses MinHash signatures + LSH banding for approximate
near-duplicate detection. Has tunable false-negative / false-positive
trade-offs via the `bands` and `rows_per_band` parameters; see the
docstring at `src/eval_toolkit/text_dedup.py` for the LSH theory.

### `JaccardNgramStrategy`

Brute-force Jaccard similarity on n-grams. O(n²) — only for small n
or as a diagnostic. The toolkit ships it for completeness; in
production reach for `MinHashLSHStrategy` if you need the Jaccard
sense of similarity at scale.

## Threshold tuning {#thresholds}

The threshold is strategy-specific:

- **TF-IDF cosine**: `[0, 1]`. Default 0.9. Lower for paraphrase
  detection (e.g., 0.7); higher for strict near-exact (e.g., 0.95).
- **Exact-hash**: `0.5` (any positive threshold; the strategy returns
  0.0 / 1.0).
- **Embedding cosine**: `[-1, 1]` typically. 0.85–0.95 for sentence-
  embedding strategies; depends on the embedder.
- **MinHash LSH**: `[0, 1]`. 0.7–0.9 typical. Tune `bands` + `rows_per_band`
  to control the threshold's effective probability profile.
- **Jaccard**: `[0, 1]`. 0.5–0.8 typical for short n-grams.

> **What NOT to do.** Don't compare a TF-IDF threshold of 0.9 to a
> MinHash threshold of 0.9 and conclude they're equivalent —
> different strategies put 0.9 at different points on their
> precision / recall curve.

## How dedup composes with `LeakageCheck` {#composition}

[`NearDuplicateCheck`](../../src/eval_toolkit/leakage.py) wraps
`near_dedup` with a `SimilarityStrategy`-typed `strategy` parameter.
Same for `CrossSplitLeakageCheck` wrapping `cross_dedup`. So the
strategy-pluggability flows from `text_dedup` → `leakage` →
`harness.evaluate(leakage_checks=...)`.

This means one swap point gives you semantic-dedup leakage checks:

<!-- skip: next -->
```python
# from eval_toolkit import EmbeddingCosineStrategy, NearDuplicateCheck
# # In your evaluate(...) call:
# leakage_checks = [
#     NearDuplicateCheck(threshold=0.85, strategy=EmbeddingCosineStrategy(emb)),
# ]
```

For the prompt-injection-eval case the
[`NormalizedFormLeakageCheck`](leakage.md#encoding-obfuscation) is
sometimes more useful than embedding-based dedup — it catches the
encoding-obfuscation attack class specifically.

## Cross-source dedup (train ↔ eval) {#cross-source}

[`cross_dedup`](../../src/eval_toolkit/text_dedup.py) is the lower-
level primitive: given two text lists (train, eval), returns the eval
indices to *keep* (i.e., those NOT near-duplicate to any train text).
[`CrossSplitLeakageCheck`](leakage.md#cross-split) wraps this for the
harness.

## LSH false-negative rates {#lsh}

MinHash + LSH is approximate — there's a per-pair probability that
genuine near-duplicates are missed. The probability depends on:

- Number of bands (b)
- Rows per band (r)
- True Jaccard similarity (s)

The threshold curve `1 - (1 - s^r)^b` is the probability of a band
collision given Jaccard `s`. The "S-curve" inflection point is where
your effective threshold lives; tune `(b, r)` to put it at your
target similarity.

For `bands=20, rows_per_band=5` (default), the inflection is around
Jaccard ≈ 0.55. For tighter near-duplicate detection (e.g., target
Jaccard ≈ 0.85), use `bands=10, rows_per_band=10`.

## Pitfalls / Common mistakes {#pitfalls}

- **Using `JaccardNgramStrategy` on > 1 k texts.** O(n²) brute force;
  hangs for hours on real corpora. Use `MinHashLSHStrategy` for the
  same sense of similarity at scale.
- **Forgetting that TF-IDF requires fitting on the corpus.**
  `TfidfCosineStrategy` learns the vocabulary from the input texts;
  applying a strategy fit on corpus A to corpus B will silently
  produce zero similarity for any out-of-vocabulary token. The
  toolkit's `near_dedup` always fits the strategy on the input;
  `cross_dedup` fits on the union of train + eval. Don't reuse a
  pre-fit strategy across calls.
- **Embedding cosine without normalization.** If the embedding vectors
  aren't unit-norm, the cosine reduces to a dot product. Pass
  `normalize_embeddings=True` to the embedder, or pre-normalize.
- **Threshold-on-the-test-set.** Tuning the dedup threshold on the
  test set you'll then dedup is a methodological smell. Use a
  validation slice or a small synthetic-near-duplicate ground truth.
- **Trusting LSH at default `bands` / `rows`.** The defaults are sane
  but not universal; check your false-negative rate on a small
  ground-truth fixture before relying on it for leakage detection.
- **Skipping cross-source dedup.** Within-source dedup misses the
  classic train→test-leak (an identical row in both splits). Always
  run `cross_dedup` (or `CrossSplitLeakageCheck`) between every
  (train, eval) pair.

## Further reading

- Lee, K. et al. *Deduplicating training data makes language models
  better.* ACL 2022. — empirical effect of corpus dedup on LM
  performance; the canonical "near-dup matters" reference.
- Penedo, G. et al. *RefinedWeb.* arXiv 2023. — a modern, large-scale
  dedup pipeline (TF-IDF + MinHash hybrid); good engineering reference.
- Penedo, G. et al. *FineWeb-2.* arXiv 2025. — successor; documents
  per-strategy thresholds + ablations.
- Broder, A. *On the resemblance and containment of documents.*
  SEQUENCES 1997. — the original MinHash paper.
- Indyk, P. & Motwani, R. *Approximate nearest neighbors: towards
  removing the curse of dimensionality.* STOC 1998. — LSH foundations.

See also: [leakage.md](leakage.md) (`NearDuplicateCheck`,
`CrossSplitLeakageCheck`, `NormalizedFormLeakageCheck`),
[extending.md §"Implementing a SimilarityStrategy"](../extending.md#similarity-strategy).
