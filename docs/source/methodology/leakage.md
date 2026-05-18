# Leakage

> **Background** *(skip if you've internalized this)*. Leakage is when an
> evaluation procedure lets information about the test set influence the
> model — directly (a row appears in both train and test), indirectly
> (correlated rows; same patient in both folds), or structurally (the
> model's feature pipeline saw test data during fitting). It inflates
> reported metrics by amounts that often exceed the gap between papers.
> Kapoor & Narayanan ([2023](https://arxiv.org/abs/2207.07048)) document
> 294 papers across 17 fields where leakage was the cause of
> non-replication. Treat leakage detection as a *first-class* eval step,
> not a one-off audit.

This chapter is a taxonomy of leakage classes, the eval-toolkit primitive
that catches each one, and the pitfalls that make leakage hard to spot.
Every code block runs under [Sybil](https://sybil.readthedocs.io/) — copy
them as starting points.

## Setup

```python
import pandas as pd
from eval_toolkit import EvalSlice, run_leakage_checks
```

We'll reuse two small split fixtures throughout:

```python
clean_train = pd.DataFrame(
    {"text": [f"unique_train_{i}" for i in range(20)],
     "label": [i % 2 for i in range(20)]}
)
clean_test = pd.DataFrame(
    {"text": [f"unique_test_{i}" for i in range(10)],
     "label": [i % 2 for i in range(10)]}
)
clean_splits = {
    "train": EvalSlice(name="train", df=clean_train),
    "test":  EvalSlice(name="test",  df=clean_test),
}
```

## Taxonomy

The eight classes below cover what consumers of this toolkit will hit in
binary-classification work. Subclasses (e.g., temporal, group, transfer-
learning) are grouped under the parent that catches them.

| # | Class | Toolkit primitive | Severity default |
|---|---|---|---|
| 1 | Exact duplicate | `ExactDuplicateCheck` | warning |
| 2 | Near duplicate | `NearDuplicateCheck` | warning |
| 3 | Encoding-obfuscated duplicate | `NormalizedFormLeakageCheck` | error |
| 4 | Cross-split (train↔eval) | `CrossSplitLeakageCheck` | error |
| 5 | Label conflict | `LabelConflictCheck` | error |
| 6 | Group leakage | `GroupLeakageCheck` | error |
| 7 | Temporal leakage | `TemporalLeakageCheck` | error |
| 8 | Target / pretraining / distribution shift | (out of toolkit scope) | n/a |

(exact-duplicates)=
### 1. Exact duplicates
**Definition.** Two rows whose normalized text is identical (same string
after Unicode-NFC + whitespace collapse + casefold).

**Harm.** Within-split duplicates inflate observed metrics by the amount
of *redundancy* in your data: a duplicated rare positive double-counts in
the recall numerator. Across-split duplicates are a bigger issue — see
class 4.

**Primitive.** [`ExactDuplicateCheck`](../api/leakage.md)
wraps `text_dedup.ExactNormalizedHashStrategy` (whitespace-normalized
SHA-256 buckets). Default severity is `"warning"` because exact dupes
within a split are common in real corpora; opt into `"error"` for strict
mode.

```python
from eval_toolkit import ExactDuplicateCheck

dup_train = pd.DataFrame(
    {"text": ["hello world", "hello world", "unique"], "label": [0, 0, 1]}
)
dup_splits = {"train": EvalSlice(name="train", df=dup_train)}

finding = ExactDuplicateCheck().validate(dup_splits)
print(f"severity={finding.severity}  n_affected={finding.n_affected}")
print(f"drop_indices={finding.drop_indices}")
```

(near-duplicates)=
### 2. Near duplicates
**Definition.** Pairs of rows whose textual similarity exceeds a threshold
under some sense — TF-IDF cosine, embedding cosine, MinHash Jaccard.

**Harm.** Same as exact duplicates but with a tunable strictness. Common
sources: paraphrased social-media posts, scraped duplicates with minor
HTML differences, multiple translations of the same source.

**Primitive.** [`NearDuplicateCheck`](../api/leakage.md)
takes a `strategy: SimilarityStrategy` and a `threshold` — pluggable so
the *sense of similarity* is opt-in. Defaults to `TfidfCosineStrategy` at
0.9.

```python
from eval_toolkit import NearDuplicateCheck

paraphrases = pd.DataFrame(
    {"text": [
        "the quick brown fox jumps",
        "the quick brown fox jumps!",
        "lorem ipsum dolor sit amet",
    ],
     "label": [0, 0, 1]}
)
splits = {"train": EvalSlice(name="train", df=paraphrases)}
finding = NearDuplicateCheck(threshold=0.8).validate(splits)
print(f"caught {finding.n_affected} near-dupe rows")
```

For semantic dedup, swap the strategy:

```python
# from eval_toolkit import EmbeddingCosineStrategy
# strategy = EmbeddingCosineStrategy(embedder=my_sentence_transformer)
# NearDuplicateCheck(strategy=strategy, threshold=0.85)
```

(encoding-obfuscation)=
### 3. Encoding-obfuscated duplicates (NEW in v0.7.0)
**Definition.** Rows that *look different* but normalize to the same text
under aggressive Unicode transforms — NFKC + zero-width-strip +
Symbol-Other-strip (most emoji) + casefold.

**Harm.** This is the **dominant unfixed leakage class in
prompt-injection corpora**. The PI_HackAPrompt_SQuAD work (2025) reports
that encoding-obfuscated duplicates detect at only **21.3 %** under naive
dedup but achieve **76.2 %** attack success rate against deployed
classifiers. If your eval set has zero-width-padded copies of your train
set, you are measuring nothing.

**Primitive.** [`NormalizedFormLeakageCheck`](../api/leakage.md)
applies the aggressive normalization before hashing. Default severity is
`"error"` — encoding-obfuscated overlap is dangerous enough that you
should opt *out* (to `"warning"`) only when upstream cleaning already
handles it.

```python
from eval_toolkit import NormalizedFormLeakageCheck

# Same string, with zero-width characters injected.
obfuscated = pd.DataFrame(
    {"text": ["hello world", "h​e​llo  world", "unrelated"],
     "label": [0, 1, 0]}
)
splits = {"test": EvalSlice(name="test", df=obfuscated)}
finding = NormalizedFormLeakageCheck().validate(splits)
print(f"caught {finding.n_affected} obfuscation collision(s)")
```

> **What NOT to do.** Don't rely on `text.lower().strip()` for
> prompt-injection eval. The class of attacks documented in OWASP
> [LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
> includes zero-width injection, emoji-padded payloads, alt-encoding —
> all defeated only by NFKC + Symbol-Other strip.

(cross-split)=
### 4. Cross-split (train ↔ eval) leakage
**Definition.** A row in your test slice that is identical or near-
identical to a row in your training slice.

**Harm.** The canonical leakage. Your model has memorized the test point;
your "test metric" is a memorization metric. Inflations of 5–30 % PR-AUC
have been reported on text classification benchmarks where dedup was
skipped.

**Primitive.** [`CrossSplitLeakageCheck`](../api/leakage.md)
wraps `text_dedup.cross_dedup`. Default severity `"error"` — this is the
genuinely dangerous one.

```python
from eval_toolkit import CrossSplitLeakageCheck

train = pd.DataFrame({"text": ["hello world a longer string", "lorem"], "label": [0, 1]})
test  = pd.DataFrame({"text": ["hello world a longer string", "different"], "label": [1, 0]})
splits = {
    "train": EvalSlice(name="train", df=train),
    "test":  EvalSlice(name="test",  df=test),
}
finding = CrossSplitLeakageCheck(train_split="train").validate(splits)
print(f"{finding.n_affected} test row(s) leak from train")
```

(label-conflict)=
### 5. Label conflict
**Definition.** The same (or near-same) text appearing with *different*
labels across slices, or across the source datasets you concatenated to
build a slice.

**Harm.** Annotator disagreement aside, label conflicts mean your eval is
ambiguous. The same prompt being labeled "injection" in train and "benign"
in test poisons the evaluation regardless of which label is "correct".

**Primitive.** [`LabelConflictCheck`](../api/leakage.md).
Replaces the cross-source conflict resolution that
`prompt-injection-sdd` and `prompt_injection_detector` reimplement
(~50 LOC each).

```python
from eval_toolkit import LabelConflictCheck

train = pd.DataFrame({"text": ["x", "y"], "label": [0, 1]})
test  = pd.DataFrame({"text": ["x", "z"], "label": [1, 0]})  # conflict on "x"
splits = {
    "train": EvalSlice(name="train", df=train),
    "test":  EvalSlice(name="test",  df=test),
}
finding = LabelConflictCheck().validate(splits)
print(f"{finding.n_affected} conflicting rows across {len(splits)} splits")
```

(group-leakage)=
### 6. Group leakage
**Definition.** A grouping unit (patient, user, document, source) that
appears in more than one split.

**Harm.** Within-group correlation makes the test metric a *memorization
of the group*, not a generalization measure. The classic medical-imaging
failure: same patient's images in train and test → near-100 % accuracy
that vanishes on truly held-out patients.

**Primitive.** [`GroupLeakageCheck`](../api/leakage.md)
takes a `group_col`. Severity `"error"`. Use with
[`GroupKFoldSplitter`](../api/splits.md) or
[`SourceDisjointKFoldSplitter`](../api/splits.md) — see
[splits.md](splits.md).

```python
from eval_toolkit import GroupLeakageCheck

train = pd.DataFrame(
    {"text": ["a", "b", "c"], "label": [0, 1, 0], "group_id": [1, 2, 3]}
)
test = pd.DataFrame(
    {"text": ["d", "e"], "label": [1, 0], "group_id": [1, 4]}  # group 1 spans
)
splits = {
    "train": EvalSlice(name="train", df=train),
    "test":  EvalSlice(name="test",  df=test),
}
finding = GroupLeakageCheck(group_col="group_id").validate(splits)
print(f"groups spanning splits: {sorted(finding.evidence['violating_groups'].keys())}")
```

(temporal-leakage)=
### 7. Temporal leakage
**Definition.** Train data with timestamps later than test data — the
model is allowed to "see the future" during training.

**Harm.** Temporal correlation: features are non-stationary, label
distributions drift, the test metric measures interpolation rather than
forecasting. Recently formalized by Yan et al. ([Hidden Leaks in Time
Series Forecasting, 2025](https://arxiv.org/html/2512.06932v1)) which
catalogs LSTM rolling-window leakage and validation-strategy leakage as
distinct subclasses.

**Primitive.**
[`TemporalLeakageCheck`](../api/leakage.md) — given a
`time_col` and `split_order`, asserts every earlier split's `max(time)`
≤ next split's `min(time)`.

```python
from eval_toolkit import TemporalLeakageCheck

train = pd.DataFrame({"text": ["a", "b"], "label": [0, 1], "t": [10, 20]})
test  = pd.DataFrame({"text": ["c", "d"], "label": [1, 0], "t": [5, 15]})  # bad
splits = {
    "train": EvalSlice(name="train", df=train),
    "test":  EvalSlice(name="test",  df=test),
}
finding = TemporalLeakageCheck(
    time_col="t", split_order=("train", "test")
).validate(splits)
print(f"violations={len(finding.evidence['violations'])}")
```

(target-leakage)=
### 8. Target / pretraining / distribution-shift leakage
**Definition (umbrella).** Three related classes that the toolkit *does
not* try to detect automatically:

- **Target leakage**: a feature trivially predicts the label because it's
  computed *after* the label is determined (e.g., "user clicked
  'unsubscribe'" as a feature for a churn label).
- **Transfer-learning / pretraining-corpus leakage**: a pretrained model
  has already seen part of your eval set during pretraining (Don't Push
  the Button, [Pellizzoni et al. 2025](https://link.springer.com/article/10.1007/s10462-025-11326-3)).
  Common with HuggingFace pretrained transformers and public benchmarks.
- **Distribution shift**: the production population differs from your eval
  population in ways that change the metric's interpretation (covariate
  shift, label shift, concept drift; Recht et al. 2019).

**Why the toolkit doesn't catch these.** Target leakage requires
domain-specific feature semantics. Pretraining leakage requires access to
the pretraining corpus. Distribution shift requires the production
distribution. All three are *consumer-side* concerns, not eval-toolkit
concerns. The pointers above are required reading.

(pytorch-pitfalls)=
## PyTorch & transformer-specific pitfalls
### Tokenization-level duplicates

Two strings can look distinct in plain text but tokenize identically once
a transformer's BPE / SentencePiece / WordPiece tokenizer is applied.
Don't dedup on raw text only when your downstream model is a transformer
— dedup on the tokenizer's output too.

Since v0.37.0 the toolkit ships `TokenizationLeakageCheck` for this
directly. It takes a tokenizer callable (any HuggingFace
`PreTrainedTokenizerBase`, or any `Callable[[str], Mapping]` returning
HF-style `{"input_ids": [...]}` output) and dedups on the `input_ids`
tuple per row:

```text
from transformers import AutoTokenizer
from eval_toolkit.leakage import TokenizationLeakageCheck

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
check = TokenizationLeakageCheck(tokenizer=tokenizer)
finding = check.validate(splits)
```

Optional install via the `[transformers]` extra (intentionally **not**
in `[all]` / `[dev]` — transformers transitively pulls torch (~700MB)
per the [embeddings] precedent). The check itself does not import
transformers; consumers pass an already-instantiated tokenizer.

**Pin the tokenizer for audit reproducibility.** Different `transformers`
releases can emit different `input_ids` for the same text (added-vocab
changes between minors), which would silently flip dedup outcomes.
Capture both the package version and a SHA-256 of the `tokenizer.json`
in your `RunManifest`.

For consumers who want to avoid the `[transformers]` extra entirely:
emulate the check by adding a `tokens` column to your dataframe and
pointing `ExactDuplicateCheck` at it via a custom feature view on the
`EvalSlice`. The `TokenizationLeakageCheck` is just sugar over that
pattern with explicit tokenizer-pin guidance.

### Pretraining contamination

Public benchmarks (GLUE, SuperGLUE, MMLU, even niche prompt-injection
corpora) often appear verbatim in pretraining corpora like Common Crawl.
A "9X.X % accuracy" on a public benchmark with a public-pretrained
backbone is rarely a generalization claim. Two mitigations:

1. **Audit publication dates.** A model pretrained in 2024-Q3 has seen
   anything published before then. Use eval sets curated *after* your
   model's pretraining cutoff.
2. **Use embedding-space dedup.** [`EmbeddingCosineStrategy`](../api/text_dedup.md)
   wrapped in a `NearDuplicateCheck` lets you flag eval rows that
   embed-near any train row, including rough paraphrases. Doesn't catch
   pretraining contamination but *does* catch your own train-set
   contamination of an eval set.

(leakage-pitfalls)=
## Pitfalls / Common mistakes
- **Running checks AFTER training.** By then the harm is done. Run
  leakage checks at *load time*, before any model fits anything. The
  recommended pattern is to pass `leakage_checks=[...]` to
  [`evaluate(...)`](../api/harness.md) which fails the
  run on `error`-severity findings before scoring starts.
- **Trusting per-source dedup as cross-source dedup.** Two corpora that
  each look clean can still leak across each other. Always run
  `CrossSplitLeakageCheck` between every (train, eval) pair, not just
  within each.
- **Picking thresholds without understanding the strategy.** A
  `TfidfCosineStrategy` threshold of 0.9 is *not* the same fraction as a
  `JaccardNgramStrategy` threshold of 0.9. Read the strategy's docstring
  before tuning.
- **Conflating `severity="warning"` with "ignore"**. Warnings still
  appear in the manifest's `leakage_report`; downstream auditors will
  see them. They don't gate the run, but they're not invisible.
- **Believing CV alone gives an OOD claim.** K-fold CV with random
  partitioning measures interpolation across your sample, not
  generalization to a new population. For OOD claims, combine
  [`SourceDisjointKFoldSplitter`](../api/splits.md) +
  `CrossSplitLeakageCheck` + a held-out *final* test set never used in
  development.

## Putting it all together

```python
from eval_toolkit import (
    EvalSlice, ExactDuplicateCheck, NormalizedFormLeakageCheck,
    LabelConflictCheck, CrossSplitLeakageCheck, run_leakage_checks,
)

train = pd.DataFrame({"text": ["clean a", "clean b"], "label": [0, 1]})
test  = pd.DataFrame({"text": ["clean c", "clean d"], "label": [1, 0]})
splits = {
    "train": EvalSlice(name="train", df=train),
    "test":  EvalSlice(name="test",  df=test),
}

report = run_leakage_checks(
    [
        ExactDuplicateCheck(),                            # warning
        NormalizedFormLeakageCheck(),                     # error
        LabelConflictCheck(),                             # error
        CrossSplitLeakageCheck(train_split="train"),      # error
    ],
    splits,
)
print(f"clean splits: has_errors={report.has_errors()}")
```

## Further reading

- Kapoor, S. & Narayanan, A. *Leakage and the reproducibility crisis in
  machine-learning-based science.* Patterns 4(9), 2023.
  [arXiv:2207.07048](https://arxiv.org/abs/2207.07048)
- Pellizzoni, S. et al. *Don't push the button! Data leakage risks in
  ML and transfer learning.* Springer AI Review, 2025.
  [DOI](https://link.springer.com/article/10.1007/s10462-025-11326-3)
- Yan, X. et al. *Hidden Leaks in Time Series Forecasting.* arXiv 2025.
  [arXiv:2512.06932](https://arxiv.org/html/2512.06932v1)
- OWASP. *LLM01:2025 Prompt Injection.*
  [genai.owasp.org](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- PI_HackAPrompt_SQuAD analysis (2025).
  [arXiv:2505.04806](https://arxiv.org/html/2505.04806v1)
- Recht, B. et al. *Do ImageNet classifiers generalize to ImageNet?*
  ICML 2019.

See also: [splits.md](splits.md), [reproducibility.md](reproducibility.md).
