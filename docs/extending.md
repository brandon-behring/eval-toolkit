# Extending eval-toolkit

This guide is the build-side complement to
[`docs/methodology/`](methodology/README.md). The methodology docs say
what good evaluation looks like; this guide says how to plug your code
into eval-toolkit's harness.

> **Three tiers, three entry points.** The toolkit is layered so you can
> start at the level of abstraction your task actually needs:
>
> - **Tier 1 — functional core.** Pure functions on `(y_true, y_score)`
>   arrays. No model coupling. Use this when you have predictions
>   already and just want metrics + CIs.
> - **Tier 2 — protocols.** Implement [`Scorer`](#scorer),
>   [`LeakageCheck`](#leakage-check), [`Splitter`](#splitter),
>   [`ThresholdSelector`](#threshold-selector),
>   [`DatasetLoader`](#dataset-loader),
>   [`SimilarityStrategy`](#similarity-strategy). Use these when you
>   want the harness to orchestrate.
> - **Tier 3 — reproducibility scaffolding.**
>   [`build_manifest`](api/manifest.md),
>   [`set_global_seeds`](api/seeds.md),
>   `provenance.*`. Use these regardless of which tier you build at.

## Setup (used throughout)

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice
```

## Tier 1 — functional core (no Protocol needed) {#functional-core}

When you already have predictions, just call the metrics directly:

```python
from eval_toolkit import pr_auc, roc_auc, bootstrap_ci, paired_bootstrap_diff

rng = np.random.default_rng(42)
y = rng.binomial(1, 0.3, size=200)
s = np.clip(0.6 * y + rng.normal(0, 0.25, size=200), 0, 1)

ci = bootstrap_ci(y, s, pr_auc, n_resamples=500, seed=42)
print(f"PR-AUC: {ci.point_estimate:.3f}  CI [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")
```

This is the fastest path — no Protocol implementation, no harness
orchestration, no manifest. Useful for ad-hoc analysis and notebooks.

## Implementing a `Scorer` {#scorer}

The [`Scorer` Protocol](api/harness.md) is anything
exposing `predict_proba(X) -> np.ndarray of P(positive)`.

### sklearn classifier

Trivial — sklearn's `LogisticRegression`, `RandomForestClassifier`, etc.
already satisfy the Protocol. Wrap to return only the positive-class
column:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class TfidfLogisticScorer:
    """sklearn pipeline as an eval_toolkit.Scorer."""
    version = "0.1.0"  # captured into RunManifest.versioned_objects

    def __init__(self) -> None:
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("lr", LogisticRegression(max_iter=200, random_state=42)),
        ])

    def fit(self, X: list[str], y: np.ndarray) -> None:
        self.pipe.fit(X, y)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]


# Demo on tiny synthetic data:
scorer = TfidfLogisticScorer()
texts = [f"good text {i}" for i in range(50)] + [f"bad attack {i}" for i in range(50)]
labels = np.array([0] * 50 + [1] * 50)
scorer.fit(texts, labels)
preds = scorer.predict_proba(texts[:5])
print(f"first 5 scores: {preds.round(3)}")
```

Note the `version` attribute — implementing the
[`Versioned`](api/leakage.md) opt-in Protocol means
`build_manifest(versioned={...})` auto-captures it, so cross-version
metric comparisons can be invalidated. See
[`methodology/versioning.md`](methodology/versioning.md) for the full
story (when to expose `version`, how to choose a version string, the
lm-evaluation-harness pattern this mirrors).

### LLM-judge with cost control

The [`SliceAwareScorer`](api/harness.md) Protocol's
`should_score_slice(name)` hook lets the harness skip slices the
scorer doesn't need to score — critical for expensive LLM judges:

```python
class _LLMJudgeStub:
    """Pretend LLM-judge that runs only on the headline slice."""
    version = "claude-haiku-2026-q1"

    def predict_proba(self, X):
        # In production: a batched LLM call returning P(injection).
        return np.full(len(X), 0.5)

    def should_score_slice(self, slice_name: str) -> bool:
        # Cost-control: don't burn budget on subgroup / OOD slices.
        return slice_name == "test"


judge = _LLMJudgeStub()
print(f"score 'test' slice? {judge.should_score_slice('test')}")
print(f"score 'ood_lakera' slice? {judge.should_score_slice('ood_lakera')}")
```

`evaluate(..., scorers={'judge': judge})` calls `should_score_slice`
before scoring; skipped slices land in `RunResult.by_slice[name]
.by_scorer[scorer_name] = {"skipped": "<reason>"}`.

### PyTorch + transformer + LoRA scorer

See [pytorch_scorer_example.md](examples/pytorch_scorer_example.md) for
the worked example. The shape is the same — wrap an `nn.Module` so its
forward+softmax returns a numpy array.

## Implementing a `LeakageCheck` {#leakage-check}

[`LeakageCheck`](api/leakage.md) takes
`Mapping[str, EvalSlice]` and returns a `LeakageFinding`. The uniform
input shape means within-split and cross-split checks share one
contract.

```python
from dataclasses import dataclass
from collections.abc import Mapping
from eval_toolkit import LeakageFinding

@dataclass(frozen=True, slots=True)
class TextLengthOutlierCheck:
    """A toy LeakageCheck: flag rows whose text is >5x the median length."""
    severity: str = "warning"

    @property
    def name(self) -> str:
        return "TextLengthOutlierCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        drop: dict[str, list[int]] = {}
        n_affected = 0
        for split_name, slice_ in splits.items():
            lengths = np.array([len(t) for t in slice_.features])
            if len(lengths) == 0:
                continue
            median = float(np.median(lengths))
            mask = lengths > 5 * max(median, 1)
            if mask.any():
                drop[split_name] = sorted(np.where(mask)[0].tolist())
                n_affected += int(mask.sum())
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,  # type: ignore[arg-type]
            drop_indices=drop,
            evidence={"rule": "len(text) > 5 * median"},
            message=(f"{n_affected} length-outlier rows" if n_affected
                     else "no length outliers"),
            n_affected=n_affected,
        )

# Demo:
df = pd.DataFrame({"text": ["short"] * 10 + ["x" * 10000], "label": [0] * 10 + [1]})
splits = {"test": EvalSlice(name="test", df=df)}
finding = TextLengthOutlierCheck().validate(splits)
print(f"{finding.check_name}: {finding.message}")
```

The toolkit's reference impls in `leakage.py` are the canonical
patterns to mirror — see
[methodology/leakage.md](methodology/leakage.md) for which check goes
with which problem.

## Implementing a `Splitter` {#splitter}

[`Splitter`](api/splits.md) yields fold-dicts ready for
`evaluate(...)`. The simplest implementation wraps an existing sklearn
splitter:

```python
from collections.abc import Iterator
from dataclasses import dataclass
from sklearn.model_selection import StratifiedShuffleSplit

@dataclass(frozen=True, slots=True)
class StratifiedShuffleSplitter:
    """A small Splitter wrapping sklearn.StratifiedShuffleSplit."""
    n_splits: int = 5
    test_size: float = 0.2
    seed: int = 42

    def iter_folds(
        self, slice_, *, groups=None
    ) -> Iterator[dict[str, EvalSlice]]:
        sss = StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size,
            random_state=self.seed,
        )
        y = slice_.y_true
        x_dummy = np.arange(len(y)).reshape(-1, 1)
        for train_idx, test_idx in sss.split(x_dummy, y):
            yield {
                "train": EvalSlice(
                    name="train", df=slice_.df.iloc[train_idx].reset_index(drop=True),
                    feature_col=slice_.feature_col, label_col=slice_.label_col,
                    strata_col=slice_.strata_col,
                ),
                "test": EvalSlice(
                    name="test", df=slice_.df.iloc[test_idx].reset_index(drop=True),
                    feature_col=slice_.feature_col, label_col=slice_.label_col,
                    strata_col=slice_.strata_col,
                ),
            }

    def get_n_splits(self, slice_) -> int:
        return self.n_splits


# Demo:
df = pd.DataFrame({"text": [f"r{i}" for i in range(40)],
                   "label": [i % 2 for i in range(40)]})
parent = EvalSlice(name="all", df=df)
spl = StratifiedShuffleSplitter(n_splits=3, test_size=0.25)
for i, fold in enumerate(spl.iter_folds(parent)):
    print(f"  fold {i}: train={len(fold['train'].df)} test={len(fold['test'].df)}")
```

## Implementing a `ThresholdSelector` {#threshold-selector}

[`ThresholdSelector`](api/thresholds.md) returns a
[`ThresholdResult`](api/metrics.md). A custom selector
is one short class:

```python
from dataclasses import dataclass
from eval_toolkit import ThresholdResult, metrics_at_threshold

@dataclass(frozen=True, slots=True)
class FixedThresholdSelector:
    """Always returns a caller-supplied threshold."""
    threshold: float

    @property
    def criterion(self) -> str:
        return f"fixed_{self.threshold:.3f}"

    def select(self, y_true, y_score) -> ThresholdResult:
        m = metrics_at_threshold(y_true, y_score, self.threshold)
        return ThresholdResult(
            threshold=float(self.threshold), f1=float(m["f1"]),
            precision=float(m["precision"]), recall=float(m["recall"]),
            criterion=self.criterion,
        )

# Demo:
y = np.array([0, 0, 1, 1, 0, 1])
s = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
result = FixedThresholdSelector(0.5).select(y, s)
print(f"fixed 0.5: F1={result.f1:.3f}  P={result.precision:.3f}")
```

When threshold-selection variance matters, pair with
`paired_bootstrap_op_point_diff` — see
[methodology/thresholds.md §"When to refit threshold per resample"
](methodology/thresholds.md#bootstrap-refit).

## Implementing a `DatasetLoader` {#dataset-loader}

[`DatasetLoader`](api/loaders.md) returns
`dict[str, EvalSlice]` (HF `DatasetDict` shape) plus a Croissant-
compatible `describe()`. Tensor-agnostic: torch users tokenize inside
the `Scorer`, not the loader.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TwoListLoader:
    """Loader for the simplest case: two pre-split lists of texts/labels."""
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    name: str = ""

    def load_splits(self) -> dict[str, EvalSlice]:
        return {
            "train": EvalSlice(
                name="train",
                df=pd.DataFrame({"text": self.train_texts, "label": self.train_labels}),
            ),
            "test": EvalSlice(
                name="test",
                df=pd.DataFrame({"text": self.test_texts, "label": self.test_labels}),
            ),
        }

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name or "TwoListLoader",
            "description": "",
            "citeAs": "",
            "license": "",
            "url": "",
            "distribution": [],
            "n_train": len(self.train_texts),
            "n_test": len(self.test_texts),
        }


# Demo:
loader = TwoListLoader(
    train_texts=["a", "b"], train_labels=[0, 1],
    test_texts=["c"], test_labels=[1], name="demo",
)
splits = loader.load_splits()
print(f"keys={list(splits.keys())}  describe.name={loader.describe()['name']}")
```

## Implementing a `SimilarityStrategy` {#similarity-strategy}

This Protocol predates v0.7.0; see
[`text_dedup.py`](api/text_dedup.md)'s docstring and
the existing reference impls (TfidfCosineStrategy,
ExactNormalizedHashStrategy, EmbeddingCosineStrategy,
JaccardNgramStrategy, MinHashLSHStrategy). The shape:
`pairs_within(texts, k_neighbors)` → similarity / index arrays. Pluggable
backend for `near_dedup` and `cross_dedup` and (transitively)
`NearDuplicateCheck` / `CrossSplitLeakageCheck`.

## Recipe: full custom eval harness in ~50 lines {#recipe}

Combines every Tier-2 Protocol + the manifest:

```python
from eval_toolkit import (
    EvalSlice, evaluate_folded,
    NormalizedFormLeakageCheck, LabelConflictCheck,
    StratifiedKFoldSplitter, MaxF1Selector,
    build_manifest, write_manifest, set_global_seeds,
)

set_global_seeds(42)

# 1. Dataset (use the loader you've built or the TwoListLoader above).
df = pd.DataFrame({
    "text": [f"benign_{i}" if i < 30 else f"injection_{i}" for i in range(60)],
    "label": [0 if i < 30 else 1 for i in range(60)],
})
parent = EvalSlice(name="all", df=df)

# 2. Scorer (use the TfidfLogisticScorer above or your transformer adapter).
class _DummyScorer:
    version = "0.0.0"
    def predict_proba(self, X):
        return np.array([0.7 if "injection" in t else 0.2 for t in X])

# 3. Run K-fold + leakage checks + auto CV-CI summary.
result = evaluate_folded(
    {"dummy": _DummyScorer()},
    StratifiedKFoldSplitter(k=3, seed=42),
    parent,
    run_id="custom-harness-demo",
    leakage_checks=[NormalizedFormLeakageCheck(), LabelConflictCheck()],
    on_leakage="record",
    eval_split_names=("test",),
)

print(f"folds: {len(result.by_fold)}")
fs = result.fold_summary["test"]["dummy"]["pr_auc"]
print(f"PR-AUC: {fs['mean']:.3f}  CI [{fs['ci_low']:.3f}, {fs['ci_high']:.3f}]")

# 4. Reproducibility manifest.
import tempfile
m = build_manifest(
    run_id="custom-harness-demo",
    config={"k": 3, "seed": 42, "scorer": "dummy"},
    seeds={"global": 42, "bootstrap": 42},
    versioned={"dummy": _DummyScorer()},
)
with tempfile.TemporaryDirectory() as d:
    write_manifest(m, d)
    print(f"manifest captured: schema={m.schema_version}")
```

That's the full pipeline: dataset loading, leakage validation,
splitting, scoring, CV-CI aggregation, manifest emission. Every step
above is replaceable with a custom Protocol implementation.

## Project layout for downstream consumers {#project-layout}

The [`prompt_injection_classifier_showcase`
](https://github.com/brandon-behring/prompt_injection_classifier_showcase)
repo is the canonical worked example. Mirror its layout:

```
your_project/
  src/your_project/
    scorers.py       # Scorer implementations
    data.py          # DatasetLoader (only if not using built-in loaders)
    evaluate.py      # thin script: load → check → split → score → manifest
  tests/
    test_scorers.py            # smoke + reference-equivalence
    test_evaluate_smoke.py     # end-to-end on a tiny fixture
  evals/
    run_<timestamp>/
      results.json
      results_full.json
      manifest.json
  pyproject.toml     # pin eval-toolkit>=0.7.0,<0.8
```

The harness owns the orchestration; your project owns scorer
implementations, data loading, and the (small) script that wires them
together.

## Further reading

- [`docs/methodology/`](methodology/README.md) — concept-by-concept
  guide for what the Protocols actually operationalize.
- [`docs/examples/prompt_injection_walkthrough.md`](examples/prompt_injection_walkthrough.md)
  — end-to-end PI workflow on a synthetic fixture.
- [`docs/examples/pytorch_scorer_example.md`](examples/pytorch_scorer_example.md)
  — Transformer + LoRA Scorer adapter.
- The toolkit's own `tests/` — smoke tests for every reference impl,
  pattern templates for property tests in PR 1.5.
