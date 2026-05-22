# Getting Started

A linear walkthrough from "I have a trained model" to "I have a
`results.json` a stakeholder can read." Aimed at Python-fluent readers
new to eval-toolkit; no prior sklearn-eval experience assumed.

If you're already comfortable with sklearn-style evaluation, you can
skim the conceptual sections (marked **What is...**) and read the code
blocks directly.

## Table of contents

1. [What is an eval, and what does this toolkit do?](#what-is-an-eval)
2. [Install](#install)
3. [The Scorer concept](#scorer)
4. [The EvalSlice concept](#eval-slice)
5. [Run `evaluate()` and read the output](#evaluate)
6. [Persist results](#persist)
7. [(Optional) Validate the JSON](#validate)
8. [(Optional) Add a claim](#add-a-claim)
9. [(Optional) Render a plot](#plot)
10. [Common errors](#common-errors)
11. [Where to go next](#where-to-go-next)

(what-is-an-eval)=
## What is an eval, and what does this toolkit do?
**An evaluation** is the process of turning a model's predictions into
**calibrated metrics with uncertainty**. The numbers (PR-AUC, ROC-AUC,
precision-at-recall-X) are the surface. The *calibration* (does the
score 0.8 actually mean 80% chance of positive?) and the *uncertainty*
(is the +5 pp PR-AUC lift over baseline likely real or noise?) are the
substance.

This toolkit sits between two things you already have:

- **A model that produces probability scores.** Could be sklearn,
  PyTorch, an API call to a hosted model, a regex — anything that
  takes inputs and returns `P(positive)`.
- **Labeled data to evaluate it on.** Rows with a binary label and a
  text (or feature) column.

What you get back:

- Headline metrics (`pr_auc`, `roc_auc`, `brier_score`, ...)
- Bootstrap confidence intervals on those metrics
- Per-slice breakdowns (dev vs test, by source, by strata)
- Paired-difference CIs when comparing two models on the same rows
- A reproducible manifest (`git_sha`, seed, GPU info, dataset hashes)
- A `results.json` and `manifest.json` that downstream consumers can
  parse against a versioned JSON Schema

The toolkit does **not** ship report templates, dashboard renderers,
or claim copy — those are domain-specific and belong in your
consumer code.

(install)=
## Install
```bash
pip install eval-toolkit
```

Or with optional extras:

```bash
pip install "eval-toolkit[dataframe,plotting,validation]"
```

Common extras:

- `dataframe` — `pandas`. Required if you want to pass `pd.DataFrame`
  to `EvalSlice` (the easy path; this guide assumes it).
- `plotting` — `matplotlib` + `pillow`. Required for the `plot_*`
  helpers.
- `validation` — `jsonschema`. Required for `validate_payload(...)`.
- `property` — `hypothesis`. Only if you write property tests against
  the toolkit itself.
- `all` — everything optional, the kitchen-sink install.

This guide uses `dataframe` and `validation`. Plotting is optional
section [(9)](#plot).

(getting-started-scorer)=
## The Scorer concept
**A `Scorer`** is anything that exposes a `predict_proba(X)` method
returning one probability per input row, where `probability` ∈ [0, 1]
and represents `P(positive class)`.

That's the entire contract. It's deliberately Protocol-based: you
don't subclass anything, you just implement the method. Your model
class probably already does this (`sklearn` estimators do;
`transformers` pipelines do not natively but it's a one-liner wrapper).

### Example: a minimal Scorer

```python
import numpy as np


class LengthScorer:
    """Scores longer texts higher. Useful only as a demo Scorer."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        # Map length to a [0, 1] score via a saturating function.
        lengths = np.array([len(x) for x in X], dtype=float)
        return lengths / (lengths + 10.0)


scorer = LengthScorer()
probs = scorer.predict_proba(["hi", "hello world"])
assert probs.shape == (2,)
assert (0.0 <= probs).all() and (probs <= 1.0).all()
```

That's a fully valid `Scorer`. No registration, no base class.

If you have an sklearn pipeline:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# sklearn pipelines already implement predict_proba(X) → (n, n_classes).
# Wrap to return only the positive-class column.

class SklearnBinaryScorer:
    def __init__(self, pipe):
        self.pipe = pipe

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]


pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200)),
])
# (you'd fit pipe on training data here)
```

If your model is async / behind an API: cache the responses upfront,
then have `predict_proba` look up the cached scores. The toolkit
doesn't care.

(eval-slice)=
## The EvalSlice concept
**An `EvalSlice`** is *the unit of evaluation*: a named, labeled
subset of data that you want metrics computed on. You typically have
several:

- `dev` and `test` (the standard split)
- `by_source` (predictions on different data sources)
- `by_strata` (predictions on different label-balanced strata)
- OOD slices, regression slices, stress-test slices, etc.

Each slice is constructed from a pandas DataFrame with at minimum a
`text` and `label` column. `label` must be `{0, 1}`.

### Example: building two slices

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice

# Synthetic dev set: 100 rows, balanced classes.
rng = np.random.default_rng(42)
n = 100
labels = rng.integers(0, 2, size=n)
# Texts whose length correlates with the label.
texts = [
    "x" * (3 + int(label) * 8 + int(rng.integers(0, 4)))
    for label in labels
]
dev_df = pd.DataFrame({"text": texts, "label": labels})

dev_slice = EvalSlice(name="dev", df=dev_df)
assert dev_slice.name == "dev"
assert len(dev_slice.df) == 100
```

The constructor validates the shape: `text` and `label` columns must
exist, labels must be in `{0, 1}` (other label encodings raise a
`ValueError`), and the DataFrame must be non-empty.

If you have multiple sources to evaluate per-source:

```python
import pandas as pd
from eval_toolkit import EvalSlice

# Tag each row with its source, then build one slice per source.
df = pd.DataFrame({
    "text": ["a", "b", "c", "d", "e", "f"],
    "label": [0, 1, 0, 1, 0, 1],
    "source": ["A", "A", "B", "B", "C", "C"],
})

slices = [
    EvalSlice(name=f"source_{src}", df=sub.reset_index(drop=True))
    for src, sub in df.groupby("source")
]
assert len(slices) == 3
```

(evaluate)=
## Run `evaluate()` and read the output
`evaluate(...)` is the orchestrator. Given a mapping of scorers and a
list of slices, it computes the full headline-metric battery per
(slice, scorer) pair, runs bootstrap CIs, and returns a `RunResult`.

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, evaluate


class LengthScorer:
    def predict_proba(self, X):
        lengths = np.array([len(x) for x in X], dtype=float)
        return lengths / (lengths + 10.0)


rng = np.random.default_rng(0)
n = 100
labels = rng.integers(0, 2, size=n)
texts = ["x" * (3 + int(label) * 8) for label in labels]
df = pd.DataFrame({"text": texts, "label": labels})
dev_slice = EvalSlice(name="dev", df=df)

result = evaluate(
    {"length": LengthScorer()},
    [dev_slice],
    run_id="demo-run",
    n_resamples=50,  # small for the doctest; use 1000+ in real runs
    seed=42,
)

assert result.run_id == "demo-run"
assert "dev" in result.by_slice
```

### Reading the output

`result.by_slice` is a nested dict:

```
by_slice
├── "dev"
│   ├── "n"          : 100
│   ├── "n_positive" : ~50 (depends on RNG)
│   ├── "by_scorer"
│   │   └── "length"
│   │       ├── "pr_auc"           : float in [0, 1]
│   │       ├── "roc_auc"          : float in [0, 1]
│   │       ├── "pr_auc_ci"        : BootstrapCI dict (v0.48+ schema)
│   │       │   ├── "point"           : float
│   │       │   ├── "low"             : float   (or "skipped" if n<30)
│   │       │   ├── "high"            : float
│   │       │   ├── "confidence"      : 0.95
│   │       │   ├── "n_resamples"     : 50
│   │       │   └── "method"          : "BCa" | "percentile"
│   │       ├── "ece"              : float (expected calibration error)
│   │       └── ...                 (other metrics, plus operating_points)
│   └── "paired_diffs" : {}  (empty unless paired_diffs= explicitly set)
```

Access a metric:

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, evaluate


class _Scorer:
    def predict_proba(self, X):
        return np.array([len(x) / (len(x) + 10) for x in X])


# Bootstrap CIs require n >= 30; use a bigger slice than the toy 3-row.
rng = np.random.default_rng(0)
n = 40
labels = rng.integers(0, 2, size=n)
texts = ["x" * (3 + int(label) * 8) for label in labels]
df = pd.DataFrame({"text": texts, "label": labels})
result = evaluate({"m": _Scorer()}, [EvalSlice(name="dev", df=df)], run_id="r", n_resamples=20)

pr_auc = result.by_slice["dev"]["by_scorer"]["m"]["pr_auc"]
ci = result.by_slice["dev"]["by_scorer"]["m"]["pr_auc_ci"]
assert 0.0 <= pr_auc <= 1.0
# ci is a BootstrapCI dict — v0.48 schema has `point` + `low` + `high`
# (replaced pre-v0.48 `point_estimate` + `ci_95: [low, high]`). See
# migration/v0.48.md for the schema change rationale.
assert "low" in ci or ci.get("status") == "skipped"
```

### Comparing two scorers

When you want a paired-difference CI between two scorers on the same
rows, pass `paired_diffs=[(baseline, candidate)]` to `evaluate(...)`:

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, evaluate


class A:
    def predict_proba(self, X):
        return np.array([0.3 + 0.4 * (i % 2) for i in range(len(X))])


class B:
    def predict_proba(self, X):
        return np.array([0.4 + 0.5 * (i % 2) for i in range(len(X))])


df = pd.DataFrame({"text": ["x"] * 40, "label": [0, 1] * 20})
result = evaluate(
    {"a": A(), "b": B()},
    [EvalSlice(name="dev", df=df)],
    run_id="r",
    n_resamples=20,
    paired_diffs=[("a", "b")],  # explicit baseline → candidate pair
)
diffs = result.by_slice["dev"]["paired_diffs"]
assert ("a", "b") in diffs or "a__minus__b" in diffs or len(diffs) >= 1
```

(persist)=
## Persist results
`RunResult.to_dict()` produces a strict-JSON-safe payload:

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from eval_toolkit import EvalSlice, evaluate
from eval_toolkit.artifacts import write_json_strict


class _S:
    def predict_proba(self, X):
        return np.linspace(0.1, 0.9, len(X))


df = pd.DataFrame({"text": [f"row_{i}" for i in range(10)], "label": [0, 1] * 5})
result = evaluate({"m": _S()}, [EvalSlice(name="dev", df=df)], run_id="demo", n_resamples=10)

out_path = Path(tempfile.gettempdir()) / "demo_results.json"
write_json_strict(result.to_dict(), out_path)

# What the on-disk JSON looks like:
data = json.loads(out_path.read_text())
assert data["run_id"] == "demo"
assert "schema_version" in data
```

`write_json_strict` uses `allow_nan=False` and runs the payload
through `sanitize_for_json` first — NaN / Inf becomes a structured
`skipped_metric(...)` payload rather than producing invalid JSON.

(validate)=
## (Optional) Validate the JSON
Validate against the bundled JSON Schema to catch shape regressions
between your harness and consumer parsers:

```python
# Requires: pip install "eval-toolkit[validation]"
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from eval_toolkit import EvalSlice, evaluate
from eval_toolkit.artifacts import validate_payload, write_json_strict


class _S:
    def predict_proba(self, X):
        return np.linspace(0.1, 0.9, len(X))


df = pd.DataFrame({"text": [f"r{i}" for i in range(10)], "label": [0, 1] * 5})
result = evaluate({"m": _S()}, [EvalSlice(name="dev", df=df)], run_id="demo", n_resamples=10)

# This is a no-op on success; raises jsonschema.ValidationError on a bad shape.
validate_payload(result.to_dict(), schema_name="results.v1.json")
```

You can also validate from the CLI without writing Python:

```bash
eval-toolkit validate run_dir/results.json results.v1
```

(See [docs/schemas.md](schemas.md) for the field-by-field reference.)

(add-a-claim)=
## (Optional) Add a claim
**A claim** is a release-time go/no-go assertion: "PR-AUC is supported
on the dev slice with at least 100 positives and 100 negatives, and
the metric value is above 0.7." Claims are *not* exploratory metrics —
they're preregistered preconditions that the renderer reads to decide
whether to print "we claim X" or "we cannot claim X."

```python
import numpy as np
import pandas as pd

from eval_toolkit import EvalSlice, evaluate
from eval_toolkit.claims import (
    ClaimSpec,
    evaluate_claims,
    metric_threshold_gate,
    minimum_slice_size_gate,
    required_metric_gate,
)
from eval_toolkit.harness import with_claim_report


class _S:
    def predict_proba(self, X):
        # Score = 0.9 if label-marker, else 0.1
        return np.array([0.9 if "P" in x else 0.1 for x in X])


df = pd.DataFrame({
    "text": ["P_a", "P_b", "N_a", "N_b"] * 50,
    "label": [1, 1, 0, 0] * 50,
})
result = evaluate({"m": _S()}, [EvalSlice(name="dev", df=df)], run_id="demo", n_resamples=20)

claim = ClaimSpec(
    name="dev_pr_auc_supported",
    gates=(
        required_metric_gate("dev", "m", "pr_auc"),
        minimum_slice_size_gate("dev", min_n=100, min_positive=20, min_negative=20),
        metric_threshold_gate("dev", "m", "pr_auc", op=">=", threshold=0.7),
    ),
)

report = evaluate_claims(result, [claim])
assert report.has_failures() is False

# Attach the claim report to the RunResult for the renderer to read:
result_with_claim = with_claim_report(result, report)
assert result_with_claim.claim_report is not None
```

Each of the three gate calls above (`required_metric_gate`,
`minimum_slice_size_gate`, `metric_threshold_gate`) is a factory
that returns an `EvidenceGate` instance — a frozen dataclass bundling
a callable check, a name, and a severity. Custom gates are written by
constructing `EvidenceGate` directly with your own check function;
the [`claims_and_gates`](examples/claims_and_gates.md) example walks
through both reference and custom gates end-to-end.

See [methodology/claims.md](methodology/claims.md) for the full
contract — exception handling, severity policy, custom gates.

(plot)=
## (Optional) Render a plot
```python
# Requires: pip install "eval-toolkit[plotting]"
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / docs runs

import numpy as np
import tempfile
from pathlib import Path

from eval_toolkit.plotting import plot_metric_bars, save_figure

# Synthetic per-scorer metric summary:
values = {"baseline": 0.65, "candidate_v1": 0.78, "candidate_v2": 0.82}
fig = plot_metric_bars(values, ylabel="PR-AUC", title="Dev slice")
out_path = Path(tempfile.gettempdir()) / "pr_auc_bars.png"
saved = save_figure(fig, out_path)
assert saved.exists()
```

The plotting module's API and visual conventions are documented in
each helper's docstring. See `eval_toolkit.plotting.__all__` for the
full list (`plot_pr_curve`, `plot_reliability_diagram`,
`plot_confusion_matrix_grid`, `plot_score_histograms`, `plot_lift_ci`,
`plot_bootstrap_distribution`).

(common-errors)=
## Common errors
A handful of mistakes are statistically more likely than the rest when
you're starting out:

(error-labels)=
### `ValueError: labels must be in {0, 1}`
Your DataFrame has labels other than `0` / `1` — strings, booleans
encoded as integers, or `-1` sentinel values. eval-toolkit treats
binary classification as `{0, 1}` only.

Fix: convert before constructing the slice.

```python
import pandas as pd

raw = pd.DataFrame({"text": ["a", "b"], "label": ["pos", "neg"]})
raw["label"] = (raw["label"] == "pos").astype(int)
# Now raw["label"] is {0, 1}.
assert set(raw["label"]) <= {0, 1}
```

(error-strata)=
### `KeyError: missing strata column 'X'`
You passed `strata_col="X"` to `EvalSlice` but the DataFrame has no
column named `X`. Either remove the `strata_col=` argument or add the
column.

(error-wide-ci)=
### Bootstrap CIs are very wide
Either `n_resamples` is too low (default in this guide is 50 for
docs-speed; use **1000+** in real runs), or your slice has very few
positives or negatives. The CI width is a function of *both* the
resampling budget *and* the underlying sample size — adding more
resamples won't help if you only have 5 positives.

Check the slice composition:

```python
import pandas as pd
from eval_toolkit import EvalSlice

df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 0, 1]})
slc = EvalSlice(name="dev", df=df)
n_positive = int(slc.df["label"].sum())
n_negative = len(slc.df) - n_positive
assert n_positive >= 1 and n_negative >= 1  # else PR-AUC is undefined
```

(error-pr-curve)=
### `RuntimeError: PR curve has no thresholds`
Your `predict_proba` returned a constant value for every input. PR /
ROC curves are undefined for a single threshold. Fix: check that your
model isn't outputting the same score for every row.

(error-pandas)=
### `'TYPE_CHECKING' import error` for pandas
The `dataframe` extra (`pip install "eval-toolkit[dataframe]"`)
installs pandas. Without it, you can still use `EvalSlice` with
DataFrames — pandas is a soft dep — but `import pandas` will fail in
your harness code. Install the extra if you're using DataFrames at all
(this guide assumes you are).

(where-to-go-next)=
## Where to go next
You now have a working `RunResult` and `results.json`. Recommended
next reading depending on what you're doing:

- **Building a real eval pipeline.** Read three methodology chapters
  in this order:
  1. [`leakage.md`](methodology/leakage.md) — making sure your eval
     data isn't contaminated by training data.
  2. [`splits.md`](methodology/splits.md) — choosing between holdout
     and K-fold, source-disjoint splitting.
  3. [`thresholds.md`](methodology/thresholds.md) — picking a decision
     threshold once your scorer ranks well.

- **Adding release-time claims.** Read
  [`methodology/claims.md`](methodology/claims.md) for the full gate
  contract and severity policy.

- **Replaying old evals.** Read
  [`methodology/artifacts.md`](methodology/artifacts.md) for the
  `PredictionArtifactRef` contract that lets you recompute metrics
  without re-running inference.

- **Writing a custom Scorer/Splitter/Gate.** Read
  [`extending.md`](extending.md).

- **Migrating from an older version.** Read
  [`MIGRATION.md`](MIGRATION.md).

- **Browsing the JSON Schemas.** Read
  [`schemas.md`](schemas.md) for the field-by-field reference, or run
  `eval-toolkit schemas list` from the CLI.

The [methodology curriculum index](methodology/README.md) covers 16
chapters total — read them in order if you want the full conceptual
map.
