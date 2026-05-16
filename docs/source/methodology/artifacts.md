# Prediction Artifacts and Metric States

This chapter covers `eval_toolkit.artifacts` — the layer that lets you
**separate predictions from the metrics computed on them**. The toolkit
exposes three contracts: `PredictionArtifactRef` (manifest pointer to
on-disk predictions), `PredictionColumns` (column-mapping schema), and
`MetricState` (typed status wrapper for metrics that may be skipped or
errored). Together they answer a single question: *can a downstream
consumer replay this evaluation without re-running inference?*

> **Background.** This chapter assumes you've produced a `RunResult`
> (see [getting-started](../getting-started.md)) and have either a CSV,
> JSONL, or Parquet file of per-row predictions. The
> [methodology curriculum](README.md) covers the metric kernels
> ([API: metrics](../api/metrics.md)), bootstrap CI shapes
> ([bootstrap.md](bootstrap.md)), and the broader claim model
> ([claims.md](claims.md)). What's *not* covered elsewhere is the
> data contract that ties them together: how the bytes of a
> `predictions.csv` connect to a typed reference in
> `manifest.json` and the optional schema-validated payload that
> ships next to it.

(why)=
## Why a separate artifact layer?
Computing metrics from predictions in-memory is cheap. The problem
appears later, when a stakeholder asks:

- "What were the actual scores for the rows that landed in the FPR-low
  region of the curve?"
- "Did the v0.8 release use the same eval predictions as the v0.9
  release, or did the scorer drift?"
- "Can we re-compute PR-AUC under a new bootstrap method without
  re-running inference on 50K rows?"

The fix is to **persist predictions as a first-class artifact** with a
stable on-disk contract. That artifact then carries enough metadata
(`uri`, `media_type`, `columns`, `sha256`, `n_rows`) for any future
analysis pipeline to load, validate, and recompute against it.

(data-model)=
## The data model
Three frozen dataclasses make up the public contract:

(prediction-columns)=
### PredictionColumns
The column-mapping schema. Required: `label` and `score`. Optional but
strongly recommended for paired-diff workflows: `row_id` and
`content_hash`.

```python
from eval_toolkit.artifacts import PredictionColumns

columns = PredictionColumns(
    label="y_true",
    score="prob_positive",
    row_id="example_id",
    content_hash="text_sha256",
)
assert columns.label == "y_true"
```

`row_id` lets downstream code align predictions across two runs of
the same scorer for paired bootstrap diffs. `content_hash` detects
silent label-drift or text-mutation: same `row_id` + different
`content_hash` means the row is *not* the same row.

(artifact-ref)=
### PredictionArtifactRef
The manifest-level pointer to a persisted prediction file. Carries the
URI, MIME-style media type, the column mapping, plus integrity
metadata (`sha256`, `n_rows`).

```python
from eval_toolkit.artifacts import PredictionArtifactRef, PredictionColumns

ref = PredictionArtifactRef(
    uri="s3://my-eval-bucket/run-42/predictions.csv",
    media_type="text/csv",
    columns=PredictionColumns(label="y", score="s"),
    sha256="a" * 64,
    n_rows=10_000,
    role="locked_eval_predictions",
)
serialized = ref.to_dict()
assert serialized["uri"].startswith("s3://")
```

The `role` field is free-form — useful when a manifest carries
predictions for multiple slices or scorers and the renderer needs to
distinguish them.

(metric-state)=
### MetricState
A typed wrapper for metrics that may not be computable: too few rows,
single-class slice, NaN inputs, downstream tool exception. The status
column is `"ok"`, `"skipped"`, or `"error"`.

```python
from eval_toolkit.artifacts import MetricState, error_metric, skipped_metric

# Successful metric:
ok = MetricState(value=0.82, status="ok").to_dict()
assert ok["value"] == 0.82

# A slice that's too small to bootstrap:
skipped = skipped_metric("n_resamples below floor", n=5)
assert skipped["status"] == "skipped"

# A metric whose computation hit a divide-by-zero:
errored = error_metric("ZeroDivisionError", numerator=0, denominator=0)
assert errored["status"] == "error"
```

Consumers reading a metric should pattern-match on `status` before
trusting `value`. The point is to make the *absence* of a number an
explicit, JSON-typed state instead of a `null` that could mean a
dozen different things.

(worked-walkthrough)=
## Worked walkthrough
Producing predictions, registering them as an artifact, computing a
bootstrap CI from the artifact, and computing a paired diff against a
second artifact — all without re-running inference.

```python
import csv
import numpy as np
import tempfile
from pathlib import Path

from eval_toolkit.analysis import (
    bootstrap_metric_from_predictions,
    load_prediction_arrays,
    paired_diff_from_prediction_refs,
)
from eval_toolkit.artifacts import PredictionArtifactRef, PredictionColumns

tmpdir = Path(tempfile.mkdtemp())

# Synthetic predictions for two scorers on the same dev rows.
rng = np.random.default_rng(0)
n = 200
labels = rng.integers(0, 2, size=n)
baseline_scores = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.15, n), 0, 1)
candidate_scores = np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.10, n), 0, 1)


def _write_csv(path: Path, scores: np.ndarray) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["label", "score", "row_id", "content_hash"]
        )
        writer.writeheader()
        for i, (y, s) in enumerate(zip(labels, scores, strict=True)):
            writer.writerow(
                {"label": int(y), "score": float(s), "row_id": f"r{i}", "content_hash": f"h{i}"}
            )


baseline_path = tmpdir / "baseline_preds.csv"
candidate_path = tmpdir / "candidate_preds.csv"
_write_csv(baseline_path, baseline_scores)
_write_csv(candidate_path, candidate_scores)

columns = PredictionColumns(
    label="label", score="score", row_id="row_id", content_hash="content_hash"
)
baseline_ref = PredictionArtifactRef(
    uri=str(baseline_path),
    media_type="text/csv",
    columns=columns,
)
candidate_ref = PredictionArtifactRef(
    uri=str(candidate_path),
    media_type="text/csv",
    columns=columns,
)

# Re-load the predictions later (could be a separate process):
arrays = load_prediction_arrays(baseline_ref.to_dict())
assert arrays.labels.shape == (n,)
assert arrays.scores.shape == (n,)

# Bootstrap PR-AUC CI directly from the on-disk artifact:
ci = bootstrap_metric_from_predictions(baseline_ref.to_dict(), n_resamples=50, seed=1)
assert ci["n_resamples"] == 50
low, high = ci["ci_95"]
assert low <= high

# Paired diff: candidate − baseline (rows align by row_id + content_hash):
diff = paired_diff_from_prediction_refs(
    baseline_ref.to_dict(),
    candidate_ref.to_dict(),
    n_resamples=50,
    seed=1,
)
assert diff["n_resamples"] == 50
```

The diff helper enforces matching `row_id` and `content_hash` between
the two artifacts. Mismatches raise `ValueError` rather than silently
producing a meaningless diff.

(metric-state-vs-error)=
## MetricState vs error_metric helpers
Three flavors of metric reporting, in order of escalation:

1. **`MetricState(value=x, status="ok")`** — the metric was computed.
   Use this directly when you control the metric pipeline.

2. **`skipped_metric(reason, **details)`** — the metric was *not*
   computed for a stated reason. Examples: single-class slice (`PR-AUC
   undefined`), too few rows for bootstrap, missing positives at a
   target FPR. The metric was *expected* to be unavailable.

3. **`error_metric(reason, **details)`** — the metric raised an
   unexpected exception. The metric was *expected* to be computable
   but wasn't. The renderer should highlight these as bugs to
   investigate.

`skipped` is the normal "principled absence." `error` is the
"something is wrong here" state. Don't conflate them — a renderer that
treats all non-`ok` states identically loses important signal.

(validation)=
## The `validation` extra and validate_payload
`validate_payload` runs a payload against a bundled JSON Schema using
the optional `jsonschema` dependency. The dependency is opt-in so the
toolkit's core deps stay at numpy / scipy / sklearn.

```python
# Install: pip install "eval-toolkit[validation]"
from eval_toolkit.artifacts import validate_payload

result_payload = {
    "run_id": "demo",
    "schema_version": "v1",
    "config": {},
    "by_slice": {
        "dev": {"n": 100, "n_positive": 50, "by_scorer": {"model": {"pr_auc": 0.8}}}
    },
}
validate_payload(result_payload, schema_name="results.v1.json")  # raises on bad shape
```

The schemas live at `eval_toolkit/schemas/*.json` and follow the
`additive-fields, additionalProperties: true` policy documented in
[versioning.md § schema-evolution](versioning.md#schema-evolution): old
consumers see new fields as inert; new consumers see old payloads as
valid.

(pitfalls)=
## Pitfalls / Common mistakes
**Do include `content_hash` when running paired diffs.** Without it,
`paired_diff_from_prediction_refs` can only check `row_id` equality —
which means a row whose underlying text changed silently between runs
will compare as the same row. The diff's variance estimate is then
meaningless. The fix: hash the input text and persist that hash with
the prediction.

**Don't build artifact dicts inline.** Construct `PredictionColumns`
and `PredictionArtifactRef` instances and serialize via `.to_dict()`.
Hand-written `{"uri": ..., "media_type": ...}` dicts skip the
`__post_init__` validation that catches typos (`"row_ids"` vs
`"row_id"`) and bad shapes (`n_rows` as a `bool`).

**Don't treat `media_type` as decorative.** It routes the prediction
reader selection in `load_prediction_arrays`. The two media types the
toolkit recognizes out of the box are `text/csv` and
`application/jsonl`; other types raise a clear "no built-in reader"
ValueError. Setting `media_type` correctly is the difference between
a working pipeline and a debugging session.

**Don't silently swallow non-finite metrics.** `sanitize_for_json`
converts NaN / +Inf / -Inf to a `skipped_metric(...)` payload so
`json.dumps(allow_nan=False)` succeeds. The downstream cost: a renderer
that doesn't look at `status` thinks the metric is missing. Always
pattern-match on `status` first.

**Don't rely on the `validation` extra in production code paths.**
`validate_payload` raises `ImportError` if `jsonschema` is not
installed. Catch that explicitly in any consumer code that wants to
degrade gracefully when running in environments where the extra isn't
present (e.g., a lightweight inference-only image).

## See also

- [claims.md](claims.md) — the gate layer that reads from these
  artifacts to produce go/no-go verdicts.
- [evidence.md](evidence.md) — broader source-role and claim-mode
  framing.
- [versioning.md § schema-evolution](versioning.md#schema-evolution) —
  the additive-fields contract for the bundled `.v1.json` schemas.
- `eval_toolkit.analysis` module docstring — implementation details
  for the CSV / JSONL prediction readers and the
  `bootstrap_metric_from_predictions` / `paired_diff_from_prediction_refs`
  helpers.
