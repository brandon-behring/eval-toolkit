# Schema Reference

Field-by-field reference for the JSON Schemas bundled with eval-toolkit.
The schemas live at `src/eval_toolkit/schemas/` and ship inside the
installed wheel — they're discoverable programmatically and
validatable via the optional `[validation]` extra or the
`eval-toolkit` CLI.

> **Scope.** This doc describes *what's in each schema today*. For the
> *evolution policy* (when fields are added, when `.vN` bumps, how
> consumers should treat `additionalProperties: true`), read
> [methodology/versioning.md § schema-evolution](methodology/versioning.md#schema-evolution).
> For the *data model* underlying `prediction_artifacts` and
> `claim_report`, read [methodology/artifacts.md](methodology/artifacts.md)
> and [methodology/claims.md](methodology/claims.md).

## Inventory

Three schemas. All Draft 2020-12. All
`"additionalProperties": true` at the top level — old consumers
gracefully ignore new optional fields.

| Schema | Top-level shape | When produced | Consumers |
|---|---|---|---|
| `results.v1.json` | `RunResult` with per-row `scores` stripped | `eval_toolkit.harness.write_run_result(...)` writes `run_dir/results.json` | Report renderers, lightweight downstream parsers |
| `results_full.v1.json` | Same as above but `by_slice[*].by_scorer[*].scores` arrays retained | Same harness call, second file (`results_full.json`) | Replay pipelines, post-hoc bootstrap on the original scores |
| `manifest.v1.json` | `RunManifest` — env / seeds / hashes / source roles | `eval_toolkit.manifest.write_manifest(...)` writes `run_dir/manifest.json` | Reproducibility audits, NeurIPS-aligned report templates |

## `results.v1.json`

Required: `schema_version`, `run_id`, `config`, `by_slice`.

| Field | Type | Required | Since | Semantics |
|---|---|---|---|---|
| `schema_version` | `"v1"` (const) | yes | v0.5 | Stable version label. Bumps on breaking output changes. |
| `run_id` | string (min length 1) | yes | v0.5 | Caller-supplied unique identifier for the run. |
| `git_sha` | string \| null | optional | v0.5 | Captured via `capture_git_sha()`. `null` if not in a git repo or git is unavailable. |
| `config` | object | yes | v0.5 | Eval-time config: `n_resamples`, `seed`, scorers, slices, `paired_diffs`, `on_scorer_error`, optional `leakage_report`. Free-form within the toolkit's contract. |
| `by_slice` | object of slice-blocks | yes | v0.5 | Per-slice nested results. See below. |
| `by_fold` | object | optional | v0.7 | Per-fold raw `RunResult` payloads when `evaluate_folded(...)` is used. Empty for single-fold runs. |
| `fold_summary` | object | optional | v0.7 | CV-CI summary indexed by `[slice][scorer][metric]` → `{mean, ci_low, ci_high, n_folds}`. Empty for non-folded runs. |
| `claim_report` | object | optional | v0.9 | `ClaimReport.to_dict()` payload from `evaluate_claims(...)`. |
| `prediction_artifacts` | array of objects | optional | v0.9 | List of `PredictionArtifactRef.to_dict()` entries. |
| `evidence_axes` | array of objects | optional | v0.9 | Each entry `{name, value}` — see `EvidenceAxis`. |
| `pairing_metadata` | object | optional | v0.9 | `PairingMetadata.to_dict()` payload. |
| `aggregate_evidence` | object | optional | v0.9 | `{status: "inferential"/"descriptive"/"diagnostic"/"unsupported", ...}`. |
| `threshold_policy` | object | optional | v0.9 | `ThresholdPolicyMetadata.to_dict()` payload — calibration slice, score column, selector, constraints, claim-enabled flag. |

### Slice-block shape

Each `by_slice[<slice_name>]` is an object with the following
properties (no formal schema enforcement on inner shape — it's
documented here rather than typed in the JSON Schema because the
metric set varies by toolkit minor version):

- `n` (int) — total rows in the slice
- `n_positive` (int) — positive labels in the slice
- `by_scorer` (object) — `{scorer_name: per-scorer-block}`
- `paired_diffs` (object) — `{(baseline, candidate): diff-payload}`,
  populated only when `paired_diffs=[(a, b)]` was passed to `evaluate`

Each per-scorer block carries:

- `pr_auc`, `roc_auc`, `brier_score` (floats)
- `pr_auc_ci`, `roc_auc_ci`, `brier_score_ci` (`BootstrapCI` dicts —
  v0.48+ self-describing schema:
  `{point, low, high, confidence, n_resamples, method}`. The pre-v0.48
  `{point_estimate, ci_95: [low, high], ...}` shape was rewritten so
  the key names don't lie when `confidence != 0.95`; see
  `migration/v0.48.md`.)
- `ece`, `ece_equal_mass`, `ece_equal_width`, `ece_equal_mass_error`
  (calibration errors)
- `precision_at_prior` (float)
- `operating_points` (object) — selector-keyed threshold metadata
- `transferred_operating_points` (object, optional) — populated when
  `OperatingPointSpec` was used
- `scores` — *stripped* from `results.v1`, retained in
  `results_full.v1`
- `is_single_class` (bool) — true when the slice has only one label
  class (PR-AUC undefined; metrics may be skipped)

## `results_full.v1.json`

Same top-level shape as `results.v1.json`. The only difference is
that `by_slice[*].by_scorer[*].scores` arrays are retained, enabling
re-computation of metrics or bootstrap CIs from the raw scores
without re-running inference.

Use this variant when you want full replay capability. Use the
compact variant when you only need the headline metrics.

## `manifest.v1.json`

Required: `schema_version`, `run_id`, `code_versions`, `env`.

| Field | Type | Required | Since | Semantics |
|---|---|---|---|---|
| `schema_version` | `"v1"` (const) | yes | v0.5 | Stable version. |
| `run_id` | string | yes | v0.5 | Matches `results.run_id`. |
| `git_sha` | string \| null | optional | v0.5 | Captured at run time. |
| `dirty_flag` | bool | optional | v0.5 | True iff `git status --porcelain` had output. NeurIPS clean-replay concern. |
| `code_versions` | `{package: version}` | yes | v0.5 | At minimum `{eval_toolkit: "x.y.z"}`. Use `Versioned`-implementing objects to enrich. |
| `seeds` | `{source: seed}` | optional | v0.5 | Global / bootstrap / torch / dataloader RNG seeds. |
| `data_hashes` | `{name: "sha256:..."}` | optional | v0.5 | SHA-256 of each input artifact (CSV, JSONL, Parquet, ...). |
| `config_hash` | string | optional | v0.5 | SHA-256 over canonical-JSON encoding of the eval config. |
| `env` | object | yes | v0.5 | Environment fingerprint: `python`, `platform`, key dep versions. |
| `gpu_info` | object | optional | v0.5 | `{name, count, memory_gb}` from `nvidia-smi`. Empty when unavailable. |
| `cuda_version` | string \| null | optional | v0.5 | Reported by `nvidia-smi`. |
| `wall_clock_seconds` | number \| null | optional | v0.5 | Optional run duration. |
| `versioned_objects` | `{name: version}` | optional | v0.7 | Auto-collected from Tier-2 implementations exposing `version` (see methodology/versioning.md). |
| `leakage_report` | object \| null | optional | v0.7 | `LeakageReport.to_dict()` payload from `run_leakage_checks(...)`. |
| `source_roles` | array of `SourceRoleRecord` dicts | optional | v0.8 | Each entry: `{source, role, n_rows?, notes?, metadata?}`. |
| `guardrails` | array of strings | optional | v0.8 | Predeclared free-form guardrails (e.g., `["no threshold tuning on locked_holdout"]`). |
| `prediction_artifacts` | array of `PredictionArtifactRef` dicts | optional | v0.9 | `{uri, media_type, columns, sha256?, n_rows?, role?, metadata?}` — see `methodology/artifacts.md`. |

### `leakage_report` shape

When present, conforms to:

```
leakage_report.findings: [
  {
    "check_name": "exact_duplicate" | "near_duplicate" | ...,
    "severity": "error" | "warning" | "info",
    "drop_indices": object,
    "evidence": object,
    "message": string,
    "n_affected": int (≥ 0)
  },
  ...
]
```

### `source_roles` shape

```
source_roles: [
  {
    "source": "main_train",
    "role": "train",
    "n_rows": 50000,
    "notes": "v0.8 train pool",
    "metadata": {"data_card_url": "..."}
  },
  ...
]
```

Recommended roles: `train`, `calibration`, `locked_eval`,
`external_diagnostic`, `excluded`. Custom roles are allowed
(`additionalProperties: true` on each entry).

## Versioning + forward compatibility

All three schemas are at `.v1`. The filename includes the version
suffix (`results.v1.json`), and the `$id` URL encodes it too. A
future incompatible change ships at `.v2` (`results.v2.json`) and
**leaves `.v1` in the wheel** so both versions remain validatable
side by side.

Within a major schema version, the rule is **additive only**:

- New optional fields can appear at any time.
- `"additionalProperties": true` is set at the top level (and on
  nested object schemas where reasonable), so old consumers see new
  fields as inert.
- Removing a field, renaming a field, narrowing an enum, tightening
  a `required` array, or changing a type are *all* breaking changes
  that trigger a `.vN+1` bump.

See [methodology/versioning.md § schema-evolution](methodology/versioning.md#schema-evolution)
for the full policy and concrete v0.7→v0.9 case studies.

## Programmatic discovery

```python
from pathlib import Path
import eval_toolkit


schemas_dir = Path(eval_toolkit.__file__).parent / "schemas"
names = sorted(p.stem for p in schemas_dir.glob("*.json"))
assert "results.v1" in names
assert "manifest.v1" in names
```

Or from the CLI (v0.10.0+):

```bash
eval-toolkit schemas list
# → manifest.v1
# → results.v1
# → results_full.v1

eval-toolkit schemas show results.v1
# Pretty-prints the schema as JSON
```

## Validation example

```python
# Requires: pip install "eval-toolkit[validation]"
from eval_toolkit.artifacts import validate_payload

payload = {
    "schema_version": "v1",
    "run_id": "demo",
    "config": {"n_resamples": 100},
    "by_slice": {
        "dev": {
            "n": 100,
            "n_positive": 50,
            "by_scorer": {"model": {"pr_auc": 0.82}},
        }
    },
}

# No-op on success; raises jsonschema.ValidationError on a bad shape.
validate_payload(payload, schema_name="results.v1.json")
```

Or from the CLI:

```bash
eval-toolkit validate run_dir/results.json results.v1
# → run_dir/results.json: OK against results.v1
```

Exit codes:

- `0` — valid
- `1` — schema validation failed
- `2` — file or schema not found
- `3` — `[validation]` extra not installed

## See also

- [methodology/versioning.md § schema-evolution](methodology/versioning.md#schema-evolution)
  — the policy that governs when fields are added vs when `.vN` bumps.
- [methodology/artifacts.md](methodology/artifacts.md) — the
  `PredictionArtifactRef` / `PredictionColumns` contract that the
  `prediction_artifacts` field references.
- [methodology/claims.md](methodology/claims.md) — the `ClaimReport`
  contract that the `claim_report` field references.
- [getting-started.md § Validate the JSON](getting-started.md#validate)
  — first-touch validation workflow.
