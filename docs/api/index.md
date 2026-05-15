# API reference

This API reference is auto-generated from NumPy-style docstrings in
`src/eval_toolkit/`. It is organized by the README's
[three-tier architecture](../index.md#three-tier-architecture):

- **Tier 1 — Functional core**: pure metric / bootstrap / calibration
  primitives. Take numpy arrays in, return numpy arrays / floats /
  dataclasses out. No filesystem, no IO.
- **Tier 2 — Protocol-based orchestration**: composable building blocks
  (Scorer, Splitter, LeakageCheck, ThresholdSelector, DatasetLoader,
  EvidenceGate). The harness (`evaluate`) wires them together.
- **Tier 3 — Reproducibility scaffolding**: NeurIPS-aligned manifests,
  versioned JSON schemas, seed management.

Use the navigation to drill into any module. Below, the headline symbols
per tier with one-line summaries:

## Tier 1: Functional core

### Metrics ([`metrics`](metrics.md))

- `pr_auc(y, score)` — area under the precision-recall curve
- `roc_auc(y, score)` — area under the ROC curve
- `brier_score(y, score)` — strictly-proper scoring rule (mean
  squared error between probabilities and labels)
- `expected_calibration_error` (+ debiased / L2 / equal-mass variants)
- `headline_metrics(y, score)` — bundled
  `{pr_auc, roc_auc, brier, ece, n, n_positive}` for harness output
- `metrics_at_threshold(y, score, t)` — precision / recall / F1 at a
  fixed decision threshold

### Bootstrap & inference ([`bootstrap`](bootstrap.md))

- `bootstrap_ci(y, score, metric=...)` — 95% BCa or percentile CI on
  any metric
- `paired_bootstrap_diff(y, s_a, s_b, metric=...)` — significance test
  on the *difference* of two scorers (preserves within-sample
  correlation)
- `cv_clt_ci(fold_metrics)` — CLT-based CI on cross-validated point
  estimates
- `mde_from_ci(result, alpha, power)` — minimum detectable effect
- `delong_roc_variance(y, s_a, s_b)` — DeLong's nonparametric variance

### Calibration ([`calibration`](calibration.md))

- `fit_platt_calibrator(y, score)` — sigmoid scaling (Platt 1999)
- `fit_isotonic_calibrator(y, score)` — monotone non-parametric fit
- `fit_temperature(...)` — single-parameter temperature scaling
- `bayes_optimal_threshold(prior, fp_cost, fn_cost)` — analytic
  cost-optimal threshold

## Tier 2: Protocol-based orchestration

### Harness ([`harness`](harness.md))

- `evaluate(scorers, slices, run_id=...)` — slice-aware orchestrator;
  returns `RunResult`
- `evaluate_folded(splitter, scorers, slice_, ...)` — CV variant
- `EvalSlice` — DataFrame wrapper with configurable column names
- `RunResult` — JSON-serializable run container (schema-versioned)
- `write_run_result(result, run_dir)` — persist + schema-validate

### Splitters ([`splits`](splits.md))

- `Splitter` Protocol
- `StratifiedKFoldSplitter`, `GroupKFoldSplitter`,
  `SourceDisjointKFoldSplitter`, `TimeSeriesSplitter`,
  `HoldoutSplitter`, `PurgedKFoldSplitter` (with embargo)
- `compute_label_overlap(t_train, t_test, horizon)` — audit utility

### Leakage detection ([`leakage`](leakage.md))

- `LeakageCheck` Protocol
- `ExactDuplicateCheck`, `NormalizedFormLeakageCheck`,
  `NearDuplicateCheck`, `CrossSplitLeakageCheck`, `GroupLeakageCheck`,
  `LabelConflictCheck`, `TemporalLeakageCheck`
- `run_leakage_checks(checks, splits) -> LeakageReport`

### Threshold selection ([`thresholds`](thresholds.md))

- `ThresholdSelector` Protocol
- `MaxF1Selector`, `TargetRecallSelector`, `TargetPrecisionSelector`,
  `TargetFPRSelector`, `YoudenJSelector`, `CostSensitiveSelector`,
  `CISafeThresholdSelector`

### Loaders ([`loaders`](loaders.md))

- `DatasetLoader` Protocol
- `DataFrameLoader`, `ParquetGlobLoader`, `SingleSliceLoader`,
  `HFDatasetsLoader`

### Claims + evidence ([`claims`](claims.md), [`evidence`](evidence.md))

- `ClaimSpec` + `evaluate_claims(result, [claim])`
- Pre-built gates: `headline_present_gate`,
  `metric_threshold_gate`, `minimum_slice_size_gate`,
  `paired_diff_present_gate`, `no_leakage_errors_gate`,
  `no_scorer_errors_gate`, `required_scorer_gate`,
  `low_fpr_feasibility_gate`, `strict_artifact_gate`, ...
- `EvidenceAxis`, `AggregateEvidence` for typed aggregation

## Tier 3: Reproducibility scaffolding

### Manifest ([`manifest`](manifest.md))

- `RunManifest`, `build_manifest`, `write_manifest`
- NeurIPS Reproducibility Checklist-aligned (git_sha, seeds,
  code_versions, env, data_hashes, contamination_flags, etc.)
- v3 schema with `validate_manifest(payload)`

### Artifacts ([`artifacts`](artifacts.md))

- `validate_manifest`, `validate_results`,
  `validate_prediction_artifact_ref`
- `write_json_strict(path, payload)` — atomic write with NaN/Inf
  rejection
- `sanitize_for_json(obj)` — recursive cleanup of numpy types

### Seeds + provenance ([`seeds`](seeds.md), [`provenance`](provenance.md))

- `set_global_seeds(seed, strict_torch_determinism=...)`
- `capture_git_sha()`, `compute_file_hash(path)`, `make_run_dir(...)`

---

## Full module list

Click any module name for the auto-generated full API reference.
