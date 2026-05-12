---
template: classification_project_worksheet
variant: long
version: 1.0
eval_toolkit_minimum_version: 0.9.0
filled: false
project_name: ""
---

# Classification Project Spec — Long Variant

> **Fill instructions**:
> - Replace `___` with your value. Check `[ ]` boxes for choices made.
> - `[OPTIONAL]` items are advanced or context-dependent; skip if not applicable.
> - The YAML block at the bottom is the parseable source of truth (mirrors `config/baseline.yaml`); checklists above are for thinking through the decision.
> - Save the filled copy as `<your-project>/spec.md` and set `filled: true` in the frontmatter above.
> - Cross-references: `eval-toolkit/src/eval_toolkit/__init__.py` is the source of truth for public symbols; `eval-toolkit/CHANGELOG.md` documents version-by-version feature provenance.

---

## 1. Project identity

- **Project name**: ___
- **Slug** (kebab-case, used for directory + package names): ___
- **One-liner** (≤140 chars): ___
- **Domain / motivation** (3–5 sentences): ___
- **Status**: [ ] design  [ ] implementation  [ ] eval  [ ] locked  [ ] deprecated
- **Owner(s)**: ___
- **Target completion date**: ___
- **Stakeholders / consumers**: ___

> **Why this project exists** (1 paragraph): ___

## 2. Problem framing

- **Class 0 (negative)** = ___
- **Class 1 (positive)** = ___
- **Class balance**:
  - [ ] balanced  [ ] moderate (1:5–1:20)  [ ] severe (>1:20)  [ ] unknown
  - Measured prior P(y=1): ___
- **Hypotheses** (each ≤1 sentence, drives experimental design):
  - H1 (primary, ties to stopping rule): ___
  - H2 (secondary): ___
  - H3 (interpretive grid, e.g., "fine-tuning gains over frozen pretrained ≥ X"): ___
- **Stopping rule** (declared up front, not post-hoc):
  ___
  > Default per ADR D4: "stop iterating when paired-bootstrap 95% CI on Δ-primary-metric overlaps zero." Other valid forms: "headline metric ≥ 0.X with CI half-width < 0.0Y", or "fixed iteration budget of N runs".
- **Out-of-scope (anti-scope)** — explicitly NOT covered by v0: ___

> **Why these hypotheses + stopping rule?** ___

## 3. Data

### 3.1 Schema (maps to `EvalSlice` from `eval_toolkit.harness`)
- `feature_col`: ___
- `label_col`: ___
- `strata_col` (optional, drives `stratified_recall`): ___

### 3.2 Sources

One row per source (training, validation, diagnostics, final holdout,
excluded sources):

| Source | Source role (`train` / `validation` / `external_diagnostic` / `locked_final_holdout` / `excluded`) | License | Rows | Class balance | URL |
|---|---|---|---|---|---|
| ___ | ___ | ___ | ___ | ___ | ___ |
| ___ | ___ | ___ | ___ | ___ | ___ |
| ___ | ___ | ___ | ___ | ___ | ___ |

### 3.3 Loaders (`eval_toolkit.loaders`)
- [ ] `DataFrameLoader` (in-memory pandas)
- [ ] `ParquetGlobLoader`
- [ ] `HFDatasetsLoader` (HuggingFace datasets)
- [ ] `SingleSliceLoader`
- [OPTIONAL] Custom `DatasetLoader` subclass: ___

### 3.4 Splits (`eval_toolkit.splits`)
- [ ] `HoldoutSplitter` — single train/val/test
- [ ] `StratifiedKFoldSplitter` — preserves class balance
- [ ] `SourceDisjointKFoldSplitter` — recommended (see ADR D6)
- [ ] `GroupKFoldSplitter` — group-aware
- [ ] `TimeSeriesSplitter` — temporal

Configuration:
- `k = ___`
- `seeds = [___]` (≥3 for variance estimation per T2)
- Train fraction: ___, Val fraction: ___, Test fraction: ___

### 3.5 Dedup (`near_dedup` from `eval_toolkit.text_dedup`)

Choose one strategy:
- [ ] `ExactNormalizedHashStrategy` — exact-paraphrase via SHA256 + normalization
- [ ] `JaccardNgramStrategy` — set-based n-gram (token-order invariant)
- [ ] `TfidfCosineStrategy` — lexical near-dedup (default)
- [ ] `EmbeddingCosineStrategy` — semantic; caller supplies embedder
- [ ] `MinHashLSHStrategy` — production-scale (Broder 1997 / Indyk-Motwani 1998)

- Threshold: `___` (default 0.9 — see ADR D7)
- [OPTIONAL] Label-aware (preserve cross-label adversarial pairs, remove within-label duplicates)
- [ ] Run `cross_dedup` between splits to scrub leakage

### 3.6 Leakage gates (`run_leakage_checks` from `eval_toolkit.leakage`)
- [ ] `ExactDuplicateCheck`
- [ ] `NearDuplicateCheck`
- [ ] `NormalizedFormLeakageCheck`
- [ ] `LabelConflictCheck`
- [ ] `CrossSplitLeakageCheck`
- [OPTIONAL] [ ] `GroupLeakageCheck`
- [OPTIONAL] [ ] `TemporalLeakageCheck`
- Report stored in `LeakageReport`; failures block training

### 3.7 OOD probes (held out entirely)

For each, list source + rationale + expected challenge:

- Probe 1: ___
- Probe 2: ___
- Probe 3: ___

### 3.8 Manifest (`eval_toolkit.manifest`)
- [ ] Build with `build_manifest`, store as `RunManifest`, write with `write_manifest`
- [ ] Include `SourceRoleRecord` entries and `guardrails` for claim discipline
- Captured fields: per-source row count, label distribution, split paths, dedup threshold, content SHA256

> **Why this split scheme?** ___
> **Why this dedup strategy?** ___

## 4. Scorer ladder

Anti-overengineering: must include at least one heuristic or classical baseline before any neural model. Each scorer implements the `Scorer` Protocol from `eval_toolkit.harness` (or `SliceAwareScorer` for slice-conditional scoring).

| Rung | Name | Architecture | Training config | Source | Hyperparams locked? |
|---|---|---|---|---|---|
| Heuristic | ___ | ___ | ___ | ___ | [ ] |
| Classical | ___ | ___ | ___ | ___ | [ ] |
| Pretrained-frozen | ___ | ___ | ___ | ___ | [ ] |
| Public reference | ___ | ___ | ___ | ___ | [ ] |
| Fine-tuned candidate | ___ | ___ | ___ | ___ | [ ] |

> **Why each rung?** ___

## 5. Metrics

### 5.1 Primary metric (exactly one)
- [ ] `pr_auc` (recommended for rare positive — see ADR D1)
- [ ] `roc_auc` (balanced classes, ranking quality)
- [ ] F1 @ chosen threshold
- [ ] `precision_at_prior` (prior-shift correction)
- [ ] `quantile_stratified_pr_auc` (per-quantile diagnostics)
- [ ] Custom: ___

> **Why this primary metric?** ___

### 5.2 Headline bundle
- [ ] `headline_metrics` — returns PR-AUC + ROC-AUC + ECE in one call

### 5.3 Operating points (`select_threshold` + selector class from `eval_toolkit.thresholds`)
- [ ] `MaxF1Selector`
- [ ] `TargetRecallSelector(value=___)`
- [ ] `TargetPrecisionSelector(value=___)`
- [ ] `TargetFPRSelector(value=___)`
- [ ] `YoudenJSelector`
- [OPTIONAL] [ ] `CostSensitiveSelector` with `CostMatrix(prior=___, fp_cost=___, fn_cost=___)` (Elkan 2001)
- Per-threshold metrics via `metrics_at_threshold`
- Single-class slice safety via `single_class_threshold_metrics`
- Threshold transfer via `OperatingPointSpec`: fit on `___`, apply to `___`

### 5.4 Calibration error
- [ ] `expected_calibration_error` — plug-in L1 ECE
- [ ] `expected_calibration_error_debiased` — Monte-Carlo debiased L1 (see ADR D3)
- [ ] `expected_calibration_error_l2` — equal-mass L2 (RMSE)
- [ ] `expected_calibration_error_l2_debiased` — Kumar 2019 closed-form
- [ ] `expected_calibration_error_equal_mass` — equal-count binning
- `n_bins=___`, `n_sweep=___` (for debiased), binning strategy: ___

### 5.5 Decomposition diagnostics
- [OPTIONAL] [ ] `brier_score`
- [OPTIONAL] [ ] `brier_decomposition` — reliability/resolution/uncertainty (Murphy 1973)

### 5.6 Stratified analysis
- [ ] `stratified_recall(strata_col=___, with_ci=True, confidence=0.95)` — Wilson CI per stratum
- [ ] `quantile_stratified_pr_auc(n_quantiles=___)` — PR-AUC per score quantile
- [ ] `score_distribution_summary` — deciles/percentiles

> **Why these metrics?** ___

## 6. Statistical inference

### 6.1 Single-condition CI
`bootstrap_ci(method=___, n_resamples=___, confidence=___)`
- Method: [ ] BCa (default — see ADR D2)  [ ] percentile  [ ] studentized (Algeshiemer 2024)
- `n_resamples=1000` (default)
- `confidence=0.95` (default)
- Returns `BootstrapCI` (point_estimate, ci_low, ci_high)

### 6.2 Paired contrasts

| Baseline | Alternative | Metric | Function | Returns |
|---|---|---|---|---|
| ___ | ___ | PR-AUC | `paired_bootstrap_diff` | `PairedBootstrapCI` |
| ___ | ___ | ECE | `paired_bootstrap_ece_diff` | `PairedBootstrapCI` |
| ___ | ___ | Op-point F1 | `paired_bootstrap_op_point_diff` | `PairedBootstrapCI` |

### 6.3 Power / MDE
- [ ] `mde_from_ci` after pilot to size full eval N (returns `MDEEstimate`)
- [ ] `paired_mde` for paired-contrast power

### 6.4 Cross-validation CI
- [ ] `cross_validate_metric(metric=___, k=___, stratified=True, seed=___)` to produce per-fold metric array
- [OPTIONAL] [ ] `cv_clt_ci(fold_metrics)` — Bayle 2020 Theorem 3.1 CI on fold-level mean

### 6.5 Claim gates
- [ ] `evaluate_claims` over the result payload and optional manifest
- [ ] `headline_present_gate` for required headline comparisons
- [ ] `minimum_slice_size_gate` for total / positive / negative evidence count
- [ ] `low_fpr_feasibility_gate` for low-FPR claims before observed FPR is interpreted
- [ ] `metric_threshold_gate` for hard-negative FPR or other domain-neutral caps
- [ ] `no_scorer_errors_gate` and `no_leakage_errors_gate`
- [ ] Attach the resulting `ClaimReport` with `with_claim_report` before `write_run_result`

> **Why this inference setup?** ___

## 7. Calibration & decision

### 7.1 Calibration fit (on val set)
- [ ] None (use raw probabilities)
- [ ] `fit_temperature` — Guo 2017 single-parameter scaling on logits
- [ ] `fit_isotonic_calibrator` — non-parametric (Niculescu-Mizil 2005)
- [ ] `fit_platt_calibrator` — canonical Platt 1999
- [ ] `fit_beta_calibrator` — Kull 2017 3-parameter Beta
- [OPTIONAL] [ ] `fit_temperature_oracle` — diagnostic upper bound only (warns if used outside diagnostics)

### 7.2 Reliability diagnostics
- [ ] `reliability_curve(n_bins=___, strategy=___)` — bin-level data for plotting

### 7.3 Decision threshold
- See Section 5.3 selectors
- [OPTIONAL] [ ] Cost-sensitive: `bayes_optimal_threshold(score, prior=___, fp_cost=___, fn_cost=___)` per Elkan 2001
- [OPTIONAL] `CostMatrix.expected_cost(...)` to evaluate cost-sensitive deployments

> **Why this calibration method?** ___

## 8. Plotting & output

### 8.1 Figures (`eval_toolkit.plotting`)
- [ ] `plot_pr_curve`
- [ ] `plot_reliability_diagram`
- [ ] `plot_confusion_matrix_grid`
- [ ] `plot_metric_bars` (per slice)
- [ ] `plot_score_histograms`
- [ ] `plot_lift_ci`
- [ ] `plot_bootstrap_distribution` (CI shape diagnostics)

### 8.2 Style
- Palette: `make_palette(negative=___, positive=___, accent=___, baseline=___)` (or accept defaults from `PALETTE`)
- [OPTIONAL] Extra semantic roles via `**extras`: `make_palette(..., benign=___, injection=___, emphasis=___)` accepts arbitrary kwargs
- [OPTIONAL] [ ] Override default `PLOT_STYLE` via `set_plot_style(...)`
- Default `figsize`: `DEFAULT_FIGSIZE` ((10, 6))

### 8.3 Output
- `save_figure(fig, path, dpi=___, provenance={...}, permitted_suffixes={___})`
- `permitted_suffixes` (default `{".png", ".pdf", ".svg"}`): ___
- [OPTIONAL] `skip_env_var=___` (default `EVAL_TOOLKIT_SKIP_SAVEFIG`) — override with project-specific name
- [ ] Provenance dict via `figure_metadata` includes git SHA, dataset SHA256, run ID

## 9. Reproducibility & provenance

- [ ] `set_global_seeds(seed=___)` at run start
- [OPTIONAL] [ ] `set_global_seeds(seed=___, strict_torch_determinism=True)` for bit-exact torch
- [ ] `capture_git_sha()` recorded in run output (require non-None: ___)
- [ ] `file_sha256(path, strict=True)` on:
  - [ ] data manifest
  - [ ] config file
  - [ ] model weights
  - [ ] split files
- [ ] Run directory via `make_run_dir(base="evals/", prefix="run")`
- [ ] Data manifest via `RunManifest` + `build_manifest` + `write_manifest`
- [ ] Manifest source roles and guardrails via `SourceRoleRecord`
- [ ] Config via `@frozen_config` dataclass loaded with `from_yaml(path, cls)`
- [ ] `figure_metadata(provenance=..., dpi=...)` injected into all saved figures

## 10. Threat audit

| # | Threat | Mitigation | Severity (critical / standard / note) |
|---|---|---|---|
| T1 | Test-set statistical power | `mde_from_ci` to size N before locking | ___ |
| T2 | Seed variance | ≥3 seeds per fold; report seed-aware CIs | ___ |
| T3 | OOD generalization | Held-out OOD probes (Section 3.7) | ___ |
| T4 | Calibration drift | ECE on each slice; recalibrate if needed | ___ |
| T5 | Train/test contamination | `cross_dedup` + `CrossSplitLeakageCheck` | ___ |
| T6 | Operating-point sensitivity | Report ≥3 thresholds; PR-AUC is threshold-free | ___ |
| T7 | Language / domain coverage | Document scope; flag English-only etc. | ___ |
| T8 | Label noise | Inter-annotator agreement; spot audit | ___ |
| T9 (custom) | ___ | ___ | ___ |

## 11. Deliverables / artifact map

- [ ] `spec.md` (this filled worksheet)
- [ ] `DECISIONS.md` (Section 12 ADRs + new ADRs added during build)
- [ ] `assumptions.md` (unverified-assumption register tagged by severity)
- [ ] `config/baseline.yaml` (seeded by YAML block below; loaded with `from_yaml`)
- [ ] `data/manifest.json` (via `RunManifest` + `write_manifest`)
- [ ] `src/<pkg>/data.py` (loaders, dedup, splitter)
- [ ] `src/<pkg>/train.py` (training + serialization)
- [ ] `src/<pkg>/classify.py` (inference CLI)
- [ ] `src/<pkg>/metrics.py` (project-specific orchestration on top of toolkit)
- [ ] `src/<pkg>/plotting.py` (palette, figure helpers)
- [ ] `evals/run_<ts>/results.json` (`RunResult` via `evaluate` + `write_run_result`)
- [ ] `evals/run_<ts>/results.json::claim_report` (`ClaimReport` via `evaluate_claims` + `with_claim_report`)
- [ ] `evals/run_<ts>/results_full.json` (per-example predictions, optional)
- [ ] `tests/` with property + invariant + smoke tests
- [ ] `Makefile` with targets: `install`, `verify-data`, `train`, `eval`, `report`
- [ ] `README.md` with: one-liner, headline result + CI, key findings, problem & motivation, approach, results table, threats to validity, reproducibility, next steps

## 12. Seeded decision log

Default ADRs encode lessons from prior projects. For each, mark "accept default" or "override → my choice + why".

**D1: Primary metric = PR-AUC**
*Rationale*: Rare positive class makes ROC-AUC deceptively flattering.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D2: Bootstrap = BCa, n=1000, conf=0.95**
*Rationale*: Bias-corrected accelerated handles skewed metric distributions; 1000 is the standard for headline CIs.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D3: ECE = `expected_calibration_error_debiased` (n_sweep=200)**
*Rationale*: Small test sets bias plain ECE upward; Monte-Carlo correction recovers honest estimate. Available since v0.5.0.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D4: Stopping rule = paired-bootstrap Δ 95% CI overlaps zero ⇒ stop**
*Rationale*: Prevents overengineering past statistical detectability.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D5: Scorer ladder must include heuristic OR LR-TFIDF baseline**
*Rationale*: Anti-overengineering check — simple often captures most signal.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D6: Splits = `SourceDisjointKFoldSplitter` + held-out OOD probes**
*Rationale*: Random splits leak via shared distributional artifacts; source-disjoint mirrors deployment.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D7: Dedup = label-aware near-dedup at threshold 0.9 (`TfidfCosineStrategy` or `EmbeddingCosineStrategy`)**
*Rationale*: Preserves cross-label adversarial pairs while removing within-label duplicates that inflate metrics.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D8: Reproducibility floor = `capture_git_sha` + `set_global_seeds` + `file_sha256(strict=True)` on data manifest**
*Rationale*: Three-line invariant; without these, no result is replayable.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D9 (project-specific)**: ___
*Rationale*: ___
*Decision*: ___

**D10**: ___ · *Rationale*: ___ · *Decision*: ___

**D11**: ___ · *Rationale*: ___ · *Decision*: ___

**D12**: ___ · *Rationale*: ___ · *Decision*: ___

**D13**: ___ · *Rationale*: ___ · *Decision*: ___

**D14**: ___ · *Rationale*: ___ · *Decision*: ___

**D15**: ___ · *Rationale*: ___ · *Decision*: ___

---

## YAML summary (source of truth for `config/baseline.yaml`)

```yaml
# === LOCKED DECISIONS — populate config/baseline.yaml from this block ===
project:
  name: ""
  slug: ""
  status: design  # design | implementation | eval | locked

data:
  feature_col: ""
  label_col: ""
  strata_col: null
  loader: DataFrameLoader  # or ParquetGlobLoader, HFDatasetsLoader, etc.
  splitter: SourceDisjointKFoldSplitter
  k: 3
  seeds: [42, 43, 44]
  train_frac: 0.8
  val_frac: 0.1
  test_frac: 0.1
  dedup:
    strategy: TfidfCosineStrategy
    threshold: 0.9
    label_aware: true
    cross_split_scrub: true
  leakage_checks:
    - ExactDuplicateCheck
    - NearDuplicateCheck
    - LabelConflictCheck
    - CrossSplitLeakageCheck
  ood_probes: []  # list of probe names

metrics:
  primary: pr_auc
  headline_bundle: true
  operating_points:
    - {selector: MaxF1Selector}
    - {selector: TargetRecallSelector, value: 0.90}
    - {selector: TargetPrecisionSelector, value: 0.90}
  calibration_error: expected_calibration_error_debiased
  ece_n_bins: 10
  ece_n_sweep: 200
  stratified_recall: true
  brier: false  # optional decomposition

inference:
  bootstrap_method: BCa  # BCa | percentile | studentized
  bootstrap_n_resamples: 1000
  bootstrap_confidence: 0.95
  paired_comparisons: []  # populate from Section 6.2 table
  mde: true
  cv_clt_ci: false

calibration:
  fit_method: temperature  # none | temperature | isotonic | platt | beta
  binning: quantile  # quantile | equal_mass

cost_matrix:                          # OPTIONAL — enable cost-sensitive thresholds
  enabled: false
  prior: 0.01
  fp_cost: 1.0
  fn_cost: 10.0
  abstain_cost: null

plotting:
  figures:
    - plot_pr_curve
    - plot_reliability_diagram
    - plot_confusion_matrix_grid
  palette:
    negative: "#004488"
    positive: "#BB5566"
    accent: "#DDAA33"
    baseline: "#999999"
  permitted_suffixes: [".png", ".pdf", ".svg"]
  skip_env_var: EVAL_TOOLKIT_SKIP_SAVEFIG

reproducibility:
  capture_git_sha: true
  global_seed: 42
  strict_torch_determinism: false
  file_sha256_strict: true
  hash_targets: ["data/manifest.json", "config/baseline.yaml"]
  run_dir_base: "evals/"
  run_dir_prefix: "run"
```
