---
template: classification_project_worksheet
variant: medium
version: 1.0
eval_toolkit_minimum_version: 0.9.0
filled: false
project_name: ""
---

# Classification Project Spec — Medium Variant

> **Fill instructions**:
> - Replace `___` with your value. Check `[ ]` boxes for choices made.
> - `[OPTIONAL]` items are advanced; skip on first pass if unsure.
> - The YAML block at the bottom is the parseable source of truth (mirrors `config/baseline.yaml`); checklists above are for thinking through the decision.
> - Save the filled copy as `<your-project>/spec.md` and set `filled: true` in the frontmatter above.

---

## 1. Identity

- **Project name**: ___
- **Slug** (kebab-case, used for directory + package names): ___
- **One-liner** (≤140 chars): ___
- **Domain / motivation** (3–5 sentences): ___
- **Status**: [ ] design  [ ] implementation  [ ] eval  [ ] locked  [ ] deprecated
- **Owner**: ___
- **Target completion**: ___

## 2. Problem framing

- **Class 0 (negative)** = ___
- **Class 1 (positive)** = ___
- **Class balance**: [ ] balanced  [ ] moderate (1:5–1:20)  [ ] severe (>1:20)  [ ] unknown — measured: ___
- **Hypotheses**:
  - H1 (primary, drives stopping rule): ___
  - H2 (secondary): ___
  - H3 (interpretive grid, optional): ___
- **Stopping rule** (pre-registered, not post-hoc): ___
  > Default per ADR D4 below: "stop iterating when paired-bootstrap 95% CI on Δ-primary-metric overlaps zero".

## 3. Data

**Schema** (maps to `EvalSlice` from `eval_toolkit.harness`):
- `feature_col`: ___
- `label_col`: ___
- `strata_col` (optional): ___

**Sources** (one row per source):

| Source | Source role (`train` / `validation` / `external_diagnostic` / `locked_final_holdout` / `excluded`) | License | Rows | URL |
|---|---|---|---|---|
| ___ | ___ | ___ | ___ | ___ |
| ___ | ___ | ___ | ___ | ___ |

**Splits** (use one of `eval_toolkit.splits`):
- [ ] `HoldoutSplitter`
- [ ] `StratifiedKFoldSplitter`
- [ ] `SourceDisjointKFoldSplitter` (recommended — see ADR D6 in Section 12-equivalent of Long variant)
- [ ] `GroupKFoldSplitter`
- [ ] `TimeSeriesSplitter`
- `k=___`, `seeds=[___]`

**Dedup** (`near_dedup` from `eval_toolkit.text_dedup`):
- Strategy: [ ] `ExactNormalizedHashStrategy`  [ ] `JaccardNgramStrategy`  [ ] `TfidfCosineStrategy`  [ ] `EmbeddingCosineStrategy`  [ ] `MinHashLSHStrategy`
- Threshold: ___
- [OPTIONAL] Label-aware (preserve cross-label adversarial pairs, remove within-label duplicates)
- [ ] Cross-split leakage scrub via `cross_dedup`

**Leakage gates** (`run_leakage_checks` from `eval_toolkit.leakage`):
- [ ] `ExactDuplicateCheck`
- [ ] `NearDuplicateCheck`
- [ ] `LabelConflictCheck`
- [ ] `CrossSplitLeakageCheck`

**OOD probes** (held out entirely, never touch train/val):
- ___
- ___

**Source-role guardrails** (`build_manifest(source_roles=..., guardrails=...)`):
- ___
- ___

> **Why this split scheme?** ___

## 4. Scorer ladder

Anti-overengineering: must include at least one simple baseline. Each scorer implements the `Scorer` Protocol from `eval_toolkit.harness`.

| Rung | Name | Training config | Hyperparams locked? |
|---|---|---|---|
| Heuristic / classical baseline | ___ | ___ | [ ] Y [ ] N |
| Pretrained-frozen | ___ | ___ | [ ] Y [ ] N |
| Public reference | ___ | ___ | [ ] Y [ ] N |
| Fine-tuned candidate | ___ | ___ | [ ] Y [ ] N |

> **Why each rung?** ___

## 5. Metrics

**Primary metric** (one):
- [ ] `pr_auc` (recommended for rare positive — see ADR D1)
- [ ] `roc_auc`
- [ ] F1 @ threshold
- [ ] Custom: ___

**Headline bundle**: [ ] `headline_metrics` (PR-AUC + ROC-AUC + ECE)

**Operating points** (`select_threshold` with one of these selectors from `eval_toolkit.thresholds`):
- [ ] `MaxF1Selector`
- [ ] `TargetRecallSelector(value=___)`
- [ ] `TargetPrecisionSelector(value=___)`
- [ ] `TargetFPRSelector(value=___)`
- [ ] `YoudenJSelector`
- [OPTIONAL] [ ] `CostSensitiveSelector` with `CostMatrix(prior=___, fp_cost=___, fn_cost=___)` for cost-based threshold
- [ ] `OperatingPointSpec` for validation-fit thresholds applied to OOD / hard-negative slices

**Calibration error**:
- [ ] `expected_calibration_error_debiased` (recommended on small test sets — see ADR D3), `n_bins=___`, `n_sweep=___`
- [OPTIONAL] [ ] `expected_calibration_error_l2_debiased` (Kumar 2019 closed-form)

**Stratified analysis**: [ ] `stratified_recall(strata_col=___, with_ci=True)`

## 6. Statistical inference

**Single-condition CI**: `bootstrap_ci(method="___", n_resamples=___, confidence=___)` — see ADR D2

**Paired contrasts** (one row per comparison):

| Baseline | Alternative | Metric | Function |
|---|---|---|---|
| ___ | ___ | ___ | `paired_bootstrap_diff` |
| ___ | ___ | ECE | `paired_bootstrap_ece_diff` |

**MDE / power**: [ ] run `mde_from_ci` after pilot to size full eval

**Cross-validation CI**: [ ] `cv_clt_ci` on `cross_validate_metric` output

**Claim gates** (`eval_toolkit.claims`):
- [ ] `headline_present_gate`
- [ ] `minimum_slice_size_gate`
- [ ] `low_fpr_feasibility_gate`
- [ ] `metric_threshold_gate` for hard-negative FPR
- [ ] Attach with `with_claim_report` before `write_run_result`

## 7. Calibration & decision

- **Fit method on val set**: [ ] none  [ ] `fit_temperature` (logits)  [ ] `fit_isotonic_calibrator`  [ ] `fit_platt_calibrator`  [ ] `fit_beta_calibrator`
- **Reliability binning**: [ ] quantile  [ ] equal-mass via `expected_calibration_error_equal_mass`
- > **Why this method?** ___

## 8. Plotting & output

- Figures (each from `eval_toolkit.plotting`):
  - [ ] `plot_pr_curve`  [ ] `plot_reliability_diagram`  [ ] `plot_confusion_matrix_grid`
  - [ ] `plot_metric_bars`  [ ] `plot_score_histograms`  [ ] `plot_lift_ci`
- Palette: `make_palette(negative=___, positive=___, accent=___, baseline=___)`
- Output formats via `save_figure`: `permitted_suffixes={___}` (default `{".png", ".pdf", ".svg"}`)

## 9. Reproducibility & provenance

- [ ] `set_global_seeds(seed=___)` (optionally `strict_torch_determinism=True`)
- [ ] `capture_git_sha()` recorded in run output
- [ ] `file_sha256(path, strict=True)` on data manifest + locked configs
- [ ] Run directory via `make_run_dir(base="evals/", prefix="run")`
- [ ] Data manifest via `RunManifest` + `build_manifest` + `write_manifest`
- [ ] Manifest source roles and guardrails via `SourceRoleRecord`
- [ ] Config via `@frozen_config` dataclass loaded with `from_yaml`

## 10. Threat audit

| # | Threat | Mitigation | Severity |
|---|---|---|---|
| T1 | Test-set statistical power | `mde_from_ci` to size N | ___ |
| T2 | Seed variance | ≥3 seeds per fold | ___ |
| T3 | OOD generalization | Held-out OOD probes (Section 3) | ___ |
| T4 | Calibration drift | ECE on each slice | ___ |
| T5 | Train/test contamination | `cross_dedup` + `CrossSplitLeakageCheck` | ___ |
| T6 | Operating-point sensitivity | Report ≥3 thresholds | ___ |

## 11. Deliverables / artifact map

- [ ] `spec.md` (this filled worksheet)
- [ ] `DECISIONS.md` (Section 12 ADRs + new ADRs added during build)
- [ ] `assumptions.md`
- [ ] `config/baseline.yaml` (seeded by YAML block below)
- [ ] `data/manifest.json` (via `RunManifest`)
- [ ] `src/<pkg>/{data,train,classify,plotting,metrics}.py`
- [ ] `evals/run_<ts>/{results.json, results_full.json}` (via `evaluate` + `write_run_result`)
- [ ] `evals/run_<ts>/results.json::claim_report` (via `evaluate_claims` + `with_claim_report`)
- [ ] `tests/`
- [ ] `Makefile` targets: `install`, `verify-data`, `train`, `eval`, `report`
- [ ] `README.md`: one-liner, headline result + CI, key findings, approach, results table, threats to validity, reproducibility

## 12. Seeded decisions (D1–D5)

For each, mark "accept default" or "override → my choice".

**D1: Primary metric = PR-AUC**
*Rationale*: Rare positive class makes ROC-AUC deceptively flattering.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D2: Bootstrap = BCa, n=1000, conf=0.95**
*Rationale*: Bias-corrected accelerated handles skewed metric distributions; 1000 is the standard for headline CIs.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D3: ECE = `expected_calibration_error_debiased` (n_sweep=200)**
*Rationale*: Small test sets bias plain ECE upward; Monte-Carlo correction recovers honest estimate.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D4: Stopping rule = paired-bootstrap Δ 95% CI overlaps zero ⇒ stop**
*Rationale*: Prevents overengineering past statistical detectability.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

**D5: Scorer ladder must include heuristic OR LR-TFIDF baseline**
*Rationale*: Anti-overengineering check — simple often captures most signal.
*Override?* [ ] No · [ ] Override → ___ · *Why*: ___

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
  splitter: SourceDisjointKFoldSplitter  # or StratifiedKFoldSplitter, etc.
  k: 3
  seeds: [42, 43, 44]
  dedup:
    strategy: TfidfCosineStrategy
    threshold: 0.9

metrics:
  primary: pr_auc
  headline_bundle: true
  operating_points:
    - {selector: MaxF1Selector}
    - {selector: TargetRecallSelector, value: 0.90}
  calibration: expected_calibration_error_debiased
  ece_n_bins: 10
  ece_n_sweep: 200

inference:
  bootstrap_method: BCa
  bootstrap_n_resamples: 1000
  bootstrap_confidence: 0.95

reproducibility:
  capture_git_sha: true
  global_seed: 42
  file_sha256_strict: true
```
