---
template: classification_project_worksheet
variant: short
version: 1.0
eval_toolkit_minimum_version: 0.9.0
filled: false
project_name: ""
---

# Classification Project Spec — Short Variant

> **Fill instructions**:
> - Replace `___` with your value. Check `[ ]` boxes.
> - Save the filled copy as `<your-project>/spec.md` and set `filled: true` above.
> - Longer rationale goes in `<your-project>/DECISIONS.md` (created later).
> - All eval-toolkit functions named below come from `from eval_toolkit import ...`.

---

## 1. Identity

- **Project name**: ___
- **Slug** (kebab-case): ___
- **One-liner** (≤140 chars): ___
- **Status**: [ ] design  [ ] implementation  [ ] eval  [ ] locked

## 2. Problem framing

- **Class 0 (negative)** = ___
- **Class 1 (positive)** = ___
- **Class balance**: [ ] balanced  [ ] moderate (1:5–1:20)  [ ] severe (>1:20)  [ ] unknown
- **Primary hypothesis (H1)**: ___
- **Stopping rule** (declare up front, not post-hoc): ___
  > E.g., "paired-bootstrap 95% CI on Δ-PR-AUC overlaps zero" or "headline metric ≥ 0.X with CI half-width < 0.0Y"

## 3. Data

- **Feature column**: ___
- **Label column**: ___
- **Strata column** (optional): ___
- **Sources** (one line each, format: `name | source_role | rows | URL/license`; recommended roles include `train`, `validation`, `external_diagnostic`, `locked_final_holdout`, `excluded`):
  - ___
  - ___
- **Manifest guardrails** (`build_manifest(source_roles=..., guardrails=...)`):
  - ___
- **Split scheme**: [ ] `HoldoutSplitter`  [ ] `StratifiedKFoldSplitter`  [ ] `SourceDisjointKFoldSplitter`  [ ] `TimeSeriesSplitter` — `k=___`, `seeds=[___]`
- **Dedup**: use `near_dedup` from `eval_toolkit.text_dedup`; strategy = ___ (e.g., `TfidfCosineStrategy`); threshold = ___
- **Cross-split leakage check**: [ ] run `cross_dedup` between splits
- > **Why this split scheme?** ___

## 4. Scorer ladder

Required: at least one simple baseline (heuristic or LR-TFIDF) before any neural model. Each implements the `Scorer` Protocol from `eval_toolkit.harness`.

- Baseline: ___
- Candidate: ___
- Public reference (if any): ___

## 5. Metrics + statistical inference

- **Primary metric**: ___ (e.g., `pr_auc` for rare positive; `roc_auc` for balanced)
- > **Why**: ___
- **Headline bundle** (`headline_metrics`): [ ] PR-AUC + ROC-AUC + ECE
- **Operating points** (`select_threshold` with one of the selectors from `eval_toolkit.thresholds`):
  - [ ] `MaxF1Selector`
  - [ ] `TargetRecallSelector(value=___)`
  - [ ] `TargetPrecisionSelector(value=___)`
- **Threshold transfer**: [ ] use `OperatingPointSpec` to fit thresholds on validation and apply to OOD / hard-negative slices
- **Calibration error**: use `expected_calibration_error_debiased` (Monte-Carlo debiased L1 ECE; preferred over plain ECE on small test sets), `n_bins=___`
- **Bootstrap**: `bootstrap_ci(method="BCa", n_resamples=1000, confidence=0.95)`
- **Paired comparison**: `paired_bootstrap_diff` between baseline and candidate on the primary metric
- **Claim gates**: [ ] run `evaluate_claims` and attach with `with_claim_report`; include [ ] headline gate [ ] slice-size gate [ ] low-FPR feasibility gate [ ] hard-negative FPR gate

## 6. Reproducibility

- [ ] `set_global_seeds(seed=___)` at start of run
- [ ] `capture_git_sha()` recorded in run output
- [ ] `file_sha256(path, strict=True)` on data manifest before training
- [ ] `build_manifest(..., source_roles=..., guardrails=...)`
- **Run directory** via `make_run_dir(base="evals/", prefix="run")`

## 7. Threat audit

| # | Threat | Mitigation | Severity |
|---|---|---|---|
| T1 | Test-set statistical power | `mde_from_ci` after pilot to size N | ___ |
| T2 | Seed variance | ≥3 seeds per fold | ___ |
| T3 | OOD generalization | Held-out OOD probes (Section 3) | ___ |
| T4 | Train/test contamination | `cross_dedup` between splits | ___ |

## 8. Deliverables

- [ ] `spec.md` (this filled worksheet)
- [ ] `config/baseline.yaml`
- [ ] `data/manifest.json` (via `RunManifest` + `write_manifest` from `eval_toolkit.manifest`)
- [ ] `src/<pkg>/{data,train,classify,metrics}.py`
- [ ] `evals/run_<ts>/results.json` (via `evaluate` + `write_run_result` from `eval_toolkit.harness`)
- [ ] `evals/run_<ts>/results.json::claim_report` (via `evaluate_claims` + `with_claim_report`)
- [ ] `tests/`
- [ ] `Makefile` targets: `install`, `train`, `eval`
- [ ] `README.md` with: one-liner, headline result + CI, key findings, threats to validity, reproducibility
