# Statistical Inference for Binary Classification Evaluation — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained reference for binary-classification inference methods used by the eval-toolkit library. Read this README first. -->

**Purpose:** Ground future Claude agents working in the `eval-toolkit` repo in the primary literature for bootstrap confidence intervals, ROC/AUC variance, cross-validation variance, probability calibration, threshold selection, and power analysis for binary classification. Designed for dual consumption — humans (reading directly) and future LLM agents (grounding reasoning in this literature).
**Primary intended consumer:** Future Claude Code / LLM agents working in `eval-toolkit` or adjacent binary-classification eval projects who need detailed primary-source context. Secondary consumers: humans reading the material directly.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself. Move it elsewhere and it still works.
**Scope:** Binary classification; 1950–2024. 7 sub-areas (A1–A2, B1, C1, D1–D2, E1–E3) across 5 topic files. 24 primary-source entries.
**Coverage:** 24 entries across 5 topic files; structured 5-bullet entries (Source / Code / Mechanism / Result / Status).
**Last updated:** 2026-05-14.

## ⚠️ Scope boundary

This cluster covers **binary classification only**. The following are explicitly out of scope and live (or will live) elsewhere:

- **Multi-class extensions** (one-vs-rest ECE, multi-class Brier, macro/micro AUC) — methodology overlaps but literature diverges; deserves a separate dossier if needed.
- **Regression metrics** (R², RMSE, calibration-as-regression) — different statistical regime.
- **LLM-specific evaluation** (HELM, EleutherAI lm-eval-harness, AISI Inspect AI patterns) — see the sibling `../eval-ecosystem/` cluster.
- **Data-integrity / leakage / splits** (Kapoor & Narayanan 2023, Bates et al. 2024 on CV variance is here, but Bates et al. on leakage detection is in `../data-integrity/`) — see the sibling `../data-integrity/` cluster.
- **Survival / time-to-event metrics** (C-statistic time-dependent variants) — different censoring assumptions.
- **Bayesian calibration / posterior predictive checks** — different paradigm.
- **Fairness-aware threshold selection** (equal-opportunity, predictive parity) — deserves dedicated coverage.
- **Conformal prediction** — different framework for uncertainty quantification.

**Cross-vol overlap convention:** when an entry is methodologically relevant to multiple dossiers, pick ONE primary location and reference it from the others' scope-boundary callouts. Do NOT duplicate entries across dossiers.

**Adjacent dossiers in this repo:**
- `../data-integrity/` — splits / CV / leakage taxonomy / text dedup (not yet built)
- `../eval-ecosystem/` — eval harness patterns, reproducibility, Croissant (not yet built)
- `../prompt-injection/` — OWASP LLM01, PINT, attack/defense literature (not yet built)
- `../../datasets/` — eval benchmarks and corpora (not yet built)

## How this is organized

Sub-section anchors use a per-file letter prefix (`## A1.` in file 01, `## B1.` in file 02, etc.). Lookup recipes in this README reference these anchors.

| File | Topic | When to read |
|---|---|---|
| `01_bootstrap_and_cv_variance.md` | Bootstrap CI foundations (A1) + CV variance (A2) | You need CI construction for any non-AUC metric, or you're working on K-fold CV variance |
| `02_roc_variance.md` | ROC / AUC variance and comparison tests (B1) | You need DeLong-test correlated-AUC variance or paired-AUC comparison |
| `03_calibration_methods.md` | Post-hoc probability calibration methods (C1) | You're implementing or comparing Platt / isotonic / temperature / beta calibration |
| `04_calibration_metrics.md` | Proper scoring rules (D1) + ECE estimator bias (D2) | You need Brier score decomposition or debiased ECE estimators |
| `05_thresholds_power_foundations.md` | Threshold selection (E1) + power analysis (E2) + foundational text (E3) | You're picking thresholds for F1 / Youden / cost-sensitive, or computing sample size |

Raw 7-column dossier tables (more detail per entry) live in `_dossier/01_*.md` … `_dossier/05_*.md`. The source bibliography is `bib_ledger.yml`; the research plan is `research_plan.md`.

## Lookup recipes

Routes by question type. Each points to a specific file and section anchor.

### Foundational papers
- **"What's the foundational paper for bootstrap CIs?"** → `01_bootstrap_and_cv_variance.md` § A1 (DiCiccio & Efron 1996, *Bootstrap confidence intervals*).
- **"What's the DeLong test for ROC AUC?"** → `02_roc_variance.md` § B1 (DeLong, DeLong & Clarke-Pearson 1988).
- **"What's the original Platt scaling paper?"** → `03_calibration_methods.md` § C1 (Platt 1999).
- **"What's the original Brier score paper?"** → `04_calibration_metrics.md` § D1 (Brier 1950).
- **"What's the Youden J statistic?"** → `05_thresholds_power_foundations.md` § E1 (Youden 1950).
- **"Where did temperature scaling come from?"** → `03_calibration_methods.md` § C1 (Guo et al. 2017, *On calibration of modern neural networks*).
- **"What's the original Murphy Brier decomposition?"** → `04_calibration_metrics.md` § D1 (Murphy 1973).
- **"What's the Lipton F1-thresholding paper?"** → `05_thresholds_power_foundations.md` § E1 (Lipton, Elkan & Narayanaswamy 2014).

### Method / technique by name
- **"How do I compute BCa bootstrap intervals?"** → `01_bootstrap_and_cv_variance.md` § A1 (DiCiccio & Efron 1996; see Hastie, Tibshirani & Friedman 2009 ch. 8 in E3 for textbook treatment).
- **"How do I compare two ROC AUCs from the same data?"** → `02_roc_variance.md` § B1 (DeLong et al. 1988 for the closed-form; Sun & Xu 2014 for the O(N log N) implementation).
- **"How do I correct K-fold CV CIs for under-coverage?"** → `01_bootstrap_and_cv_variance.md` § A2 (Bates, Hastie & Tibshirani 2024 nested-CV procedure).
- **"Why are naive K-fold CIs under-covered?"** → `01_bootstrap_and_cv_variance.md` § A2 (Bengio & Grandvalet 2004 negative result + Bates et al. 2024).
- **"What's debiased ECE?"** → `04_calibration_metrics.md` § D2 (Kumar, Liang & Ma 2019 scaling-binning + plug-in debiased estimator; Roelofs et al. 2022 ECE_sweep).
- **"How do I correct class-prior shift at decision time?"** → `05_thresholds_power_foundations.md` § E1 (Saerens, Latinne & Decaestecker 2002 EM procedure).
- **"What's the cost-sensitive Bayes-optimal threshold?"** → `05_thresholds_power_foundations.md` § E1 (Elkan 2001; general form in entry, simplified to p\* = c₁₀ / (c₁₀ + c₀₁) under zero-cost correct classifications).
- **"What threshold maximizes F1?"** → `05_thresholds_power_foundations.md` § E1 (Lipton, Elkan & Narayanaswamy 2014; equals half the optimal F1 score when probabilities are calibrated).

### eval-toolkit code-mapping
- **"I'm working on `bootstrap.delong_variance` — primary refs?"** → `02_roc_variance.md` § B1 (DeLong 1988 + Sun & Xu 2014).
- **"I'm working on `bootstrap.kfold_clt_corrected_variance` — primary refs?"** → `01_bootstrap_and_cv_variance.md` § A2 (Bates et al. 2024; Bengio & Grandvalet 2004 for the negative result).
- **"I'm working on `calibration.reliability_diagram_data` — primary refs?"** → `04_calibration_metrics.md` § D1 (Murphy 1973 reliability/resolution decomposition).
- **"I'm working on ECE-variant metrics in `metrics.py` — primary refs?"** → `04_calibration_metrics.md` § D2 (Kumar et al. 2019, Roelofs et al. 2022; Naeini et al. 2015 introduces ECE in modern form in `03_calibration_methods.md` § C1).
- **"I'm working on `thresholds.MaxF1Selector` — primary refs?"** → `05_thresholds_power_foundations.md` § E1 (Lipton et al. 2014).
- **"I'm working on `calibration.platt_scaling` — primary refs?"** → `03_calibration_methods.md` § C1 (Platt 1999 + Niculescu-Mizil & Caruana 2005 for empirical comparison).
- **"I'm working on `calibration.temperature_scaling` — primary refs?"** → `03_calibration_methods.md` § C1 (Guo et al. 2017).
- **"I'm working on MDE / sample-size planning code — primary refs?"** → `05_thresholds_power_foundations.md` § E2 (Obuchowski 1998).

### Concept / glossary
- **"What does 'proper scoring rule' mean?"** → README glossary (canonical statement is Gneiting & Raftery 2007, not in this cluster; foundational examples Brier 1950 in `04_calibration_metrics.md` § D1).
- **"What does 'reliability' mean in calibration?"** → README glossary; primary source is Murphy 1973 decomposition in `04_calibration_metrics.md` § D1.
- **"Equal-width vs equal-mass ECE binning — which is less biased?"** → `04_calibration_metrics.md` § D2 (Roelofs et al. 2022 — equal-mass).
- **"How does ROC AUC relate to the Wilcoxon statistic?"** → `02_roc_variance.md` § B1 (Hanley & McNeil 1982).

### Out of scope routing
- **"Where are multi-class calibration methods?"** → Out of scope (see Scope boundary). For multi-class extensions (Dirichlet calibration, matrix scaling), build a separate dossier.
- **"Where's conformal prediction?"** → Out of scope (see Scope boundary). Different paradigm — own dossier if needed.
- **"Where are fairness-aware threshold selectors?"** → Out of scope (see Scope boundary).
- **"Where are LLM-eval-harness patterns (HELM, lm-eval, Inspect AI)?"** → See sibling `../eval-ecosystem/` dossier (not yet built as of last-updated date).

## Glossary

- **AUC (ROC AUC)**: Area under the ROC curve; equals the probability a random positive example is ranked above a random negative (Wilcoxon-Mann-Whitney statistic). Source: Hanley & McNeil 1982 (`02_roc_variance.md` § B1).
- **BCa interval**: Bias-corrected and accelerated bootstrap CI; second-order accurate via two scalar adjustments. Source: DiCiccio & Efron 1996 (`01_bootstrap_and_cv_variance.md` § A1).
- **Bayes-optimal threshold**: Threshold derived from a 2x2 cost matrix; general form p\* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁), reducing to p\* = c₁₀ / (c₁₀ + c₀₁) under zero-cost correct classifications. Source: Elkan 2001 (`05_thresholds_power_foundations.md` § E1).
- **Brier score**: Mean squared error between predicted probability and binary outcome; strictly proper scoring rule. Source: Brier 1950 (`04_calibration_metrics.md` § D1).
- **Calibration**: Predicted probabilities match observed frequencies (P(Y=1 | p̂=p) = p). Methods: Platt, isotonic, temperature, beta (`03_calibration_methods.md` § C1).
- **DeLong test**: Nonparametric paired test for two correlated ROC AUCs based on generalized U-statistics. Source: DeLong et al. 1988 (`02_roc_variance.md` § B1).
- **ECE (Expected Calibration Error)**: Expected absolute difference between confidence and accuracy, averaged over a binning of the score range. Source: Naeini et al. 2015 (`03_calibration_methods.md` § C1); debiased estimators in Kumar et al. 2019 / Roelofs et al. 2022 (`04_calibration_metrics.md` § D2).
- **Isotonic regression** (calibration): Non-parametric monotonic mapping from scores to probabilities. Source: Zadrozny & Elkan 2002 (`03_calibration_methods.md` § C1).
- **K-fold cross-validation**: Partition data into K folds; train on K-1, evaluate on the held-out fold. Variance is NOT unbiasedly estimable: Bengio & Grandvalet 2004 (`01_bootstrap_and_cv_variance.md` § A2).
- **MDE (Minimum Detectable Effect)**: Smallest effect size detectable with given power and sample size. For AUC differences: Obuchowski 1998 (`05_thresholds_power_foundations.md` § E2).
- **Nested CV**: Outer CV loop for evaluation, inner for model selection; corrected coverage per Bates et al. 2024 (`01_bootstrap_and_cv_variance.md` § A2).
- **Platt scaling**: Fit a 2-parameter sigmoid on held-out scores to produce calibrated probabilities. Source: Platt 1999 (`03_calibration_methods.md` § C1).
- **Prior shift (label shift)**: Train- and test-time class priors differ. Correction: Saerens, Latinne & Decaestecker 2002 EM procedure (`05_thresholds_power_foundations.md` § E1).
- **Proper scoring rule**: Scoring rule minimized in expectation by reporting the true probability. Brier and log-loss are strictly proper. Foundational: Brier 1950 (`04_calibration_metrics.md` § D1).
- **Reliability (Murphy decomposition)**: Calibration-error component of the Brier score; equals zero iff predictions are perfectly calibrated. Source: Murphy 1973 (`04_calibration_metrics.md` § D1).
- **Resolution (Murphy decomposition)**: Discrimination component of the Brier score; measures how well predictions separate positive from negative cases.
- **Temperature scaling**: Single-parameter divisor on the logits before softmax; calibrates without changing accuracy. Source: Guo et al. 2017 (`03_calibration_methods.md` § C1).
- **Wilcoxon-Mann-Whitney statistic**: U-statistic equal to ROC AUC; foundational nonparametric two-sample test. Source: Hanley & McNeil 1982 (`02_roc_variance.md` § B1).
- **Youden J statistic**: J = sensitivity + specificity − 1; threshold-optimization criterion giving equal weight to FPs and FNs. Source: Youden 1950 (`05_thresholds_power_foundations.md` § E1).

## Verification & limits

- Citations resolved as of 2026-05-14.
- All 24 entries are `status: verified` in `bib_ledger.yml` after round 1 audit (see audit-trail note below).
- This synthesis is a snapshot. The bootstrap, ROC, and calibration foundations are stable (most pre-2020); the ECE-debiasing literature is more recent and may evolve. Re-check after 12 months.
- 2 audit rounds are scheduled for this cluster (inference is a load-bearing math-heavy cluster per the dossier roadmap). Round 1 complete; round 2 pending.

**Independent audit, round 1 (2026-05-14):** A standard complementary-scope review pass focused on attribution correctness for the 17 entries marked `status: unverified` after the gather stage, plus a 2-entry spot-check on already-verified entries. Prior rounds covered: (none — this is round 1). Findings: 0 dropped, 0 corrected, 2 flagged. All 17 unverified entries passed WebFetch/WebSearch confirmation of title / first-author surname / year and were promoted to `verified`. The two flags were (1) generalization of "ten classifier families" → "multiple classifier families (SVMs, boosted trees, neural nets, bagged trees, naive Bayes, etc.)" for `niculescumizil2005calibration` (applied), and (2) a bibkey naming-convention note that `efron1996bootstrap` has Efron as second author rather than first (DiCiccio is first author) — kept as-is for round 1 since the `authors:` field correctly renders "DiCiccio & Efron (1996)". ~9 primary URLs were paywall/bot-blocked (RSNA, JSTOR, IEEE, Springer, ACM DL, AMS, Sage, MIT Press) requiring WebSearch cross-validation; this is normal for older journal papers and not URL-rot worth flagging. Recommendation: re-run with focus "Mechanism / Result bullet claim accuracy — do specific quantitative or mechanistic claims in the synthesis match the source's abstract?".

**Independent audit, round 2 (2026-05-14):** A standard complementary-scope review pass focused on Mechanism / Result bullet claim accuracy in the 5 synthesis files (01_*.md through 05_*.md), plus a check that every lookup recipe in this README points to a valid file-anchor. Prior rounds covered: round 1 (attribution correctness). Findings: 0 dropped, 0 corrected, 2 flagged (both applied). The flags were (1) the Elkan 2001 closed-form threshold equation `p* = c₁₀ / (c₁₀ + c₀₁)` only holds under zero-cost correct classifications; corrected to also include the general four-cell form `p* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁)` in both the entry and the glossary; (2) the Kumar et al. 2019 entry asserted the standard ECE estimator is "upward-biased" — the upward direction is supported by Roelofs et al. 2022 but not in the Kumar abstract, so generalized to "biased" with attribution. SPOT-CHECKS confirmed: O(N log N) complexity (Sun & Xu 2014), half-of-optimal-F1 threshold (Lipton et al. 2014), scaling-binning calibrator (Kumar et al. 2019), ECE_sweep + equal-mass < equal-width bias (Roelofs et al. 2022), single-parameter Platt variant (Guo et al. 2017), nested-CV coverage correction (Bates et al. 2024), Niculescu-Mizil & Caruana 2005 classifier-family findings, no-unbiased-estimator (Bengio & Grandvalet 2004), AUC=Wilcoxon (Hanley & McNeil 1982). All lookup recipes in the README point to valid file-anchor combinations (A1, A2, B1, C1, D1, D2, E1, E2, E3 all exist). Recommendation: Clean — stop here.

## Attribution

Synthesized from a research dossier maintained by the research_toolkit (`~/Claude/research_toolkit/`). Source URLs link to primary sources (arXiv, journal DOIs, conference proceedings, GitHub). No local file paths are referenced. Companion raw dossier tables live in `_dossier/`; source ledger is `bib_ledger.yml`; research plan is `research_plan.md`.
