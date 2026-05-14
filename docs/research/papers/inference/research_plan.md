# Research Plan: Statistical inference for binary classification evaluation

This research grounds the `eval-toolkit` library's statistical machinery — bootstrap confidence intervals, probability calibration, and threshold selection — in primary literature so future Claude agents can reason about correctness, edge cases, and alternative methods. Narrow scope: ~20 papers across 7 sub-areas. Covers binary classification only; multi-class, regression, survival, and LLM-specific eval are out of scope.

## Sub-areas

- A1. Bootstrap confidence intervals (foundations)
  - Source types: monograph (Efron & Tibshirani), journal (Annals of Statistics, JASA), arXiv
  - Notes: BCa, percentile, paired-difference CIs. Bias correction and acceleration constants. Studentized vs un-studentized. Excludes Bayesian bootstrap, residual bootstrap.

- A2. ROC / AUC variance and comparison tests
  - Source types: journal (Biometrics, Statistics in Medicine), arXiv reproductions, software notes
  - Notes: Closed-form variance for AUC (Hanley-McNeil), correlated-AUC variance (DeLong-DeLong-Clarke-Pearson 1988), fast DeLong algorithm (Sun & Xu 2014). Paired ROC tests. Excludes non-parametric U-statistic theory beyond what's needed for software implementation.

- A3. Cross-validation variance and CI construction
  - Source types: arXiv, journal (JMLR, Annals of Statistics)
  - Notes: K-fold CLT-corrected variance (Bates, Hastie, Tibshirani 2024 "Cross-validation: what does it estimate and how well does it do it?"). Nested CV. Why naive K-fold CIs are wrong. Excludes bootstrap-CV hybrids beyond .632/.632+ if mentioned.

- A4. Probability calibration methods
  - Source types: conference (NeurIPS, ICML, AISTATS), journal
  - Notes: Platt scaling (Platt 1999), isotonic regression (Zadrozny & Elkan 2002; Naeini et al. 2015 BBQ), temperature scaling (Guo et al. 2017), beta calibration (Kull et al. 2017), Dirichlet/matrix scaling. Excludes Bayesian neural network calibration, conformal prediction (different paradigm).

- A5. Calibration metrics and diagnostic decompositions
  - Source types: conference, arXiv, journal
  - Notes: ECE variants (equal-width, equal-mass, L2/quadratic, debiased — Kumar et al. 2019, Roelofs et al. 2022), Brier score decomposition (Murphy 1973, DeGroot & Fienberg 1983), reliability diagrams. Test-set bias of plug-in ECE estimators.

- A6. Threshold selection and decision theory
  - Source types: arXiv, conference, journal (Biometrics)
  - Notes: F1 thresholding (Lipton, Elkan, Naryanaswamy 2014), Youden J statistic (Youden 1950), cost-sensitive Bayes-optimal thresholds, prior-shift correction at decision time. Excludes Neyman-Pearson constrained optimization, conformal prediction sets.

- A7. Power analysis and minimum detectable effect (MDE)
  - Source types: journal (Statistics in Medicine, Biostatistics), arXiv, software docs
  - Notes: Sample size planning for AUC differences, MDE estimation for binary outcomes. Includes simulation-based power for paired comparisons. Excludes adaptive/sequential testing.

## Out-of-scope

- Multi-class extensions (one-vs-rest ECE, multi-class Brier, macro/micro AUC) — deserves a separate plan if needed; methodology overlaps but the literature diverges enough to warrant standalone treatment.
- Regression metrics (R², RMSE, calibration-as-regression) — different statistical regime.
- LLM-specific evaluation (HELM, lm-eval, Inspect AI patterns) — covered in the `eval-ecosystem/` cluster.
- Survival / time-to-event metrics (C-statistic time-dependent variants) — different censoring assumptions.
- Bayesian calibration / posterior predictive checks — different paradigm; could be a follow-up plan.
- Fairness-aware threshold selection (equal-opportunity, predictive parity) — deserves dedicated coverage; deferred.
- Conformal prediction — different framework for uncertainty quantification; out of scope for this cluster.

## Claim family taxonomy

- `bootstrap_ci` — bootstrap-style CI methods (BCa, percentile, paired)
- `roc_variance` — closed-form / asymptotic variance and comparison tests for ROC AUC
- `cv_variance` — cross-validation variance and CI construction (K-fold CLT, nested CV)
- `calibration_method` — methods that produce calibrated probabilities from uncalibrated scores
- `calibration_metric` — metrics that quantify calibration quality (ECE variants, Brier decomposition)
- `threshold_decision` — threshold selection rules and decision-theoretic frameworks
- `power_analysis` — sample-size planning and minimum detectable effect estimation
- `foundational_text` — textbooks and monographs that span multiple sub-areas (Efron-Tibshirani, ESL)

## Known landmark papers

- `efron1993bootstrap` — Efron & Tibshirani 1993 "An Introduction to the Bootstrap" (CRC) — foundational monograph on bootstrap CIs including BCa.
- `delong1988auc` — DeLong, DeLong & Clarke-Pearson 1988 "Comparing the areas under two or more correlated receiver operating characteristic curves" (Biometrics) — DeLong test for correlated AUCs.
- `platt1999probabilistic` — Platt 1999 "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods" — Platt scaling.
- `niculescumizil2005calibration` — Niculescu-Mizil & Caruana 2005 "Predicting good probabilities with supervised learning" (ICML) — first systematic comparison of calibration methods.
- `naeini2015bbq` — Naeini, Cooper, Hauskrecht 2015 "Obtaining well calibrated probabilities using Bayesian binning" (AAAI) — BBQ isotonic + ECE.
- `guo2017calibration` — Guo, Pleiss, Sun, Weinberger 2017 "On calibration of modern neural networks" (ICML) — temperature scaling and modern-NN ECE results.
- `kumar2019verified` — Kumar, Liang, Ma 2019 "Verified uncertainty calibration" (NeurIPS) — debiased ECE estimators with formal guarantees.
- `lipton2014thresholding` — Lipton, Elkan, Naryanaswamy 2014 "Optimal thresholding of classifiers to maximize F1 measure" (ECML PKDD) — closed-form F1-optimal threshold theory.
- `bates2024crossvalidation` — Bates, Hastie, Tibshirani 2024 "Cross-validation: what does it estimate and how well does it do it?" (JASA) — modern treatment of CV variance and CI construction.
