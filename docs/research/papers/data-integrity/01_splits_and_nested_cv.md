# Splits, Cross-Validation Strategies, and Nested CV — Synthesis

This file synthesizes A1 (splits and CV strategies for structured/time-series data) and A2 (nested CV for unbiased HP-tuned evaluation). Companion raw-table dossier: `_dossier/01_splits_and_nested_cv.md`. General CV variance theory (K-fold CIs, nested-CV coverage corrections) lives in `../inference/01_bootstrap_and_cv_variance.md`.

---

## A1. Splits and CV strategies for structured data

- **On the use of cross-validation for time series predictor evaluation** — Bergmeir & Benitez (Information Sciences 2012).
  - **Source:** https://doi.org/10.1016/j.ins.2011.12.028
  - **Code:** —
  - **Mechanism:** Empirical and theoretical comparison of K-fold CV, blocked CV, and out-of-sample evaluation for time-series forecasting with ML models.
  - **Result:** Demonstrates that standard K-fold CV is valid for time-series ML when the residuals are uncorrelated (the i.i.d. condition is on residuals, not raw observations); empirically favorable vs OOS for stationary predictors. Provides decision rules for blocked vs random K-fold.
  - **Status:** Verified (no widely-known repo).

- **Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure** — Roberts et al. (Ecography 2017).
  - **Source:** https://doi.org/10.1111/ecog.02881
  - **Code:** —
  - **Mechanism:** Practical guide to block CV for ecological data with intrinsic dependence (temporal / spatial / hierarchical / phylogenetic).
  - **Result:** Establishes that random K-fold underestimates predictive error when train/test points are dependent; block CV is the appropriate remedy. Foundational for source-disjoint and group-disjoint splits in the eval-toolkit `splits` module.
  - **Status:** Verified (no widely-known repo).

## A2. Nested CV for unbiased HP-tuned evaluation

- **Bias in error estimation when using cross-validation for model selection** — Varma & Simon (BMC Bioinformatics 2006).
  - **Source:** https://link.springer.com/article/10.1186/1471-2105-7-91
  - **Code:** —
  - **Mechanism:** Empirical demonstration that using CV both to tune hyperparameters and to estimate generalization error produces an optimistically biased estimate.
  - **Result:** Establishes the nested-CV recipe — outer loop for evaluation, inner loop for HP tuning — that reduces selection bias to near-zero. Foundational reference for the eval-toolkit's nested-CV / `PoolBuilder` split pattern.
  - **Status:** Verified (no widely-known repo).

- **On over-fitting in model selection and subsequent selection bias in performance evaluation** — Cawley & Talbot (JMLR 2010).
  - **Source:** https://www.jmlr.org/papers/v11/cawley10a.html
  - **Code:** —
  - **Mechanism:** Theoretical and empirical analysis of how variance in the model-selection criterion drives selection bias in performance evaluation.
  - **Result:** Demonstrates that low variance in the model-selection criterion is at least as important as unbiasedness for honest performance estimation; concrete worked examples on SVMs and kernel-ridge regression.
  - **Status:** Verified (no widely-known repo).
