# Splits, Cross-Validation Strategies, and Nested CV

This file covers A1 (splits and CV strategies for structured/time-series/grouped data) and A2 (nested CV for unbiased HP-tuned evaluation). General CV variance theory (K-fold under-coverage, Bates et al. 2024) lives in `../../inference/_dossier/01_bootstrap_and_cv_variance.md`.

---

## A1. Splits and CV strategies for structured data

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| On the use of cross-validation for time series predictor evaluation | Bergmeir & Benitez (2012) | Information Sciences 191 | DOI:10.1016/j.ins.2011.12.028 | — | Empirical and theoretical comparison of CV strategies for time-series forecasting; shows standard K-fold can be valid for stationary ML predictors | Demonstrates that K-fold CV is valid for time-series ML when residuals are uncorrelated; provides decision rules for when to use blocked/temporal vs random K-fold |
| Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure | Roberts et al. (2017) | Ecography 40(8) | DOI:10.1111/ecog.02881 | — | Practical guide to block CV for ecological data with intrinsic dependence structure | Establishes that random K-fold underestimates predictive error when training/test points are dependent; block CV is the appropriate remedy. Foundational for source-disjoint and group-disjoint splits |

## A2. Nested CV for unbiased HP-tuned evaluation

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Bias in error estimation when using cross-validation for model selection | Varma & Simon (2006) | BMC Bioinformatics 7 | DOI:10.1186/1471-2105-7-91 | — | Shows that using CV both to tune hyperparameters and to estimate error gives an optimistically biased estimate | Established the nested-CV recipe — outer loop for evaluation, inner loop for HP tuning — that reduces selection bias to near-zero. Foundational reference for the eval-toolkit's nested-CV split pattern |
| On over-fitting in model selection and subsequent selection bias in performance evaluation | Cawley & Talbot (2010) | JMLR 11 | (no arXiv) | — | Analyzes how variance in the model-selection criterion drives selection bias in performance evaluation | Demonstrates that low variance in the model-selection criterion is at least as important as unbiasedness; concrete examples on SVMs and kernel-ridge regression |
