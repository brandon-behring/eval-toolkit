# ROC / AUC Variance and Comparison Tests

Closed-form and resampling-based variance estimation for ROC AUC, plus paired and correlated AUC comparison tests. General bootstrap CI theory lives in `01_bootstrap_and_cv_variance.md`. Threshold-based summaries of ROC (Youden J, F1) live in `05_thresholds_power_foundations.md`.

---

## B1. AUC theory and variance estimation

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| The meaning and use of the area under a receiver operating characteristic (ROC) curve | Hanley & McNeil (1982) | Radiology 143(1) | DOI:10.1148/radiology.143.1.7063747 | — | Establishes that AUC equals the probability a random positive is ranked above a random negative (Wilcoxon-Mann-Whitney) | Foundational interpretation linking ROC AUC to the Wilcoxon nonparametric statistic; closed-form variance under binormal assumption |
| A method of comparing the areas under receiver operating characteristic curves derived from the same cases | Hanley & McNeil (1983) | Radiology 148(3) | DOI:10.1148/radiology.148.3.6878708 | — | Method for comparing ROC AUCs from paired data via correlation tables | First paired-difference test for ROC AUC that accounts for the correlation induced by shared cases; converts observed rating correlations into an area correlation via a lookup table |
| Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach | DeLong, DeLong & Clarke-Pearson (1988) | Biometrics 44(3) | DOI:10.2307/2531595 | — | Nonparametric covariance for multiple correlated ROC AUCs via generalized U-statistics | Distribution-free covariance estimator that eliminates the Hanley-McNeil correlation table; standard reference for paired-AUC comparison and the canonical "DeLong test" |
| Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves | Sun & Xu (2014) | IEEE Signal Processing Letters 21(11) | DOI:10.1109/LSP.2014.2337313 | — | Reduces DeLong covariance computation from O(N²) to O(N log N) using mid-rank equivalence | Linearithmic algorithm makes DeLong's test practical for large datasets; basis for most modern open-source implementations |
