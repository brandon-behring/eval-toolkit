# ROC / AUC Variance and Comparison Tests — Synthesis

This file synthesizes B1 (AUC theory and variance estimation). Companion raw-table dossier: `_dossier/02_roc_variance.md`. General bootstrap CI theory (BCa, K-fold CLT) lives in `01_bootstrap_and_cv_variance.md`. Threshold-based summaries derived from ROC (Youden J, F1-thresholding) live in `05_thresholds_power_foundations.md` § E1.

---

## B1. AUC theory and variance estimation

- **The meaning and use of the area under a receiver operating characteristic (ROC) curve** — Hanley & McNeil (Radiology 1982).
  - **Source:** https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747
  - **Code:** —
  - **Mechanism:** Establishes that ROC AUC equals the probability that a randomly chosen positive case is ranked above a randomly chosen negative case, the same quantity estimated by the nonparametric Wilcoxon-Mann-Whitney statistic.
  - **Result:** Foundational interpretation of AUC as the Wilcoxon nonparametric statistic; provides a closed-form variance under the binormal assumption that remained standard for two decades.
  - **Status:** Verified (no widely-known repo).

- **A method of comparing the areas under receiver operating characteristic curves derived from the same cases** — Hanley & McNeil (Radiology 1983).
  - **Source:** https://pubs.rsna.org/doi/10.1148/radiology.148.3.6878708
  - **Code:** —
  - **Mechanism:** Refines paired-AUC comparison by accounting for the correlation induced by shared cases; uses a lookup table that converts observed rating correlations into AUC correlations under a binormal assumption.
  - **Result:** First paired-difference test for ROC AUC that correctly handles correlation; introduced the practice (later superseded by DeLong et al. 1988) of working in the correlation-of-areas space.
  - **Status:** Verified (no widely-known repo).

- **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach** — DeLong, DeLong & Clarke-Pearson (Biometrics 1988).
  - **Source:** https://doi.org/10.2307/2531595
  - **Code:** —
  - **Mechanism:** Derives a nonparametric covariance estimator for multiple correlated ROC AUCs using the theory of generalized U-statistics. No distributional assumptions on the score distribution.
  - **Result:** Distribution-free covariance estimator that eliminates the Hanley-McNeil correlation table; standard reference for paired/correlated AUC comparison and the canonical "DeLong test." Underlies `bootstrap.delong_variance` in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo).

- **Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves** — Sun & Xu (IEEE Signal Processing Letters 2014).
  - **Source:** https://ieeexplore.ieee.org/document/6851192
  - **Code:** —
  - **Mechanism:** Reduces DeLong covariance computation from O(N²) to O(N log N) using an equivalent relationship between the Heaviside function and the mid-ranks of samples.
  - **Result:** Linearithmic algorithm makes DeLong's test practical for large datasets; basis for most modern open-source DeLong implementations (Python, R, MATLAB).
  - **Status:** Verified (no widely-known repo).
