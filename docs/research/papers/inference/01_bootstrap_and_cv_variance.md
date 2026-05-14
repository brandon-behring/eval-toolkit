# Bootstrap CIs and Cross-validation Variance — Synthesis

This file synthesizes A1 (Bootstrap CI foundations) and A2 (Cross-validation variance and CI construction). Companion raw-table dossier: `_dossier/01_bootstrap_and_cv_variance.md`. ROC-AUC-specific variance (DeLong family) lives in `02_roc_variance.md`. The Hastie/Tibshirani/Friedman textbook (general reference for bootstrap and CV) is indexed in `05_thresholds_power_foundations.md` § E3.

---

## A1. Bootstrap CI foundations

- **Bootstrap confidence intervals** — DiCiccio & Efron (Statistical Science 1996).
  - **Source:** https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full
  - **Code:** —
  - **Mechanism:** Surveys four bootstrap CI methods — BCa, bootstrap-t, ABC, and calibration — providing theoretical and practical comparisons.
  - **Result:** Canonical reference for the bias-corrected and accelerated (BCa) interval; achieves second-order accuracy via two scalar adjustments for bias and skewness, allowing routine application to complicated problems.
  - **Status:** Verified (no widely-known repo).

## A2. Cross-validation variance and CI construction

- **Inference for the generalization error** — Nadeau & Bengio (Machine Learning 2003).
  - **Source:** https://link.springer.com/article/10.1023/A:1024068626366
  - **Code:** —
  - **Mechanism:** Studies the variance of cross-validation-based generalization-error estimators, accounting for both training-set and test-set sources of variability.
  - **Result:** Introduces the "corrected resampled t-test" with a variance correction factor that produces better-calibrated p-values for classifier comparison than the naive paired t-test.
  - **Status:** Verified (no widely-known repo).

- **No unbiased estimator of the variance of K-fold cross-validation** — Bengio & Grandvalet (JMLR 2004).
  - **Source:** https://jmlr.csail.mit.edu/papers/v5/grandvalet04a.html
  - **Code:** —
  - **Mechanism:** Proves no universal (distribution-free) unbiased estimator exists for the variance of K-fold cross-validation.
  - **Result:** Negative result motivating approximation-based variance estimators and CLT-corrected CIs; explains why naive K-fold variance under-covers and why downstream methods must approximate.
  - **Status:** Verified (no widely-known repo).

- **Cross-validation: what does it estimate and how well does it do it?** — Bates, Hastie & Tibshirani (JASA 2024).
  - **Source:** https://arxiv.org/abs/2104.00673
  - **Code:** https://github.com/stephenbates19/nestedcv
  - **Mechanism:** Proves CV does not estimate the prediction error of the model at hand fit to the training data; rather it estimates the average prediction error across training sets drawn from the same population. Provides a nested-CV procedure.
  - **Result:** Standard K-fold CIs have severe under-coverage because per-point train/test correlation makes the usual variance estimate too small; the nested-CV procedure gives corrected coverage.
  - **Status:** Verified.
