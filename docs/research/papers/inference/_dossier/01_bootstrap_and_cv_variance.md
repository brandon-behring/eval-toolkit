# Bootstrap CIs and Cross-validation Variance

Bootstrap confidence-interval foundations (A1) together with cross-validation variance and confidence-interval construction (A2). The DeLong family for ROC-specific variance lives in `02_roc_variance.md`. Calibration metrics that use bootstrap CIs live in `04_calibration_metrics.md`.

---

## A1. Bootstrap CI foundations

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Bootstrap confidence intervals | DiCiccio & Efron (1996) | Statistical Science 11(3) | DOI:10.1214/ss/1032280214 | — | Surveys four bootstrap CI methods (BCa, bootstrap-t, ABC, calibration) with theoretical and practical comparisons | Canonical reference for the bias-corrected and accelerated (BCa) interval; second-order accuracy with bias and skewness correction via two scalar adjustments |

## A2. Cross-validation variance and CI construction

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Inference for the generalization error | Nadeau & Bengio (2003) | Machine Learning 52 | DOI:10.1023/A:1024068626366 | — | Studies variance of cross-validation-based generalization-error estimators, accounting for training- and test-set randomness | "Corrected resampled t-test" with a variance correction factor that produces better-calibrated p-values for classifier comparison than the naive paired t-test |
| No unbiased estimator of the variance of K-fold cross-validation | Bengio & Grandvalet (2004) | JMLR 5 | (no arXiv) | — | Shows no universal unbiased estimator exists for the variance of K-fold CV | Negative result that motivates approximation-based variance estimators and CLT-corrected CIs; explains why naive K-fold variance under-covers |
| Cross-validation: what does it estimate and how well does it do it? | Bates, Hastie & Tibshirani (2024) | JASA 119(546) | arXiv:2104.00673 | stephenbates19/nestedcv | Proves CV does not estimate prediction error of the model at hand but the average across training sets drawn from the same population | Shows naive K-fold CIs have severe under-coverage; provides a nested-CV procedure with corrected coverage and identifies the source of bias |
