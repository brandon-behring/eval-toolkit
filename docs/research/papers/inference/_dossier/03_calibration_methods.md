# Probability Calibration Methods

Post-hoc methods for producing calibrated probability estimates from uncalibrated classifier scores. Metrics for quantifying calibration quality (ECE variants, Brier decomposition) live in `04_calibration_metrics.md`. Conformal prediction is out of scope (see `research_plan.md` § Out-of-scope).

---

## C1. Post-hoc calibration methods

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods | Platt (1999) | Advances in Large Margin Classifiers (MIT Press) | (no arXiv) | — | Maps SVM decision scores to probabilities by fitting a 2-parameter sigmoid on held-out data via maximum likelihood | Original "Platt scaling" recipe — first widely-adopted post-hoc calibrator; assumes Gaussian per-class margin scores |
| Transforming classifier scores into accurate multiclass probability estimates | Zadrozny & Elkan (2002) | KDD 2002 | DOI:10.1145/775047.775151 | — | Applies binning, Platt scaling, and isotonic regression to recalibrate classifier outputs in the multiclass setting | Established the post-hoc calibration recipe (Platt or isotonic) that remained the default for two decades; introduced one-vs-rest decomposition for multiclass |
| Predicting good probabilities with supervised learning | Niculescu-Mizil & Caruana (2005) | ICML 2005 | DOI:10.1145/1102351.1102430 | — | Empirical comparison of Platt and isotonic regression across multiple classifier families (SVMs, boosted trees, neural nets, bagged trees, naive Bayes, etc.) on real datasets | Established that boosted trees and SVMs need calibration while bagged trees and well-trained neural nets are reasonably calibrated; identified characteristic sigmoid distortion of margin-based methods |
| Obtaining well calibrated probabilities using Bayesian binning | Naeini, Cooper & Hauskrecht (2015) | AAAI 2015 | (no arXiv) | — | Bayesian model averaging over equal-frequency binning schemes (BBQ) | Introduces ECE in its modern AAAI form; BBQ provides a Bayesian non-parametric calibrator competitive with isotonic regression |
| On calibration of modern neural networks | Guo et al. (2017) | ICML 2017 | arXiv:1706.04599 | — | Shows modern deep nets are systematically overconfident and proposes temperature scaling | Established temperature scaling as the strong default post-hoc baseline for neural nets; popularized ECE as an evaluation metric for deep learning |
| Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers | Kull, Silva Filho & Flach (2017) | AISTATS 2017 | (no arXiv) | — | Replaces Platt's logistic link with a 3-parameter Beta-family link tailored to scores already in [0,1] | Strictly more flexible than Platt while remaining identifiable; avoids the bias Platt has on already-near-calibrated scores |
