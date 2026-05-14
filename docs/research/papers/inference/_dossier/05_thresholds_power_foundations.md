# Thresholds, Power Analysis, and Foundational Texts

Threshold selection theory (E1), sample-size planning for binary-classifier comparison (E2), and foundational textbooks spanning the rest of this cluster (E3). The DeLong family for ROC-specific paired comparison tests lives in `02_roc_variance.md`.

---

## E1. Threshold selection and decision theory

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Index for rating diagnostic tests | Youden (1950) | Cancer 3(1) | DOI:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3 | — | Defines J = sensitivity + specificity − 1 as a single threshold-optimization criterion | The Youden J statistic — gives equal weight to FPs and FNs; corresponds to the threshold at the point on the ROC curve closest to (0,1) |
| The foundations of cost-sensitive learning | Elkan (2001) | IJCAI 2001 | DOI:10.5555/1642194.1642224 | — | Characterizes when cost matrices are coherent and derives the cost-sensitive Bayes-optimal threshold | Closed-form rule for the optimal threshold given a 2x2 cost matrix (general form p\* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁); reduces to p\* = c₁₀ / (c₁₀ + c₀₁) under zero-cost correct classifications); argues classifier rebalancing is rarely necessary if probability estimates are calibrated |
| Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure | Saerens, Latinne & Decaestecker (2002) | Neural Computation 14(1) | DOI:10.1162/089976602753284446 | — | EM-based procedure for adjusting classifier outputs when test-time class priors differ from training-time priors | Iterative prior-shift correction without retraining; foundational reference for label-shift adaptation at decision time |
| Thresholding classifiers to maximize F1 score | Lipton, Elkan & Narayanaswamy (2014) | arXiv (also ECML PKDD 2014 as "Optimal Thresholding of Classifiers to Maximize F1 Measure") | arXiv:1402.1892 | — | Derives the closed-form relationship between F1-optimal threshold and best achievable F1 | For well-calibrated probabilities the F1-optimal threshold equals half the optimal F1 score; gives a plug-in F1-thresholding algorithm and analyzes the binary and multilabel cases |

## E2. Power analysis and minimum detectable effect

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Sample size calculations in studies of test accuracy | Obuchowski (1998) | Statistical Methods in Medical Research 7(4) | DOI:10.1177/096228029800700405 | — | Surveys sample-size formulas for several accuracy indices: sensitivity/specificity, partial AUC, fixed-FPR sensitivity, likelihood ratio | Comprehensive reference for sample-size and minimum-detectable-effect calculations in diagnostic-accuracy studies; widely cited in medical imaging |

## E3. Foundational text

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| The Elements of Statistical Learning: Data Mining, Inference, and Prediction | Hastie, Tibshirani & Friedman (2009) | Springer (2nd ed.; free PDF from authors) | (no arXiv) | — | Comprehensive textbook covering supervised learning, model selection, ensemble methods, and statistical learning theory | Standard reference for binary classification, cross-validation, and the bias-variance tradeoff; chapters 7 (model assessment) and 8 (bootstrap) are particularly load-bearing for this inference cluster |
