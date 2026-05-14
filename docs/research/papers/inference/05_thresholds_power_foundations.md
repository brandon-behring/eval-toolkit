# Thresholds, Power Analysis, and Foundational Texts — Synthesis

This file synthesizes E1 (Threshold selection and decision theory), E2 (Power analysis and minimum detectable effect), and E3 (Foundational text). Companion raw-table dossier: `_dossier/05_thresholds_power_foundations.md`. ROC-specific paired comparison tests (DeLong family) live in `02_roc_variance.md`.

---

## E1. Threshold selection and decision theory

- **Index for rating diagnostic tests** — Youden (Cancer 1950).
  - **Source:** https://pubmed.ncbi.nlm.nih.gov/15405679/
  - **Code:** —
  - **Mechanism:** Defines J = sensitivity + specificity − 1 as a single criterion summarizing a diagnostic test's performance at a threshold.
  - **Result:** The Youden J statistic — gives equal weight to false positives and false negatives; corresponds geometrically to the point on the ROC curve farthest from the chance diagonal. Underlies threshold selectors that maximize informedness.
  - **Status:** Verified (no widely-known repo).

- **The foundations of cost-sensitive learning** — Elkan (IJCAI 2001).
  - **Source:** https://dl.acm.org/doi/10.5555/1642194.1642224
  - **Code:** —
  - **Mechanism:** Characterizes when a cost matrix is economically coherent and derives the cost-sensitive Bayes-optimal decision threshold from calibrated posterior probabilities.
  - **Result:** Closed-form rule for the optimal threshold given a 2x2 cost matrix. Under zero-cost correct classifications, p\* = c₁₀ / (c₁₀ + c₀₁); see the paper for the general four-cell form, p\* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁). Argues that classifier rebalancing is rarely necessary when probability estimates are calibrated — the optimal decision is just a threshold shift. Underlies cost-sensitive threshold selection in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo).

- **Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure** — Saerens, Latinne & Decaestecker (Neural Computation 2002).
  - **Source:** https://direct.mit.edu/neco/article/14/1/21/6577/Adjusting-the-Outputs-of-a-Classifier-to-New-a
  - **Code:** —
  - **Mechanism:** EM-based procedure for adjusting classifier posterior outputs when test-time class priors differ from training-time priors; iteratively re-estimates the new priors and re-normalizes the posteriors.
  - **Result:** Iterative prior-shift correction without retraining; foundational reference for label-shift adaptation at decision time. Informs prior-shift projection in the eval-toolkit `metrics` module.
  - **Status:** Verified (no widely-known repo).

- **Thresholding classifiers to maximize F1 score** — Lipton, Elkan & Narayanaswamy (arXiv 2014; published in ECML PKDD 2014 as "Optimal Thresholding of Classifiers to Maximize F1 Measure").
  - **Source:** https://arxiv.org/abs/1402.1892
  - **Code:** —
  - **Mechanism:** Derives the closed-form relationship between the F1-optimal threshold and the best achievable F1 score for both binary and multilabel classification.
  - **Result:** For well-calibrated probabilities the F1-optimal threshold equals half the optimal F1 score; provides a plug-in F1-thresholding algorithm. Underlies `thresholds.MaxF1Selector` in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo).

## E2. Power analysis and minimum detectable effect

- **Sample size calculations in studies of test accuracy** — Obuchowski (Statistical Methods in Medical Research 1998).
  - **Source:** https://journals.sagepub.com/doi/abs/10.1177/096228029800700405
  - **Code:** —
  - **Mechanism:** Surveys sample-size formulas for several accuracy indices: sensitivity / specificity, partial AUC, sensitivity at a fixed FPR, likelihood ratio. Covers both single-test and two-test paired comparison settings.
  - **Result:** Comprehensive reference for sample-size and minimum-detectable-effect calculations in diagnostic-accuracy studies; widely cited in medical imaging and clinical biostatistics. Informs MDE estimation in the eval-toolkit `bootstrap` module.
  - **Status:** Verified (no widely-known repo).

## E3. Foundational text

- **The Elements of Statistical Learning: Data Mining, Inference, and Prediction** — Hastie, Tibshirani & Friedman (Springer 2009, 2nd ed.).
  - **Source:** https://hastie.su.domains/ElemStatLearn/
  - **Code:** —
  - **Mechanism:** Comprehensive textbook on supervised learning, model selection, ensemble methods, and statistical learning theory. Free PDF maintained by the authors at the canonical Stanford URL.
  - **Result:** Standard textbook reference for binary classification, cross-validation, the bias-variance tradeoff, and the bootstrap. Chapter 7 (model assessment / cross-validation) and Chapter 8 (bootstrap, model averaging) are particularly load-bearing for this inference cluster.
  - **Status:** Verified (no widely-known repo) (uncertain venue — textbook, not a paper).
