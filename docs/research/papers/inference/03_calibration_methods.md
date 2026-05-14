# Probability Calibration Methods — Synthesis

This file synthesizes C1 (Post-hoc calibration methods). Companion raw-table dossier: `_dossier/03_calibration_methods.md`. Metrics for quantifying calibration quality (ECE variants, Brier decomposition, debiased estimators) live in `04_calibration_metrics.md`. Conformal prediction is out of scope (see README § Scope boundary).

---

## C1. Post-hoc calibration methods

- **Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods** — Platt (Advances in Large Margin Classifiers / MIT Press 1999).
  - **Source:** https://www.semanticscholar.org/paper/Probabilistic-Outputs-for-Support-vector-Machines-Platt/42e5ed832d4310ce4378c44d05570439df28a393
  - **Code:** —
  - **Mechanism:** Maps SVM decision scores to probabilities by fitting a 2-parameter sigmoid on a held-out set via maximum likelihood. Assumes Gaussian per-class margin scores.
  - **Result:** Original "Platt scaling" recipe — the first widely-adopted post-hoc calibrator. Foundational reference for `calibration.platt_scaling` in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo) (uncertain venue — original is a book chapter with no separate DOI).

- **Transforming classifier scores into accurate multiclass probability estimates** — Zadrozny & Elkan (KDD 2002).
  - **Source:** https://dl.acm.org/doi/10.1145/775047.775151
  - **Code:** —
  - **Mechanism:** Extends Platt scaling and isotonic regression from binary to multiclass calibration via one-vs-rest decomposition; also analyzes histogram binning.
  - **Result:** Established the post-hoc calibration recipe (Platt or isotonic) that remained the default for two decades; introduces isotonic regression as a non-parametric post-hoc calibrator.
  - **Status:** Verified (no widely-known repo).

- **Predicting good probabilities with supervised learning** — Niculescu-Mizil & Caruana (ICML 2005).
  - **Source:** https://dl.acm.org/doi/10.1145/1102351.1102430
  - **Code:** —
  - **Mechanism:** Empirical comparison of Platt scaling and isotonic regression across multiple classifier families (SVMs, boosted trees, neural nets, bagged trees, naive Bayes, etc.) on real datasets.
  - **Result:** Established that boosted trees and SVMs need calibration while bagged trees and (well-trained) neural nets are reasonably calibrated; identified the characteristic sigmoid distortion of margin-based methods that motivates Platt scaling.
  - **Status:** Verified (no widely-known repo).

- **Obtaining well calibrated probabilities using Bayesian binning** — Naeini, Cooper & Hauskrecht (AAAI 2015).
  - **Source:** https://ojs.aaai.org/index.php/AAAI/article/view/9602
  - **Code:** —
  - **Mechanism:** Bayesian model averaging over equal-frequency binning schemes (BBQ); post-processes any binary classifier's outputs.
  - **Result:** Introduces ECE in its modern AAAI form; BBQ provides a Bayesian non-parametric calibrator competitive with isotonic regression. The paper is the most-cited modern reference for ECE.
  - **Status:** Verified (no widely-known repo).

- **On calibration of modern neural networks** — Guo et al. (ICML 2017).
  - **Source:** https://arxiv.org/abs/1706.04599
  - **Code:** —
  - **Mechanism:** Shows modern deep nets are systematically overconfident; proposes temperature scaling — a single-parameter variant of Platt scaling that divides logits before softmax.
  - **Result:** Established temperature scaling as the strong default post-hoc baseline for neural nets; popularized ECE as an evaluation metric for deep learning. Foundational for `calibration.temperature_scaling` in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo).

- **Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers** — Kull, Silva Filho & Flach (AISTATS 2017).
  - **Source:** https://proceedings.mlr.press/v54/kull17a.html
  - **Code:** —
  - **Mechanism:** Replaces Platt's logistic link with a 3-parameter Beta-family link tailored to scores already in [0,1].
  - **Result:** Strictly more flexible than Platt while remaining identifiable; avoids the bias Platt produces on already-near-calibrated scores (such as those from neural networks or Naive Bayes after rescaling).
  - **Status:** Verified (no widely-known repo).
