# Calibration Metrics and Diagnostic Decompositions — Synthesis

This file synthesizes D1 (Proper scoring rules and Brier decomposition) and D2 (ECE estimator bias and debiasing). Companion raw-table dossier: `_dossier/04_calibration_metrics.md`. Calibration methods that produce calibrated probabilities live in `03_calibration_methods.md`. Note: Naeini et al. 2015 introduces ECE in its modern AAAI form but is indexed under `03_calibration_methods.md` § C1 because BBQ is primarily a calibration method.

---

## D1. Proper scoring rules and Brier decomposition

- **Verification of forecasts expressed in terms of probability** — Brier (Monthly Weather Review 1950).
  - **Source:** https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml
  - **Code:** —
  - **Mechanism:** Introduces the squared-error scoring rule for probabilistic weather forecasts; sums (p − y)² over all forecast/outcome pairs.
  - **Result:** The Brier score — a strictly proper scoring rule still used as the default calibration-aware accuracy metric. Foundational to calibration evaluation.
  - **Status:** Verified (no widely-known repo).

- **A new vector partition of the probability score** — Murphy (Journal of Applied Meteorology 1973).
  - **Source:** https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml
  - **Code:** —
  - **Mechanism:** Decomposes the Brier score into three additive components: uncertainty (intrinsic to the events), reliability (calibration error), and resolution (discrimination).
  - **Result:** First three-part decomposition isolating calibration error from baseline uncertainty and discrimination; allows direct reading of calibration vs sharpness contributions. Underlies `calibration.reliability_diagram_data` in the eval-toolkit harness.
  - **Status:** Verified (no widely-known repo).

## D2. ECE estimator bias and debiasing

- **Verified uncertainty calibration** — Kumar, Liang & Ma (NeurIPS 2019 Spotlight).
  - **Source:** https://arxiv.org/abs/1909.10155
  - **Code:** https://github.com/p-lambda/verified_calibration
  - **Mechanism:** Combines parametric scaling with binning ("scaling-binning"): first fit a parametric function to scores, then bin the function values. Also introduces a debiased plug-in ECE estimator with sample-complexity guarantees.
  - **Result:** First post-hoc calibrator with provably low calibration error; demonstrates that prior temperature/Platt methods are less calibrated than reported because the standard plug-in ECE estimator is biased (Roelofs et al. 2022 later quantifies the upward direction).
  - **Status:** Verified.

- **Mitigating bias in calibration error estimation** — Roelofs et al. (AISTATS 2022).
  - **Source:** https://arxiv.org/abs/2012.08668
  - **Code:** —
  - **Mechanism:** Quantifies the upward bias of plug-in ECE estimators in finite samples; proposes ECE_sweep — equal-mass bins with bin-count chosen as large as possible while preserving monotonicity in the calibration function.
  - **Result:** Establishes that equal-mass bins have lower bias than equal-width; provides a framework for choosing the right ECE estimator given the evaluation-set size. Directly informs the equal-mass ECE variant in the eval-toolkit `metrics` module.
  - **Status:** Verified (no widely-known repo).
