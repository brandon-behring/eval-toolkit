# Calibration Metrics and Diagnostic Decompositions

Metrics quantifying calibration quality (Brier score, ECE variants, debiased estimators) and decompositions that separate calibration from sharpness/resolution. Calibration methods that produce calibrated probabilities live in `03_calibration_methods.md`.

---

## D1. Proper scoring rules and Brier decomposition

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Verification of forecasts expressed in terms of probability | Brier (1950) | Monthly Weather Review 78(1) | DOI:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 | — | Introduces the squared-error scoring rule for probabilistic weather forecasts | The Brier score — a strictly proper scoring rule for probabilistic prediction; foundational to calibration evaluation |
| A new vector partition of the probability score | Murphy (1973) | Journal of Applied Meteorology 12(4) | DOI:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 | — | Decomposes the Brier score into uncertainty, reliability, and resolution components | First three-part decomposition isolating calibration error (reliability) from baseline uncertainty and discrimination (resolution); enables direct reading of calibration vs sharpness contributions |

## D2. ECE estimator bias and debiasing

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Verified uncertainty calibration | Kumar, Liang & Ma (2019) | NeurIPS 2019 Spotlight | arXiv:1909.10155 | p-lambda/verified_calibration | Combines parametric scaling with binning ("scaling-binning") and provides sample-complexity guarantees | First post-hoc calibrator with provably low calibration error; introduces a debiased plug-in ECE estimator with formal coverage and demonstrates that prior temperature/Platt methods are less calibrated than reported |
| Mitigating bias in calibration error estimation | Roelofs et al. (2022) | AISTATS 2022 | arXiv:2012.08668 | — | Quantifies the upward bias of plug-in ECE estimators with finite samples | Establishes that equal-mass bins have lower bias than equal-width; proposes ECE_sweep (equal-mass with monotonicity-preserving bin-count selection) and a framework for choosing estimators based on sample size |
