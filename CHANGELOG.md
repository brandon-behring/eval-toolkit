# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-07

### Added

Initial extraction from
[`prompt_injection_detector`](https://github.com/brandon-behring/prompt-injection-detector)
PoC. Eleven modules promoted to library-grade status:

- `eval_toolkit.metrics` — PR-AUC, ROC-AUC, ECE (equal-width and equal-mass),
  threshold selection (max-F1, recall-target), prior-shift precision projection,
  stratified recall, quantile-stratified PR-AUC, headline-metrics bundle.
- `eval_toolkit.bootstrap` — BCa and percentile bootstrap CIs, paired bootstrap
  on metric deltas (with dependency-injected metric function for ECE),
  two-level operating-point bootstrap that re-fits thresholds within each
  resample, minimum-detectable-effect estimates.
- `eval_toolkit.calibration` — Reliability curves (DeGroot & Fienberg 1983),
  Bayes-optimal thresholds (Elkan 2001), isotonic regression
  (Niculescu-Mizil & Caruana 2005), Platt scaling (Platt 1999),
  temperature scaling on val logits (Guo et al. 2017),
  per-slice oracle T-scaling (diagnostic upper bound).
- `eval_toolkit.plotting` — PR curves, reliability diagrams, confusion matrix
  grids, metric bars, score-distribution histograms, lift-CI forest plots.
  Provenance-aware `save_figure` writes PNG iTXt + sidecar JSON metadata.
- `eval_toolkit.harness` — `Scorer` Protocol, slice-aware evaluation
  orchestrator with pure `evaluate(...) → RunResult` and IO `write_run_result`.
- `eval_toolkit.text_dedup` — TF-IDF near-duplicate detection, cross-source
  leakage scrubbing, canonical text hashing and normalization.
- `eval_toolkit.provenance` — File SHA-256, run-directory layout, optional
  git-SHA capture, figure-metadata builder.
- `eval_toolkit.paths` — Repo-relative path normalization.
- `eval_toolkit.seeds` — `set_global_seeds` (random + numpy + optional torch).
- `eval_toolkit.config` — `frozen_config` decorator + `from_yaml` loader.
- `eval_toolkit.docs` — Anchor-based markdown renderer with formatter registry.

### Notes

- **Binary-classification first**: v1 API focuses on the binary case.
  Multiclass, regression, and ranking are explicit future extensions.
- **PEP 561 compliant**: ships `py.typed` marker; downstream mypy users get
  free type checking.
- **Hypothesis property tests** cover the math/stat invariants for metrics,
  bootstrap, calibration, text_dedup, provenance, paths, seeds, and config.
- **Doctested Examples** required for math kernels (metrics, bootstrap,
  calibration, text_dedup); enforced via `pytest --doctest-modules`.

### Quality gates (2026-05-07)

- **Tests**: 221 passing (122 unit + smoke, 25 property, 13 docs golden,
  59 coverage-gap, 27 doctests). `pytest --doctest-modules` green.
- **Coverage**: 90.10% line + branch. Largest remaining gap is the optional
  torch branch in `seeds.py` (5 stmts unreachable when torch is not in the
  dev env).
- **Lint**: `ruff check src tests` and `black --check src tests` clean.
- **Types**: `mypy src` strict mode clean (PEP 561 `py.typed` marker shipped).
- **Public API**: 84 symbols re-exported from top-level `eval_toolkit`
  package; `from eval_toolkit import *` is well-defined.

[0.1.0]: ./
