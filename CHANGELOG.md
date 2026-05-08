# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-07

Backlog-clearance release. v0.3.0's audit document deferred 6 items to
v0.4.0+; this release ships 5 of them (CV-CLT lands as a standalone
helper rather than as part of a full CV runner; the CV-runner design
is gated on a separate v0.5 conversation).

### Added — methodology

- **`expected_calibration_error_l2`** — equal-mass L2 ECE (RMSE form).
  Companion to the L1 variant `expected_calibration_error_equal_mass`.
- **`expected_calibration_error_l2_debiased`** — Kumar 2019 §3.3
  closed-form bias-corrected L2 ECE. Removes the O(M/n) positive bias
  of the plug-in estimator; key result for production calibration
  measurement on small / mid datasets. arXiv:1909.10155.
- **`bootstrap_ci(method="studentized")`** — studentized bootstrap-t
  per Algeshiemer 2024 / Davison & Hinkley §5.2. Per-resample inner
  jackknife → pivot → CI = θ̂ − q · SE. Best CI coverage of any
  non-nested method at the cost of an extra factor-n compute.
- **`cv_clt_ci(fold_metrics, *, confidence=0.95)`** — standalone
  cross-validation CI per Bayle et al. 2020 Theorem 3.1 (Annals of
  Statistics). Caller supplies pre-computed per-fold metric estimates;
  this helper does NOT run the CV (gated on a separate v0.5 design
  conversation about fold strategy).
- **`MinHashLSHStrategy(n=3, num_perm=128, bands=16, seed=42)`** — 5th
  similarity strategy in `text_dedup`. Pure stdlib + numpy MinHash +
  LSH banding (Broder 1997 / Indyk-Motwani 1998); production-scale
  alternative to JaccardNgramStrategy. ~300 LOC of pure-numpy
  implementation; no datasketch dep.

### Added — testing infrastructure

- **`tests/test_plotting_visual.py`** + **`tests/baseline/`** —
  pytest-mpl visual-regression baselines for 7 plot helpers
  (plot_pr_curve, plot_reliability_diagram, plot_confusion_matrix_grid,
  plot_metric_bars, plot_score_histograms, plot_lift_ci,
  plot_bootstrap_distribution). Tolerance=15. Run with `pytest --mpl`.
- **`tox.ini`** + **`noxfile.py`** — local-runnable multi-Python test
  matrix (3.11/3.12/3.13). No GitHub Actions.

### Changed

- `bootstrap_ci` `method` Literal type extended from
  `Literal["BCa", "percentile"]` to
  `Literal["BCa", "percentile", "studentized"]`.

### Quality gates (2026-05-07)

- **352 tests passing** (was 329 in v0.3.0; +23: 4 ECE-debiased, 2
  studentized, 3 CV-CLT, 7 MinHash, 7 visual baselines).
- **15 doctests on math kernels** (was 13 in v0.3.0; +2 from the new
  ECE methods + cv_clt_ci).
- 95 symbols re-exported from top-level `eval_toolkit` (was 90 in v0.3.0;
  +5 new public symbols).
- ruff + black + mypy strict all clean.
- pytest-mpl visual baselines all match at tolerance=15.

### Deferred to v0.5

- Full CV orchestrator (the runner that produces fold_metrics for
  cv_clt_ci to consume) — gated on design discussion of fold strategy.
- ECE_SWEEP estimator (Roelofs 2022) — Monte-Carlo bias correction for
  L1 ECE; complementary to the L2 closed-form form already shipped.
- GitHub Actions CI workflow — tox/nox configs ship in v0.4 but actual
  CI matrix execution stays external until a public-release decision.

## [0.3.0] — 2026-05-07

Audit-driven correctness + methodology hardening release. Phase A produced
a literature-grounded research audit at `docs/v0.3_research_audit.md`
(696 lines + 33 per-method literature notes in
`~/Claude/research-kb/eval-toolkit-audit/`). Phase B locked the broad
v0.3.0 scope. Phase C executed across 12 per-category commits (this
release). 

### Added

- **`brier_score`** + **`brier_decomposition`** — Brier 1950 +
  Murphy 1973 reliability/resolution/uncertainty decomposition. Strictly
  proper scoring rule capturing what ECE alone misses.
- **`fit_beta_calibrator`** — Kull et al. 2017 Beta calibration; 3-parameter
  generalization of Platt that empirically dominates it on most real
  classifiers.
- **`CostMatrix.expected_cost(...)`** — composes the matrix's bayes_threshold
  with empirical FP/FN counts to evaluate cost-sensitive deployments.
- **`plot_bootstrap_distribution(deltas, *, ci_low=, ci_high=, ...)`** —
  histogram of bootstrap-sampled deltas with optional CI overlay; useful
  for diagnosing CI shape / normality assumption violations.
- **`fpr` and `fnr` keys** in `metrics_at_threshold` return dict
  (backwards-compatible).
- **`stratified_recall(..., *, with_ci=True, confidence=0.95)`** — opt-in
  Wilson scoring CI per stratum.
- **`set_global_seeds(..., *, strict_torch_determinism=False)`** — opt-in
  `torch.use_deterministic_algorithms(True)` matching PyTorch Lightning.
  Also sets `PYTHONHASHSEED` (with warning if pre-set).
- **`tests/test_reference_equivalence.py`** — 15 sklearn / scipy
  value-equality tests pushing the test methodology grade from B+ to A−
  (`pr_auc ≡ sklearn`, `roc_auc ≡ sklearn`, `bootstrap_ci ≡ scipy.stats.bootstrap`,
  `reliability_curve ≡ sklearn.calibration`, `fit_isotonic ≡ sklearn`,
  `fit_platt ≡ sklearn._SigmoidCalibration` post-rewrite).
- **`tests/strategies.py`** — consolidated Hypothesis strategies
  (`balanced_binary_array`, `score_array`) shared across property test
  files.

### Changed

- **`fit_platt_calibrator` rewritten to canonical Platt** (Platt 1999 §2.2
  + Lin et al. 2007). Source-verified to match
  `sklearn.calibration._SigmoidCalibration` to ~1e-6 on imbalanced data.
  Empirical delta vs v0.2.0: ~1–3% ECE on imbalanced (n ≥ 200) data.
  v0.2.0 wrapped `LogisticRegression(C=1)` which was sklearn-flavored
  logistic, NOT canonical Platt.
- **`bayes_optimal_threshold` docstring**: added qualifying paragraph
  distinguishing the prior-corrected formula from Elkan 2001 §4's
  prior-independent posterior-formula.
- **`fit_temperature_oracle`**: `.. warning::` admonition + runtime
  `warnings.warn(UserWarning)` flagging fit-on-test pitfall (Vaicenavicius
  2019, Kumar 2019, Roelofs 2022). Suite tests opt-in to suppression.
- **`save_figure`**: now supports `.pdf` and `.svg` in addition to `.png`
  (sidecar JSON written for all three; iTXt embedded metadata is
  PNG-only). Provenance JSON now built via `provenance.figure_metadata()`
  instead of inline dict.
- **`PALETTE` is now `Mapping[str, str]`** wrapped in
  `types.MappingProxyType`; mutation raises TypeError.
- **`Scorer` and `SliceAwareScorer` are `@runtime_checkable`** —
  `isinstance(obj, Scorer)` works.
- **`RunResult` is `frozen=True`** — `evaluate()` builds `by_slice`
  before construction.

### Fixed (P1 audit gaps)

- **ECE methods** (`expected_calibration_error` +
  `expected_calibration_error_equal_mass`) now reject scores outside
  `[0, 1]` with diagnostic message. Previously silently produced
  meaningless ECE on logits.
- **`metrics._validate_inputs`** now rejects NaN/Inf in `y_score`
  (harmonizes 7 metric helpers).
- **`fit_temperature`** now applies the same NaN/Inf + single-class
  validation pattern as peer calibrator fitters.
- **`EmbeddingCosineStrategy.pairs_across`** now rejects buggy embedders
  that return inconsistent feature dimensions for query vs reference.
- **`paired_bootstrap_diff`** now catches per-resample metric exceptions
  (single-class draws on rare-positive data); raises only if > 5%
  failure rate.
- **`select_threshold:205` docstring bug**: "smallest threshold" →
  "highest threshold" (returns the most precise operating point still
  meeting the recall floor).

### Breaking

- **Kwarg-only break across 6 signatures**: `bootstrap_ci`,
  `paired_bootstrap_diff`, `paired_bootstrap_op_point_diff`,
  `mde_from_ci`, `paired_mde`, and `harness.evaluate_scorer_on_slice` now
  require keyword arguments after the first positional break. Migration:
  use `bootstrap_ci(y, s, metric, n_resamples=1000, ...)` instead of
  `bootstrap_ci(y, s, metric, 1000, ...)`. sklearn made this exact change
  in 0.23 / 2020.
- **`BootstrapCI.to_dict()` JSON key rename**: `"mean"` →
  `"point_estimate"` to match the dataclass field. Trips downstream JSON
  consumers; `git grep '"mean"'` to find call sites.

### Quality gates (2026-05-07)

- 329 tests passing (was 290 in v0.2.0; +39 across new validation,
  Brier, expected_cost, Beta, plotting, sklearn-equivalence,
  strategies-refactor).
- 13 doctests on math kernels (was 9 in v0.2.0; +4 from Brier,
  brier_decomposition, expected_cost, fit_beta_calibrator).
- ruff + black + mypy strict all clean.
- Coverage maintained.
- 90 symbols re-exported from top-level `eval_toolkit` (was 89; +5 from
  new public symbols, −4 deduplicated).

### Deferred to v0.3.1 / v0.4.0

- pytest-mpl visual-regression baselines for plot helpers (audit gap
  #27; baseline-image generation workflow needs dedicated round)
- Bias-corrected ECE (Roelofs 2022 / Kumar 2019) as opt-in `estimator=`
  kwarg
- MinHashLSHStrategy (audit §3.4; gated on user signal)
- Studentized bootstrap-t (Algeshiemer 2024)
- CV-CLT (Bayle 2020)

## [0.2.0] — 2026-05-07

### Added

Pluggable similarity strategies for `text_dedup`. Different classification
projects encode different operational senses of "leakage" (lexical,
semantic, exact-after-normalization, set-based n-gram); the toolkit now
ships a `SimilarityStrategy` Protocol + four reference implementations so
users can plug in a project-specific strategy without forking.

- `SimilarityStrategy(Protocol)` — `pairs_within(texts, k)` and
  `pairs_across(query_texts, reference_texts, k)`. Runtime-checkable
  (`isinstance(obj, SimilarityStrategy)` works). Mirrors the existing
  `Scorer` Protocol pattern from `harness.py`.
- `TfidfCosineStrategy(ngram_range=(1,3), min_df=1, lowercase=True)` —
  default lexical near-dedup; bit-for-bit equivalent to v0.1.0 inline
  TF-IDF + cosine path.
- `ExactNormalizedHashStrategy(normalize=True)` — SHA-256-bucket
  exact-paraphrase dedup; similarities are exactly `{0.0, 1.0}`. Reuses
  `sha256_text` + `normalize_text_for_dedup`.
- `EmbeddingCosineStrategy(embedder)` — cosine on caller-supplied
  embeddings. Caller owns the embedder (sentence-transformers, OpenAI,
  local model) so the toolkit stays dep-free.
- `JaccardNgramStrategy(n=3, analyzer='char'|'word')` — set-based n-gram
  Jaccard; brute-force pairwise (O(n²)). Useful for token-order-invariant
  dedup (SQL fingerprints, CLI flags).

### Changed

- `near_dedup` and `cross_dedup` now accept a keyword-only
  `strategy: SimilarityStrategy | None = None` parameter. When `None`
  (default), behavior is bit-for-bit equivalent to v0.1.0; existing
  callers keep working unchanged.
- `near_dedup` orchestrator dispatches to `strategy.pairs_within(...)`
  instead of inlining `TfidfVectorizer` + `NearestNeighbors`. Forward-scan
  greedy-drop logic preserved.
- `cross_dedup` orchestrator dispatches to `strategy.pairs_across(...)`.
  `max_sim_per_eval` now from `sims.max(axis=1)` instead of
  `1 - distances.min(axis=1)` — equivalent for any strategy that returns
  the k nearest references per query.

### Tests

- 45 unit tests in `tests/test_text_dedup_strategies.py`: cross-strategy
  contract (Protocol conformance, shape, self-pair, determinism,
  empty-input edge case), strategy-specific behavior, plug-in contract
  (custom `_CountingStrategy` proves the user's strategy is dispatched).
- 24 Hypothesis property tests in `tests/test_text_dedup_props.py`,
  parameterized over all four strategies: partition invariant,
  idempotence, threshold monotonicity, `cross_dedup(X, X, t) == []`,
  `kept_indices` sorted.
- Existing `tests/test_text_dedup.py` (18 tests) passes unmodified —
  back-compat verified.

### Quality gates (2026-05-07)

- 290 tests passing (was 221 in v0.1.0; +69).
- 9 doctests passing on math/algorithmic kernels (was 5; +4 strategy
  class examples).
- ruff + black + mypy strict all clean.
- 89 symbols re-exported from top-level `eval_toolkit` (+5 strategy types
  vs. v0.1.0's 84).

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
