# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tier β #2 (RELEASING.md runbook, from v0.29.0 audit): extracted +
  expanded the release flow from CONTRIBUTING.md into a standalone
  `docs/RELEASING.md`. Documents the public-API snapshot regen
  gotcha (which bit v0.28.0), the PyPI CDN propagation delay we saw
  on v0.28.0 install verification, the GitHub Pages first-time-enable
  requirement, the v0.27.0 internal-milestone-tag conflict lesson,
  the pip-audit chicken-and-egg fix we just shipped in v0.28.1, and
  rollback policy. CONTRIBUTING.md gains a one-line pointer at the
  top of its release-flow section.

- Tier β #1 (repo hygiene, from v0.29.0 best-practice gap audit):
  added `.editorconfig` (charset, EOL, indent-per-extension rules)
  and `.github/CODEOWNERS` (default ownership: `@brandon-behring`).
  Improves contributor onboarding consistency across editors;
  CODEOWNERS unlocks GitHub's auto-reviewer assignment when external
  PRs arrive.

## [0.28.1] — 2026-05-15 — security-patch (CodeQL + pip-audit)

Tier α of the post-v0.28.0 best-practice gap audit. Pure CI/security
infrastructure additions; zero source-code or behavior changes.

### Added

- `.github/workflows/codeql.yml`: GitHub's CodeQL static analyzer
  on push/PR/weekly cron (Sundays 04:00 UTC). Uses the
  `security-extended` query suite. Findings populate the repo's
  Security → Code scanning tab.
- pip-audit step in the existing `test-base-install` CI job:
  scans the runtime-only venv (`numpy` / `scipy` / `scikit-learn` /
  `jsonschema`) for known CVEs on every PR. Fails CI on any finding.
  Dev-extras vulns (pytest, hypothesis, etc.) are not gated —
  surfaced through Dependabot. Per the v0.28.1 plan Q3=C
  (runtime-deps-only gate).

### Internal

- Audit discovered that `mypy --strict --no-implicit-reexport src/`
  already passes with zero issues on the v0.28.0 source. The
  planned Tier α #3 "chase remaining Any leaks" task was a no-op —
  no commit shipped for it.
- pip-audit on current runtime deps: zero known vulnerabilities.

## [0.28.0] — 2026-05-15 — temporalcv cross-pollination bundle

Six-section bundle adopting the highest-value patterns from the
sibling `temporalcv` project plus public-repo polish + hosted docs.
Major additions: `PurgedKFoldSplitter` for label-overlap-protected
cross-validation, nightly Monte Carlo bootstrap CI calibration
testing, 6-example documentation gallery, a hosted mkdocs-material
docs site with MathJax + tikzjax for full LaTeX + TikZ rendering,
SECURITY.md + CITATION.cff for public-repo polish, and a
documentary mutmut audit cataloguing math-kernel test strength.

### Added

- Section F (mutmut audit, from temporalcv-cross-pollination bundle):
  added `docs/internals/mutmut_audit.md` — documentary code-analysis
  audit of the 5 math kernel modules (`metrics`, `bootstrap`,
  `calibration`, `operating_points`, `thresholds`). Per Q10=A
  acceptance (audit-only, no kill-rate target), the deliverable is
  a catalog of likely surviving mutant patterns per module + an
  assessment of whether the existing test suite would catch them.
  Identifies 3 specific high-leverage gaps for future work:
  (a) calibration fit-vs-eval data isolation, (b) BCa degenerate-
  jackknife fallback assertion strengthening, (c) `empty_strategy`
  default lock-in tests. Programmatic mutmut run deferred: mutmut
  3.5.0 has a config-parsing bug in our env where `tests_dir =
  "tests/"` is splat character-by-character — revisit with mutmut
  v4 or cosmic-ray. Re-run instructions captured in the audit doc.

- Section E.2 (mkdocs link cleanup, from temporalcv-cross-pollination
  bundle): fixed 30+ broken relative links across 18 documentation
  files. Pattern: docs that link to `../src/eval_toolkit/<X>.py`
  (works on GitHub render but breaks in mkdocs) now point at the
  auto-generated API reference page (`api/<X>.md`). CHANGELOG.md
  references (also outside the docs tree) repointed to absolute
  GitHub URLs. Down from 93 warnings to 1: the remaining
  `griffe: π : float` is a documented tool limitation — griffe
  doesn't parse Unicode parameter names; the project's STYLE.md
  intentionally allows Unicode in math kernels (`π`, `α`, etc.).
  Also patched `harness.py` RunResult docstring: replaced the Sphinx
  `.. versionchanged::` directive with a NumPy "Notes" section so
  mkdocstrings renders it cleanly. `mkdocs build --strict` would
  fail on the 1 remaining griffe warning, so the docs.yml workflow
  intentionally runs without `--strict`. The link-cleanup deliverable
  is complete; the source-docstring + methodology enrichment passes
  originally scoped for E.2 are deferred (existing docstrings already
  carry References + LaTeX where it matters; methodology pages are
  already strong content-wise — only the link structure needed fixing).

- Section E.1 (hosted documentation site, from
  temporalcv-cross-pollination bundle): new mkdocs-material site at
  `https://brandon-behring.github.io/eval-toolkit/`, auto-generated
  from existing Markdown docs + `mkdocstrings`-rendered API reference.
  - `mkdocs.yml` configures the material theme (auto light/dark,
    tabs nav, code-copy buttons, full-text search) with MathJax v3 +
    tikzjax loaded from CDN for full LaTeX + TikZ rendering
    (per Q12=B).
  - `docs/index.md` — site landing page
  - `docs/api/index.md` — curated API landing organized by README's
    three-tier architecture (Tier 1 functional core, Tier 2 protocol
    orchestration, Tier 3 reproducibility scaffolding); per Q8=C.
  - `docs/api/<module>.md` — 22 per-module auto-gen stubs invoking
    `::: eval_toolkit.<module>` mkdocstrings directives.
  - `docs/javascripts/mathjax-config.js` — MathJax v3 init script
    matching mkdocs-material's pymdownx.arithmatex (generic: true).
  - `.github/workflows/docs.yml` deploys to GitHub Pages on every
    push to main + every tag push. Single-version site (no `mike`,
    per Q11=A).
  - `[docs]` optional extra added to `pyproject.toml` listing
    mkdocs-material, mkdocstrings[python], pymdown-extensions.
  - `pyproject.urls.Documentation` repointed at the hosted-docs URL.
  - README badge added: `Docs` linking to the GitHub Pages site.
  - `.gitignore` extended to exclude the mkdocs build output (`/site/`).
  - **Known follow-up**: 30+ relative-link warnings in
    `docs/methodology/*.md` files (links to `../../src/...` and
    `../../CHANGELOG.md`). Workflow temporarily runs without
    `--strict`; Section E.2 will fix these and re-enable strict mode.

- Section D (public-repo polish from temporalcv-cross-pollination bundle):
  added `SECURITY.md` (security disclosure policy with response SLAs,
  scope, and reporter-credit policy); added `CITATION.cff` (machine-
  readable academic citation metadata, exposing the GitHub web UI
  "Cite this repository" button — methodology-relevant primary
  references listed for `bootstrap_ci`, `brier_score`,
  `fit_platt_calibrator`, `delong_roc_variance`, `PurgedKFoldSplitter`).
  Added four trust-set badges to README (CI status, PyPI version,
  Python ≥3.13, License MIT). Extended `pyproject.urls` with a
  `Documentation` key pointing at `docs/getting-started.md` (the
  hosted-docs URL replaces this in Section E.1). Module-docstring
  audit across all 22 `src/eval_toolkit/*.py` modules — all already
  carry adequate module-level docstrings; no patches needed.

- Section C (example gallery from temporalcv-cross-pollination bundle):
  six new minimal worked examples in `docs/examples/`, each one
  concept per file, Sybil-validated end-to-end in CI:
  - `metrics_and_bootstrap.md` — `pr_auc` / `roc_auc` / `brier_score`
    + `bootstrap_ci` (BCa vs percentile)
  - `evaluate_harness.md` — slice-aware `evaluate(...)` with two
    scorers, `write_run_result(...)`, JSON schema validation
  - `calibration.md` — Platt + isotonic recalibration, ECE before/after
  - `leakage_detection.md` — `ExactDuplicateCheck` +
    `NormalizedFormLeakageCheck` + `LabelConflictCheck` on a
    contaminated train/test pair
  - `claims_and_gates.md` — `EvidenceGate` composition (metric
    threshold + minimum slice size) for release-decision gating
  - `paired_comparison.md` — `paired_bootstrap_diff` for two-scorer
    significance + `mde_from_ci` for power analysis
  - `index.md` — examples landing page mapping each example to the
    capability it demonstrates + the minimum extras required
  Total: 28 sybil-validated code blocks. Each is the headline-import
  → usable-output minimum surface; together they cover the public API
  surface a new user needs to be productive.

- Section B (PurgedKFold splitter from temporalcv-cross-pollination bundle):
  `PurgedKFoldSplitter(n_splits, purge_gap, embargo_pct, time_col)` and a
  standalone `compute_label_overlap(t_train, t_test, horizon)` helper, both
  now public via `from eval_toolkit import ...`. Time-aware k-fold with
  explicit purge gap straddling the test fold + post-test embargo —
  prevents label-window leakage when labels have a forward horizon
  (e.g., H-step forward returns). The standalone helper audits arbitrary
  train/test overlap independent of the splitter. Adapted from López de
  Prado (2018) Chapter 7 via temporalcv's `cv_financial.py`; API names
  preserved verbatim for cross-library muscle memory. Public-API
  drift-guard snapshot regenerated for the two new exports.

### Internal

- Section A (Monte Carlo bootstrap CI calibration, from temporalcv-cross-pollination
  bundle): added `tests/test_bootstrap_calibration_mc.py` (slow-marker) that runs
  500-replicate MC validation of `bootstrap_ci` coverage + bias across 5 cases
  (pr_auc / roc_auc × balanced / imbalanced × n=200 / n=1000 × BCa / percentile
  method). Asserts empirical coverage ∈ [0.90, 0.99] for nominal 95% CIs and
  |bias| < 0.05. Complements Tier 1's golden tests: goldens pin exact numerical
  output (drift detection), MC tests validate that the math is correct (a buggy
  implementation producing self-consistent wrong values fails MC but passes
  goldens). Also added CI width-scaling test (width should shrink as ~1/√n).
  New workflow `.github/workflows/nightly-mc.yml` triggers this suite weekly
  on Sundays at 03:00 UTC (plus `workflow_dispatch` for manual runs). Harness
  pattern adapted from temporalcv's `tests/conftest.py` Monte Carlo helpers.

- Test coverage (Tier 1 — math kernel correctness + integration backbone):
  added end-to-end pipeline tests (`tests/test_pipeline_e2e.py`) that
  exercise loader → `evaluate` → `write_run_result` → JSON schema
  validation for `DataFrameLoader` and `SingleSliceLoader` (incl.
  paired-diffs path). Extended `tests/test_metrics_props.py` with
  Brier-score bounds + label/score inversion symmetry properties. Added
  bootstrap CI golden tests (`tests/test_bootstrap_golden.py`, fixture at
  `tests/golden/bootstrap_ci/cases.json`) pinning BCa/percentile output
  on 6 canonical stress points (balanced, imbalanced 5%, small-n=10,
  tied scores) to ±1e-9. Expanded the `golden` pytest marker doc.

- Test coverage (Tier 3 — resilience moat): added multi-slice
  fault-injection tests (`tests/test_harness_fault_injection.py`) that
  exercise `on_scorer_error="record"` across three slices where the
  scorer succeeds on the middle one and fails on the outer two —
  asserts per-(slice, scorer) independence (no error-state bleed) plus
  a healthy-vs-faulting scorer parity check against a no-fault control.
  Added exactness tests for `TargetFPRSelector`
  (`tests/test_thresholds.py`): analytical answer on
  perfectly-separable data plus a golden-style pinned threshold value
  for a canonical (n=500, seed=42) overlapping distribution across
  target FPRs 0.01 / 0.05 / 0.10 / 0.20, with a monotonicity invariant.
  Added calibration determinism tests
  (`tests/test_calibration_determinism.py`): same `(y, score)` produces
  bit-identical Platt fit `a`/`b` parameters and isotonic transform
  output across runs, parametrized over 1% / 50% / 99% positive
  prevalence. Added NaN/+inf/-inf rejection tests for `pr_auc`,
  `roc_auc`, `brier_score` to `tests/test_metrics_props.py` —
  parametrized; locks the input-validation contract.

- Test coverage (Tier 2 — public-contract + integration breadth):
  added public-API drift guard (`tests/test_public_api.py`, fixture at
  `tests/golden/public_api/snapshot.json`) that snapshots all 199 names
  in `eval_toolkit.__all__` with signatures, class bases, first
  docstring lines, and primitive-value summaries. Drift now requires an
  explicit golden-regeneration commit. Extended
  `tests/test_pipeline_e2e.py` with `ParquetGlobLoader` round-trip
  (synthetic parquet → glob → load → evaluate → schema-validate; gated
  on `pyarrow`). Extended `tests/test_artifacts.py` with four manifest
  v2↔v3 dispatcher tests: v3 well-formed accepted; v3 missing
  `contamination_flags` rejected; v3 with unknown enum value rejected;
  v2 payloads still routed to v2 schema (no eager v3 demotion).

## [0.27.2] — 2026-05-15 — fix base-install pandas import

Base install of `eval-toolkit` (no extras) was broken in 0.27.1: every
attempt to `from eval_toolkit import evaluate` raised
`ModuleNotFoundError: No module named 'pandas'` because four modules
imported pandas at top level despite pandas being declared in the
`[dataframe]` optional extra. This patch restores the documented
contract: base install gives pure-numpy primitives; pandas is only
needed for the `[dataframe]` / `[parquet]` capabilities.

### Fixed

- `src/eval_toolkit/harness.py`: moved `import pandas as pd` under
  `if TYPE_CHECKING:` (annotation-only use at `EvalSlice.df`).
- `src/eval_toolkit/loaders.py`: moved top-level pandas import under
  `if TYPE_CHECKING:` for the `DataFrameLoader.df` annotation; added a
  function-local `import pandas as pd` inside
  `ParquetGlobLoader.load_splits` where `pd.read_parquet` / `pd.concat`
  are actually called. Calling the parquet path still requires pandas
  (via the `[parquet]` extra) — that's documented behavior, not a
  regression.
- `src/eval_toolkit/leakage.py`: removed dead top-level pandas import.
  Doctest examples already `>>> import pandas as pd` themselves.
- `src/eval_toolkit/splits.py`: removed dead top-level pandas import
  and unused `Sequence` import. Doctest example imports pandas itself.

### Added

- `.github/workflows/ci.yml`: new `test-base-install` job. Creates a
  fresh Py3.13 venv, `pip install .` with no extras, then exercises
  every public top-level import. Sanity check fails if pandas leaks
  into the base venv. Regression guard against this entire class of
  bug. Closes #13.

## [0.27.1] — 2026-05-15 — first PyPI release (Trusted Publishing)

Functionally identical to v0.27.0; bumped to 0.27.1 because the `v0.27.0`
git tag was already used as an internal milestone tag before PyPI publishing
infrastructure existed. The PyPI debut version is 0.27.1.

### Added

- `.github/workflows/publish.yml`: automated PyPI / TestPyPI publishing via
  OIDC Trusted Publishing (`pypa/gh-action-pypi-publish@release/v1`). Tag
  push trigger (`v*`), with PEP 440 prerelease tags (`v*rc*`, `v*a*`,
  `v*b*`, `v*dev*`) routing to TestPyPI and stable `vX.Y.Z` tags routing
  to PyPI. Sigstore attestations enabled. Workflow validates that the
  tag's base release matches `_version.py` and overrides the source
  version with the tag-derived prerelease version for prerelease builds.

### Changed

- `pyproject.toml`: switched from static `version = "X.Y.Z"` to
  `dynamic = ["version"]` with `[tool.hatch.version]` reading from
  `src/eval_toolkit/_version.py`. Eliminates the version-drift class by
  making `_version.py` the single source of truth — distribution
  metadata, `importlib.metadata.version('eval-toolkit')`, and the runtime
  `eval_toolkit.__version__` now agree by construction.
- `CONTRIBUTING.md` "Release flow" section rewritten to describe the
  tag-driven automation, the TestPyPI rehearsal step for prereleases,
  and the PyPI no-reupload rollback semantics (yank-and-bump).

## [0.27.0] — 2026-05-14 — harness decomposition + exception carve-outs

Internal refactor + small behavior change in error-recording. No public
API change. Output schema unchanged (the existing `additionalProperties:
true` on `by_slice` already accommodates the recorded-error fields).

### Internal

- Extracted two private helpers from `harness.py`:
  - `_resolve_y_score(scorer, slice_, precomputed_scores, *, on_scorer_error, attack_style)`
    — resolves `y_score` from the scorer or a precomputed array; returns
    the ndarray on success or the full error-dict on caught failure.
    Replaces the inline precomputed-shape-check + try/except block in
    `evaluate_scorer_on_slice` (prior lines 473–498).
  - `_compute_paired_diffs(slice_, scores_by_scorer, scorers, paired_diffs, *, n_resamples, seed)`
    — per-slice paired bootstrap on `pr_auc(b) - pr_auc(a)`; returns
    a dict keyed by `f"{b}_minus_{a}"`. Pure; no caches mutated.
    Replaces the inline paired-diff block in `evaluate()` (prior lines
    678–708).
- Net line count: `evaluate_scorer_on_slice` 143 → ~115; `evaluate`
  190 → ~150. Helpers were chosen for clean boundaries; other
  decomposition candidates (`_build_run_config`, `_run_leakage_phase`,
  `_score_slice`, `_merge_calibrated_metrics`) were deliberately
  skipped per STYLE.md §5 anti-overengineering (no second use site).

### Fixed

- `evaluate_scorer_on_slice(on_scorer_error="record")` no longer
  swallows `MemoryError` or `AssertionError`. Both now propagate even
  in `record` mode:
  - `MemoryError` signals an environment failure (OOM, resource
    exhaustion), not a scorer bug.
  - `AssertionError` signals an internal-invariant violation that
    should surface loudly.
  - Other exceptions (`RuntimeError`, `ValueError`, etc.) continue to
    be recorded under `record` mode as before. `KeyboardInterrupt`
    and `SystemExit` already propagated (they inherit from
    `BaseException`, not `Exception`).
- Docstring updated to document the new carve-outs.

### Added — tests

- `tests/test_harness_internals.py` (NEW): 5 invariant-focused tests
  (per /exploring-options Option 3: minimal coverage):
  - 2 parametrized skip-condition cases for `_compute_paired_diffs`
    (scorer skipped this slice; single-class slice).
  - 1 dedicated test for the n<30 skip-condition.
  - 2 regression-guards for the v0.27.0 exception carve-outs
    (`MemoryError` and `AssertionError` propagate in record mode).
  Happy paths for both helpers remain covered by existing public-API
  tests (`test_harness_v22.py`, `test_harness_v07.py`,
  `test_harness_smoke.py`).

## [0.26.0] — 2026-05-14 — test completeness

Builds on v0.25.1 (`Raises:` sweep). Test additions only; no behavior
or API change. Closes the gaps surfaced by the v0.26.0 toolkit-
completeness audit (research-grounded validation, error-path coverage,
cross-module integration). Pytest count: 1085 → 1161 (+76).

### Added — research-grounded test upgrades (Class B → A)

- `tests/test_calibration_research_grounded.py` (extended):
  Niculescu-Mizil & Caruana 2005 §4 — isotonic dominates Platt on
  saturated tree-ensemble-shaped scores; counter-test on smooth
  sigmoidal scores confirms the dominance is distribution-shape-
  dependent.
- `tests/test_bootstrap_research_grounded.py` (NEW): DiCiccio & Efron
  1996 §4 BCa transformation-respecting coverage on Beta(2, 5)
  skewed-mean fixture (100-seed loop); BCa coverage proximity vs
  percentile method; Bayle 2020 / Bates 2024 cv_clt_ci coverage
  validity on K=5 i.i.d. folds (200-seed loop) plus closed-form
  formula sanity check.
- `tests/test_thresholds_research_grounded.py` (NEW): Lipton, Elkan,
  Naryanaswamy 2014 Theorem 1 — F1-optimal threshold ≈ F1\*/2 on
  well-calibrated probabilities (10-seed parametrized + aggregate
  test); MaxF1Selector argmax sanity check.

### Added — error-path coverage (top-10 audit gaps)

- `tests/test_leakage_error_paths.py` (NEW): 8 KeyError tests
  covering every `raise KeyError` site in `leakage.py` (target_splits
  validation, CrossSplit train/eval missing, validate_label_split
  train/eval missing, GroupLeakage missing column, TemporalLeakage
  missing split + missing time_col).
- `tests/test_evidence_validators.py` (NEW): 3 dataclass
  `__post_init__` tests for empty-name / empty-value / empty-method
  ValueError raises.
- `tests/test_thresholds_constant_score.py` (NEW): explicit
  positive coverage that constant-score inputs do NOT raise on
  MaxF1Selector / YoudenJSelector + monkeypatch coverage of the
  defensive `len(thresholds) == 0` raises in
  `_pr_curve_trim` / `_roc_curve_trim`.
- `tests/test_calibration_optimization_failures.py` (NEW): 2
  monkeypatched tests for L-BFGS-B / minimize_scalar convergence
  failure paths in `fit_platt_calibrator` / `fit_temperature` plus
  positive controls.
- `tests/test_bootstrap_edge_cases.py` (NEW): 7 tests covering
  n<10 guards across `bootstrap_ci` / `paired_bootstrap_diff` /
  `paired_bootstrap_ece_diff`, shape-mismatch raises, the
  >5%-degenerate-resamples gate (using a strict-single-class
  metric helper), and confidence-out-of-(0,1) raises.
- `tests/test_metrics_stratified_subsets.py` (NEW): 5 tests on
  `quantile_stratified_pr_auc` (shape mismatch, invalid bounds,
  empty window via NaN stratifier, imbalanced subset, positive
  control).
- `tests/test_harness_v22.py` (extended): 1 new test exercising the
  full v0.22 kwarg cross-product (calibrator + bootstrap_roc_auc +
  fpr_ladder + compute_mce + compute_brier + attack_style
  + precomputed_scores).
- `tests/test_manifest_contamination_round_trip.py` (NEW): 6 tests
  pinning `RunManifest → JSON → validate_manifest → reload` cycle
  for every contamination_flags enum value, plus invalid-value
  rejection at build time and multi-scorer round-trip.
- `tests/test_numeric_edge_cases.py` (NEW): 9 tests across
  metrics / bootstrap / calibration covering n=1, constant
  y_score, np.int32 / np.float32 dtypes, and Python list inputs.

### Added — cross-module integration tests

- `tests/test_splits_leakage_integration.py` (NEW): 4 tests verifying
  `StratifiedKFoldSplitter` output produces no false-positive
  findings under `ExactDuplicateCheck` / `CrossSplitLeakageCheck` /
  `NormalizedFormLeakageCheck` on a clean unique-text corpus, plus
  a positive control with deliberate duplication.
- `tests/test_dedup_split_leakage_chain.py` (NEW): 3 tests on the
  end-to-end dedup → stratify → cross-split-leakage chain
  (TF-IDF and MinHashLSH backends), plus undedup positive control.
- `tests/test_calibration_bootstrap_chain.py` (NEW): 2 tests on the
  uncalibrated → fit-Platt → calibrated bootstrap-CI workflow,
  asserting calibrated point ECE drops below uncalibrated in
  ≥ 70% of 50 seeds (per the v0.25.0 flake-mitigation policy)
  plus a single-seed point-comparison sanity check.

### Notes on plan deviations

- The originally-planned "CV-CLT-CI width-dominance over naive
  percentile bootstrap" test was restructured to a coverage test
  after inspection found `cv_clt_ci` uses the standard `mean ±
  z·σ/√K` formula (Bayle 2020 proves its asymptotic validity, but
  doesn't add a width-improvement correction). The right test of
  the claim is coverage, not width.
- The originally-planned "MaxF1Selector / YoudenJSelector raise on
  constant y_score" test was restructured after empirical
  investigation found sklearn 1.x returns at least one threshold on
  constant input; the selectors handle constant scores without
  raising. The defensive `len(thresholds) == 0` raises are pinned
  via monkeypatch instead.
- The Niculescu-Mizil 2005 counter-test tolerance was loosened to
  0.02 (above the small-n calibration-noise floor); a strict
  flip-direction counter-test would require n ≳ 1000 calibration
  data per Niculescu-Mizil 2005 §5, which would balloon the test
  runtime.

## [0.25.1] — 2026-05-14 — docstring `Raises:` sweep

Builds on v0.25.0. Docs-only patch; no behavior or API change. Surfaced
by a three-agent toolkit-completeness audit (code-quality /
research-to-code mapping / test-coverage) that found 44 public
functions raising exceptions without a corresponding `Raises:` section
in their docstrings — the only systematic code-quality issue across
the 13,733-LOC package.

### Docs

- Added `Raises:` sections to **44 public functions** spanning
  `analysis.py`, `bootstrap.py`, `calibration.py`, `claims.py`,
  `docs.py`, `harness.py`, `leakage.py`, `loaders.py`, `manifest.py`,
  `metrics.py`, `operating_points.py`, `plotting.py`, `seeds.py`,
  `splits.py`, `text_dedup.py`, and `thresholds.py`.
- Added `scripts/audit_raises_sections.py` — an AST-based check that
  walks every public function, compares its `raise` sites against the
  NumPy-style `Raises:` block, and prints mismatches. Used to drive
  the v0.25.1 sweep; retained for future regression prevention.

### Tests

- `tests/test_seeds.py:54` — replaced placeholder
  `assert True  # smoke check` with a real assertion that
  `set_global_seeds(0)` actually seeds the numpy global RNG (compares
  `np.random.rand(5)` outputs across two seeded calls).
- `tests/conftest.py` — added 4 shared input fixtures
  (`balanced_binary_inputs`, `imbalanced_binary_inputs`,
  `single_class_inputs`, `constant_score_inputs`) defining the common
  scaffolds copy-pasted across 15+ test files. Adoption rollout
  deferred to v0.26.0 to keep this release docs-only.

## [0.25.0] — 2026-05-14 — research-grounded test additions

Builds on v0.24.1 (research-dossier docs hygiene) by validating
implementations against the methodology cited in the dossier. Tests
only — no API or schema changes.

### Added

- `tests/test_calibration_research_grounded.py` (new file, ~250 LOC):
  - **Beta dominates Platt on miscalibrated fixture** (Kull et al.
    2017 §5; `inference/_dossier/` § C1 entry `kull2017beta`).
    Asymmetric mixture fixture where Platt's symmetric sigmoid
    cannot apply correction in only one tail; Beta's 3-parameter
    log-feature form can. Seed-loop guard rails: 12 seeds, dominance
    in ≥ 9/12, margin > 0.5σ.
  - **ECE plug-in upward bias on miscalibrated small-n** (Roelofs
    2022 + Kumar 2019; `inference/` § D2 entries
    `roelofs2022mitigating`, `kumar2019verified`). Plug-in ECE
    over-estimates the debiased variant on small-n miscalibrated
    samples; bias amplified by miscalibration vs. calibrated
    counter-test.
- `tests/test_text_dedup_strategies.py` (extended, +120 LOC):
  - **MinHashLSH approximation bound vs exact Jaccard** (Broder
    1997 + Indyk-Motwani 1998; `data-integrity/` § C1 entries
    `broder1997minhash`, `indyk1998lsh`). Per-pair MinHash estimates
    within Hoeffding 95% bound ε = 1.96 / √num_perm of exact Jaccard
    for ≥ 85% of pairs above the LSH band-curve flip threshold.
- `tests/test_reproducibility_integration.py` (new file, ~180 LOC):
  - **End-to-end harness output bit-identity under replay** (Pineau
    et al. 2021; `eval-ecosystem/` § B1 entry `pineau2021reproducibility`).
    Same-seed re-runs of `evaluate_scorer_on_slice` produce
    `np.array_equal`-identical metric dicts (with `equal_nan=True` for
    degenerate-CI NaN preservation). Negative control: different seed
    produces different output. Cross-call isolation: bit-identity
    holds even when sibling harness calls happen between replays.
- `tests/test_leakage.py` (extended, +90 LOC):
  - **Kapoor 2023 L2 (illegitimate features) — partial coverage**
    via `LabelConflictCheck`. Tests the same-text-conflicting-labels
    sub-case of L2; explicit docstring caveat that this does NOT
    cover general illegitimate-feature detection.

### Deferred (Kapoor 2023 leakage taxonomy gaps)

Per Phase-0 audit of `src/eval_toolkit/leakage.py`, the following
Kapoor 2023 leaf-level leakage modes lack detector machinery in
v0.25.0 and are deferred to a future release:

- **L1.2** preprocessing on combined train+test data — requires
  new `PreprocessingLeakageCheck` (detect StandardScaler / similar
  fit on combined data via statistics-drift signature).
- **L1.3** feature selection on combined train+test — requires
  new `FeatureSelectionLeakageCheck` (detect SelectKBest / similar
  fit on combined data via rank-drift signature).
- **L3.3** sampling bias (different distributions train vs. test) —
  requires new `SamplingBiasCheck` (KS-test or similar
  distribution-shift detector).
- **L2-general** illegitimate features (post-prediction sources,
  target-derived aggregates) — requires generalized
  `IllegitimateFeatureCheck` beyond `LabelConflictCheck`'s
  same-text-conflict sub-case.

Reference: Kapoor & Narayanan, "Leakage and the reproducibility
crisis in ML-based science," Patterns 4(9), 2023; arXiv:2207.07048;
Table 2 (8-leaf taxonomy).

## [0.24.1] — 2026-05-14 — research-dossier docs hygiene

Builds on v0.24.0 (manifest.v3 + contamination_flags). No code surface
changes; docs-only.

### Docs

- `docs/research/README.md`: new "Scope: which clusters inform code vs. consumer repos" section making the intentional split explicit — `inference/` and `data-integrity/` clusters map to library code (bootstrap, calibration, splits, leakage, dedup, manifest); `prompt-injection/` is reference material for downstream consumer repos (e.g., `prompt-injection-v4`); `datasets/` is mixed.
- `docs/research/`: add 5 entries previously flagged by `RECONCILIATION.md` as gaps:
  - **Yan et al. 2025** (`yan2025timeseries`) — *Hidden Leaks in Time Series Forecasting* — `data-integrity/` § A1.
  - **Pellizzoni et al. 2025** (`pellizzoni2025leakage`) — *Don't push the button! Data leakage risks in ML and transfer learning* — `data-integrity/` § B1.
  - **HackAPrompt × SQuAD 2025** (`hackaprompt_squad_2025`) — naive-dedup detection for PI benchmarks — `prompt-injection/` § C1.
  - **DataSentinel + PromptLocate 2025** (`datasentinel_promptlocate_2025`) — strict-normalization PI defense — `prompt-injection/` § B1.
  - **Open-Prompt-Injection (Liu et al.)** (`open_prompt_injection_liu`) — attack-prompt dataset — `datasets/dataset_ledger.yml`.
- `docs/research/papers/data-integrity/02_leakage_and_contamination.md`: cross-link the Sainz 2023 § B2 entry to `RunManifest.contamination_flags` (manifest.v3, v0.24.0) — eval-toolkit's response to the per-benchmark contamination-disclosure norm; closes V4 audit issue A6.
- `docs/research/RECONCILIATION.md`: append v0.24.1 status block documenting which gap-entries were absorbed.
- Entry counts: 69 → 74 (data-integrity 15 → 17, prompt-injection 10 → 12, datasets 11 → 12).

## [0.24.0] — 2026-05-14 — manifest.v3 + contamination_flags (V4.4 D7)

Closes V4 audit issue A6 (contamination flag docstring-only) by promoting
per-scorer contamination posture to a required manifest field. Drives V4.4's
audit closure phase before V5 paper-canonical. Lands on top of v0.23.0's
Python 3.13 floor.

### Added

- `schemas/manifest.v3.json` — new schema with required `contamination_flags`
  field (object mapping scorer-name → enum string of `verified_disjoint`,
  `suspected_contamination`, `vendor_black_box`, `unknown`). May be an empty
  object for runs not tracking contamination.
- `RunManifest.contamination_flags: dict[str, str]` field on the dataclass.
- `build_manifest(..., contamination_flags=...)` parameter; validates enum
  membership at build time (raises `ValueError` for invalid values).
- v3 schema permits `guardrails` entries to be either strings (v2 back-compat)
  or non-empty objects (forward-compatibility for structured sub-fields like
  V4.4 D5's `source_freshness_check`).

### Changed

- `MANIFEST_SCHEMA_VERSION` constant flipped from `"v2"` to `"v3"`. New
  manifests written by `build_manifest()` default to schema_version `"v3"`.
  Callers that need v2 explicitly should construct `RunManifest(...,
  schema_version="v2")` directly.
- `validate_manifest()` knows `"v3"` in `_KNOWN_MANIFEST_VERSIONS`; existing
  v1/v2 manifests continue to validate against their declared schema.
- `guardrails` field on `RunManifest` typed as `list[object]` to permit
  dict entries; build-time validation rejects empty strings AND empty dicts.

## [0.23.0] — 2026-05-14 — Python 3.13+ floor; restore green CI

### BREAKING

- **Python floor raised to `>=3.13`** (was `>=3.11,<3.14`). Consumers
  on Python 3.11 or 3.12 must either upgrade to 3.13+ or pin
  `eval-toolkit<0.23`. The upper bound is now open — 3.14 and later
  are nominally allowed but not yet smoke-tested in CI.
- `[tool.{black,ruff,mypy}]` `target-version` / `python_version` all
  raised to `py313` / `3.13`. Tooling now emits 3.13-only idiom
  suggestions (`datetime.UTC` over `datetime.timezone.utc`, etc.).
- Trove classifiers `Programming Language :: Python :: 3.11` and
  `… 3.12` dropped.

### Changed

- CI matrix slimmed: `ubuntu-latest` + `macos-latest` + `windows-latest`,
  all at Python 3.13. Drops the prior 3.11 / 3.12 Ubuntu jobs (the
  cross-OS coverage added in v0.10.0 is preserved). Removes the
  `include:` block from `.github/workflows/ci.yml`.
- `eval_toolkit.config`: `frozen_config` and `from_yaml` migrated to
  PEP 695 type-parameter syntax (`def frozen_config[T](...)`). Drops
  the module-level `TypeVar("T")` declaration. Caller-visible behavior
  unchanged; surfaces a cleaner generic signature in IDE hover / docs.

### Fixed

- Restore green CI on `main`. Five ruff lint violations accumulated
  across v0.13.0 → v0.22.0 were blocking the CI lint step (which
  fast-fails before black / mypy / pytest):
  - `bootstrap.py:1285` (C401) — set generator → set comprehension.
  - `manifest.py:256` (SIM105) — `try / except ValueError: pass` →
    `contextlib.suppress(ValueError)`.
  - `manifest.py:372` (UP017) — `_dt.timezone.utc` → `_dt.UTC`.
  - `plotting.py:27` (F401) — drop unused `typing.Literal` import.
  - `tests/test_harness_v22.py:32` (F841) — drop dead `rng` (the
    fixture's docstring incorrectly claimed gaussian scores; tests
    inject scores per-case via `np.linspace`). Docstring tightened.
- Mechanical black 26 reformat across 12 files (line re-wrapping +
  string quote normalization). Pure formatting; no semantic changes.

### Added

- `docs/repo-strategy.md` — repo organization strategy document.
  Captures the v0.10.0 dependency-graph audit, the 6-bucket
  in-place reorganization (slipped past v0.11.0 to a later minor;
  re-target TBD), the 4-question machine-checkable checklist for
  "should we extract sub-package X?", and the audit cadence
  (every 3 minor releases; next at v0.13.0). Linked from
  `README.md` Documentation section.

### Migration notes for downstream consumers

Consumers pinning `eval-toolkit>=0.22,<0.23` on Python 3.11 / 3.12:

1. Bump the Python floor of the consumer to `>=3.13` (or stay on
   `eval-toolkit<0.23` and accept that v0.22 won't receive backports).
2. Re-run `uv sync` / `pip install -e .` — the lockfile resolves to
   `eval-toolkit==0.23.0` once `requires-python` is compatible.
3. No Python API changes — all v0.22.x callsites work unchanged.

## [0.22.1] — 2026-05-14 — agent-grounding research dossier

### Added

- `docs/research/` — 52-file research dossier (~304K) covering inference,
  data integrity, eval-ecosystem, prompt-injection, and datasets. 69
  primary-source entries (all `status: verified` after six audit rounds);
  80 URLs HEAD-checked (0 broken, 16 paywall-blocked, 64 OK). Top-level
  cross-cluster index in `docs/research/README.md`; URL health in
  `docs/research/url-freshness-report.md`; gap-analysis against the
  existing `docs/methodology/reading_list.md` in
  `docs/research/RECONCILIATION.md` (reading_list.md unchanged).

## [0.22.0] — 2026-05-13 — expanded `evaluate_scorer_on_slice` (C11 / F8.1)

Closes F8.1 from the V4 consumer feedback log. The harness aggregator gains
six additive kwargs so a single delegate call can replace V4's bespoke
per-(scorer, slice, style) evaluation wrapper. Same metric primitives
underneath (`headline_metrics`, `pr_auc`, `roc_auc`, `brier_score`,
`maximum_calibration_error`, `TargetFPRSelector`); the value-add is
composing them once with a clean kwarg surface.

**Backward-compatible**: all v0.22 kwargs default to no-op shapes. Existing
callers (pre-v0.22 surface) get an unchanged result dict.

### Added

- `evaluate_scorer_on_slice(..., precomputed_scores: np.ndarray | None = None)` —
  skip `scorer.predict_proba` when set. Validates shape matches the slice.
- `evaluate_scorer_on_slice(..., attack_style: str | None = None)` — pass-through
  label; lands in the result dict under `"attack_style"`. No metric effect.
  Threads through the error-recording path too.
- `evaluate_scorer_on_slice(..., fpr_ladder: list[float] | None = None)` —
  emit `tpr_at_fpr: {str(fpr): tpr | None}` via :class:`TargetFPRSelector`.
- `evaluate_scorer_on_slice(..., compute_mce: bool = False)` — emit `mce`
  via :func:`maximum_calibration_error`.
- `evaluate_scorer_on_slice(..., compute_brier: bool = False)` — emit
  `brier_score` via :func:`brier_score` (`empty_strategy="return_none"`).
- `evaluate_scorer_on_slice(..., calibrator: PlattFit | None = None)` —
  apply to `y_score`, recompute every requested metric on the calibrated
  scores, merge under `*_calibrated` keys.
- `evaluate_scorer_on_slice(..., bootstrap_roc_auc: bool = False)` — when
  True (and `n_resamples > 0` and mixed-class), also bootstrap ROC-AUC CI;
  emitted under `roc_auc_ci`.
- Private helper `harness._evaluate_scores(y_true, y_score, ...)` carries
  the metric-block construction so the calibrator path can reuse it.

### Tests

- New `tests/test_harness_v22.py`: 9 unit tests covering each kwarg
  (precomputed fast-path, shape mismatch, attack_style pass-through +
  error-path threading, fpr_ladder dict shape, compute_mce / compute_brier
  on-off, bootstrap_roc_auc CI emission, calibrator emits `*_calibrated`
  block, full back-compat of pre-v0.22 surface).

## [0.21.1] — 2026-05-13 — relocate `reliability_diagram_data` to `calibration`

Patch release. The v0.21.0 introduction placed `reliability_diagram_data`
in `plotting.py`, which imports matplotlib at module load. Consumers
without matplotlib (notably the prompt-injection-v4 base venv) could not
import the helper. The function is pure data preparation (no plotting),
so the right home is `calibration.py` next to `reliability_curve`.

### Changed

- `eval_toolkit.calibration.reliability_diagram_data` (was in `plotting`).
- `__init__` lazy-export updated to point at `calibration`.

No API change for matplotlib-equipped callers: same name, same shape,
same docstring.

## [0.21.0] — 2026-05-13 — `reliability_diagram_data` structured rows

Adds the structured-bin emitter that V4 (and other downstream consumers)
needs for serializing reliability data to parquet / JSON without
re-implementing the bin-edge reshape. Companion to the existing
:func:`plot_reliability_diagram` figure renderer.

### Added

- `eval_toolkit.calibration.reliability_diagram_data(y_true, y_score, *,
  n_bins=10, strategy="quantile") -> list[dict]`. Schema per row:
  `bin_lower`, `bin_upper`, `mean_pred`, `frac_positive`, `n`. Returns
  `[]` for degenerate slices (single-class / empty). Wraps
  :func:`eval_toolkit.calibration.reliability_curve`. Lives in
  ``calibration`` (not ``plotting``) so consumers that don't pull in
  matplotlib can still import it.

## [0.20.0] — 2026-05-13 — DeLong correlated-ROC variance

Adds DeLong's correlated-ROC ΔAUC test as a Phase 4 prep deliverable
(plan §C12). Companion to ``paired_bootstrap_diff`` for cases where a
fast closed-form variance is preferred over bootstrap resampling.

### Added

- `eval_toolkit.bootstrap.DeLongResult` — frozen dataclass with
  `auc_a`, `auc_b`, `delta_auc`, `var`, `z`, `p_value`, `ci_low`,
  `ci_high`. ``to_dict()`` for manifest serialization.
- `eval_toolkit.bootstrap.delong_roc_variance(y_true, y_score_a,
  y_score_b) -> DeLongResult` — Sun & Xu 2014 fast implementation
  (midrank-based; ties handled). Returns 95% normal-approx CI on the
  delta plus two-sided z and p-value.

### Tests

- 5 new unit tests: result shape, AUC matches `sklearn.roc_auc_score`
  within 1e-8, empty-class rejection, shape-mismatch rejection,
  large-effect p-value < 0.001.

## [0.19.0] — 2026-05-13 — `PoolBuilder` Protocol + `iter_folds_with_pool`

Closes F7.1 from the V4 consumer feedback log: the `Splitter` Protocol
alone could not express domain-specific train-pool augmentation (the
canonical V4 pattern is "fold's positives + a stable benign pool, with
a stratified val carve"). Splitting the responsibility into a
``Splitter`` (rotates rows) + ``PoolBuilder`` (augments train, carves
val) lets generic CV machinery compose with research-specific pool
semantics.

### Added

- `eval_toolkit.splits.PoolBuilder` — runtime-checkable Protocol with
  `build(train: EvalSlice, *, fold_idx: int) -> dict[str, EvalSlice]`.
  Implementations carry pool state in instance attributes (so the
  composition helper can configure once outside the fold loop).
- `eval_toolkit.splits.iter_folds_with_pool(splitter, slice_, *,
  pool_builder, groups=None) -> Iterator[dict[str, EvalSlice]]` — yields
  per-fold dicts combining the PoolBuilder's `train`/`val` with the
  Splitter's `test`. Additional keys returned by the PoolBuilder are
  forwarded verbatim. Validates the contract: PoolBuilder must return
  at least `{"train", "val"}`.

### Tests

- 2 new unit tests covering the happy path (composition yields
  `{train, val, test}`) and the contract violation (PoolBuilder missing
  `val` raises).
- 1 doctest in `iter_folds_with_pool` exercising the trivial pool
  pattern.

## [0.18.0] — 2026-05-13 — `LeakageFinding.drop_indices` optional

Closes F6.2 from the V4 consumer feedback log. ``LeakageFinding.drop_indices``
becomes ``dict[str, list[int]] | None``: ``None`` signals a
pair-tally audit (the check found leakage but did not localize rows to
drop), while ``{}`` still means "this check found nothing".

V4's pre-C9 emitters wrote ``drop_indices={}`` for pair-count findings
(SHA256 overlap counts, label-aware near-dup tallies), which is
ambiguous against "the check ran and found no rows to drop." ``None``
makes the distinction explicit.

### Changed

- `LeakageFinding.drop_indices: dict[str, list[int]] | None`.
- `LeakageFinding.to_dict()` emits `null` when `drop_indices` is `None`
  (RFC 8259 strict JSON; ``write_json_strict`` already handles ``None``
  -> ``null``).
- `manifest.v2.json`: `leakage_report.findings[*].drop_indices` is now
  `{"type": ["object", "null"]}`. ``manifest.v1.json`` kept unchanged
  (legacy reruns still require an object).

## [0.17.0] — 2026-05-13 — `label_aware` leakage findings

Closes F6.1 from the V4 consumer feedback log. ``NearDuplicateCheck``
and ``CrossSplitLeakageCheck`` gain a ``label_aware: bool = False``
field and a new ``validate_label_split(splits) -> tuple[LeakageFinding,
LeakageFinding]`` method that decomposes near-duplicate hits into
same-label and cross-label findings with independent severities.

**Backward-compatible**: the existing ``validate(splits) ->
LeakageFinding`` single-finding contract is preserved. Callers
explicitly opt into the dual emission by invoking
``validate_label_split``.

Motivation: within-split near-duplicate pairs that share a label are a
mild label-noise signal; pairs with opposing labels are conflicting
supervision (catastrophic for training). Cross-split near-duplicates
that share a label are a memorization signal; opposing-label matches
are memorization + supervision conflict. Treating the two cases
identically (single severity) loses signal.

### Added

- `NearDuplicateCheck.label_aware: bool` (default `False`),
  `severity_same_label: Severity` (default `"warning"`),
  `severity_cross_label: Severity` (default `"error"`).
- `NearDuplicateCheck.validate_label_split(splits) -> tuple[
  LeakageFinding, LeakageFinding]` — emits `(same_label, cross_label)`
  findings. `check_name` becomes `"NearDuplicateCheck.same_label"` /
  `"NearDuplicateCheck.cross_label"`; evidence carries `label_polarity`.
- `CrossSplitLeakageCheck.label_aware: bool` (default `False`),
  `severity_same_label`, `severity_cross_label` — same shape.
- `CrossSplitLeakageCheck.validate_label_split(splits)` — same dual
  emission, splitting by matched train neighbor's label.
- `eval_toolkit.text_dedup.cross_dedup_pairs(train_texts, eval_texts,
  threshold, k_neighbors, *, strategy) -> list[tuple[int, int, float]]`
  — exposes the `(eval_idx, train_idx, similarity)` tuples backing
  `cross_dedup`. Required for the label-aware cross-split decomposition.

### Tests

- 5 new unit tests covering the dual emission (same/cross-label
  separation), default severities, single-finding back-compat, and
  `cross_dedup_pairs` shape.

## [0.16.0] — 2026-05-13 — `jsonschema` promoted to a hard dependency

Closes F9.1 from the V4 consumer feedback log. Schema validation is the
NeurIPS-aligned manifest contract, not an optional polish; consumers
should not need to install ``eval-toolkit[validation]`` to call
``validate_manifest`` / ``validate_results`` / ``validate_payload``.

### Changed

- `jsonschema>=4.21` moved from `[project.optional-dependencies.validation]`
  to base `[project.dependencies]`.
- `[project.optional-dependencies.validation]` is now an empty list,
  preserved as a transitive no-op so existing
  `pip install eval-toolkit[validation]` invocations keep resolving. The
  empty extra may be removed in a future minor after a deprecation
  window.
- `validate_payload` and `validate_prediction_artifact_ref` drop their
  try/except `ImportError` ladders around `from jsonschema import
  Draft202012Validator`. The "install the optional extra" `ImportError`
  branch is gone.

### Tests

- Pre-existing `pytest.importorskip("jsonschema")` calls become harmless
  no-ops; left in place for defensive intent.

## [0.15.0] — 2026-05-13 — FileHash sentinel + PredictionArtifactRef.role union

Closes F5.1 (file_sha256 None ambiguity) and F5.2 (PredictionArtifactRef.role
single-string) from the V4 consumer feedback log.

**Backward-compatible**: existing string-role and `file_sha256` callers
keep working unchanged.

### Added

- `eval_toolkit.provenance.FileHash(sha256: str)` and
  `eval_toolkit.provenance.FileHashMissing(reason: str, path: str)` —
  frozen dataclasses for sentinel-style file-hash results.
- `eval_toolkit.provenance.compute_file_hash(path) -> FileHash |
  FileHashMissing` — the sentinel-returning helper. Pattern-match on
  the union members instead of testing for `None`. Missing files emit
  `FileHashMissing(reason="missing")`; non-file paths (directories)
  emit `FileHashMissing(reason="not_a_file")`.
- `PredictionArtifactRef.role` now accepts `str | list[str]`. Single-
  artifact references that span multiple slices / fold-roles can list
  them explicitly instead of synthesizing a single-string role and a
  parallel `metadata["slices"]` array.
- `manifest.v2.json` schema patches `prediction_artifacts[*].role` to
  the same union via `oneOf`. Inline schema in
  `validate_prediction_artifact_ref` mirrors the patch.

### Changed

- `file_sha256(path, strict=False) -> str | None` is now a thin wrapper
  over `compute_file_hash`. Behavior is preserved (returns string digest
  on hit, `None` on miss when not strict, `FileNotFoundError` on miss
  when strict). Docstring marks it as legacy-but-supported and points
  to `compute_file_hash` for new code.
- `eval_toolkit.manifest.build_manifest` internal `data_hashes` builder
  pattern-matches on `compute_file_hash` instead of testing
  `file_sha256() is not None`. Output JSON is byte-equivalent.

### Tests

- 5 new unit tests for `compute_file_hash` (success, missing, directory,
  short-digest validator, back-compat with `file_sha256`).
- 5 new unit tests for `PredictionArtifactRef.role` (list path,
  back-compat string path, rejection paths for empty list / empty
  entries, validate-helper accepts list).

## [0.14.2] — 2026-05-13 — relaxed source_roles uniqueness ((source, role) pair)

Closes F4.5 from the V4 consumer feedback log: ``validate_source_roles``
relaxes its uniqueness contract from "unique ``source``" to
"unique ``(source, role)`` pair". The same upstream source may now appear
multiple times as long as each occurrence carries a distinct role.

This unlocks the natural usage pattern of one dataset feeding multiple
slices (training pool + OOD diagnostic, calibration + locked eval, etc.)
without consumers having to synthesize per-slice source names just to
satisfy the validator.

### Changed

- `eval_toolkit.manifest.validate_source_roles` — uniqueness is now on
  the `(source, role)` pair. Error message wording changes from
  ``"duplicate source"`` to ``"duplicate (source, role) pair"``.
- `manifest.v2.json` — no schema change required (uniqueness was always
  a producer-side contract, not a schema constraint).

### Tests

- 1 new unit test covering the relaxed contract (same source + different
  roles passes).
- 1 renamed test (`flags_duplicate_source_role_pair`) covering the
  duplicate-pair detection. The pre-v0.14.2 test was
  `flags_duplicate_source` and asserted on a (source) pair only.

## [0.14.1] — 2026-05-13 — explicit `git_sha` kwarg on `build_manifest`

Patch release adding an opt-in override to bypass
:func:`capture_git_sha`. Closes F10.1 from the V4 consumer feedback log:
pods / CI runners that rsync source without ``.git/`` no longer need to
post-mutate ``manifest_payload["git_sha"]`` after building.

### Added

- `build_manifest(..., git_sha=...)` — explicit kwarg. When ``None``
  (default), behavior is unchanged: ``capture_git_sha(repo_root)`` runs.
  When provided, the kwarg is used directly as
  :attr:`RunManifest.git_sha`.

### Tests

- 2 new unit tests covering the explicit-override path and the
  fall-back-to-capture path.

## [0.14.0] — 2026-05-13 — manifest.v2 + typed validate_* helpers

Manifest schema migrates v1 → v2 with three additive structural fixes and
two new typed validation entry points. Closes F4.2 (auto-captured
timestamp), F4.3 (`gpu_info.memory_gb` typing), F4.4
(`extra_code_versions` split), and F9.2 (typed schema-validation helpers)
from the V4 consumer feedback log.

**Forward-compat**: `manifest.v1.json` ships unchanged. `validate_manifest`
dispatches on `payload["schema_version"]`, so legacy V4.2-era manifests
revalidate against v1; new manifests produced by `build_manifest()` carry
`schema_version="v2"`.

### Added

- `manifest.v2.json` — new top-level fields `captured_at` (ISO-8601 UTC,
  required), `data_revisions: object<string,string>`, `metadata:
  object<string,string>`. `gpu_info.count` tightened from `string` to
  `integer`; `gpu_info.memory_gb` from `string` to `number`. Other v1
  fields are byte-equivalent.
- `RunManifest.captured_at: str` — auto-populated by `build_manifest()`
  with the current UTC time. Separates wall-clock capture from caller-
  meaningful `run_id`.
- `RunManifest.data_revisions: dict[str, str]` and
  `RunManifest.metadata: dict[str, str]` — replaces the v1 pattern of
  cramming dataset/model revisions and run-time labels into
  `code_versions` under `hf_dataset:` / `hf_model:` / `meta:` prefixes.
- `build_manifest(..., data_revisions=..., metadata=...)` — new kwargs.
  Backward-compatible (defaults: empty dicts).
- `eval_toolkit.artifacts.validate_manifest(payload)` — dispatches on
  `payload["schema_version"]` (v1 or v2); falls back to v2 when absent;
  raises `ValueError` on unknown versions.
- `eval_toolkit.artifacts.validate_results(payload)` — thin wrapper over
  `validate_payload(payload, "results.v1.json")`.
- `eval_toolkit.artifacts.validate_prediction_artifact_ref(payload)` —
  validates a single `PredictionArtifactRef` payload against the inline
  schema (the same shape embedded in `manifest.v2.json` for
  `prediction_artifacts` items). Useful for callers handing out refs
  outside of a manifest.

### Changed

- `gpu_info()` now returns `dict[str, object]` where `count: int` and
  `memory_gb: float` (was `dict[str, str]` with stringified values).
  Caller-visible only when `nvidia-smi` is present (no behavior change on
  CPU-only environments where it returned `{}`).
- `RunManifest.gpu_info` field annotation widened from `dict[str, str]`
  to `dict[str, object]` to accommodate the mixed value types.
- `MANIFEST_SCHEMA_VERSION` bumped from `"v1"` to `"v2"`.
- `_version.py` bumped from `0.10.0` to `0.14.0`, catching up to
  `pyproject.toml` (the two had drifted across v0.11–v0.13; both now sync
  on every release).

### Tests

- 5 new manifest-side unit tests covering `captured_at` auto-population,
  `data_revisions` / `metadata` round-trip, default-empty behavior,
  schema_version invariant.
- 6 new artifacts-side unit tests covering `validate_manifest` v1 / v2
  dispatch, unknown-version rejection, default-to-current fallback,
  `validate_results` happy path, `validate_prediction_artifact_ref` happy
  path + missing-columns rejection.
- 2 new schema tests: v2 schema accepts `data_revisions`/`metadata`; v2
  rejects string `memory_gb`; v1 schema still accepts legacy payloads.
- 2 updated `gpu_info` tests reflecting the new int/float types.
- Existing manifest schema tests retargeted from `manifest.v1.json` to
  `manifest.v2.json`.

## [0.13.0] — 2026-05-13 — empty_strategy / nan_strategy kwargs

Adds opt-in degenerate-input handling to the Tier-1 metric primitives and
`sanitize_for_json`. Closes F1.2 and F4.1 from the V4 consumer feedback log:
callers no longer need to wrap eval-toolkit primitives in `safe_*` shims to
get `None` on empty / single-class slices, and can opt into JSON-`null` or
loud-`raise` non-finite handling without re-implementing `sanitize_for_json`.

**Backward-compatible**: all defaults (`empty_strategy="raise"`,
`nan_strategy="skipped"`) preserve pre-v0.13 behavior. Full eval-toolkit
test suite (1006+ tests) green without modification.

### Added

- `eval_toolkit.metrics.pr_auc` / `roc_auc` / `brier_score` now accept an
  `empty_strategy: Literal["raise", "return_none", "skipped_metric"] = "raise"`
  kwarg. `"return_none"` short-circuits with `None` on empty or single-class
  `y_true` (for AUC metrics; only `n=0` for `brier_score` since brier is
  valid on single-class). `"skipped_metric"` returns a structured
  `skipped_metric` dict so the reason threads through JSON artifacts. Each
  function has three `@overload` variants so the return-type narrows
  correctly under each strategy.
- `eval_toolkit.artifacts.sanitize_for_json` now accepts a
  `nan_strategy: Literal["skipped", "null", "raise"] = "skipped"` kwarg.
  `"null"` replaces non-finite values with JSON `null` (useful for plot
  consumers that expect numeric-or-null). `"raise"` raises on first
  non-finite value, surfacing scoring bugs that the default's silent
  structured replacement would mask. `nan_strategy` is threaded through
  recursive descent (mappings, sequences, numpy arrays, dataclasses).
- New private helper `eval_toolkit.metrics._empty_strategy_guard` and
  typed sentinel `_SentinelOk` for the three metric functions to share
  the degenerate-input check logic.

### Tests

- 5 new metric-side unit tests covering default-raise / return_none /
  skipped_metric branches, validator for invalid strategy values, brier's
  asymmetric single-class handling, and pass-through on normal input.
- 4 new artifacts-side unit tests covering nan_strategy=null,
  nan_strategy=raise, validator for invalid strategy, and recursion
  through nested structures.

## [0.12.0] — 2026-05-13 — PlattFit dataclass

`fit_platt_calibrator()` now returns a `PlattFit` frozen dataclass that
exposes the fitted `(a, b)` parameters alongside the `transform` callable.
Closes F2.2 from the V4 consumer feedback log: previous versions returned
a plain `Callable[[np.ndarray], np.ndarray]`, forcing serialization callers
to recover `(a, b)` by probing the closure at `s=0` and `s=1` and inverting
the sigmoid (the "logit-probe" trick in
`prompt-injection-v4/src/pid/orchestrate.py` `fit_platt_scaler`).

**Backward-compatible**: `PlattFit.__call__` delegates to `transform`, so any
caller annotated as `Callable[[np.ndarray], np.ndarray]` and using the
return value as `g(scores)` continues to work unchanged. Existing tests
exercising this pattern (`test_calibration_props.py:88,103`) all pass
without modification.

### Added

- `eval_toolkit.calibration.PlattFit(transform: Callable, a: float, b: float)`
  — frozen, slots-equipped dataclass returned from `fit_platt_calibrator`.
  Re-exported via `__all__`; module docstring's Public surface section
  references it inline with `fit_platt_calibrator`.

### Changed

- `eval_toolkit.calibration.fit_platt_calibrator` return type annotation
  `Callable[[np.ndarray], np.ndarray]` → `PlattFit`. The transform behavior
  is unchanged; the return wrapping happens at the final `return`.

### Tests

- Five new unit tests in `tests/test_calibration_unit.py`: returns
  `PlattFit` dataclass with float `a`/`b`, `__call__` delegates to
  `transform` (back-compat), `(a, b)` parameterize `sigmoid(a·s + b)` per
  the closed form, frozen-dataclass write protection. The existing
  `test_fit_platt_matches_sklearn_canonical` and property tests continue
  to pass — they exercise the `__call__` back-compat path implicitly.

## [0.11.0] — 2026-05-13 — Maximum Calibration Error

Adds `maximum_calibration_error()` as a top-level public symbol —
companion scalar to ECE that surfaces the *worst-bin* calibration
gap, so a model with low ECE but one very-poorly-calibrated bin is
not given a clean bill of health. Closes F1.3 from the V4 consumer
feedback log (`prompt-injection-v4/docs/eval_toolkit_feedback.md`).

### Added

- `eval_toolkit.calibration.maximum_calibration_error(y_true,
  y_score, *, n_bins=10, strategy="quantile") -> float | None` —
  MCE per Naeini & Cooper 2014. Returns `None` for single-class
  slices (calibration is degenerate when one class is absent).
  Parameter signature mirrors `reliability_curve` for consistency.
  Re-exported via `eval_toolkit.calibration.__all__` and the
  module docstring's Public surface section.

### Tests

- Five new unit tests in `tests/test_calibration_unit.py`: unit-
  interval bound, single-class returns `None`, input validation
  (shape mismatch, empty input, `n_bins<=1`, invalid strategy),
  reference-impl agreement with `reliability_curve`'s max-gap
  reconstruction, `MCE >= ECE` inequality (max of gaps dominates
  any weighted mean of gaps).

## [0.10.0] — 2026-05-12 — Maturity Release

A docs-and-hygiene minor release. Closes the v0.9 documentation gap
(new methodology chapters for claims + artifacts; new getting-started
tutorial; new schemas field reference) and the per-module coverage
gap (every `src/eval_toolkit/*.py` is now ≥90% individually;
aggregate ~95%). Adds an `eval-toolkit` console script for schema
discovery + payload validation, expands CI to macOS and Windows,
caps Python at `<3.14`, and adds a `make fast` / `nox -s fast` target
that skips `@pytest.mark.slow` tests for the local iteration loop.
No breaking changes; no API removals.

### Changed

- `pyproject.toml` `requires-python = ">=3.11,<3.14"` (upper bound).
  Reflects what's actually tested in CI; re-evaluate when Python 3.14
  stabilizes.
- `pyproject.toml` `all` extra refactored to reference sub-extras via
  PEP 685 self-reference: `all =
  ["eval-toolkit[dataframe,plotting,property,yaml,parquet,validation]"]`
  (was a flat list of every dep). Eliminates drift risk when a dep is
  added to a sub-extra. The `dev` extra now references `all` (no
  longer `all,parquet` since `all` includes parquet directly).
- Aggregate coverage floor raised from 90% to 92%
  (`pyproject.toml [tool.coverage.report] fail_under`, `tox.ini`,
  `noxfile.py`, `.github/workflows/ci.yml`, `Makefile`). Reflects the
  new per-module ≥90% baseline; current aggregate ~95%.
- CI matrix: previously Linux-only on Python 3.11/3.12/3.13. Now adds
  `macos-latest` and `windows-latest` jobs on Python 3.13 (5 jobs
  total). Backs up the `OS Independent` package classifier with
  actual platform validation.
- Methodology curriculum total updated to 16 chapters (was 12;
  `claims.md` + `artifacts.md` added in this release; the migration
  guide and methodology README cross-links updated accordingly).

### Added

- `make fast` and `nox -s fast` — fast iteration loop that runs
  `pytest -m 'not slow'`. CI continues to run the full suite on
  every push.
- `@pytest.mark.slow` markers added to tests >2s identified via
  `pytest --durations=50`:
  `test_manifest_props.test_config_hash_invariant_to_key_order`,
  `test_manifest_props.test_config_hash_changes_when_config_changes`,
  `test_bootstrap_props.test_paired_bootstrap_diff_anti_symmetry`,
  `test_bootstrap_unit.test_bootstrap_ci_width_shrinks_with_n`.
- `seeds.py` and `docs.py` added to the doctest pass in
  `.github/workflows/ci.yml`, `Makefile`, `tox.ini`, and
  `noxfile.py`. Previously listed in roadmap.md as deferred; now
  caught by CI.
- `eval-toolkit` console script (`python -m eval_toolkit ...`) with
  three subcommands: `schemas list` (enumerate bundled schemas),
  `schemas show <name>` (pretty-print a single schema, accepts both
  `results.v1` and `results.v1.json`), and `validate <file>
  <schema>` (dogfoods `[validation]` extra; exit codes 0/1/2/3 for
  ok/validation-failed/bad-arg/missing-extra). Stdlib argparse —
  no new runtime deps.
- `docs/getting-started.md` — linear newcomer-first end-to-end
  walkthrough (~480 lines): install, define a Scorer, build slices,
  run `evaluate()`, read the output, persist results, validate JSON,
  add a claim, render a plot, common errors, where-to-go-next.
  Sybil-runnable.
- `docs/schemas.md` — field-by-field reference for the three bundled
  JSON Schemas (`results.v1.json`, `results_full.v1.json`,
  `manifest.v1.json`): inventory table, per-field type/required/
  semantics/since-version, programmatic discovery snippet,
  validation example. Sybil-runnable.
- `README.md` — new top-level Documentation section linking the four
  docs entry points (getting-started, methodology curriculum, schema
  reference, migration, extending). Updated methodology link text to
  reflect 16 chapters (was 12).
- `conftest.py` — Sybil now collects `docs/getting-started.md` and
  `docs/schemas.md` in addition to the existing docs tree.
- `docs/methodology/claims.md` — worked-walkthrough chapter for the
  v0.9 claims pipeline (`ClaimSpec` / `EvidenceGate` / `GateResult` /
  `ClaimReport`). Covers the exception-handling contract, severity
  policy, and common pitfalls. Sybil-runnable code blocks.
- `docs/methodology/artifacts.md` — worked-walkthrough chapter for
  the v0.9 prediction-artifact contract (`PredictionArtifactRef`,
  `PredictionColumns`, `MetricState`). Covers `validate_payload` and
  the `[validation]` extra, paired-diff `content_hash` requirements,
  and `MetricState` status taxonomy (`ok` / `skipped` / `error`).
  Sybil-runnable code blocks.
- `docs/methodology/README.md`: reading-order rows 15-16 for the
  two new chapters.
- Per-module coverage tests pushing every `src/eval_toolkit/*.py` to
  ≥90% individually: `tests/test_claims_coverage.py` (gate exception
  paths, defensive `not isinstance(Mapping)` branches, every
  `_compare` operator); `tests/test_text_dedup_coverage.py` (all 5
  similarity strategies on degenerate inputs, MinHash LSH edge cases,
  audit-pair filter combinations, `_similarity_relation` mode table);
  `tests/test_thresholds_coverage.py` (every selector's
  `__post_init__` guards, `WilsonInterval` validation, no-eligible
  RuntimeErrors); `tests/test_plotting_edge.py` (shape/NaN/empty
  validation across all plot helpers, optional rendering paths);
  `tests/test_loaders_coverage.py` (HFDatasetsLoader mocked via
  `sys.modules['datasets']` injection, ParquetGlobLoader column
  errors); `tests/test_misc_coverage.py` (evidence, config,
  `__init__` lazy-import, `OperatingPointSpec` validation). Net +153
  tests; per-module coverage now ranges 91–100%.

## [0.9.1] — 2026-05-12

Post-v0.9.0 polish patch from an independent fresh-eyes audit. No
public-API behavior changes; fixes config drift between CI and local
runners, deduplicates Protocol definitions, and adds the v0.8→v0.9
migration guide.

### Fixed

- Coverage threshold drift between CI (90%) and local `tox` / `nox`
  (was 85%). `tox.ini` and `noxfile.py` now both enforce 90% to match
  `.github/workflows/ci.yml` and `pyproject.toml` `[tool.coverage.report]`.
- Doctest module list inconsistency: `Makefile`, `tox.ini`, and
  `noxfile.py` all listed 4 modules (`metrics`, `bootstrap`,
  `calibration`, `text_dedup`) while CI runs 9 (adds `thresholds`,
  `leakage`, `manifest`, `paths`, `provenance`). Local runners now
  match CI.

### Changed

- `claims.EvidenceGate.evaluate` exception handler narrowed from bare
  `except Exception` to specific runtime/data errors
  (`KeyError`, `ValueError`, `TypeError`, `RuntimeError`,
  `AttributeError`, `LookupError`). Gate-implementer bugs
  (`NameError`, `AssertionError`, `ImportError`, …) now surface as
  real exceptions instead of being silently coerced into
  "gate failed" messages. Intentional gate failures (caller-raised
  `ValueError` / `KeyError` from missing payload fields) still
  produce graceful `GateResult` records.
- `Scorer` and `SliceAwareScorer` Protocols are now defined
  exclusively in `eval_toolkit.protocols` (removed shadowed copies
  from `eval_toolkit.harness`). The canonical type signature
  for `Scorer.predict_proba` is now
  `Sequence[str] | np.ndarray | pd.Series` — pandas remains a
  type-only import (`TYPE_CHECKING`), so `protocols.py` retains
  zero runtime pandas dependency. Re-exports from
  `eval_toolkit.harness` and the top-level `eval_toolkit` namespace
  are preserved; consumer imports require no changes.
- `pyproject.toml`: deduplicated `jsonschema>=4.21` listing in the
  `dev` extra (it was already pulled transitively via
  `eval-toolkit[all]`).

### Added

- `docs/migration/v0.9.md` — comprehensive v0.8 → v0.9 migration
  guide covering the new evidence-core modules, `RunResult` field
  additions, schema additions, the `validation` extra, common
  pitfalls, and an end-to-end worked harness walkthrough.
- `docs/methodology/versioning.md` § Schema evolution policy —
  documents the `.vN.json` filename convention, the additive-fields
  forward-compatibility contract (`additionalProperties: true`), and
  when a filename bump is required.
- `tests/test_claims_props.py` — Hypothesis property tests for
  `ClaimSpec` / `EvidenceGate` invariants (all-pass, any-fail,
  warning-severity isolation, exception → typed-failure, order
  independence, `GateResult.name` round-trip).
- `tests/test_operating_points_props.py` — Hypothesis property
  tests for `fit_operating_points` / `apply_operating_points`
  (rank preservation, fit→apply round-trip with tolerance,
  degenerate input handling, determinism under seed).
- `tests/test_analysis.py`, `tests/test_manifest_validation.py`,
  `tests/test_harness_edge_cases.py`, plus expanded
  `tests/test_artifacts.py` — coverage-chase suites that close
  the 90% global gate which v0.9.0 had inadvertently dropped to
  86.46%. The chase brought `analysis.py` 68 → 98%, `artifacts.py`
  82 → 99%, `manifest.py` 71 → 96%, and added targeted
  evidence-of-error tests across `harness.py`'s operating-point
  branches. Total coverage now sits at 90.01% with 807 tests.

### Notes

- See `~/.claude/plans/examine-remote-and-look-parsed-starfish.md`
  for the audit rationale, the eight findings (none blocking), and
  the four user scoping decisions that shaped this patch.

## [0.9.0] — 2026-05-12

Evidence-core release. Introduces the `claims`, `artifacts`,
`evidence`, `operating_points`, `analysis`, and `protocols` modules
along with `RunResult` field additions for evidence axes, paired
metadata, aggregate evidence, threshold policy, and claim reports.

### Added

- v1-prelude evidence core: cross-slice operating-point transfer,
  source-role / guardrail manifest metadata, generic claim gates,
  `RunResult.claim_report` attachment, and low-FPR feasibility gating.

## [0.8.3] — 2026-05-09

Pure-polish patch from a fresh-eyes audit of the v0.8.2 surface. Three
parallel Explore agents surfaced 18 raw findings; **11 were rejected on
verification** (sklearn PR-curve index semantics, integer arithmetic
equivalence at `k // 2`, intentional determinism `==` assertions, and a
wheel-inclusion claim disproved by direct `zipfile` inspection). The 4
remaining items were all LOW severity. No public-API or behavior
changes.

### Fixed

- `README.md` modules table no longer mislabels `eval_toolkit.schemas`
  as an importable Python module (it's a JSON resource directory with
  no `__init__.py`). Row reworded to point at
  `importlib.resources.files("eval_toolkit") / "schemas"`.

### Tests

- Deduped the autouse `_close_figures_after_each_test` fixture into
  `tests/conftest.py`; removed the local copies in
  `tests/test_plotting_smoke.py` and `tests/test_plotting_visual.py`
  (also dropped the now-unused `import matplotlib.pyplot as plt` from
  `test_plotting_visual.py`).
- `tests/test_paths.py::test_path_for_config_outside_repo` tightened:
  `assert out is not None` → `assert out == elsewhere` (was passing on
  any non-None value); replaced hardcoded `/tmp/some_external_path.txt`
  with `tmp_path.parent / "outside.txt"` for full tmp-isolation.
- `tests/test_bootstrap_unit.py` adds 2 degenerate-input boundary tests
  for `bootstrap_ci`: `n_resamples=2` produces a (degenerate) CI without
  crashing; `n_resamples=0` is rejected (currently delegated to
  scipy.stats.bootstrap; test is belt-and-braces against silent NaN
  regressions).

### Notes

- See `~/.claude/plans/my-goal-is-to-enchanted-origami.md` for the audit
  rationale, the 11 rejected claims with verification evidence, and the
  signal that a fresh audit found 0 HIGH/MED issues across the v0.8.2
  surface (i.e., the v0.8.0/.1/.2 sweep was thorough).

## [0.8.2] — 2026-05-08

Follow-up patch closing the post-v0.8.1 audit's residual items: extends
the Protocol conformance harness to cover all 8 Protocols (was ⅝),
boosts coverage in two under-covered modules to 98–100%, adds a
`slow` test marker for opt-out of the studentized-bootstrap tests, and
applies `Final` annotations to module-level defaults.

### Added

- Conformance harness extended (`tests/test_protocol_conformance.py`):
  `Scorer`, `SliceAwareScorer`, and `SimilarityStrategy` now have
  contract assertions + negative-isinstance tests. 9 new tests; harness
  now covers 8/8 `@runtime_checkable` Protocols (was 5/8).
- `tests/test_seeds.py` adds a torch-installed code-path test via
  `monkeypatch` of `builtins.__import__`. Covers `seeds.py:103-109`
  (`manual_seed` / cuda branch / cudnn flags / `use_deterministic_algorithms`)
  even when torch isn't a dev dep.
- `tests/test_docs_props.py` adds 8 unit tests for `walk_path` list-index
  branch, render_text/render_files type guards, render_files check-mode
  drift+errors aggregation, and per-helper None handling on `_fmt_signed_3`,
  `_fmt_signed_4`, `_fmt_4`.
- `tests/test_bootstrap_unit.py` now houses 3 diagnostic regression tests
  (relocated from `test_coverage_gap.py`) covering the v0.8.1
  `first_failure` capture in `cross_validate_metric` + `_bootstrap_t_ci`,
  PLUS a new test for the *inner* LOO failure capture (a metric that
  succeeds on the full resample but fails on every leave-one-out subset
  — exercises `bootstrap.py:343-345`).
- `slow` pytest marker registered in `pyproject.toml`. Three tests
  (`test_bootstrap_ci_studentized_runs`,
  `test_bootstrap_ci_studentized_deterministic`,
  `test_evaluate_folded_multi_seed`) are now marked `@pytest.mark.slow`
  — opt out via `pytest -m "not slow"` (saves ~10s on default runs).

### Changed

- Module-level constants in `metrics.py`, `bootstrap.py`, `calibration.py`,
  `harness.py` now annotated with `typing.Final` (`DEFAULT_ASSUMED_PRIORS`,
  `DEFAULT_N_RESAMPLES`, `DEFAULT_CONFIDENCE`, `DEFAULT_METHOD`,
  `DEFAULT_SEED`, `DEFAULT_N_BINS`, `DEFAULT_STRATEGY`, `DEFAULT_PRIOR`,
  `DEFAULT_FP_COST`, `DEFAULT_FN_COST`, `DEFAULT_BOOTSTRAP_RESAMPLES`,
  `RUN_RESULT_SCHEMA_VERSION`). Catches accidental rebinding under mypy
  `--strict`. No runtime change.

### Coverage

- `seeds.py`: 74% → 98%
- `docs.py`: 84% → 100%
- Total module coverage maintained ≥ 90% gate.

### Notes

- Audit's "split `test_coverage_gap.py` into topical files" recommendation
  was rejected on inspection: the file is intentionally organized
  chronologically by version (v0.3 → v0.4 → v0.5 → v0.7 → v0.8 capability
  tags) and a topical reorg would lose that audit trail. The 3 v0.8.x
  bootstrap-diagnostic tests were relocated to `test_bootstrap_unit.py`
  where they topologically belong, but the chronological structure of
  `test_coverage_gap.py` is preserved.
- Audit's "narrow `Any` in `docs.py` formatters" was rejected: the
  formatters are dispatched dynamically on user-supplied anchor keys
  with arbitrary leaf types — narrowing would over-constrain.

## [0.8.1] — 2026-05-08

Post-v0.8.0 quality sweep. Surfaces bootstrap diagnostics that were
previously silent, removes type-discipline anti-patterns in `loaders.py`,
adds a generic Protocol conformance test harness consumers can adapt,
makes JSON schemas self-describing via a `version` root property, and
closes a handful of docs / packaging hygiene gaps.

### Added

- `tests/test_protocol_conformance.py` — generic conformance harness
  (23 tests) covering all 5 v0.7 Protocols (`ThresholdSelector`,
  `LeakageCheck`, `Splitter`, `DatasetLoader`, `Versioned`). Doubles as
  a copy-paste template for downstream consumers validating their own
  custom impls before plugging into the harness.
- `version: "1"` root property in `src/eval_toolkit/schemas/*.json`.
  Allows programmatic schema-version checking without filename parsing.
- README "Reproducibility manifest" quickstart block — runnable
  `build_manifest` / `write_manifest` example covered by Sybil doctest.
- `Programming Language :: Python :: 3.13` trove classifier in
  `pyproject.toml` (CI matrix already exercises 3.13 via
  `.github/workflows/ci.yml`).

### Changed

- `bootstrap.py:_bootstrap_t_ci` and `cross_validate_metric` now
  capture the *first* underlying exception when their inner `try/except`
  swallows resample / fold failures, and quote it in the eventual
  guard-rail `ValueError`. Previously these raises only said "likely
  single-class" — a guess that was unhelpful when the real cause was
  a different upstream error. Behavior is purely additive (extra text
  in the error message); no return-shape changes.

### Fixed

- `loaders.py:_load_dataset` is now annotated `-> Mapping[str, Any]`
  (with a single `cast(...)` at the boundary) instead of returning
  `object` and forcing `# type: ignore[attr-defined]` /
  `# type: ignore[index]` at every call site. Net `# type: ignore`
  count in `loaders.py` drops 3 → 1 (only the soft-import is now
  ignored). No runtime behavior change.
- `loaders.py` removed `_ = field` anti-pattern (the comment claimed
  `field` was used in dataclass defaults, but no calls existed —
  the import was just dead code).
- `metrics.py:_coerce` (inside the strata report's groupby) removed a
  dead `try/except TypeError: pass` that was unreachable because the
  preceding `isinstance(v, float)` guard prevents `np.isnan` from ever
  raising `TypeError`. Behavior unchanged; intent now visible.
- `metrics.py:metrics_at_threshold` docstring `Returns` section was
  missing `fpr` and `fnr` keys (the function returns them but the
  docstring claimed only TN/FP/FN/TP/F1/precision/recall/accuracy).
- `bootstrap.py` dropped `import contextlib` (now unused).
- `CHANGELOG.md` adds the standard `## [Unreleased]` placeholder per
  Keep-a-Changelog 1.1.0.

### Notes

- `harness.py:281` `except Exception` was flagged in audit but
  verified safe: `BaseException` subclasses (`KeyboardInterrupt`,
  `SystemExit`) are NOT swallowed by `except Exception`; left as-is.

## [0.8.0] — 2026-05-08

Post-v0.7.1 best-practices sweep. Closes one real bug (the v0.7.1
`__version__` mismatch), formalizes the ECE input contract with
parametric regression tests, breaks `pyarrow` out into its own
`[parquet]` extra, adds a new helper + four methodology chapters +
per-version migration guides + a roadmap.

The behavior change (ECE functions raise `ValueError` on uncalibrated
logits) is the reason this is v0.8.0 and not v0.7.2 — the validation
was *already* enforced in code (the helper was wired in pre-v0.8) but
v0.8 locks it in via parametric regression tests so the contract
can't silently regress in future releases.

### BREAKING (small)

- **`__version__` mismatch fixed.** v0.7.1 shipped with
  `pyproject.toml = "0.7.1"` but
  `src/eval_toolkit/__init__.py:13 __version__ = "0.7.0"`. v0.8.0
  closes the mismatch and bumps to `"0.8.0"`. Any consumer code
  branching on `eval_toolkit.__version__` would have seen the wrong
  value in v0.7.1.

### Added

- **`metrics.quantile_stratified_report`** — 10-LOC wrapper around
  `pr_auc` + `quantile_stratified_pr_auc` returning the four-field
  SDD reporting shape `{full, trimmed, gap, gap_flag}`. Closes
  prompt-injection-clean Gap 2. See
  `docs/methodology/length_stratification.md`.

- **Four new methodology chapters**:
  - `docs/methodology/bootstrap.md` — BCa derivation, paired vs
    unpaired, MDE, two-level bootstrap, K-fold CV-CI, resample
    budgets.
  - `docs/methodology/text_dedup.md` — when to use each
    `SimilarityStrategy`; threshold tuning; LSH false-negative rates;
    composition with `LeakageCheck`.
  - `docs/methodology/versioning.md` — the `Versioned` Protocol;
    `lm-evaluation-harness` `VERSION`-field pattern; how to choose a
    version string convention; threading into
    `RunManifest.versioned_objects`.
  - `docs/methodology/length_stratification.md` —
    `quantile_stratified_report` motivation; McClish 1989 partial-AUC
    framing; SDD `gap_flag` convention.

- **`docs/MIGRATION.md`** + **`docs/migration/v0.7.md`** +
  **`docs/migration/v0.8.md`** — per-version migration guides with
  copy-pasteable before/after blocks. Index file at
  `docs/MIGRATION.md`.

- **`docs/roadmap.md`** — forward-looking tracker; v0.9 candidates;
  v1.0.0 path with explicit gating criteria; consumer gap-doc
  cross-links (`prompt-injection-clean/docs/eval_toolkit_gaps.md`).

- **`tests/test_coverage_gap.py::test_all_ece_variants_reject_out_of_range_scores`**
  — parametric regression test asserting all 5 ECE variants raise
  `ValueError` on out-of-range scores. Closes v0.3 audit P1 #2.

- **`tests/test_reference_equivalence.py::test_brier_score_matches_sklearn`**
  — adds `brier_score ≡ sklearn.metrics.brier_score_loss` to the
  existing equivalence-test sweep.

- **`tests/test_seeds.py`** — three new tests covering the optional
  torch path:
  `test_set_global_seeds_torch_path_when_available` (skipped if torch
  absent), `test_set_global_seeds_strict_torch_raises_when_torch_absent`
  (mocks the import), and
  `test_set_global_seeds_strict_torch_with_torch_installed` (skipped
  if torch absent).

### Changed

- **`pyproject.toml`** — bump to `0.8.0`. Add `[parquet]` extra:
  ```toml
  parquet = ["pyarrow>=15.0"]
  ```
  Move `pyarrow` from `[dev]`-only to the new `[parquet]` extra;
  `[dev]` continues to depend on `eval-toolkit[parquet]` so CI still
  exercises `ParquetGlobLoader`. Consumers can now
  `pip install eval-toolkit[parquet]` without pulling the entire
  test/lint stack.

- **`STYLE.md`** — §4 "Type hints" updated. The Protocol-seam list
  now reflects all 7 v0.7 / v0.8 seams (`Scorer`, `SliceAwareScorer`,
  `LeakageCheck`, `Splitter`, `ThresholdSelector`, `DatasetLoader`,
  `SimilarityStrategy`, `Versioned`); documents the
  `@runtime_checkable` + `frozen+slots` reference-impl convention.

- **`docs/methodology/thresholds.md`** Pitfalls section — added an
  entry explaining the `recall@p` semantics divergence (smallest- vs
  highest-threshold-meeting-floor) for downstream migrators.

- **`docs/methodology/README.md`** — index updated for the four new
  chapters; renumbered reading-order table (now 13 chapters).

- **`docs/extending.md`** — cross-link to
  `methodology/versioning.md` from the Versioned-Protocol callout.

- **`README.md`** — methodology link list expanded (13 chapters);
  added `docs/MIGRATION.md` and `docs/roadmap.md` links.

- **`tests/conftest.py` + `conftest.py` (root)** — Sybil patterns
  expanded to include `docs/MIGRATION.md` and `docs/migration/*.md`.

### Bug fixes

- **`fit_platt_calibrator`** docstring confirmed accurate
  (canonical Platt with Lin 2007 Laplace smoothing) — the v0.3
  research audit's P1 #3 was already closed in v0.3.0; the
  audit reported a stale state.

- **`bayes_optimal_threshold`** docstring confirmed accurate (already
  notes the prior-corrected vs prior-independent distinction) — v0.3
  audit P1 #4 was already closed.

### Migration notes for downstream consumers

Most consumers pinning `eval-toolkit>=0.7.0,<0.8` should bump to
`>=0.8.0,<0.9` and run their tests. The two changes that may surface:

1. `eval_toolkit.__version__` now reads `"0.8.0"` (not `"0.7.0"`).
2. ECE on uncalibrated logits now raises `ValueError`. Apply
   `softmax`/`sigmoid`/`np.clip(scores, 0, 1)` first. See
   `docs/migration/v0.8.md` for the worked example.

## [0.7.1] — 2026-05-08

Property-test follow-up to v0.7.0 (PR 1.5 in the release plan).
Restores the 90 % coverage gate after the new v0.7.0 modules
(thresholds / leakage / splits / loaders / manifest) ship with full
Hypothesis property-test suites. No API changes.

### Added

- **`tests/test_thresholds_props.py`** — invariants for every
  `ThresholdSelector` reference impl: `MaxF1Selector` returns
  F1-optimal, `TargetRecallSelector` / `TargetPrecisionSelector` /
  `TargetFPRSelector` meet their constraint, `CostSensitiveSelector`
  matches the closed-form Bayes-optimal threshold, constructors reject
  out-of-range parameters.

- **`tests/test_leakage_props.py`** — invariants for `LeakageReport`
  (empty checks → clean report, merged_drop_indices = union),
  `NormalizedFormLeakageCheck` (zero-width-injected variants always
  flagged), `_aggressive_normalize` is idempotent, `LeakageFinding`
  to_dict round-trip, `GroupLeakageCheck` cross-split detection.

- **`tests/test_splits_props.py`** — Splitter Protocol invariants:
  HoldoutSplitter yields exactly 1 fold; StratifiedKFoldSplitter test
  partitions are pairwise disjoint and union to the full slice;
  GroupKFoldSplitter / SourceDisjointKFoldSplitter keep groups /
  sources train↔test-disjoint per fold; TimeSeriesSplitter respects
  max(train_t) < min(test_t); k-fold constructors reject k < 2.

- **`tests/test_loaders_props.py`** — DataFrameLoader split keys match
  unique split_col values + total rows preserved; describe() always
  returns the Croissant key set; SingleSliceLoader emits only "all";
  constructor rejects missing columns.

- **`tests/test_manifest_props.py`** — config_hash invariant to dict
  key order; changes when any value mutates; schema_version always
  "v1"; data_hashes always sha256-prefixed; JSON round-trip preserves
  run_id / schema_version / seeds; Versioned-protocol opt-in pattern
  collects only objects exposing `version`.

### Changed

- **`[tool.coverage.report] fail_under` 85 → 90** restored. The
  v0.7.0 release temporarily relaxed the gate while smoke tests
  shipped without property coverage; this restores the historical
  gate.

- **`pyarrow >= 15.0` added to `[dev]`** so `ParquetGlobLoader`'s
  load + hash path is exercised in CI (gated 90 % coverage on
  `loaders.py` requires it).

## [0.7.0] — 2026-05-08

Methodology-aware harness release. Promotes eval-toolkit from a
"library of metrics" to an opinionated **evaluation harness for binary
classification**. Adds five new Protocol-based extension surfaces
(`LeakageCheck`, `Splitter`, `ThresholdSelector`, `DatasetLoader`,
`Versioned`), a NeurIPS-aligned `RunManifest`, versioned JSON
schemas, and a multi-file consumer-facing methodology curriculum.

The four `prompt_injection_*` consumer projects (PRs 2–4 in the
release plan) migrate atomically once v0.7.0 is on PyPI.

### BREAKING

- **`metrics.select_threshold(criterion=str)` removed.** The string
  form (`"max_f1"` / `"recall_0.90"` / `"recall_0.95"`) is gone.
  `criterion` now requires a `ThresholdSelector` instance. Passing a
  string raises `TypeError` with the migration mapping in the message.

  Migration:

  | v0.6 | v0.7 |
  |---|---|
  | `criterion="max_f1"` | `criterion=MaxF1Selector()` |
  | `criterion="recall_0.90"` | `criterion=TargetRecallSelector(0.90)` |
  | `criterion="recall_0.95"` | `criterion=TargetRecallSelector(0.95)` |
  | `criterion="precision@0.90"` *(prompt-injection-sdd local fork)* | `criterion=TargetPrecisionSelector(0.90)` *(new)* |
  | `criterion="recall@0.90"` *(prompt-injection-sdd local fork)* | `criterion=TargetRecallSelector(0.90)` |

  `select_threshold` itself moved from `eval_toolkit.metrics` to
  `eval_toolkit.thresholds`; it remains re-exported at the package
  level (`from eval_toolkit import select_threshold` keeps working).
  The `OperatingPoint` Literal alias is removed.

- Internal callers in `eval_toolkit.metrics.headline_metrics` and the
  toolkit's own test suite migrated atomically in this commit.

### Added — new modules

- **`eval_toolkit.thresholds`** — `ThresholdSelector` Protocol +
  reference impls: `MaxF1Selector`, `TargetRecallSelector(recall)`,
  `TargetPrecisionSelector(precision)` (new), `TargetFPRSelector(fpr)`,
  `YoudenJSelector`, `CostSensitiveSelector(cost_matrix)`. All return
  the existing `ThresholdResult` dataclass; all are
  `runtime_checkable` and dataclass-frozen-with-slots.

- **`eval_toolkit.leakage`** — `LeakageCheck` Protocol + uniform
  `validate(splits: Mapping[str, EvalSlice]) -> LeakageFinding`
  contract; two-tier `LeakageReport` with severity gating
  (`error` / `warning` / `info`); `Versioned` opt-in Protocol mirroring
  `lm-evaluation-harness`'s task `VERSION` field; `run_leakage_checks`
  aggregator. Reference impls: `ExactDuplicateCheck`,
  `NearDuplicateCheck`, `NormalizedFormLeakageCheck` (NEW: NFKC +
  zero-width / Symbol-Other strip — catches the encoding-obfuscation
  attack class documented in PI_HackAPrompt_SQuAD 2025 at 21.3 %
  detection / 76.2 % ASR), `CrossSplitLeakageCheck`,
  `LabelConflictCheck`, `GroupLeakageCheck`, `TemporalLeakageCheck`.

- **`eval_toolkit.splits`** — `Splitter` Protocol with
  `iter_folds(slice, *, groups=None) -> Iterator[dict[str, EvalSlice]]`
  + `get_n_splits`. Reference impls: `HoldoutSplitter` (k=1 unifies
  holdout into the same iterator shape as K-fold),
  `StratifiedKFoldSplitter`, `GroupKFoldSplitter`,
  `SourceDisjointKFoldSplitter` (generalizes the source-disjoint
  pattern that `prompt-injection-sdd` hand-rolled), `TimeSeriesSplitter`.

- **`eval_toolkit.loaders`** — `DatasetLoader` Protocol with HF-
  `DatasetDict`-shaped `load_splits() -> dict[str, EvalSlice]` +
  Croissant-compatible `describe()`. Reference impls:
  `DataFrameLoader`, `SingleSliceLoader`, `ParquetGlobLoader`,
  `HFDatasetsLoader` (soft-imports `datasets` only if installed).

- **`eval_toolkit.manifest`** — `RunManifest` dataclass aligned with
  the [NeurIPS Reproducibility Checklist](https://neurips.cc/public/guides/PaperChecklist):
  `run_id`, `git_sha`, `dirty_flag`, `code_versions`, `seeds`,
  `data_hashes` (sha256-prefixed), `config_hash`, `env`, `gpu_info`
  (via `nvidia-smi --query-gpu`, graceful fallback), `cuda_version`,
  `wall_clock_seconds`, `versioned_objects` (auto-collected from any
  Tier-2 instance exposing a `version` attribute), `leakage_report`,
  `schema_version="v1"`. `build_manifest` (pure builder) +
  `write_manifest` (sole IO sink) mirror the existing
  `harness.evaluate` / `write_run_result` pure/IO split.

- **`eval_toolkit.schemas/`** — versioned JSON Schemas
  (`results.v1.json`, `results_full.v1.json`, `manifest.v1.json`,
  draft 2020-12). `tests/test_schemas.py` validates every JSON
  output against the schemas; a breaking shape change without a
  `schema_version` bump fails CI.

### Added — `harness` extensions (additive; backward-compatible)

- **`evaluate(..., leakage_checks: Sequence[LeakageCheck] = (),
  on_leakage: Literal["raise", "record", "skip"] = "raise")`** —
  inline leakage validation. Default fail-fast on error-severity
  findings; `"record"` captures the report in `RunResult.config`;
  `"skip"` runs checks without recording. Mirrors the DVC / Great
  Expectations declarative pattern.

- **`evaluate(..., on_scorer_error: Literal["raise", "record"] =
  "raise")`** — catch `Scorer.predict_proba` exceptions per (slice,
  scorer) when `"record"`; the run completes with `{"error",
  "exc_type", "traceback"}` in the per-scorer block. Subsumes the
  `_safe_select_threshold` workaround pattern from
  `prompt-injection-sdd`.

- **`evaluate_folded(scorers, splitter, slice_, ...)`** — fold
  aggregator. Loops `splitter.iter_folds(slice_)` × seeds, calls
  `evaluate(...)` per fold, populates `RunResult.by_fold` (raw per-
  fold results) and `RunResult.fold_summary` (auto-computed
  `cv_clt_ci` per `(slice, scorer, metric)` triple, with graceful
  `{"skipped": ...}` fallback for degenerate folds).

- **`RunResult.by_fold` / `fold_summary` / `schema_version="v1"`** —
  additive fields; default empty / `"v1"` so existing `evaluate(...)`
  callers see no behavior change.

- **`RUN_RESULT_SCHEMA_VERSION = "v1"`** + **`MANIFEST_SCHEMA_VERSION
  = "v1"`** module-level constants.

### Added — docs

- **`docs/methodology/`** — multi-file consumer-facing methodology
  curriculum: `README.md` (index), `leakage.md`, `splits.md`,
  `thresholds.md`, `calibration.md`, `comparison.md`, `fairness.md`,
  `reproducibility.md`, `testing.md`, `reading_list.md`. Hybrid
  expert + learner audience with collapsible *Background*
  admonitions and explicit *Pitfalls / Common mistakes* sections per
  chapter. All Python code blocks runnable end-to-end under
  [Sybil](https://sybil.readthedocs.io/); PyTorch / HuggingFace
  blocks marked `<!-- skip: next -->` with explicit rationale.

- **`docs/extending.md`** — Protocol-by-Protocol guide for plugging
  custom Scorers / LeakageChecks / Splitters / ThresholdSelectors /
  DatasetLoaders into the harness. ~50-line full-harness recipe;
  project-layout pointer to the showcase repo.

- **`docs/examples/prompt_injection_walkthrough.md`** — end-to-end
  prompt-injection eval on a synthetic 12-prompt OWASP LLM01:2025
  fixture (direct, indirect, encoded/obfuscated, system-prompt-leak,
  multi-stage). Cross-links to the showcase repo.

- **`docs/examples/pytorch_scorer_example.md`** — HuggingFace
  transformer + LoRA `Scorer` adapter. Marked `<!-- skip: next -->`
  since `torch` is consumer-side.

- **`README.md` reframed** — three-tier architecture diagram,
  Methodology / Extending / Examples link blocks, expanded module
  table.

### Changed — testing infrastructure

- **Sybil 10.x added to `[dev]`** (and a root-level `conftest.py`
  registering `pytest_collect_file`) so every `python` doc-block
  under `docs/` and `README.md` runs in CI as part of `pytest`.

- **`jsonschema 4.21+` added to `[dev]`** for
  `tests/test_schemas.py`.

- **80+ new smoke / Protocol-instanceof tests** across the new
  modules.

- **Coverage gate temporarily lowered to 85 %** (was 90 %) for this
  release. Property tests for the new modules — and the gate
  restoration to 90 % — land in v0.7.1 (PR 1.5 in the release plan).

### Migration notes for downstream consumers

- **eval-toolkit-pinning consumers** (the four `prompt_injection_*`
  repos): bump pin to `eval-toolkit>=0.7.0,<0.8`. The breaking
  `select_threshold` change has a mechanical migration; a
  `TypeError` at runtime points at the exact mapping.

- **Consumers carrying their own `select_threshold` /
  `_safe_select_threshold` / `data.py` (loader+leakage) shims**
  (especially `prompt-injection-sdd`): delete the local copies, use
  the toolkit's reference impls. The new `NormalizedFormLeakageCheck`
  + `LabelConflictCheck` + `SourceDisjointKFoldSplitter` cover the
  ~200 LOC duplicated across consumer repos today.

- **Consumers building bespoke fairness / drift / McNemar / DeLong
  metrics**: still consumer-side. eval-toolkit deliberately does not
  ship these; `docs/methodology/fairness.md` and
  `docs/methodology/comparison.md` document the conventions and point
  to `fairlearn` / `scipy.stats`.

## [0.6.0] — 2026-05-08

Downstream-extensibility release. Surfaces three previously hard-coded
defaults (palette role names, save_figure permitted suffixes, file_sha256
missing-file behavior) as configurable parameters so projects can adopt
``eval_toolkit`` without local-shim wrappers. Driven by the
prompt-injection-detector PoC's toolkit-migration follow-up.

### Added

- **`make_palette(*, negative='#004488', positive='#BB5566', accent='#DDAA33', baseline='#999999', **extras: str) -> Mapping[str, str]`**
  in ``eval_toolkit.plotting``. Factory for project-specific semantic
  palettes. Returns a frozen ``MappingProxyType``. ``**extras`` accepts any
  number of additional named keyword arguments for project-specific roles
  (e.g. ``make_palette(benign=..., injection=..., emphasis=...)`` for
  prompt-injection framing).

- **`save_figure(..., permitted_suffixes: Container[str] = {".png", ".pdf", ".svg"})`**
  parameter. Downstream projects can restrict to a single format
  (``permitted_suffixes={".png"}``) for stable artifact pipelines.

- **`save_figure(..., skip_env_var: str = "EVAL_TOOLKIT_SKIP_SAVEFIG")`**
  parameter. Downstream projects can pass their own opt-out env-var name.

- **`file_sha256(path, *, strict: bool = False)`** parameter. When ``True``,
  raises ``FileNotFoundError`` instead of returning ``None`` for missing
  paths. Useful when caller invariants require the digest to exist.

### Changed

- ``save_figure`` validation message now reflects the configured
  ``permitted_suffixes`` rather than a hard-coded set.

### Backward compatibility

All four changes are purely additive — defaults match v0.5.0 behavior. No
breaking changes; existing call sites unaffected.

## [0.5.0] — 2026-05-07

Closes the v0.4 deferral list: cross-validation orchestrator,
debiased L1 ECE, and GitHub Actions CI.

### Added

- **`cross_validate_metric(y_true, y_score, *, metric, k=5, stratified=True, seed=42)`**
  — eval-only K-fold orchestrator. Uses sklearn `StratifiedKFold` by
  default for class-balance preservation; computes the metric per fold;
  returns shape-(K,) array (NaN for single-class folds). Pairs with
  `cv_clt_ci` (v0.4) for end-to-end CV inference. Eval-only by design;
  the toolkit does not own model training.
- **`expected_calibration_error_debiased(y_true, y_score, n_bins=10, *, n_sweep=200, seed=42)`**
  — Monte-Carlo simulated-H0 bias correction for L1 ECE. Companion to
  v0.4's closed-form `expected_calibration_error_l2_debiased` (Kumar
  2019); the L1 form has no closed-form correction so we estimate the
  bias by drawing y_b ~ Bernoulli(s) under H0, computing plug-in ECE_b,
  averaging over `n_sweep` resamples. Same conceptual move as Roelofs
  2022's ECE_SWEEP estimator (which uses CV instead of simulated-H0);
  trades fidelity to the literal SWEEP construction for substantial
  implementation simplicity.
- **`.github/workflows/ci.yml`** — GitHub Actions CI on push + PR to
  main. Matrix Python 3.11/3.12/3.13. Steps: ruff + black + mypy strict
  + pytest with coverage gate + doctests on math kernels + pytest-mpl
  visual regression.

### Quality gates (2026-05-07)

- 357 tests passing (was 352 in v0.4.0; +5 new C1+C2 tests).
- 22 doctests across math kernels (was 15 in v0.4.0; +7 across new
  functions and updated examples).
- ruff + black + mypy strict all clean.
- 99 symbols re-exported from top-level `eval_toolkit` (was 97 in
  v0.4.0; +2 new public symbols).
- Visual baselines (pytest-mpl) all pass at tolerance=15.

### Deferred (no remaining v0.4 backlog)

The v0.4 deferral list is now complete. Future enhancements
(no concrete commitment): full train+eval CV (caller-supplied
`fit_fn`); literal Roelofs 2022 ECE_SWEEP via cross-validation;
PyPI publication.

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
