# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Aggregate coverage floor raised from 90% to 92%
  (`pyproject.toml [tool.coverage.report] fail_under`, `tox.ini`,
  `noxfile.py`, `.github/workflows/ci.yml`). Reflects the new per-module
  ≥90% baseline; current aggregate ~95%.

### Added

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
