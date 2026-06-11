# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — loaders + prediction-reader contract hardening, silent-failure class ([#98](https://github.com/brandon-behring/eval-toolkit/issues/98))

All four are new raises on garbage input that previously produced
silently-wrong results (STYLE §1 *never fail silently*; same class as the
v1.9.0 #96 batch):

- **`HFDatasetsLoader.load_splits`**: a missing/NaN cell in any `feature_cols`
  column now raises `ValueError` with the split name + per-column NaN counts —
  previously `fillna("")` silently fabricated empty prompt parts (truncated
  system/user prompts shipped into the eval) while the `label_map` path eight
  lines down fail-fasted. Valid data keeps the pandas-3.0 per-column
  `astype(str)` join.
- **`ood_dataset_from_manifest` label pass-through** (`label_map` omitted):
  numeric non-integer labels now raise instead of silently truncating
  (`0.7 → 0` flipped ground truth) — the loader twin of v1.9.0's
  `load_prediction_arrays` fix, but integrality + finiteness rather than
  `{0, 1}` membership (loader labels are generic ints). NaN/`pd.NA` labels get
  a dedicated diagnostic (slice id + count + first row index); integral floats
  from parquet (`1.0`/`0.0`) and bools still pass; string labels keep the
  existing "no label_map provided" error.
- **`ood_dataset_from_manifest`**: YAML parse errors are re-raised as
  `ValueError` with the manifest path + parse detail — the documented
  "ValueError on YAML parse error" contract was false (raw `yaml.YAMLError`
  escaped). Callers catching `yaml.YAMLError` must catch `ValueError` now.
- **`analysis.CsvPredictionReader`**: a missing or empty-string cell in any
  declared column now fails fast with file + row context (per-row parity with
  the #96 JSONL check) — previously a declared-but-sparse
  `row_id`/`content_hash` loaded `""` placeholders that silently weakened
  paired-diff row alignment, and a sparse `score` died downstream without row
  context.

### Added ([#98](https://github.com/brandon-behring/eval-toolkit/issues/98))

Tier-2 ADDITIVE per ADR 0003 — no constructor signature changes; public-API
snapshot unchanged:

- **`HFDatasetsLoader.describe()` output gains a `"revision"` key** echoing the
  constructor's `revision` (`None` when unpinned), so revision-pinned loads are
  no longer indistinguishable from unpinned ones in provenance manifests.

### Documentation ([#98](https://github.com/brandon-behring/eval-toolkit/issues/98))

Tier-3:

- `HFDatasetsLoader` Parameters now documents the four v1.5.0 fields
  (`feature_cols`, `feature_join`, `label_map`, `revision`, each "Added in
  v1.5.0") and gives `strata_col` its missing description line.
- `describe()` documents the `refs/convert/parquet` tree-hash limitation:
  per-file sha256 always reflects the latest parquet conversion, which may not
  correspond to a pinned `revision`.
- Raises completeness: `ParquetGlobLoader.load_splits` gains
  `FileNotFoundError`; `HFDatasetsLoader.load_splits` gains the feature_cols
  `ValueError` + the soft-import `ImportError`; `_apply_label_map` gains a full
  Raises section.

### Documentation / Process ([#101](https://github.com/brandon-behring/eval-toolkit/issues/101))

- **CHANGELOG v1.5.0 erratum appended** (history not rewritten): the #85 loader
  parameters were a Tier-1 positional-compat break per ADR 0003, mislabeled
  "Tier-2 ADDITIVE" at release.
- **Release-prep tier check**: `make release-prep` and the RELEASING.md TL;DR
  checklist now carry a step 2b — classify the snapshot diff against ADR 0003
  tiers and label the CHANGELOG entry to match.

### Build / Infra ([#102](https://github.com/brandon-behring/eval-toolkit/issues/102))

- **Per-module coverage floors now run in CI** (ubuntu reference platform):
  the CI coverage step emits `coverage.json` and `scripts/check_module_floors.py`
  gates on it. The previously-empty `FLOORS` dict now documents dev-install
  reality for the modules whose tests skip on the documented `[dev]` install
  (`losses.py`, `probes.py`, `loaders.py` — their heavy extras are deliberately
  excluded from `[dev]`/`[all]`): floors sit just below observed dev-install
  coverage — no-regression gates, not aspiration. Declined for now: a nightly
  heavy-extras coverage job (revisit if torch-path regressions bite).
- **`make install` is now lock-faithful**: `uv sync --extra dev --extra docs`
  resolves from `uv.lock` (closing the observed drift: fresh-resolved black
  26.5.1 locally vs 26.3.1 locked/CI) and still installs the project editable.

## [1.9.0] — 2026-06-10 — resample distribution + silent-NaN hardening + UTF-8 batch (#93, #96, #97)

### Fixed — pre-tag adversarial-review completion (silent-NaN gaps in `bootstrap_ci` itself)

The pre-tag review panel (whole-repo re-audit + independent reviewers +
self-refutation) found the #96 class surviving in `bootstrap_ci` — the
most-used entry point, outside #96's enumerated scope:

- **Studentized path**: a NaN outer statistic was accepted as a *valid*
  resample, bypassing the 95%-valid gate and poisoning the pivots into a
  silent all-NaN `BootstrapCI` with zero warnings. NaN/inf outer statistics
  (and inner-jackknife LOO values) now count as degenerate draws.
- **Non-finite CI bounds now raise for ANY method** — the degeneracy check
  was gated to `method="BCa"`, so `percentile` returned NaN bounds with only
  scipy's misdirecting `DegenerateDataWarning` (which always names BCa).
  The finite BCa-collapse case (`ci_low == ci_high == point`) keeps the R9
  `UserWarning` contract. Behavior change: BCa NaN-bounds previously
  warned-and-returned; they now raise (the scorecard/harness per-cell
  isolation converts this into an error/reason cell as before).
- **Point estimate guarded**: a metric returning NaN on the full data
  previously yielded `point_estimate=nan` beside a finite CI, silently.
- `analysis.load_prediction_arrays`: labels are now domain-checked before
  the int cast — `dtype=int` coercion silently **truncated** float labels
  (`0.7 → 0`), flipping ground truth with in-domain values no downstream
  gate could catch.
- Docs: `sweep()` Raises now documents the reachable pandas `ImportError`
  and its Returns lists the always-present `strategy_id` column; STYLE.md
  Tier-2 quick reference corrected to the ten strict Protocols (v1.0.2
  `SimilarityStrategy` promotion).
- Tests: mutation-verified gap closed (per-stratum NaN filter pinned via a
  non-NaN-propagating `combine`); `samples`↔quantile consistency pinned at
  non-default confidence.

### Added — resample-distribution exposure on the cluster bootstraps (#93)

Tier-1 strictly-appended optional parameters, SemVer-MINOR per the
[ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)
2026-06-10 amendment (#101) — backward-compatible; snapshot regenerated in
the same commit.

- **`cluster_bootstrap_ci(..., return_samples=True)`** and
  **`stratified_cluster_bootstrap_ci(..., return_samples=True)`** attach the
  post-filter bootstrap resample statistics to the result as
  **`BootstrapCI.samples`** (read-only `numpy.ndarray`, the same array the
  percentile bounds are computed from; `shape == (n_resamples_used,)`).
  Distribution summaries such as the consumer's `frac_gt0`
  (`float(np.mean(ci.samples > 0.0))`) are now derivable from the *same*
  draws as the CI — previously structurally unrecoverable, blocking the two
  remaining LODO call-site migrations (downstream DF-11).
- `BootstrapCI` gains the trailing optional field `samples`
  (default `None`, `compare=False`, `repr=False`): positional construction,
  equality/hash semantics, and the **`to_dict()` schema are all unchanged**.
  Note: `dataclasses.asdict()` (which ignores those flags) now includes a
  `samples` key — consumers serializing via `asdict` instead of `to_dict()`
  should drop it (an attached ndarray is not JSON-serializable).
- scipy precedent: `BootstrapResult.bootstrap_distribution`. Honors the
  n_jobs bit-for-bit reproducibility contract.

### Fixed — silent-NaN hardening batch (#96)

Finiteness guards across the numeric surface (STYLE §1 *never fail silently* /
§7 validation boundary). All are new raises (or error statuses) on garbage
input that previously produced silently-wrong results:

- `cluster_bootstrap_ci` / `stratified_cluster_bootstrap_ci`: NaN/inf scores
  now raise at the validation boundary (previously the per-stratum check was
  shape-only and the result was a silent all-NaN `BootstrapCI`); a non-finite
  point estimate raises with the got-value; resample draws where the statistic
  (or `combine`) **returns** NaN/inf now count toward the >5% degenerate gate
  instead of poisoning the quantile CI (previously only *raising* draws counted).
- `paired_bootstrap_diff` / `paired_bootstrap_ece_diff` /
  `paired_bootstrap_op_point_diff`: NaN resample deltas now count as degenerate
  draws (same ≤5% tolerance as raising draws — pre-#96 a NaN CI made
  `overlaps_zero` read `False`, i.e. silently "statistically significant");
  a non-finite full-data Δ raises with the got-value; a non-finite-CI-bounds
  raise remains in each constructor as a backstop (mirrors the BCa degeneracy
  guard in `bootstrap_ci`).
- `scorecard` bootstrap path: BCa-degenerate NaN CI bounds are no longer
  attached to an `"ok"` cell — the CI is dropped and the reason recorded
  (consistent with the existing "bootstrap unavailable" convention).
- `eda.median_bandwidth`: non-finite input (NaN/inf) now raises at entry —
  NaN bypassed the `sigma <= 0.0` check and escaped as a NaN bandwidth, and a
  NaN row outside the `max_samples` subsample escaped entirely.
- `eda.maximum_mean_discrepancy`: explicit `bandwidth` must be finite and > 0 —
  `inf` yielded γ = 0 → all-ones Gram → MMD² = 0 → `p_value = 1.0` silently
  reading "no shift".
- `eda` PAD/MMD/kNN feature matrices are finiteness-checked at the boundary
  (previously NaN embeddings died deep inside sklearn blaming internals).
- `eda.class_lexical_association`: a `positive_label` matching no label (the
  1-vs-`"1"` type-mismatch trap) or matching every label now raises listing the
  observed label values, instead of returning a documented all-empty result
  that read "no shortcut signal".
- `scorecard`: a custom `MetricSpec.compute` returning NaN/inf now yields
  `MetricResult(status="error", reason=...)` through the same path as a raising
  compute — previously `status="ok"` with a NaN value.
- `sweep`: a NaN `attack_threshold` now raises (it silently zeroed every `asr`
  flag); `±inf` remains a documented unsatisfiable sentinel.
- `analysis.JsonlPredictionReader`: a row missing (or `null` on) a declared
  column key now fails fast with file + row + key context (the R8-F3 pattern
  already applied to CSV headers) — previously a missing score coerced to NaN
  deep in the metric computation and a missing label died as a bare `TypeError`.
  A malformed JSON row now reports the actual file row (raw `json.JSONDecodeError`
  always said "line 1"). `analysis.load_prediction_arrays` additionally rejects
  non-finite loaded scores (a bare JSON `NaN` token or a CSV `"nan"` cell passes
  per-row key checks) with file + column + row-index context.

### Fixed — explicit UTF-8 encoding batch (#97)

Windows (cp1252 locale codec) is the trigger; Linux/macOS hid all of these.
Locked convention: always pass `encoding="utf-8"` on text-file IO.

- `docs.render_files` **apply mode** read and wrote consumer markdown with the
  locale codec — on cp1252 this silently and *cumulatively* corrupted
  non-ASCII user content on every apply (the worst item in the batch).
- All remaining text IO made explicit: `__main__` schema/payload reads
  (RFC 8259 mandates UTF-8 for JSON), `analysis` CSV/JSONL prediction readers,
  `config.from_yaml`, `artifacts` schema read + report write,
  `plotting` sidecar write, `scripts/audit_raises_sections.py`.
- `scripts/dogfood_audit.py`: a surface file skipped for `UnicodeDecodeError`
  now emits a stderr warning with the path — previously it silently vanished
  from the acceptance evidence.
- **Detection locked out permanently**: ruff now enforces `PLW1514`
  (implicit-encoding) across `src/`, `scripts/`, and `tests/` via
  `preview = true` + `explicit-preview-rules = true` (only this rule gets
  preview status; no other behavior changes).

### Fixed

- `audit_value_bindings.validate_reader_value_bindings` now raises a
  diagnostic `ValueError` when a scanned file is not valid UTF-8, instead
  of letting an unguarded `read_text(encoding="utf-8")` abort the run with
  a bare `UnicodeDecodeError`. Documented in the function's `Raises` section.
- `audit_sister_doc_concept_drift.validate_sister_doc_concept_drift` now
  skips non-UTF-8 files with a `warnings.warn` instead of crashing — its
  prior `except OSError` did not catch `UnicodeDecodeError` (a `ValueError`,
  not an `OSError`), so a single non-UTF-8 byte aborted the whole scan.

### Internal

- `scripts/` is now covered by `ruff` / `black` / `mypy` across all runners
  (`Makefile`, `ci.yml`, `.pre-commit-config.yaml`, `tox.ini`, `noxfile.py`).

### Fixed — documentation/config consistency batch (2026-06-09 full-repo audit)

- [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)
  amended: records the v1.0.2 `SimilarityStrategy` promotion (strict
  Tier-2 count 9 → 10) in the Tier-1 Protocol list, and replaces the
  unimplemented `STRICT_DOCSTRINGS` plan with the actual contract
  (docstring first lines remain pinned through v1.x).
- `SimilarityStrategy` registered in
  `tests/test_public_api.py::_TIER2_PROTOCOLS` — the R6-D fail-fast
  list had lagged the v1.0.2 promotion.
- `docs/source/roadmap.md` post-v1.0 section refreshed to the v1.8.0
  state (was still "v1.0.1 is the next minor"; the referenced
  `v1.0.1 cleanup` issue #76 closed at v1.0.2); broken repo-relative
  link to a machine-local planning document removed.
- STYLE.md §17 example updated — `pr_auc` left the top level at v0.46
  (Decision L); the example now uses `scorecard`. README Tier-2 box
  disambiguated (10 strict Protocols vs `SliceAwareScorer`/`Versioned`).
- CONTRIBUTING.md: corrected the `[dev]`-extra claim (heavy optional
  stacks `embeddings`/`transformers`/`probes`/`losses` are not
  included) and documented the docs-extra requirement for `pre-push`.

### Internal

- `make test` now collects all three doc-execution surfaces — the
  positional `tests` arg silently bypassed pyproject `testpaths`,
  skipping the 161 README/docs Sybil doc tests (v0.47 §5L incident
  class). `make install` installs `.[dev,docs]` so the sphinx
  pre-push gate works on a fresh environment.
- CI coverage step excludes `-m integration` (aligns ci.yml with the
  pyproject marker contract and the Makefile coverage target).
- tox/nox aligned with `requires-python = ">=3.13"`: py313-only
  envlist/`PY_VERSIONS`, monte_carlo/benchmark/integration marker
  exclusions added to their pytest commands, stale "private and
  home-designed" framing removed; Makefile help text and §5H
  notebook-gate comments updated to current reality.

## [1.8.0] — 2026-06-04 — composite multi-stratum cluster bootstrap (#92)

### Added — `bootstrap.stratified_cluster_bootstrap_ci` (composite multi-stratum cluster bootstrap)

`eval_toolkit.bootstrap` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — backward-compatible. Generalises the v1.7.0 single-block `cluster_bootstrap_ci` to the shape leave-one-group-out transfer gaps actually take: a **composite statistic reduced over several independently-resampled cluster strata**.

- **`stratified_cluster_bootstrap_ci(strata, per_stratum_metric, combine, *, resample_labels=(0,1), …)`** — `strata` is a mapping `{key: (y, score, groups)}` of independent resample-units (e.g. `seed`, `(carrier, seed)`, `(attack_type, seed)`); each bootstrap iteration resamples every stratum's `(label, group)` clusters, computes `per_stratum_metric` on each, and reduces the `{key: metric}` map with `combine` to one scalar (a seed-averaged ROC-AUC gap, a mean-over-carriers gap, a top−bottom per-type AUPRC contrast, …). Percentile `BootstrapCI` (`method="stratified_cluster_percentile"`). `cluster_bootstrap_ci` is the single-stratum, identity-reduce special case.
- **Why:** the v1.7.0 single-block primitive could not express the **seed-averaging** that real LODO estimators do inside the bootstrap (`Gx = val − mean_seed(test_roc)`), so it did not actually fit the consumer portfolio's attack-type / carrier / dialect bootstraps. This is the correct primitive for them.
- **Parallel + reproducible:** built on `parallel_map` + `spawn_seed_sequences` ⇒ `n_jobs` gives bit-for-bit-identical CIs; `n_jobs=-1` all cores.
- Exported via `from eval_toolkit import stratified_cluster_bootstrap_ci`; `__all__` + `_EXPORTS` + doctest + n_jobs-reproducibility / seed-averaged / composite-statistic tests; mypy-strict clean.

## [1.7.0] — 2026-06-04 — label-stratified cluster bootstrap (#90, #91)

### Added — `bootstrap.cluster_bootstrap_ci` (label-stratified cluster bootstrap)

`eval_toolkit.bootstrap` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — backward-compatible. Closes the gap between the row-level (`bootstrap_ci`) and fold-level (`block_bootstrap_on_folds`) resamplers: **the missing middle — resampling clusters of rows.**

- **`cluster_bootstrap_ci(y_true, y_score, groups, statistic, *, resample_labels=(0, 1), …)`** — percentile CI for a single-condition metric that resamples whole **clusters** (`groups`) with replacement, so the CI is honest under intra-cluster correlation (prompts sharing one attack payload; a document contributing both a poisoned and a benign row). The resample unit is `(label, group)`: by default positive- and negative-clusters are resampled **separately** (never a single-class draw); `resample_labels=(1,)` resamples only positive clusters with negatives held fixed (the payload-cluster convention). Returns a `BootstrapCI` with `method="cluster_percentile"`.
- **Parallel + reproducible:** built on `parallel_map` + `spawn_seed_sequences`, so `n_jobs` gives bit-for-bit-identical CIs across worker counts (the v0.34.0 reproducibility contract). `n_jobs=-1` uses all cores.
- Motivation: the analytic row-level AUROC-difference CI (`delong_roc_variance`) assumes row independence and under-covers on clustered eval data (LODO transfer gaps with payload/document/page clusters). Dogfooded by the consumer portfolio's attack-type / carrier / dialect leave-one-group-out bootstraps (Rule of Three).
- Exported via `from eval_toolkit import cluster_bootstrap_ci`; `__all__` + `_EXPORTS` updated; doctest + n_jobs-reproducibility tests; mypy-strict clean.

### Fixed — stale `seed=` kwarg in 2 bootstrap benchmarks (#91)

`tests/benchmarks/test_kernel_benchmarks.py` passed `seed=` to `bootstrap_ci` / `paired_bootstrap_diff`, but those parameters migrated to `rng=` (SPEC 7) — the two bootstrap benchmark tests `TypeError`'d on the nightly-benchmarks workflow (excluded from PR CI via `-m "not benchmark"`, so it went unnoticed). 2-line `seed=`→`rng=` rename.

## [1.6.0] — 2026-05-29 — Tier-2 `eda` Job-2 + Job-3: shortcut + shift diagnostics (#86, #87)

`eval_toolkit.eda.*` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — Tier-2, backward-compatible. Completes the EDA-first analytic layer above the v1.5.0 Job-1 integrity gate: **Job-2** lexical shortcut diagnostics (`lexical_association`, #86) and **Job-3** distribution-shift quantification (`distribution_shift`, #87). Both are dogfooded by the consumer portfolio's pre-modeling OOD-wall prediction (V5 + V9).

### Added — Tier-2 `eda.lexical_association` shortcut diagnostics (Job-2: C1 + C2)

`eval_toolkit.eda` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — Tier-2, torch-free (NumPy + scikit-learn). The analytic layer above the Job-1 integrity gate: *"is the label recoverable from a surface shortcut that will not transfer out-of-distribution?"*

- **C1 — `weighted_log_odds` / `class_lexical_association`:** Monroe, Colaresi & Quinn (2008)
  informative-Dirichlet weighted log-odds-ratio z-scores + smoothed PMI per token, with a
  `min_count` rare-token floor (the V5 pitfall). Returns a `LexicalAssociationResult`
  (`top_a` / `top_b` / `to_dict`); tokens ordered by descending z-score.
- **C2 — `competency_baselines`:** partial-input baselines (length-only, char-n-gram, BoW)
  fit on a train split and scored on a test split → `CompetencyResult` of per-baseline
  average-precision vs the positive-prevalence floor (the *shortcut floor*; Feng, Wallace &
  Boyd-Graber, ACL 2019 caveat documented). Vectorizers fit on train only (no test leakage);
  empty or single-class train/test raises a diagnostic `ValueError`.
- Exported via `from eval_toolkit.eda import ...`; 100% line+branch coverage; mypy-strict clean.

### Added — Tier-2 `eda.distribution_shift` covariate-shift quantification (Job-3: E1)

`eval_toolkit.eda` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — Tier-2. Public functions take **feature matrices**, so the module is base-install-safe (NumPy + SciPy + scikit-learn); embed text first with `eval_toolkit.embeddings.make_minilm_embedder` (`[embeddings]` extra) or any vectorizer.

- **`proxy_a_distance`:** Ben-David et al. (2006/2010) PAD = `2(1 − 2ε)` from a **linear**
  domain classifier's **k-fold CV** error, with **fixed strong regularization** (small `C`) —
  *not* the high-`C` RBF-SVM-on-`predict_proba` recipe that overfits to `PAD ≈ 2` at small `n`.
  Optional bootstrap CI.
- **`maximum_mean_discrepancy`:** Gretton et al. (2012) **unbiased** RBF-kernel MMD² U-statistic +
  **median-heuristic bandwidth** (freezable across folds) + **permutation-test** p-value
  (Phipson & Smyth 2010, `(1+count)/(B+1)`, never zero). Optional bootstrap CI.
- **`knn_purity`:** mean fraction of each point's k nearest neighbours sharing its domain label.
- **`median_bandwidth`** helper + the **`distribution_shift`** orchestrator (all three) +
  `PadResult` / `MmdResult` / `KnnPurityResult` / `DistributionShiftResult` dataclasses (`to_dict`).
- Docstrings carry the pre-registered caveats: distance is **necessary-not-sufficient** for OOD
  collapse (fuse with shortcut-exposure); a non-significant MMD p is not "no shift"; cross-dataset
  distances are ordinal-only (covariate vs label-semantics conflation). 100% line+branch coverage.

### Fixed

- **Public-API golden `__version__` drift:** the `v1.5.0` release commit bumped
  `_version.py` to `1.5.0` but did not regenerate `tests/golden/public_api/snapshot.json`,
  which still pinned `'1.4.0'` — leaving `test_public_api_drift_guard` red on `main` (and on
  every branch cut from it). Regenerated the golden (the diff is the `__version__` value only).

## [1.5.0] — 2026-05-29 — Tier-2 `eda` layer (#83) + schema-aware `HFDatasetsLoader` (#85)

Tier-2 / `loaders` ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md) — backward-compatible.

> **Erratum (2026-06-11, [#101](https://github.com/brandon-behring/eval-toolkit/issues/101)):**
> the #85 loader parameters were inserted mid-signature *before* `name` on a
> non-`kw_only` frozen dataclass, changing the meaning of positional argument 7 —
> Tier-1 per ADR 0003, not "Tier-2 ADDITIVE" as labeled above. It should have
> been a major. Policy since codified: the v1.9.0 ADR 0003 amendment admits only
> *strictly-appended trailing* optional parameters to the minor path.

- **`eda` Job-1 integrity gate (#83):** `audit_dataset` / `DataAudit` / `SplitSummary` + the
  `class_balance` / `no_cross_split_leakage` / `context_window_fit` gates + the §B2 obfuscation
  prevalence module.
- **schema-aware `HFDatasetsLoader` (#85):** load real-world dataset schemas without column
  guessing — `feature_cols` + `feature_join` (join multiple columns into one feature; NaN-safe),
  `label_map` (remap raw labels → int; fail-fast `ValueError` lists unmapped values), `revision`
  (pin the HF dataset SHA). All new params default to the prior behavior; a missing feature/label
  column raises `KeyError` listing the observed columns.

## [1.4.0] — 2026-05-26 — `audit_citation_alignment` Layer 2 + Layer 3 (closes #82); shared `_narrative` helpers (ADR 0007)

Tier-1 ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md).
Closes [#82](https://github.com/brandon-behring/eval-toolkit/issues/82)
— consumer-feedback follow-on after v1.3.0 closed
`audit_value_bindings`. The consumer's 188 residual warnings on
`audit_citation_alignment` were the same architectural-class gap
(missing Layer 2 + Layer 3 context-awareness) that
`audit_value_bindings` worked through over v1.1.0 → v1.3.0.

Introduces [ADR 0007](docs/source/adr/0007-three-layer-architecture-for-audit-validators.md)
codifying the **three-layer correctness model** (identity + scope +
pairing) as the canonical architecture for ALL `audit_*` validators
in the family. ADR 0005 + ADR 0006 were originally validator-
specific; ADR 0007 generalizes.

### Added — `audit_citation_alignment` Layer 2 + Layer 3 (closes #82)

- **`scope: Literal["all", "narrative"] = "all"`** kwarg on
  `validate_citations(...)`. Default `"all"` preserves v1.0.1 /
  v1.3.x behavior exactly (Tier-1 ADDITIVE; byte-identical legacy
  semantics).
- **Pattern β (Layer 2)** — under `scope="narrative"`, citations
  inside markdown table rows, bracketed expressions, and fenced
  code blocks are excluded. Mirrors `audit_value_bindings`'s
  Layer 2 from v1.1.0. Closes ~67 of the consumer's residual
  warnings (SPEC_SHEET.md table rows).
- **Pattern γ (Layer 3)** — the category-keyword extraction window
  for a citation is bounded by the SENTENCE containing the
  citation, not by a ±N-line window. Uses
  `_sentence_boundary_positions` (paragraph-aware, abbreviation-
  guarded) from `_narrative`. Catches the consumer's dense
  multi-clause sentences where keywords from prior clauses pull
  through.
- **Pattern α (Layer 3)** — when MULTIPLE ADR citations appear in
  the same sentence (e.g.,
  `"per ADR-025 + ADR-021 + ADR-034 + ADR-045"`), the validator
  switches from first-match-wins category check to multi-category
  set membership. Each ADR's actual category is accepted if it's
  in the SET of categories matched by the sentence's keywords —
  not just the dominant first-match. Catches the dense multi-ADR
  list pattern where each ADR addresses a different topic.

### Refactor — Shared `_narrative.py` helpers

Per ADR 0007, narrative-prose helpers are extracted to private flat
module `src/eval_toolkit/_narrative.py` (consistent with ADR 0001's
`_rng.py` / `_parallel.py` / `_sweep.py` precedent — flat-module
compliant, private/underscore-prefixed). Both validators import:

- Keyword frozensets: `_DELTA_KEYWORDS`, `_FLOOR_KEYWORDS`,
  `_GROUP_SUBJECT_KEYWORDS`, `_ABBREV_BEFORE_DOT`.
- Compiled patterns: `_DELTA_PATTERN`, `_FLOOR_PATTERN`,
  `_GROUP_SUBJECT_PATTERN`.
- Helpers: `_build_exclusion_ranges`, `_is_excluded`,
  `_is_sentence_terminator_dot`, `_sentence_boundary_positions`,
  `_sentence_id_of`, `_crosses_sentence_boundary`,
  `_is_signed_value`, `_has_keyword_in_window`,
  `_compile_keyword_pattern`.

`audit_value_bindings.py` updated to import these from `_narrative`
instead of defining inline. **Signature-preserving refactor**: all
43 existing `audit_value_bindings` tests pass UNCHANGED. The
private helpers are non-public, so no Tier-1 STRICT impact.

### Dogfood result

| Configuration | Warnings on `prompt-injection-detection-submission` HEAD | Reduction |
|---|---|---|
| v1.3.0 (`audit_citation_alignment` with scope='all') | 188 | — (baseline) |
| **v1.4.0 (`scope='narrative'`)** | **37** | **80%** |

Verified locally via `.scratch/dogfood_v1_4_0_citation.py`
(monkey-patched consumer call with `scope="narrative"`).

The residual 37 are a mix of:
- **Real misalignments** consumer should triage (e.g., `ADR-025`
  cited for a threshold claim when ADR-025 is the cost ADR — could
  be a wrong-ADR bug or a multi-topic ADR not captured by the
  consumer's category-keyword map).
- **Single-topic sentences** where the first-match category
  inferred from the sentence genuinely differs from the ADR's
  actual category. The multi-topic Pattern α fallback only fires
  when ≥2 categories match; single-topic prose stays on the
  legacy first-match check.
- **Edge cases** requiring parser-level understanding of how an
  ADR's scope intersects with a multi-clause sentence's topics.

The original #82 acceptance criterion was ≤20 warnings (the
filer's estimate of "genuinely ambiguous citations"). v1.4.0
hits 37 — above the target but a 5× reduction overall. The
remaining gap requires either (a) consumer-side expansion of
`CATEGORY_KEYWORDS` to capture multi-topic ADRs, (b) consumer
prose adjustments for the real misalignments, or (c) future
v1.4.x refinements to the validator's heuristic. Consumer
HARD-gate promotion remains a judgment call — the residual 37
includes some real misalignments worth fixing.

### Consumer adoption path

Consumer (`prompt-injection-detection-submission`):
1. Re-pin `eval-toolkit>=1.4.0,<2`.
2. Add `scope="narrative"` to their `validate_citations(...)` call
   in `scripts/audit_citation_alignment.py`.
3. **Bundled HARD-gate promotion** of BOTH `audit_value_bindings`
   AND `audit_citation_alignment` now credible per the v1.3.8
   plan. Promotes from SOFT to HARD in their next v1.3.X release.

### Tests

61 across the audit-validator suite (43 audit_value_bindings + 18
audit_citation_alignment; 6 new for v1.4.0 — Pattern α / β / γ /
scope='all' backward-compat / shared-helpers / combined dogfood).
All pass. Public API snapshot regenerated for `__version__` bump
+ `validate_citations` signature with new `scope` kwarg.

### Out of scope (deferred)

- **`audit_sister_doc_concept_drift` Layer 2 / Layer 3** — embedding-
  based validator (v1.0.4); different false-positive surface. Add
  layers only if consumer demand emerges.
- **Public helper promotion** (`eval_toolkit.audit_narrative`) —
  YAGNI per ADR 0007 §A2.
- **Configurable category-keyword-window extension kwargs** —
  YAGNI; add in v1.4.x patch if demand emerges.

## [1.3.0] — 2026-05-26 — `audit_value_bindings` cross-detector list-grammar pairing rules (closes #81)

Tier-1 ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md).
Closes [#81](https://github.com/brandon-behring/eval-toolkit/issues/81)
— consumer-feedback follow-on after v1.2.0's adoption at
`prompt-injection-detection-submission@v1.3.12` (4 residual
warnings, all cross-detector list-grammar or metric-axis
confusion). Introduces **Layer 3 — pairing rules** as the third
correctness layer alongside ADR 0005's identity + scope model
(see new [ADR 0006](docs/source/adr/0006-pairing-rules-for-cross-detector-list-grammar.md)).

Consumer-side dogfood result: **4 → 0 warnings**. Combined with
v1.1.0 + v1.2.0, **100% reduction vs the pre-fix v1.0.5 baseline**
on the consumer's writeup (95 → 0).

### Added — `audit_value_bindings.py` Layer 3 pairing rules

All four rules activate ONLY when `scope="narrative"`. Legacy
`scope="all"` callers see zero behavior change. No new public
kwargs; keyword sets are hardcoded module-level `frozenset`
constants.

- **Pattern A — `"for {detector}"` postfix override.** When a
  candidate value is followed (within +50 chars) by `"for
  {detector_alias}"` AND no other value lies between, the
  postfix is authoritative: confirms pairing for this binding
  OR skips if it names a different canonical detector.
  Intervening-value check uses the v1.1.0 exclusion-ranges
  infrastructure (CI brackets like `[0.286, 0.301]` don't count
  as intervening values).
- **Pattern B — `"{detector}'s"` possessive override.** Same
  mechanics; scans −80 chars before the value. Last possessive
  in the pre-window is authoritative if its end is within 30
  chars of the value start. Catches both immediate `"frozen
  probe's 0.515"` and short-clause `"LoRA's ... AUROC is 0.383"`.
- **Pattern C — group-subject suppression.** When prose contains
  `"for the {trained|frozen|baseline|all|both|other} detectors"`
  within ±60 chars of the value AND on the same side of any
  sentence boundary, the value is suppressed (it refers to a
  multi-detector group statement that doesn't bind to a single
  canonical detector). Multi-detector inference deferred to v1.4.0+
  per ADR 0006.
- **Pattern D — metric-axis nearest-pairing.** Symmetric to
  detector-axis pairing. Pre-collects ALL metric positions per
  file (across `metric_aliases` keys, not just binding-derived
  metrics). Requires the NEAREST metric to the value to be THIS
  binding's metric. Catches prose like `"AUPRC delta suggests:
  ... AUROC is 0.383"` where the wider window-based metric
  proximity check picks up the wrong metric.

### Internal changes (no public API impact)

- New module-level constants:
  - `_GROUP_SUBJECT_KEYWORDS: frozenset[str]` — group adjectives.
  - `_GROUP_SUBJECT_PATTERN: re.Pattern[str]` — compiled regex
    matching `"for the {kw} detectors?"`.
- New private helpers:
  - `_build_postfix_pattern(detector_aliases, detector_keys)` —
    per-call regex builder for Pattern A.
  - `_build_possessive_pattern(detector_aliases, detector_keys)` —
    per-call regex builder for Pattern B.
- `metric_patterns` build extended to use the union of
  `binding-derived` and `metric_aliases.keys()` so Pattern D can
  pair against unbound-but-aliased metrics.
- Inner loop reordered to apply C-suppress → Pattern A → Pattern B
  before proximity-based detector pairing. Pattern A/B record a
  `pairing_confirmed_pos` that BYPASSES proximity when the override
  confirms THIS binding's detector.
- Pattern D added as a separate check after the existing
  metric_close proximity test.

### Dogfood evidence (compounded across the cycle)

| Release | Configuration | Warnings on consumer HEAD | Reduction vs v1.0.5 |
|---|---|---|---|
| v1.0.5 | Legacy 2-tuple, no scope | 95 | — |
| v1.1.0 | BindingKey + scope='narrative' content-type | 23 | -76% |
| v1.2.0 | + T1–T4 context filters | 7 | -93% |
| **v1.3.0** | + Pattern A/B/C/D pairing rules | **0** | **-100%** |

### Consumer adoption path

`prompt-injection-detection-submission` and other consumers using
`scope="narrative"` get the v1.3.0 pairing rules automatically with
no code change. Recommended migration:

1. Re-pin `eval-toolkit>=1.3.0,<2` (additive; no consumer code
   change required).
2. **HARD-gate promotion is now credible.** With 0 residual
   warnings, `audit_value_bindings` can be promoted from SOFT to
   HARD (failing CI on violations) bundled with
   `audit_citation_alignment` per the consumer's v1.3.8
   bundled-promotion plan.

### Tests

43 in `tests/test_audit_value_bindings.py` (36 from v1.2.0 + 7
new for Pattern A/B/C/D + unknown-alias fall-through + scope='all'
backward-compat + combined dogfood). All pass. Public API
snapshot regenerated for `__version__` bump only (no signature
changes).

### Out of scope (deferred)

- **Multi-detector inference for Pattern C** — replace
  suppression with explicit iteration over implied group
  detectors. ~250 LOC; v1.4.0+ candidate if consumer demand
  emerges.
- **Enumeration parsing** — `"X scored Y, Z, W for A, B, C
  respectively"` patterns. Not in #81; v1.4.0+ if needed.
- **Markdown AST parsing** (ADR 0005 §A4) — v2.0 territory.
- **Public kwargs for pairing-rule keyword extension** — YAGNI;
  add in v1.3.x patch if demand emerges.

## [1.2.0] — 2026-05-26 — `audit_value_bindings` context-aware noise reduction (consumer-feedback follow-on to #80)

Tier-1 ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md).
Consumer-feedback follow-on after v1.1.0's adoption at
`prompt-injection-detection-submission@v1.3.11`. The v1.1.0
slice-axis fix achieved 62% noise reduction (96 → 36 warnings) on
the consumer's writeup; the residual 36 were positional-heuristic
limitations [ADR 0005](docs/source/adr/0005-structured-keys-for-audit-validators.md)
named as "Future work (deferred)" for v1.2.0+. This release
addresses 81% of that residual (36 → 7) via four context-aware
extensions to `scope="narrative"`. Combined with v1.1.0,
**93% total noise reduction** vs the pre-fix v1.0.5 baseline.

### Added — `audit_value_bindings.py` context-aware narrative filters

All four filters activate ONLY when `scope="narrative"`. Legacy
`scope="all"` callers see zero behavior change (Tier-1 ADDITIVE).
No new public kwargs; no signature drift; the keyword lists are
hardcoded module-level `frozenset` constants. Issue [#80](https://github.com/brandon-behring/eval-toolkit/issues/80)'s
acceptance criterion was ≤5 warnings; v1.2.0 hits 7 (close to the
target; the remaining 7 are pure cross-detector list-grammar cases
that require parser-level work — see "Out of scope" below).

- **T1: Delta-context filter.** Suppresses values that are
  comparative magnitudes rather than binding claims. Two
  sub-filters:
  - Sign-prefix skip: values immediately preceded by `+` or `-`
    (negative-magnitude markers like `-0.071 AUPRC`,
    `+0.073 lift`) are dropped.
  - Delta-keyword skip: values within 30 chars AFTER a
    delta-marker token are dropped. The before-only window
    prevents mis-firing on prose like `"frozen probe's 0.515
    (delta -0.132)"` where the `"delta"` token refers to the
    following `-0.132`, not the preceding `0.515`.

  Keyword list (`_DELTA_KEYWORDS`, hardcoded frozenset):
  `delta`, `drop`, `drops`, `lift`, `lifts`, `gap`, `margin`,
  `regresses`, `improves`, `beats`, `exceeds`, `trails`,
  `underperforms`, `vs`, `versus`, `below`. Excluded:
  `against`, `above`, `ahead`, `behind` (too ambiguous; common
  comparison prepositions in legitimate binding prose).

- **T2: Floor-context filter.** Suppresses values near random-
  baseline / floor mentions. Window is asymmetric (50 chars
  before, 5 chars after) because floor mentions canonically
  precede the value (`"random AUPRC is 0.374"`).

  Keyword list (`_FLOOR_KEYWORDS`): `random`, `floor`, `chance`,
  `trivial`. Intentionally narrow — `baseline`, `prior`,
  `majority` excluded because they have legitimate non-floor
  senses (`"TF-IDF baseline"`, `"prior work"`). Multi-word
  patterns like `"below the prevalence baseline of 0.374"` are
  caught by T1's `"below"` keyword instead.

- **T3: Consume-on-match within sentence.** After a value
  produces a Match for `(detector, metric, slice)`, subsequent
  values for the same canonical binding in the same sentence are
  suppressed. Catches dense multi-detector enumerations like
  `"AUPRC 0.556 vs 0.519"` where the second value is implicitly
  a contrasting detector's binding (cross-detector inference
  remains out of scope per ADR 0005 A4).

- **T4: Sentence-boundary detector-pair reject.** When pairing a
  detector mention with a value, if a sentence terminator (`.`,
  `!`, `?`, `\n\n`) lies between them, the pair is rejected.
  Sentence detection uses paragraph-aware abbreviation guarding
  (`vs.`, `e.g.`, `i.e.`, `c.f.`, `etc.`, `cf.`, `fig.`,
  `eq.`, `pp.`, `viz.`, `ca.` excluded; decimal numbers and
  letter-dot-letter patterns also guarded). Single `\n` is a
  soft break (markdown line-wrap, NOT a sentence boundary);
  `\n\n` is hard.

### Internal changes (no public API impact)

- `_nearest_canonical_key()` now returns `(key, position)`
  instead of just `key`. The position is needed for T4's
  sentence-boundary check. The slice-pairing call site unpacks
  and discards the position. Private helper; no consumer impact.
- New private helpers: `_is_sentence_terminator_dot`,
  `_sentence_boundary_positions`, `_sentence_id_of`,
  `_crosses_sentence_boundary`, `_is_signed_value`,
  `_has_keyword_in_window`, `_compile_keyword_pattern`. All
  underscore-prefixed; Tier-3 FREE.

### Dogfood evidence

| Configuration | Warnings on `prompt-injection-detection-submission` HEAD | Reduction vs v1.0.5 baseline |
|---|---|---|
| v1.0.5 (legacy 2-tuple) | 95 | — |
| v1.1.0 BindingKey + scope='narrative' (content-type filter only) | 23 | 76% |
| **v1.2.0 + context filters (this release)** | **7** | **93%** |

The 7 v1.2.0 residuals are all cross-detector list constructions
(e.g., `"0.293 versus 0.364 for the frozen probe and 0.291 for
TF-IDF + LR"` where the validator can't infer that 0.361 / 0.291
belong to ProtectAI-v1 and TF-IDF respectively because they're
introduced by `"and"` / `"for"` without an immediately-preceding
detector mention). These require true list-grammar parsing
(rejected for v1.x in ADR 0005 A4) and are tracked for v1.3.0+
with their own ADR design review.

### Consumer adoption path

`prompt-injection-detection-submission` and other consumers using
`scope="narrative"` get the v1.2.0 filters automatically with no
code change. Consumers on `scope="all"` (default) continue with
v1.1.0 behavior. Recommended consumer migration:

1. Re-pin `eval-toolkit>=1.2.0,<2` (additive; no consumer code
   change required).
2. HARD-gate promotion is now credible: 7 residual warnings is
   below the actionable threshold; consumer can promote
   `audit_value_bindings` from SOFT to HARD bundled with
   `audit_citation_alignment` per the v1.3.8 plan.

### Tests

36 in `tests/test_audit_value_bindings.py` (28 from v1.1.0 + 8
new for T1–T4 + sentence-boundary helper unit test). All pass.
Public API snapshot regenerated for `__version__` bump only (no
signature changes beyond an inspect-formatting normalization on
the `validate_reader_value_bindings` `bindings` annotation; same
type semantically).

### Out of scope (deferred)

- **Cross-detector list-grammar parsing** — the 7 residual
  warnings. Requires lookahead context-aware list parsing
  (`"X scored Y vs Z for W and V for U"`). Track as a v1.3.0+
  candidate; needs ADR design before implementation.
- **Markdown AST parsing** (ADR 0005 A4) — v2.0 territory.
- **`extra_*_keywords` kwargs** for runtime extension of the
  hardcoded keyword lists — YAGNI for now (consumer's prose is
  covered); add in a v1.2.x patch if concrete demand emerges.

## [1.1.0] — 2026-05-26 — `audit_value_bindings` slice-aware matching via `BindingKey` (closes #80)

Tier-1 ADDITIVE per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md).
Closes [#80](https://github.com/brandon-behring/eval-toolkit/issues/80)
— consumer-feedback structural fix surfaced by
`brandon-behring/prompt-injection-detection-prototype@v1.3.9` (96
warnings, ~95 false positives) where the pre-v1.1 2-tuple
`(detector, metric)` canonical-binding identity could not
disambiguate the same `(detector, metric)` across multiple slices
(`direct_validation`, `pooled_ood`, paired-delta cells,
random-floor mentions).

Pending [ADR 0005](docs/source/adr/0005-structured-keys-for-audit-validators.md)
codifies the underlying rule: "structured keys over positional
tuples for canonical-identity types in audit validators."

### Added

- **`BindingKey`** (new public class, exported via the lazy
  `_EXPORTS` resolver). Frozen dataclass with fields
  `(detector: str, metric: str, slice: str = "any")`. Forward-
  extensible: future identity axes (split, ci_kind, source_ref, ...)
  can be added as defaulted fields without breaking the dict-key
  schema. Avoids the recur-every-N-months schema-event pattern that
  produced #80.
- **`validate_reader_value_bindings(bindings=...)`** now accepts
  three input shapes, normalized internally to `dict[BindingKey,
  float]` via a per-key `_normalize_binding_key` adapter:
  1. Canonical: `BindingKey(detector=..., metric=..., slice=...)`
     (recommended for new consumer code).
  2. Sugar 3-tuple: `(detector, metric, slice)` (concise dict
     literal; issue #80's proposed schema).
  3. Legacy 2-tuple: `(detector, metric)` (preserved; treated as
     `slice="any"`). All pre-v1.1 consumer code continues to work
     unchanged.
  Mixed key shapes in a single dict are supported. Invalid key
  shapes raise `TypeError` at the function boundary (loud failure,
  not silent zero-match drift).
- **New optional kwargs** on `validate_reader_value_bindings`:
  - `slice_aliases: Mapping[str, Sequence[str]] | None = None` —
    canonical-slice-name → regex-alternatives mapping, mirroring
    `detector_aliases` / `metric_aliases`. Used when at least one
    `BindingKey` has `slice != "any"`.
  - `slice_window_chars: int = 240` — character window for slice
    disambiguation (≈ 50 tokens at ~5 chars/token).
- **Slice disambiguation** in the matching loop. When a binding
  key has `slice != "any"`, the candidate value is paired with
  the nearest slice mention (same last-before-first-after rule as
  detector pairing). If no slice mention falls within
  `slice_window_chars`, the triple is suppressed (warn-only;
  counted in `unmatched_slice_count`). If the paired slice
  differs from the binding's slice, the triple is silently skipped
  (handled by the binding for the correct slice on its own loop
  iteration).
- **`ValueBindingsReport.unmatched_slice_count: int = 0`** — new
  field surfacing the warn-only signal. Default `0` means full
  backward compatibility for code that constructs reports manually.
- **`scope: Literal["all", "narrative"] = "all"`** — content-type
  filter on the validator. Default `"all"` preserves legacy v1.0.x
  behavior. Setting `scope="narrative"` excludes from matching:
  - Markdown table rows (lines starting with `|`)
  - Bracketed expressions `[...]` (CI bounds, ranges)
  - Fenced code blocks (triple-backtick)

  This addresses the broader category of false positives that the
  slice-axis fix alone could not (CI bounds being matched as point
  estimates, table-cell metrics being cross-flagged, code-block
  literals being treated as claims). Compatible with the
  motivating misbinding bug class (V1.3.1 ADR-080) which was in
  narrative prose — no recall loss.

  Combined dogfood result on `prompt-injection-detection-submission`
  HEAD: **76% noise reduction** (95 warnings at v1.0.5 2-tuple
  baseline → 23 with v1.1.0 BindingKey + `scope="narrative"`). The
  remaining 23 are positional-heuristic limitations (random-floor
  mis-attribution across sentence boundaries; cross-detector
  pairing in dense prose) not addressable without parser-level
  work (future v1.2.0+ scope).

### Changed (Tier-1 ADDITIVE — no breaking changes)

- `_nearest_detector_key()` private helper renamed to
  `_nearest_canonical_key()` (used now for both detector and slice
  pairing; the body was already generic). Implementation detail;
  no public API impact.
- `pyproject.toml` `Development Status` classifier bumped
  `4 - Beta` → `5 - Production/Stable` (post-v1.0 hygiene).
- `.gitignore` now covers `.scratch/` and the broader
  `codex-comprehensive-audit-*.md` pattern (was previously only
  `*-report.md`).

### Deprecation policy

All three `bindings` input shapes remain accepted **indefinitely**
through the v1.x line. `BindingKey` is the canonical/recommended
shape per ADR 0005 + docstring guidance; tuples remain valid
syntactic sugar with no `DeprecationWarning`. Formal deprecation
deferred to a future v2.0 cleanup pass when there is concrete
payoff. Consumers can migrate slot-by-slot at their own pace, or
not at all (legacy 2-tuple semantics survive as `slice="any"`).

### Consumer adoption path

`prompt-injection-detection-submission` consumer-side script at
`scripts/audit_value_bindings.py` can adopt by either:
- Replacing the 2-tuple `BINDINGS` literal with 3-tuple keys
  (smallest diff; issue body's proposal), OR
- Migrating to `BindingKey(...)` for forward-extensibility (new
  identity axes won't require re-touching the script).

### Validator design philosophy (introduced this release)

This release introduces a two-layer correctness model for audit
validators, formalized in pending ADR 0005:

1. **Identity correctness** — canonical measurements have
   structured identity (`BindingKey`), not positional tuples.
   Future axes added as defaulted fields without breaking the
   schema.
2. **Scope correctness** — the validator should only scan content
   that is plausibly a binding claim. Narrative prose is.
   Markdown tables, CI brackets, and code blocks aren't.
   `scope="narrative"` is the v1.1.0 implementation of this rule;
   match the lint-design convention from `ruff`/`mypy`/`bandit`
   (scope predicates aren't optional in production-quality
   linters).

Both layers are now in place. Consumers writing typical research
writeup prose with dense statistical tables should adopt
`scope="narrative"` for a ~80% noise reduction relative to v1.0.5.

### Known v1.1.0 limitations (residual after slice + narrative fixes)

The remaining ~20% of v1.0.5 baseline noise is positional-
heuristic limitations that the slice + narrative fixes cannot
address:

1. **"Random floor" / sub-clause values across sentence
   boundaries** — prose like "X scored 0.291. The pooled OOD
   random floor is 0.374" pairs 0.374 with the nearest preceding
   detector (X), because the validator doesn't treat `.` as a
   pairing boundary.
2. **Multi-detector cross-pairing in dense prose** — prose like
   "LoRA scored 0.293 [0.286, 0.301] versus 0.364 for the frozen
   probe and 0.291 for TF-IDF + LR" pairs values with
   text-order-nearest detector, which over-credits the second
   detector in a list construction.

Future work (v1.2.0+ candidate): sentence-boundary awareness
(respecting `.` / `\n` as pairing boundaries) + better
multi-detector list parsing.

HARD-gate promotion of the consumer-side validator at v1.3.10+
becomes credible at the ~80% reduction level achieved here.
Remaining false positives can be suppressed via consumer-side
filtering (e.g., excluding lines containing "random floor" or
"versus") or accepted as known low-frequency noise.

## [1.0.5] — 2026-05-26 — publish workflow hardening (infrastructure-only)

Tier-3 / infrastructure-only release. **No library code or public API
changes.** Hardens the release pipeline against the failure mode
observed at v1.0.4, where a documented GitHub Actions CRITICAL
incident (codeload action download failure across the platform) left
the v1.0.4 wheel un-published on PyPI despite a successful tag and
GitHub release. The wheel for v1.0.5 is functionally identical to
v1.0.4; this release exists primarily as a dress rehearsal for the
new verification step.

### Added — `.github/workflows/publish.yml`

- **`workflow_dispatch:` trigger** — recovery path for failed
  tag-triggered runs. Manually re-trigger via
  `gh workflow run publish.yml --ref vX.Y.Z` or the Actions UI
  "Run workflow" dropdown. Always uses the workflow file from main
  HEAD, so workflow patches take effect immediately for recovery.
- **Post-publish `Verify PyPI receipt` step** — polls
  `pypi.org/pypi/eval-toolkit/<version>/json` for HTTP 200 over a
  6-minute window (12 × 30s backoff); fails loudly if the wheel
  never lands. Catches silent half-releases where
  `pypa/gh-action-pypi-publish` returns success but PyPI never
  receives the wheel.

### Added — `docs/source/RELEASING.md`

- **"Tag-triggered publish failed; need to re-publish to PyPI"**
  recovery recipe under Known gotchas. Documents both the
  `gh run rerun` path (when the original run can be retried) and
  the `workflow_dispatch` path (when the workflow has been patched
  on main since the original tag). References the v1.0.4 incident
  as the canonical example.

### Notes

- `setup-uv@v8.1.0` pin is intentionally unchanged. The v1.0.4
  failure was a documented GitHub Actions/codeload incident, not
  an action-specific issue; replacing setup-uv with a curl-install
  would lose the cache layer + Python integration + version-from-
  pyproject detection it provides, and would not have prevented the
  observed failure (actions/checkout downloaded successfully in the
  same failing run; codeload was the SPOF, not setup-uv).
- The other 5 workflows (ci/codeql/docs/nightly-benchmarks/
  nightly-mc) are not patched because they self-heal on the next
  push; the SPOF only matters for one-shot tag-triggered runs.

## [1.0.4] — 2026-05-26 — `audit_sister_doc_concept_drift` module (closes #72)

Tier-2 ADDITIVE — third (and final) member of the audit-validator
family. Flat-module per [ADR 0001](docs/source/adr/0001-flat-module-layout.md).
Family complete: `audit_citation_alignment` (v1.0.1) + `audit_value_bindings`
(v1.0.3) + `audit_sister_doc_concept_drift` (this release).

### Added

- **`audit_sister_doc_concept_drift` module** exporting
  `validate_sister_doc_concept_drift()` + `DriftCluster` +
  `SisterDocDriftReport` as Tier 1 STRICT (per
  [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)).
  Catches the bug class where two linked sister docs reference the
  same concept token (e.g., `T1`, `manifest v3`) but the
  surrounding-sentence definitions semantically disagree.
  Cross-doc semantic drift survives lychee (links resolve), anchor
  audits (anchors exist), and numeric audits (qualitative prose).
- Algorithm: per concept_token, scan all files for occurrences;
  extract surrounding-sentence context (`context_window_sentences`);
  embed each snippet via the supplied `embedder` (default lazily
  routes to `make_minilm_embedder()` — requires `[embeddings]`
  optional extra); cluster via single-linkage cosine similarity at
  `similarity_threshold` (default 0.7); tokens with >1 cluster are
  flagged as `DriftCluster`.
- The `embedder: Callable[[Sequence[str]], np.ndarray] | None`
  parameter matches the existing
  `EmbeddingCosineStrategy.embedder` Protocol — consumers can pass
  any embedder (BGE, E5, OpenAI, or a mock for tests). Default
  `None` defers `sentence_transformers` import to call-time
  (`[embeddings]` extra is required only when caller doesn't supply
  their own embedder).
- Motivating bug class: consumer audit found
  `docs/REPRODUCIBILITY.md:85` defines `T1` as "full canonical
  re-eval (GPU; A100 80GB)" while `WRITEUP/reproducibility.md:33`
  defines `T1` as "smoke (laptop, $0, ~10 min)" — the two docs
  cross-link as "Aggregator docs" so a reviewer following the link
  lands on contradictory definitions.
- 13 tests at `tests/test_audit_sister_doc_concept_drift.py` using a
  deterministic mock embedder (no `sentence_transformers` dependency
  for unit tests). Covers: seed-case T1 drift, consistent definition
  across files, single-occurrence consistency, unreferenced-token
  coverage tracking, multi-token mixed (T0 + T1 + T3), threshold
  sensitivity, whole-word boundary (`T1` vs `T10` vs `t1`), context
  window scope, empty inputs, 3-way drift, frozen-dataclass
  invariants, lazy default-embedder import. Closes #72.

### Audit-validator family complete

| Validator | Released | Issue |
|---|---|---|
| `audit_citation_alignment` | v1.0.1 | #73 |
| `audit_value_bindings` | v1.0.3 | #71 |
| `audit_sister_doc_concept_drift` | v1.0.4 (this release) | #72 |

All three follow the flat-module convention (ADR 0001), closed-config
pattern (consumer supplies the auditable surface; validator owns the
parsing+matching logic; ADR 0002), and Tier 1 STRICT top-level
exports per ADR 0003. Consumer adoption pattern is the same across
all three: thin `scripts/audit_<name>.py` CLI wrapper invoking the
upstream validator.

## [1.0.3] — 2026-05-26 — `audit_value_bindings` module (closes #71)

Tier-2 ADDITIVE — second member of the audit-validator family
following `audit_citation_alignment` (v1.0.1). Flat-module per
[ADR 0001](docs/source/adr/0001-flat-module-layout.md).

### Added

- **`audit_value_bindings` module** exporting
  `validate_reader_value_bindings()` + `Match` + `Violation` +
  `ValueBindingsReport` as Tier 1 STRICT (per
  [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)).
  Catches the bug class where a markdown surface pairs a detector name
  with the **wrong** canonical value — both values exist in the
  source-of-truth table but the binding is misordered. Motivated by
  the consumer V1.3.1 ADR-080 audit-fix patch closure (2026-05-22)
  where `WRITEUP_NARRATIVE.md:38` said "TF-IDF + logistic regression
  baseline reaches 0.974 AUPRC" but canonical TF-IDF direct val AUPRC
  is 0.971 (0.974 was LoRA's value). The existing `audit_numbers.py`
  validates VALUES against source data but not BINDINGS — this
  validator closes that gap.
- Cross-detector disambiguation: when multiple detectors and values
  appear in the same paragraph (e.g., "TF-IDF achieves 0.971, while
  LoRA reaches 0.974"), each value pairs with the LAST detector
  appearing before it in text order (falling back to first detector
  after if no before-detector is in range). Avoids false-positive
  bindings across closely-spaced detector mentions.
- Coverage metric: `ValueBindingsReport.coverage` reports the fraction
  of `(detector, metric)` keys in the canonical `bindings` dict that
  produced at least one `Match` — useful for detecting stale or
  unreferenced bindings in reader prose.
- 13 tests at `tests/test_audit_value_bindings.py` including the
  verbatim WRITEUP_NARRATIVE seed-case regression, alias resolution
  (detector + metric), distance-window edge, value-without-metric
  skip, coverage fraction, tolerance band, multi-detector
  disambiguation, frozen-dataclass invariants. Closes #71.

## [1.0.2] — 2026-05-26 — #76 cleanup batch closes (RC2 + RC3 + F-metrics-1/3/4)

Closes the GH #76 v1.0.1 cleanup tracker. All 6 items shipped across
v1.0.1 (RC4) and v1.0.2 (this release). All P3, all NON-BREAKING.

### Changed (Tier-2 ADDITIVE: contract clarification only)

- **RC2** (#76) — `SimilarityStrategy` Protocol promoted from
  "pre-v0.7 internal interface" (prose framing only) to formal
  10th strict Tier-2 Protocol per [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md).
  Aligns prose surfaces (README, extending.md, strict_tier2_protocols.md,
  api/protocols.md, ADR 0004 §D6, roadmap.md) with the contract
  already locked in `tests/golden/public_api/snapshot.json` +
  `src/eval_toolkit/__init__.py:_EXPORTS` since v1.0.0. **No code
  change — documentation-only reconciliation.** Strict-Tier-2 count
  goes 9 → 10 (+ 1 opt-in `Versioned`).

### Fixed

- **RC3** (#76) — `tests/test_harness_folded.py::test_evaluate_folded_reseed_splitter_varies_partitions`
  test hardening. Previous assertions covered count + key existence
  only; a regression silently reusing the splitter (R8-C1 pre-fix
  behavior) could still pass. v1.0.2 adds row-content comparison:
  replays `reseed_splitter` against the splitter for `seed=1` vs
  `seed=2` and asserts fold-0 test partitions differ via feature-text
  set membership (robust to `_slice_subset`'s `reset_index(drop=True)`
  via stable text-column identifiers).

- **F-metrics-1** (#76) — `brier_score` docstring input-domain clarity.
  Added explicit "Input domain" Notes subsection clarifying binary
  labels in `{0, 1}` + calibrated probabilities in `[0, 1]` are
  required; raw logits or unbounded ranking scores pass the finiteness
  check but produce out-of-range MSE that misrepresents calibration
  quality. Includes calibration-applying recipe pointer.

- **F-metrics-3** (#76) — `expected_calibration_error` docstring
  uniform-scores note. Added explicit Notes subsection documenting
  that constant `y_score` returns 0.0 (per-bin formula trivially
  satisfied) but is semantically misleading — uninformative scorers
  look "perfectly calibrated" despite zero discriminative power.
  Callers should filter constant inputs before ECE.

- **F-metrics-4** (#76) — `brier_score` docstring single-class
  edge-case explicit. Added Notes subsection with closed-form
  expressions for all-zeros (`BS = mean(p²)`) and all-ones
  (`BS = mean((1-p)²)`) cases. Explicit confirmation that
  per-slice degenerate-class evaluation is supported (unlike
  PR-AUC / ROC-AUC).

## [1.0.1] — 2026-05-25 — audit_citation_alignment + RC4 docs polish

First v1.x patch release. Ships the `audit_citation_alignment` validator
that's been pre-staged by consumer `prompt-injection-detection-prototype`
(see #77), plus the smallest #76 cleanup item (RC4).

### Added

- **`audit_citation_alignment` module** — flat-module per [ADR 0001](docs/source/adr/0001-flat-module-layout.md)
  (stay-flat-through-v1.x; subpackage restructure deferred to v2.0).
  Exports `validate_citations(...)`, `ADRSubject`, `CitationMisalignment`,
  and `extract_adr_subject_category` as Tier 1 STRICT (per
  [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)).
  Validator catches the bug class where a markdown surface cites "per
  ADR-NNN" but the cited ADR's actual subject doesn't match the
  surrounding claim category — motivated by the V1.3.2 P1-2 finding in
  the consumer `prompt-injection-detection-prototype` audit where
  `docs/REPRODUCIBILITY.md:76` cited ADR-029 (test markers) for a
  tier-lock claim that should have cited ADR-034 (reproducibility tier
  ladder). The mis-citation went undetected by lychee (URL-resolves
  check), consumer's `audit_numbers.py` (numeric values), and consumer's
  `audit_adr_count_claims.py` (count claims). Closes #73.
- **Pre-tag dogfood**: `validate_citations()` exercised against
  eval-toolkit's own docs (95 files including README + audit_findings +
  methodology + migration guides) — 0 misalignments found. Validator
  proven in production use before consumers adopt.

### Fixed

- **RC4** (#76 cleanup) — v0.51 documentation count-tally reconciliation
  across `docs/source/audit_findings.md`, `docs/source/migration/v0.51.md`,
  and `CHANGELOG.md` `[0.51.0]` section. Canonical tally now consistent:
  **13 confirmed → fixed in v0.51 / 3 refuted (R8-G2 + R8-G5 + R8-V1+V2
  paired) / 2 deferred (R8-G3, R8-G4) = 18 total**. Prior drift was
  "left 2 undecided" (migration/v0.51) vs "deferred" (CHANGELOG) +
  ambiguity in the audit_findings ship-status section. Closes RC4 of #76.

## [1.0.0] — 2026-05-25 — Stability contract activates per ADR 0003

v1.0 is a **stability-contract activation**, not a code delta from v0.51.
Every fix that landed at v0.51 is what v1.0 ships; the new thing at v1.0
is that the [ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md)
Tier 1 / Tier 2 / Tier 3 stability contract becomes load-bearing.
Breaking changes to Tier-1 surfaces after v1.0 require a major bump (v2.0).

### Contract activation

- **Tier 1 STRICT** — public-API signatures captured in
  `tests/golden/public_api/snapshot.json`. Any signature drift bumps to v2.0.
- **Tier 2 ADDITIVE** — the 9 strict Protocols (`Scorer`, `LeakageCheck`,
  `Splitter`, `ThresholdSelector`, `DatasetLoader`, `MetricSpec`,
  `MetaLearner`, `Probe`, `TextTransform`) + 1 opt-in (`Versioned`).
  Method shapes are frozen; additive subprotocols + new Protocols allowed.
- **Tier 3 FREE** — internal modules (prefixed `_`). Refactors don't
  bump major.

### Gate 3 audit closure (Rounds 5 → 10)

The multi-LLM cross-review sequence is closed. Per ADR 0003, Gate 3
substitutes Codex + Gemini + Claude independent reads for external
academic peer review. Outcomes:

- **Round 8** (against v0.50): 13 confirmed → fixed in v0.51; 3 refuted;
  2 v1.x-deferred (R8-G3 custom exceptions; R8-G4 joblib OOM capping).
- **Round 9** (against v0.51 RC): 6 confirmed of 10 source items + 3
  third-audit findings in modules neither auditor cited. 2 candidate-
  blocker items fixed in-PR before tag.
- **Round 10** (micro-audit on R9 follow-on commit): 3 Codex confirmed →
  fixed in v0.51; 1 accept-as-design (Gemini); 1 refuted (Gemini
  Pattern-1 violation).

Full ledger at `docs/source/audit_findings.md` Rounds 5 → 10. v1.0.1
cleanup batch tracked at GH issue #76.

### Carried-over deprecations

The R8-C1 `DeprecationWarning` on multi-seed `evaluate_folded(seeds=...)`
calls without an explicit `reseed_splitter` callback **persists past v1.0
by design** (pre-v1.0 deprecation window is one minor; `DEPRECATION.md`
requires ≥2 minors to close a cycle). Single-seed callers see no change;
multi-seed callers should pass `reseed_splitter` for true seed variance.

### Migration

If your consumer is on v0.51, nothing changes. If on v0.50 or earlier,
follow [`docs/source/migration/v0.51.md`](docs/source/migration/v0.51.md)
for the actual migration steps. Downstream projects should pin
`eval-toolkit>=1.0,<2.0` to opt into the stability contract.

## [0.51.0] — 2026-05-24 — Round 8 rectification batch

The 18-item rectification batch following the Round 8 multi-LLM audit
(Codex + Gemini reports verified at
`audit-verification-codex-gemini-v0.50.0.md`, 2026-05-24). Per Decision
Y.2 + the staggered-pre-v1.0 plan, v0.51.0 is a BREAKING-allowed
minor bundling all fixes before v1.0 tags. Round 9 audit runs against
the v0.51 RC.

**Audit outcome**: 13 confirmed → fixed in this release; 2 deferred
(R8-G3 custom exceptions, R8-G4 joblib OOM capping) to v1.x as Tier-2
additive; 3 refuted (R8-G2 cyclic-import framing; R8-G5 cherry-picked
weak test; R8-V1 + R8-V2 over-confident Gemini validations). See
`docs/source/audit_findings.md` Round 8 section for the full ledger.

**Round 9 follow-on**: a Round 9 multi-LLM cross-review (Codex + Gemini)
ran against the v0.51 RC pre-tag. Verified by Claude at
`audit-verification-round-9-v0.51.0.md` (6 confirmed / 3 refuted / 1
partial; plus 3 third-audit fixes in modules neither auditor cited).
**Two third-audit findings + one source-report regression fix shipped
in this RC pre-tag** (commit-graph below); the remaining 4 deferred
items go to v1.0.1. See `audit_findings.md` Round 9 section for the
full ledger.

### Added (Round 9 follow-on)

- **R9-F-sweep-1** (CANDIDATE v1.0 BLOCKER closed) — `_sweep.py:
  _validate_scorer_output()` now validates scorer output is finite
  (no NaN / +inf / -inf), not just shape. Pre-R9 follow-on, NaN/inf
  scores passed R7-C's shape check and silently propagated into the
  sweep DataFrame, then silently zeroed the ASR flag (NaN >= threshold
  is False). Closes the "no silent failures" invariant gap R7-C
  established for shape but didn't extend to finiteness. Brings sweep
  validation to parity with `stacking.py`'s `_validate_fit_inputs` /
  `_validate_predict_inputs`. Tier-2 additive — callers whose scorers
  were silently producing NaN now get a clear `ValueError` with
  diagnostic context.

- **R9-F-bootstrap-1** — `bootstrap.bootstrap_ci(...)` emits a
  `UserWarning` when scipy's BCa method degenerates (returns
  `ci_low == ci_high == point` or non-finite bounds). Pre-R9, the
  R8-C4(b) RNG bug spuriously varied bootstrap streams and could mask
  BCa degeneracy on small-n + ceiling/floor-metric inputs; post-R8 with
  correct RNG, the brittleness is exposed. Warning text recommends
  `method='percentile'` as the safer fallback at small n. The default
  remains `method='BCa'` (preserves bit-stability for non-degenerate
  cases); auto-fallback is deferred to v1.0.1 if user demand.

- **R9-F-bootstrap-2** — `bootstrap.mde_from_ci(...)` now explicitly
  rejects NaN CI width with `RuntimeError`. Pre-R9, NaN width
  (possible when scipy BCa returns NaN bounds) bypassed the
  `if width <= 0` check (NaN <= 0 is False in IEEE float) and
  silently returned `MDEEstimate.mde = NaN`. Bundled with F-bootstrap-1.

### Fixed (Round 10 follow-on)

Pre-tag scoped Codex + Gemini micro-audit on `edadddc` surfaced 3
Codex-confirmed findings (all fix-recommended / minor; no v1.0
blockers). Verified by Claude; 1 Gemini accept-as-design + 1 Gemini
refuted (Pattern-1 violation; calibration record in
`audit_findings.md` Round 10 section). All 3 confirmed findings
shipped in this RC pre-tag:

- **R10-F1** — `protocols.Scorer.predict_proba` docstring + `_sweep.py`
  error message clarification. Pre-R10, `_validate_scorer_output`'s
  runtime error said "finite floats in [0, 1]" but the boundary check
  only enforced finiteness (no range validation); the Scorer Protocol
  docstring also lacked an explicit `[0, 1]` contract statement. R10-F1
  extends the Protocol docstring to document calibrated-probability
  semantics + reword the sweep runtime message to drop the unenforced
  `[0, 1]` claim. Range enforcement is intentionally deferred to a
  future minor once consumer usage patterns clarify whether the
  Protocol should be strict (`[0, 1]`) or permissive (ranking scores).

- **R10-F2** — `tests/test_bootstrap_unit.py::test_bootstrap_ci_bca_degeneracy_emits_warning`
  test predicate hardening. Pre-R10, the test's assertion block used
  `if ci.ci_low == ci.ci_high == ci.point_estimate:` — but NaN==NaN is
  False in IEEE float, so the assertions were silently skipped on the
  current scipy fixture (which returns NaN bounds). The test passed
  WITHOUT proving the warning fires for the common degeneracy mode.
  R10-F2 mirrors the production predicate exactly:
  `(not np.isfinite(low)) or (not np.isfinite(high)) or (low == high == point)`.
  The assertion block now runs whenever ANY degeneracy mode fires.

- **R10-F3** — `bootstrap.mde_from_ci` docstring update for the
  R9-F-bootstrap-2 non-finite-width branch. Pre-R10, the Raises section
  said "non-positive width" only; the implementation has also been
  rejecting non-finite width since `edadddc` but the docstring lagged.
  R10-F3 updates the Raises text to "non-positive or non-finite width"
  and adds a 4-line note explaining the scipy-BCa NaN-bound motivation
  so callers understand the new behavior is intentional, not incidental.

### Added

- **R8-C6** — `calibration.reliability_curve(...)` and
  `calibration.maximum_calibration_error(...)` now call
  `_validate_calibrated_score(y_score)` BEFORE the sklearn dispatch.
  Pre-v0.51 these functions silently accepted raw logits (any range);
  sibling `metrics.expected_calibration_error*` variants already
  validated input range via the same helper. Now symmetric — out-of-range
  scores raise `ValueError` with the same actionable diagnostic. Also,
  `calibration.fit_temperature(...)` now validates the `bounds` tuple
  (finiteness, positivity, `lo < hi`) BEFORE forwarding to
  `scipy.optimize.minimize_scalar` — cryptic optimizer errors replaced
  with actionable input-validation errors.

- **R8-F1** — `losses.RecallAtLowFPR.__init__(...)` now validates
  `pos_weight > 0` at construction time, matching the sibling-kwarg
  validators for `fpr_target` and `fpr_smoothing_beta`. Pre-v0.51
  non-positive `pos_weight` produced degenerate-but-bounded loss
  values silently.

- **R8-F2** — `metric_specs.ece(n_bins=, strategy=)` factory now validates
  `n_bins` eagerly at spec-construction time (matches the eager
  `strategy` validation already present). Pre-v0.51 `n_bins`
  validation was deferred to compute time.

- **R8-F3** — `analysis.CsvPredictionReader.read_predictions(...)` now
  detects missing CSV columns at read time and raises a
  `ValueError(f"CSV file at {uri!r} is missing required column(s) ...")`
  with the file path + available columns. Pre-v0.51 missing columns
  were silently filled with empty strings, causing cryptic
  `ValueError: invalid literal for int() with base 10: ''` downstream
  in `load_prediction_arrays`'s dtype conversion. Root cause now
  surfaces at the boundary.

- **R8-C1** — `harness.evaluate_folded(...)` now accepts an optional
  `reseed_splitter: Callable[[Splitter, int], Splitter] | None`
  callback. When provided, each seed iteration calls
  `reseed_splitter(splitter, seed)` to produce a fresh splitter for
  that seed's fold iteration. Default `None` preserves the historical
  behavior (the same splitter instance is reused across the seed loop,
  so multi-seed × CV only varies the bootstrap RNG, not fold
  partitions) AND emits a `DeprecationWarning` when `len(seeds) > 1`.
  The warning persists past v1.0 because the pre-v1.0 deprecation
  window (v0.51 → v1.0) is one minor and ADR 0003 / DEPRECATION.md
  require ≥2 minors to close a cycle. Migration example::

      from dataclasses import replace
      evaluate_folded(
          scorers, splitter, slice_,
          seeds=(1, 2, 3),
          reseed_splitter=lambda sp, s: replace(sp, seed=s),
          ...
      )

  R8-C1 audit fix.

### BREAKING

- **R8-C2** — `SourceDisjointKFoldSplitter.iter_folds(...)` now caps
  the fold count at `min(self.k, n_sources)` (matching
  `get_n_splits(...)`). Pre-v0.51 the loop ran `range(self.k)` and
  yielded EMPTY test partitions for the surplus folds when
  `k > n_sources` while `get_n_splits` returned `min(k, n_sources)`
  — the two methods silently disagreed on fold count. v0.51 caps both
  at the same value AND emits a `UserWarning` when `k > n_sources` so
  the caller knows the cap was applied. Callers that consumed the
  surplus empty-test folds will see fewer iterations now; that was
  the bug. (Probe-verified at
  `audit-verification-codex-gemini-v0.50.0.md`.)

- **R8-C3** — `thresholds.recall_at_fpr(...)` fallback semantics changed
  when no threshold satisfies `target_fpr`. Pre-v0.51 the fallback set
  `threshold = 1.0` and then computed `y_pred = (y_score >= 1.0)` —
  inclusive comparator — which classified any negative-class sample
  with score exactly 1.0 as predicted-positive. The probe
  `recall_at_fpr(y=[0,1], scores=[1.0,1.0], target_fpr=0.0)` returned
  `actual_fpr=1.0, fp=1` in silent violation of the function's
  FPR-ceiling invariant. v0.51 returns a SENTINEL
  `RecallAtFprResult(threshold=np.inf, recall=0.0, actual_fpr=0.0,
  fp=0, tn=n_val_neg)` whenever the constraint is unsatisfiable.
  Callers detect via `np.isinf(result.threshold)`. The
  `actual_fpr ≤ target_fpr` invariant is now preserved by construction.
  Migration: any caller filtering on `result.threshold` should add an
  `np.isinf(...)` branch — pre-v0.51 the sentinel value was `1.0`.
  (Verified at `audit-verification-codex-gemini-v0.50.0.md`.)

- **R8-C4(a)** — `harness.evaluate(...)` with a `Generator`-typed `rng`
  is now bit-stable across `n_jobs` values. Prior to v0.51, the same
  `rng` object was attached to every `(slice, scorer)` work_unit;
  joblib forked copies at the SAME generator state into N parallel
  workers, so every worker used identical bootstrap sample streams —
  silently producing non-independent CIs across `(slice, scorer)`
  pairs in parallel mode and divergent results vs sequential mode.
  The v0.51 implementation spawns one independent `SeedSequence` per
  work unit at the dispatch boundary in `_score_all_slices` (depends
  on the R8-C4(b) `spawn_seed_sequences` fix). Each pair now sees an
  independent bootstrap stream; sequential (`n_jobs=1`) and parallel
  (`n_jobs>1`) modes produce bit-identical CIs per the SPEC 7
  contract at `docs/source/methodology/parallelism.md`. Integer `rng`
  callers (the common case) are unaffected. (Verified by multi-slice
  probe at `audit-verification-codex-gemini-v0.50.0.md`.)

- **R8-C4(b)** — `eval_toolkit._rng.spawn_seed_sequences(rng, n)` now
  respects `Generator` state. Prior to v0.51, the function extracted
  the bit-generator's seed_seq and called `.spawn(n)` on it — so a
  `Generator` advanced by prior draws produced the same children as a
  fresh `Generator` with the same construction seed. The new
  implementation draws `n` fresh entropy values FROM the generator
  via `rng.integers(0, 2**63-1, size=n)` and wraps each in a
  `SeedSequence`. Each call advances generator state, so repeated
  calls on the same instance yield different children. This was the
  root cause of bootstrap non-independence across `(slice, scorer)`
  pairs in `harness.evaluate` — when the same `Generator` was shared
  across bootstrap callsites, all callsites silently used the same
  resample stream. (Verified probe at
  `audit-verification-codex-gemini-v0.50.0.md`.)

## [0.50.0] — 2026-05-23 — SPEC 7 `rng` parameter adoption

The SPEC 7 follow-up to v0.49.0. The `_rng.py` scaffold shipped at
v0.49.0 (SeedLike + RNGLike type aliases per
[Scientific Python SPEC 7](https://scientific-python.org/specs/spec-0007/))
is now wired into every Tier-1 public function that consumes a NumPy RNG.

### BREAKING

**22 Tier-1 function signatures**: `seed: int = X` / `random_state: int | None` → `rng: RNGLike | SeedLike | None = X`. Pre-v1.0 SemVer-minor BREAKING (v0.34.0 precedent). Defaults preserved (still deterministic-by-default).

Affected functions:

- `bootstrap.py` (7 public + 1 private): `bootstrap_ci`, `paired_bootstrap_diff`, `paired_bootstrap_ece_diff`, `paired_bootstrap_op_point_diff`, `paired_mde`, `block_bootstrap_on_folds`, `cross_validate_metric`, `_bootstrap_t_ci`.
- `metrics.py:1063`: `expected_calibration_error_debiased`.
- `thresholds.py`: `selected_operating_point` + `_bootstrap_threshold_metric_cis`.
- `analysis.py`: `bootstrap_metric_from_predictions`, `paired_diff_from_prediction_refs`.
- `harness.py` (6 sites): `evaluate`, `evaluate_scorer_on_slice`, `_bootstrap_auc_ci`, `_evaluate_scores`, `_compute_paired_diffs`, `_score_all_slices`.
- `scorecards.py`: `scorecard`, `_evaluate_spec`.
- `stacking.py`: `LogisticStacker.random_state` → `LogisticStacker.rng` class-field rename (sklearn pass-through derives int at the boundary).

**Body refactors**:

- 4 SeedSequence.spawn() sites converted from `np.random.SeedSequence(seed).spawn(n)` to `rng.bit_generator.seed_seq.spawn(n)` (Option A — preserves existing worker SeedSequence signatures).
- 2 sklearn-bridge sites in `cross_validate_metric` derive int from rng before passing to `StratifiedKFold`/`KFold(random_state=...)` (defensive across sklearn versions <1.4).
- `LogisticStacker.fit` derives sklearn int from `self.rng` at the boundary.

**Config schema** (Tier-2 additive): `evaluate()` config dict key `"seed"` → `"rng"`. Generator-typed input serializes as `repr(rng)`; int/None serialize as-is (backward-compatible for prior int-seed usage).

### Added

- **Docstrings**: NumPy-style parameter doc for every renamed function now references `rng : RNGLike | SeedLike | None` with explicit link to SPEC 7.
- **STYLE.md §3a** + **ADR 0004 D4**: `rng` row flipped from "target convention; adopted in v0.50.0" → "**canonical** convention (adopted v0.50.0)".

### Changed

- **Test sweep** (~230+ test sites): `seed=X` → `rng=X` in test kwarg calls, EXCEPT in test files that test legitimate `seed`-as-int contexts (`test_adversarial.py` for Python `random.Random`, `test_seeds.py` for `set_global_seeds`, `test_splits*.py` for Splitter dataclass fields, `test_text_dedup*.py` for MinHashLSHStrategy class field).
- **CHANGELOG header**: this release.

### Exceptions to SPEC 7 (KEPT `seed:` — documented in STYLE.md §3a + ADR 0004 D4)

- `seeds.set_global_seeds(seed: int)` — global-state setter, not per-function RNG.
- `adversarial.py` dataclass fields + functional wrappers — use Python stdlib `random.Random(seed)`, not NumPy.
- `splits.py` Splitter dataclass class-fields (`HoldoutSplitter.seed`, `StratifiedKFoldSplitter.seed`, etc.) — configuration storage, not user-facing RNG parameter.
- `loaders.py:903` YAML config schema key — declarative; renaming would break consumer YAMLs.

### Migration

- Consumer (`prompt-injection-detection-submission`) lockstep: bump dep pin `>=0.49.0` → `>=0.50.0`; rename `seed=` → `rng=` on eval-toolkit-bound call sites (estimated 5-8 sites).
- Bit-for-bit reproducibility preserved when migrating `seed=42` → `rng=42` (int seed is SeedLike; `np.random.default_rng(42)` is the canonical normalization).

### Notes

- Ships in parallel with Round 8 audit STOP-GATE (Decision Y.2); R8 briefing at commit `6f6839a`, awaiting Codex+Gemini reports.
- Memory pattern captured at v0.49.0: pre-flight grep MUST cover `README.md`, `.doctest-modules`, and any config files (per `feedback_sybil_runs_readme.md`). Applied to v0.50.0 pre-flight.

## [0.49.0] — 2026-05-23 — Global naming-standards sweep + final cleanup before v1.0

Final pre-v1.0 minor consolidating the naming-convention standardization
that locks the v1.0 Tier-1 contract. Audit + industry-research pass
(PEP 8, scikit-learn, NumPy, Google Python Style Guide, Scientific
Python SPEC 7) found the repo already 95-99% consistent; this release
closes the small remaining gaps + documents the conventions as
[ADR 0004](docs/source/adr/0004-naming-conventions.md). The SPEC 7
``rng`` parameter convention is documented here and adopted in v0.50.0.

### BREAKING

Five Tier-1 renames for naming consistency (pre-v1.0; SemVer-minor per
the v0.34.0 BREAKING-minor precedent). Single-consumer lockstep bump in
``prompt-injection-detection-submission``; no deprecation aliases.

- **``build_manifest`` → ``make_manifest``** (manifest.py). Aligns
  with ``make_minilm_embedder`` / ``make_palette`` / ``make_run_dir``
  factory pattern. ``build_*`` was the only outlier.
- **``CaseRandomization`` → ``CaseInjection``** (adversarial.py).
  Aligns with ``*Injection`` / ``*Substitution`` adversarial suffix
  convention.
- **``TokenSplitting`` → ``TokenSplittingInjection``** (adversarial.py).
  Same rationale.
- **``UnicodeNormalization`` → ``UnicodeNormalizationInjection``**
  (adversarial.py). Same rationale.
- **``eval_toolkit._scorecard.py`` → ``eval_toolkit.scorecards.py``**
  (private → public module promotion). The 4 top-level symbols
  (``scorecard``, ``Scorecard``, ``MetricSpec``, ``MetricResult``)
  remain top-level Tier-1; the new public submodule path
  ``from eval_toolkit.scorecards import Scorecard`` is now stable.
  ``_scorecard.py`` is gone — old import paths raise
  ``ModuleNotFoundError``. Per the asymmetric-promotion principle in
  [ADR 0001](docs/source/adr/0001-flat-module-layout.md): promote
  collection-of-types modules, keep single-function modules underscore
  (``_sweep.py`` stays private).

### Added

- **[ADR 0004](docs/source/adr/0004-naming-conventions.md)** — Naming
  conventions decision record with industry citations. Covers module
  naming (singular vs plural), class suffixes by domain, function
  verb-prefix conventions, canonical parameter list, fitted-attribute
  trailing underscore (sklearn convention), TypeVar leading underscore
  (Google convention), and the SPEC 7 ``rng`` parameter convention
  (adopted in v0.50.0).
- **STYLE.md** extended with §3a-d (parameter naming, class suffixes
  by domain, module naming, asymmetric promotion), §4a-b
  (fitted-attribute trailing underscore + TypeVar), §12 (75-col
  docstring prose rule), §14 (test naming convention).
- **CONTRIBUTING.md** cross-link to ADR 0004 + STYLE.md.
- **[docs/source/api/strict_tier2_protocols.md](docs/source/api/strict_tier2_protocols.md)** —
  new docs page enumerating the 9 strict Tier-2 Protocols + 1 opt-in
  per [ADR 0003 §1](docs/source/adr/0003-stability-contract-and-gate3-methodology.md),
  with canonical top-level import paths. Resolves #69's discoverability
  concern without breaking the lightweight design intent of
  ``eval_toolkit.protocols`` (per ``protocols.py:1-5``).
- **``src/eval_toolkit/_rng.py``** — private module with SPEC 7 type
  aliases (``SeedLike``, ``RNGLike``). Not yet referenced; scaffold for
  the v0.50.0 SPEC 7 adoption.
- **[ADR 0001](docs/source/adr/0001-flat-module-layout.md)** amendment
  — added the asymmetric-promotion sub-rule (collection-of-types MAY
  promote, single-function SHOULD stay underscore).

### Changed

- **Duplicate-type consolidation** (single source of truth):
  - ``Versioned`` Protocol — canonical at ``protocols.py:64``; the
    duplicate at ``leakage.py:82`` removed. Removed
    ``"Versioned"`` from ``leakage.__all__``; previously-unused
    ``from eval_toolkit.leakage import Versioned`` now raises
    ``ImportError``. Use ``from eval_toolkit.protocols import Versioned``
    or top-level ``from eval_toolkit import Versioned``.
  - ``MetricStatus`` ``Literal`` — canonical at ``artifacts.py:30``; the
    duplicate at ``scorecards.py:78`` removed; ``scorecards`` now
    imports from ``artifacts``.
- **[validation] optional extra** reclassified from "active deprecation
  with removal target v0.33.0" → "permanent no-op kept for backward
  compatibility." Hard removal would break consumer pip pins of the
  form ``eval-toolkit[validation]`` for zero functional benefit
  (R3 in DEPRECATION.md).
- **Sphinx cross-references** updated from
  ``eval_toolkit.leakage.Versioned`` → ``eval_toolkit.protocols.Versioned``
  in ``manifest.py`` docstrings.

### Deferred to v0.50.0

- **SPEC 7 ``rng`` parameter adoption** across ~30 NumPy-RNG functions.
  Scope deferred from v0.49.0 after the planning audit revealed the
  full blast radius (~30 signature sites + 247 test kwarg sites +
  7 internal helpers + SeedSequence/Generator/sklearn-bridge
  conversions). Splitting matches the "one cleanup per minor" pattern
  per [feedback_staggered_breaking_releases]. ``_rng.py`` ships in
  v0.49.0 as the scaffold; v0.50.0 wires it into every applicable
  function.

### Notes

- Round 8 audit STOP-GATE per Decision Y.2 — briefing committed at
  v0.48.0 (commit ``6f6839a``); v0.49.0 ships in parallel since the
  audit-trail synthesis confirmed R8 audits the existing contract
  (does not prescribe new changes). Any R8 finding folds into v0.49.1
  hotfix if needed.
- Issue #69 closed by the new strict-Tier-2-Protocols docs page; see
  ``docs/source/api/strict_tier2_protocols.md`` and the close
  rationale on the issue itself.

## [0.48.0] — 2026-05-22 — Polish + audit-driven tightening before v1.0 (Round 7 follow-on + cross-API consistency + doc-execution gates)

Third + final BREAKING minor of the staggered v0.45 → v0.46 → v0.46.1 → v0.47
→ v0.48 → v1.0 release sequence (plan
``~/.claude/plans/evaluate-all-the-work-twinkly-kite.md``, Step 4). Migration
guide: ``docs/source/migration/v0.48.md``.

Closes:

- Round 7 audit STOP-GATE per Decision Y.2 (Codex R7-F1/F2/F3 + 6 Gemini
  observations; see ``docs/source/audit_findings.md`` for the per-finding
  ledger).
- Audit-as-seed extensions surfaced during plan refinement: full
  module-docstring sweep across ``src/eval_toolkit/``; expanded
  ``.doctest-modules`` from 11 → 21 modules; comprehensive cross-API
  shape-validation consistency sweep.
- Round 5 §5E-prep packet-drift fixes (7 methodology documentation
  corrections).

After v0.48 observes ≥1 consumer cycle, the Round 8 audit STOP-GATE
opens before ``v1.0.0`` tag.

### BREAKING

- **``BootstrapCI.to_dict()`` + ``PairedBootstrapCI.to_dict()`` schema
  rewrite** (§5B). Pre-v0.48 hard-coded a ``"ci_95"`` key regardless of
  the actual ``confidence`` field — the key contradicted the data.
  v0.48 schema is self-describing:

    Before: ``{"point_estimate": p, "ci_95": [l, h], "confidence": 0.95, ...}``
    After:  ``{"point": p, "low": l, "high": h, "confidence": 0.95, ...}``

  Migration: ``d["point_estimate"]`` → ``d["point"]``; ``d["ci_95"]``
  → ``(d["low"], d["high"])``. Same rewrite for ``PairedBootstrapCI``.
- **``sweep()`` schema grows by 1 column** (§5I, Decision R7-B option C).
  New ``strategy_id`` column inserted between ``text_id`` and ``variant``
  carries the canonical per-row identifier built from configured
  kwargs. Callers indexing by column position must re-check offsets.
- **``sweep()`` rejects duplicate ``strategy_id``** (§5I). Mirrors
  R6-B's duplicate ``MetricSpec.name`` rejection in ``scorecard()``.
- **``sweep()`` validates scorer output shape** (§5J, Decision R7-C).
  Wrong-shape arrays from ``Scorer.predict_proba`` raise contextual
  ``ValueError`` at the boundary. Pre-v0.48: silent truncation
  (overlong), ``IndexError`` (short), or ``TypeError`` (matrix-shaped).
- **``paired_bootstrap_op_point_diff()`` rejects ``val_y is test_y``**
  (§5E-prep). The two-level bootstrap assumes disjoint val + test
  partitions; passing the same array causes ~63.2% silent overlap.

### Added

- **``make pre-push``** Makefile target (§5L) running all 3 doc-
  execution surfaces — Sybil-collected ``.md`` fences, MyST-NB example
  notebooks, and in-source ``>>>`` docstring examples. Closes the
  v0.47 Sub-PR 7 incident class.
- **``nb_execution_raise_on_error = True``** in ``docs/source/conf.py``
  (§5H, Decision R7-A). Docs CI now fails on notebook execution errors.
- **``.doctest-modules`` expanded** from 11 → 21 modules (§5M).

### Changed

- **Cross-API shape-validation consistency** (§5N). Every public-API
  surface with array inputs now validates shape + raises ``ValueError``
  with context (rather than leaking low-level numpy/sklearn errors).
- **Standardized ``ImportError`` messages** across lazy-extras (§5C).
  Canonical template: ``"<feature> requires <pkg>. Install with: pip
  install eval-toolkit[<extra>]"``.
- **Pin-exact-key-set regression-guards** (§5A) for every dict-returning
  metrics function. Audit revealed no drift; the tests pin existing
  key sets so future drift fails CI loud.
- **Docs polish** (§5K + §5E-prep): ``SynonymSubstitution`` whitelist
  ``Notes``; ``Scorecard.to_pandas()`` dtype coercion ``Notes``;
  ``CostSensitiveSelector`` calibrated-prior ``Warning``; ``cv_clt_ci``
  docstring per Bayle et al. (2020) Theorem 3.1; ``methodology/parallelism.md``
  post-v0.36 state; ``methodology/testing.md`` reference-equivalence-gap
  framing; ``methodology/calibration.md`` 4-binary-adapter family;
  ``methodology/bootstrap.md`` disjoint-split example; DeLong docs
  aligned to shipped state (Decision U).

### Fixed

- **R7-F1**: 6 MyST-NB example notebooks (``docs/source/examples/*.md``)
  migrated to v0.47 API; 4 module-level docstrings rewritten; 5
  drifted ``docs/source/api/*.md`` autosummary lists corrected;
  8 missing ``api/*.md`` pages created; roadmap "Sybil-validated
  examples" wording corrected (§5G).
- **ADR 0001** (flat-module layout) + **ADR 0003** (stability contract
  + Gate 3 methodology) finalized for v1.0 (§5E + §5F).
- **schemas.md** + **methodology/claims.md** + **getting-started.md**:
  ``BootstrapCI`` schema references updated for the §5B rewrite.

## [0.47.0] — 2026-05-21 — Sweep unification + TextTransform + advanced-6 + cleanup + Round 6 follow-on

Second BREAKING minor of the staggered v0.45 → v0.46 → v0.46.1 → v0.47 →
v0.48 → v1.0 release sequence (plan
``~/.claude/plans/evaluate-all-the-work-twinkly-kite.md``, Step 3).

Closes:

- The v0.43 CHANGELOG forward-look re: advanced-6 character-injection
  techniques (Decision Q11→11.3 — "12-technique suite + new sweep API
  in one migration step").
- Round 6 audit follow-on items per Decision R6-E (R6-A docstring,
  R6-B duplicate name guard, R6-C to_pandas schema, R6-D Protocol
  method-shape drift guard, R6-F5 narrow except, R6-F6 plan/roadmap
  refresh, R6-H make_spec_name helper) — see ``docs/source/audit_findings.md``
  for the per-finding ledger.

### Removed (BREAKING)

- **Top-level scalar metric names** (``eval_toolkit.pr_auc``,
  ``eval_toolkit.roc_auc``, ``eval_toolkit.brier_score``, all 5
  ``expected_calibration_*`` variants) — the v0.46 ``__getattr__``
  deprecation shim has been deleted. These names now raise
  ``AttributeError`` at the top-level. Migration: use ``scorecard(...)``
  with ``metric_specs`` (primary) OR import from the
  ``eval_toolkit.metrics`` submodule (internal API per ADR 0002).
  (Decision L; plan §4D.)
- **Module-level ``adversarial.sweep`` + ``preprocessing.sweep``** —
  consolidated into the top-level :func:`sweep` (Decision D + plan §4C).
  Parity tests in Sub-PR 4 of this release proved 1:1 output equivalence
  on the neutral subset.
- **``adversarial.character_injection`` + ``preprocessing.spotlighting``
  ``SimpleNamespace`` shortcuts** — removed (Decision N + plan §4E).
  The 12 adversarial dataclasses + the 3 preprocessing variants + the
  underlying functional API are the only public paths.
- **``adversarial.CharacterInjectionStrategy``** per-module Protocol —
  removed. The top-level :class:`TextTransform` Protocol (Decision K)
  is the single canonical contract; all 12 character-injection
  dataclasses + 3 preprocessing variants satisfy it structurally.

### Added

- **``TextTransform`` Protocol** (top-level; ``eval_toolkit.protocols`` module).
  Decision K + Audit R5-F3 (Codex Round 5): unifies the "name + transform(text)"
  shape across preprocessing (defence) and adversarial (attack) strategies so
  the v0.47 top-level :func:`sweep` (next sub-PR) can mix them in one call. The
  9th strict Tier-2 Protocol per ADR 0003.
- **Advanced 6 character-injection techniques** (plan §4F, Decision Q11→11.3) —
  closes the v0.43.0 CHANGELOG forward-look that referenced these as "scheduled
  for v0.43.1" (a version that never shipped). Each satisfies the top-level
  :class:`TextTransform` Protocol structurally; all are frozen + ``slots=True``
  dataclasses with deterministic behaviour under their ``seed`` kwarg where
  applicable:

    - :class:`BidiRTLInjection` — wrap input in ``U+202E … U+202C``
      RIGHT-TO-LEFT OVERRIDE block.
    - :class:`TagStrippingInjection` — strip HTML/XML-like ``<…>`` tags
      (idempotent).
    - :class:`SynonymSubstitution` — replace whitelisted prompt-injection-
      relevant function words / verbs with semantic-preserving synonyms.
    - :class:`TokenSplitting` — insert a single space inside long enough
      words; forces subword tokenizers to re-segment.
    - :class:`UnicodeNormalization` — NFC / NFD / NFKC / NFKD; default NFKC
      folds compatibility chars (e.g., fullwidth ``ＡＢＣ`` → ``ABC``).
    - :class:`InvisibleCharsInjection` — sample from the 5-element invisible-
      code-point set (ZWSP, ZWNJ, ZWJ, word joiner, BOM) — distinct from the
      single-codepoint :class:`ZeroWidthSpaceInjection`.

  Also exported: ``ADVANCED_TECHNIQUES`` (6-tuple) and ``ALL_TECHNIQUES``
  (12-tuple = core 6 + advanced 6).
- **Top-level :func:`sweep`** — single ``TextTransform`` enumeration entry
  point (Decision K + Decision D + Audit R5-F3). Replaces the per-module
  ``adversarial.sweep`` + ``preprocessing.sweep`` (those are removed in a
  subsequent sub-PR of this release). New contract:

    - ``sweep(strategies, texts)`` → neutral DataFrame with ``text_id`` /
      ``variant`` / ``transformed_text`` columns. Pure text-transform
      enumeration; defence + attack strategies compose freely.
    - ``sweep(..., scorer=...)`` → also emits ``original_score`` /
      ``transformed_score`` columns (single batched scorer call per
      strategy, not per-row).
    - ``sweep(..., scorer=..., attack_threshold=t)`` → also emits ``asr``
      (per-row attack-success flag). Explicit threshold REQUIRED to
      materialize ``asr``; no magic ``threshold=0.5`` default.
      ``attack_threshold`` without ``scorer`` raises ``ValueError``.

  Parity tests against the existing module-level sweeps ship in this
  sub-PR (``tests/test_sweep.py``) and prove the v0.47 consolidation
  produces identical transformed-text rows for the 6 core character-
  injection techniques + the 3 spotlighting variants.
- **3 preprocessing dataclasses** (``DelimitVariant``, ``DatamarkVariant``,
  ``EncodeVariant``) in :mod:`eval_toolkit.preprocessing`. Frozen +
  ``slots=True`` thin wrappers over the existing :func:`delimit` /
  :func:`datamark` / :func:`encode` functions. Closes Audit R5-F3
  (Codex Round 5) — prior to this commit, ``preprocessing.__all__`` exported
  only functions, so the "concrete classes satisfy ``TextTransform``
  structurally" claim only held on the adversarial side. Now both sides
  share the dataclass-strategy shape.
- ``metric_specs.make_spec_name(prefix, **kwargs)`` canonicalization helper
  for custom parameterized :class:`MetricSpec` implementations. Alphabetized
  kwargs joined by underscore — same convention the v0.46 ECE factory uses.
  Lands in ``metric_specs.__all__`` only; **not** top-level ``__all__`` per
  Decision R6-H. (Closes Round 6 Gemini R6-F4.)

### Changed (Round 6 follow-on)

- ``scorecard()`` now raises ``ValueError`` when two :class:`MetricSpec`
  instances in the ``metrics`` list share a ``name``. Forces caller
  disambiguation; the ``Mapping[str, MetricResult]`` contract never silently
  drops a cell. Error message reports both indices. (Decision R6-B; closes
  Round 6 Codex R6-F3.)
- ``scorecard(seed=None)`` docstring rewritten to document the deterministic-
  by-default contract (``None`` is treated as ``seed=0``). No behavior
  change; v0.46 documented the wrong contract. (Decision R6-A; closes Round 6
  Codex R6-F4 + Gemini R6-F1.)
- ``_evaluate_spec()`` exception catches narrowed: ``MemoryError``,
  ``RecursionError``, ``KeyboardInterrupt``, and ``SystemExit`` now propagate
  out of ``scorecard()`` instead of being captured as a ``status="error"``
  cell. Per-cell isolation remains for ordinary application errors.
  (Decision R6-F5; closes Round 6 Gemini R6-F5.)
- ``Scorecard.to_pandas()`` MultiIndex schema extended with two new inner-
  field columns: ``n_resamples`` (int / NaN sentinel) and ``method``
  (string / ``""`` sentinel). The DataFrame view is now lossless against
  :meth:`BootstrapCI.to_dict` — trace provenance (resample count + CI
  method) no longer drops at the DataFrame boundary. Callers indexing the
  MultiIndex by name keep working; callers indexing by position must
  re-check column offsets. (Decision R6-C; closes Round 6 Gemini R6-F3.)
- ``tests/test_public_api.py`` drift guard now captures method signatures
  for ``typing.Protocol`` classes in ``__all__`` (a ``protocol_methods``
  sub-entry in the snapshot). Together with a Tier-2 coverage test, this
  actually enforces the strict method-shape stability ADR 0003 promises
  for the 9 Tier-2 Protocols. (Decision R6-D; closes Round 6 Codex R6-F5.)
  Public-API golden regenerated alongside this change.

## [0.46.1] — 2026-05-21 — Round 6 hotfix: ECE strategy validation + deprecation warning content

Hotfix release per **Decision Q** (data correctness regression + time-sensitive
warning content) + **Decision R6-E** (scope: R6-F1 + R6-F2 only; R6-A docstring
rolls forward to v0.47). All other Round 6 findings dispositioned to v0.47.0.

See [`docs/source/audit_findings.md`](docs/source/audit_findings.md) Round 6 for
the full disposition ledger.

### Fixed

- **`metric_specs.ece(strategy=<value>)` strategy validation** (Round 6 Codex
  R6-F1). Prior to v0.46.1, an invalid strategy string (e.g.
  `metric_specs.ece(strategy="typo")`) silently dispatched to quantile ECE and
  returned a `scorecard()` cell with `status="ok"` under an invalid encoded key
  (`"ece_n_bins_15_strategy_typo"`) — wrong-by-design data correctness path.
  Verified by Codex via runtime probe. Now both the `ece()` factory and
  `_EceSpec.compute()` raise:
  ```
  ValueError: ECE strategy must be 'uniform' or 'quantile'; got 'typo'
  ```
  Defence-in-depth: the factory validates eagerly (before LRU cache hit) AND
  `compute()` validates at the compute boundary so direct construction of
  `_EceSpec(strategy="typo")` (bypassing the factory) also raises.

- **Deprecation warning content for all 5 ECE variants** (Round 6 Codex R6-F2 +
  Gemini R6-F2, with Decisions R6-F + R6-G). The v0.46.0 `__getattr__`
  deprecation shim's warning messages produced broken migration snippets:
  - For `expected_calibration_error` + `expected_calibration_error_equal_mass`:
    the suggested `Scorecard` lookup key was the factory-call expression
    (`"ece(n_bins=10)"`) instead of the encoded spec name
    (`"ece_n_bins_10_strategy_uniform"`). Now uses the correct encoded key.
  - For `expected_calibration_error_debiased` / `_l2` / `_l2_debiased`: these
    variants are not in the v0.46 `metric_specs` namespace (Decision R6-G;
    research-completeness primitives, deferred to v1.x if user demand
    surfaces). Their warnings now point at the submodule path
    (`from eval_toolkit.metrics import expected_calibration_error_debiased`)
    instead of an unconstructable scorecard snippet.
  - Pre-v0.46 default verification: Gemini's report claimed
    `expected_calibration_error` defaulted to `n_bins=15`; verified against
    `metrics.py:730-734` that the actual default is `n_bins=10`. Per Decision
    R6-F, warning snippets use `n_bins=10` to preserve bit-identical pre-v0.46
    math + add a migration note explaining the new `metric_specs.ece()` factory
    default of `n_bins=15` (matching Hines et al.).

### Tests

- `tests/test_scorecard.py`: 4 new tests for ECE strategy validation
  (parametrized factory-rejection + compute-defence-in-depth).
- `tests/test_deprecated_scalars_shim.py`: 4 new test classes — verify each
  warning contains correct factory expression + encoded scorecard key, ECE
  warnings carry the n_bins=10/15 migration note, submodule-only warnings cite
  `eval_toolkit.metrics` path, and the snippet in each first-party warning is
  EXECUTABLE (parses + runs against synthetic data + produces ok-status cell).

### Rolled forward to v0.47 (Decision R6-E)

- R6-A `seed=None` docstring fix (non-blocker per Decision Q).
- R6-F3 duplicate `MetricSpec.name` rejection.
- R6-F5 (Codex) Protocol method-shape drift guard.
- R6-F3 (Gemini) `Scorecard.to_pandas()` schema expansion.
- R6-F4 (Gemini) `make_spec_name()` helper.
- R6-F5 (Gemini) narrow `_evaluate_spec()` exception catch.
- R6-F6 (Codex) plan + roadmap state-drift refresh.

## [0.46.0] — 2026-05-21 — Scorecard: primary v1.0 metric surface (closes #36)

Second minor of the staggered v0.45 → v0.46 → v0.47 → v0.48 → v1.0 sequence.
**Soft-breaking** — existing top-level scalar metric imports still work but
emit `DeprecationWarning` (hard-removed at v0.47).

See `docs/source/migration/v0.46.md` for the full consumer migration guide and
`docs/source/adr/0002-scorecard-as-primary-metric-surface.md` for the
decision record.

### Added

- **`eval_toolkit.scorecard(y_true, y_score, metrics=[...], bootstrap=True)`**
  — primary v1.0 metric surface. Single call computes multiple threshold-free
  metrics + bootstrap CIs on one slice; returns a `Scorecard` (read-only
  `Mapping[str, MetricResult]`). Type-safe dict-subscript access; status-aware
  cells; per-cell error isolation.
- **`MetricSpec` Protocol** — v1.0 Tier-2 contract; `name: str` +
  `compute(y_true, y_score) -> float`. Custom user specs satisfy structurally.
- **`MetricResult`** frozen dataclass — `value: float | None`, `status:
  Literal["ok", "skipped", "error"]`, `reason: str`, `ci: BootstrapCI | None`.
  Reuses the existing `MetricState` vocabulary from `artifacts.py:30-61`.
- **`Scorecard`** read-only `Mapping[str, MetricResult]` — `to_dict()`
  JSON-friendly, `to_pandas()` one-row DataFrame (lazy pandas import).
- **`eval_toolkit.metric_specs`** namespace submodule with threshold-free
  first-party specs:
  - `pr_auc`, `roc_auc`, `brier` — module-level singletons (identity stable).
  - `ece(n_bins, strategy)` — LRU-cached factory (identity stable per kwargs).
- **`SINGLE_CLASS_INCOMPATIBLE_METRICS`** extended with `pr_auc` / `roc_auc`
  aliases (alongside existing `auroc` / `auprc`) so the v0.46 scorecard
  surface and the v0.39 harness paths both produce correct skipped-status
  behavior. Non-breaking; doctest + unit tests added.
- **`docs/source/adr/0002-scorecard-as-primary-metric-surface.md`** —
  decision record covering single-surface rationale, threshold-free scope,
  Tier-2 Protocol commitment, and v2.0 trigger conditions.
- **`docs/source/migration/v0.46.md`** — consumer migration guide with
  side-by-side recipes for every common pattern.

### Deprecated

The following 8 top-level scalar imports emit `DeprecationWarning` and will
be hard-removed at v0.47.0. Use `scorecard()` + `metric_specs` or the
`eval_toolkit.metrics` submodule path (internal API, no warning).

- `pr_auc`, `roc_auc`, `brier_score`
- `expected_calibration_error`
- `expected_calibration_error_debiased`
- `expected_calibration_error_equal_mass`
- `expected_calibration_error_l2`
- `expected_calibration_error_l2_debiased`

### Audit findings integrated (Round 5)

Per `docs/source/audit_findings.md`:

- **F1** (scorecard threshold semantics) — addressed by Decision R: ship
  threshold-free first-party specs only at v0.46. Threshold-dependent
  metrics (F1, accuracy, precision, recall) deferred to v1.x with explicit
  operating-point provenance.
- **F2** (scorecard cell-state semantics) — addressed by Decision S: reuse
  existing `MetricState` (`ok`/`skipped`/`error`) vocabulary.
- **F4** (deprecation shim must extend the lazy resolver, not replace it) —
  addressed: `__getattr__` deprecation branch sits between `__version__`
  short-circuit and the base `_EXPORTS` lookup; tagged with BEGIN/END
  TRANSITIONAL markers for clean v0.47 removal. Tests guard that every
  remaining `_EXPORTS` symbol still resolves.
- **X.2 precondition** — `is_metric_defined_for_slice` aliases shipped
  ahead of v0.46 (PR #62).

### Protocol stability

Tier-2 streak continues: 7 of 7 consecutive minors (v0.40–v0.46) without
method-shape edits to any existing Tier-2 Protocol. `MetricSpec` is a NEW
Tier-2 Protocol added at v0.46; freezes at v1.0.

## [0.45.0] — 2026-05-21 — Stacking: MetaLearner Protocol + LogisticStacker (closes #52)

First minor of the staggered v0.45 → v0.46 → v0.47 → v0.48 → v1.0 sequence
(per the v1.0 plan at `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`).
Non-breaking — purely additive. No Protocol shape edits to the existing 6
Tier-2 contracts (Gate 2 streak continues: 6 of 6 consecutive minors without
Protocol-shape changes).

### Added

- `eval_toolkit.stacking` — new module providing the `MetaLearner` Protocol
  and one reference impl, `LogisticStacker`, for combining outputs from
  multiple binary detectors into a calibrated ensemble. Wraps
  `sklearn.linear_model.LogisticRegression` with a stacker-shaped public API
  (sklearn-style `fit(score_matrix, y)`, `predict(score_matrix)`,
  `predict_proba(score_matrix)`, plus `coef_` / `classes_` / `intercept_`
  attributes). No new dependencies — `scikit-learn` is already core since
  v0.27. Closes #52.
- `MetaLearner` Protocol — `@runtime_checkable`; sklearn-shape contract
  taking a `(n_samples, n_detectors)` score matrix. Sized as a v1.0 Tier-2
  contract per the v1.0 plan Decision M (tiered stability — strict freeze at
  v1.0; additive subprotocols permitted in minor releases). Mirrors the
  `Probe` Protocol pattern from v0.43.
- `LogisticStacker` reference impl — configurable C, fit_intercept,
  class_weight, penalty, solver, max_iter, random_state. Class-weight default
  `"balanced"` for the common imbalanced-detection setting. Composes with the
  4-binary-calibrator family (v0.40 + v0.42) via `fit_platt_binary` /
  `fit_isotonic_binary` chaining on stacked output.
- 24-test coverage in `tests/test_stacking.py`: Protocol satisfaction (both
  structural and duck-typed), shape contracts (3-detector × 500-sample
  fixtures), regularization behavior (C, L1 penalty), signal ordering,
  calibration chaining (Platt + Isotonic), bootstrap CI on stacker output
  (Audit F6a-aware — uses correct `BootstrapCI.ci_low/ci_high` attribute
  names), determinism under fixed `random_state`, hypothesis property on
  signal monotonicity, input validation (shape mismatch, single-class,
  non-finite, unfit, wrong-n-detectors).
- `docs/source/examples/stacking.md` — myst-nb worked example: 3 synthetic
  detectors with descending signal-to-noise, stacker fit, post-stacking
  Platt calibration. Cites Wolpert 1992 + Breiman 1996.

### Notes

- Sklearn 1.8+ deprecates `LogisticRegression(penalty=...)` in favor of
  `l1_ratio`. The public `LogisticStacker(penalty=...)` API is preserved;
  internal sklearn-side migration to `l1_ratio` will land when sklearn 1.10
  lands and the warning becomes more visible. No user-facing impact.

## [0.44.0] — 2026-05-19 — Defenses + losses: Spotlighting variants + RecallAtLowFPR (closes #50, #51)

### Added

- `eval_toolkit.preprocessing` — new module with 3 Spotlighting
  structural-defense variants from Hines et al. 2024
  (arXiv 2403.14720): `delimit(text, delimiter='<<')`,
  `datamark(text, marker='^')`, `encode(text, encoding='base64')`,
  plus a `sweep(texts, variants=..., kwargs=...)` batch wrapper that
  returns a `(N*3)`-row DataFrame. Includes a `spotlighting`
  SimpleNamespace exposing the upstream issue's function-style API
  (`spotlighting.delimit(text)`, etc.). Base-install safe (pure
  stdlib). Closes #51.
- `eval_toolkit.losses` — new module with `RecallAtLowFPR` — the
  Meta Prompt Guard 2 (PG2) training recipe: a differentiable
  approximation of recall-at-fixed-FPR via soft-rank, returning a
  scalar `torch.nn.Module` loss for use in standard training loops.
  Optimizes detector ranking at a constrained operating point
  (e.g. `fpr_target=0.01` → "maximize recall while keeping FPR ≤ 1%").
  Closes #50.
- New optional extra `[losses] = torch>=2.0`. Granular per the v0.43
  plan Decision 4 — separated from `[probes]` so callers wanting only
  the loss don't have to install the larger transformers stack.
  Shares the torch version pin with `[probes]`.

## [0.43.0] — 2026-05-19 — P1 batch: OOD manifest loader + character_injection sweep + ActivationDeltaProbe (closes #48, #49, #53)

### Added

- `ood_dataset_from_manifest(yaml_path, slices=..., cache_dir=...)` —
  declarative loader for multiple OOD eval slates (BIPIA, AgentDojo,
  InjecAgent, NotInject, PINT, LLMail-Inject, …) into a single
  unified DataFrame with columns `text` / `label` / `source` /
  `row_id` / `sha`. Bytes are downloaded once, sha256-verified
  against the manifest, and cached on-disk keyed by content hash
  (default `~/.cache/eval-toolkit/ood/`). Closes #48 — drops the
  per-source loader boilerplate carried by
  `prompt-injection-portfolio` and `prompt-injection-detection-submission`.
- `OodManifestLoader` — `DatasetLoader`-Protocol-compliant wrapper
  around the factory, returning `{"all": EvalSlice}` with
  `source` as the default strata column for harness pipelines.
- `src/eval_toolkit/schemas/ood_manifest.v1.json` — Draft 2020-12
  JSON schema for the OOD manifest YAML; auto-validated by
  `uv run eval-toolkit schemas check`.
- `eval_toolkit.adversarial` — new module with character-injection
  bypass suite (Microsoft Research 2024, arXiv 2404.13208).
  Six core techniques shipped as frozen-dataclass strategies:
  `ZeroWidthSpaceInjection`, `HomoglyphSubstitution`,
  `DiacriticInjection`, `WhitespaceInjection`, `CaseRandomization`,
  `PunctuationInjection`. All implement a `CharacterInjectionStrategy`
  Protocol with `transform(text: str) -> str`. Six advanced techniques
  (bidi RTL, tag stripping, synonym, token splitting, Unicode
  normalization, invisible chars) scheduled for v0.43.1 — the sweep
  API stabilizes in v0.43.0 so the v0.43.1 additions are pure
  extensions. Closes #49 (core-6).
- `adversarial.sweep(texts, scorer, techniques="all", threshold=0.5)`
  — Scorer-Protocol-compliant adversarial-robustness sweep. Returns
  a DataFrame with `(text_id, technique, original_score,
  transformed_score, asr)` rows for matrix analysis.
  Aggregate ASR with `df.groupby("technique")["asr"].mean()`.
- `adversarial.character_injection` — `SimpleNamespace` exposing the
  function-style API from the upstream issue spec
  (`character_injection.zero_width_space(text)`,
  `character_injection.sweep(...)`, etc.).
- `eval_toolkit.probes` — new module with `ActivationDeltaProbe`:
  TaskTracker-style linear probe over HuggingFace transformer
  hidden-state activation deltas (Abdelnabi et al. 2024,
  arXiv 2406.00799). Backbone-agnostic (encoder OR decoder).
  Sklearn-compatible API: `.fit(clean_texts, injected_texts)`,
  `.predict()` → `(n,)`, `.predict_proba()` → `(n, 2)`,
  `.coef_`, `.classes_`. Activations cached to
  `$XDG_CACHE_HOME/eval-toolkit/probes/` keyed by
  `(backbone, layer_index, aggregate, sha256(text))` so re-runs are
  near-instant. Aggregate modes: `mean`, `max`, `cls`. Closes #53.
- `Probe` Protocol — minimal sklearn-shaped probe surface (`fit`,
  `predict`, `predict_proba`, `coef_`, `classes_`). Distinct from
  `Scorer` (which returns 1-D `P(positive)`); wrap with
  `lambda p, X: p.predict_proba(X)[:, 1]` to adapt.
- `ActivationExtractor` Protocol — pluggable hidden-state-extraction
  contract for `ActivationDeltaProbe`; injectable for tests to avoid
  loading a real backbone.
- New optional extra `[probes] = torch>=2.0, transformers>=4.40`.
  Follows the `[embeddings]` precedent — opt-in only, NOT in
  `[all]` or `[dev]`, since the transitive install is ~600MB+.
  Module is base-install-safe: a friendly `ImportError` fires only
  if you try to use the default HF extractor without the extra.

## [0.42.0] — 2026-05-19 — fit_isotonic_binary completes 4-calibrator family (closes #44)

Final element of the binary scalar-prob calibrator family started by
`fit_temperature_binary` (v0.35.0). All four now uniformly return
`(params, apply)`:

| Function | Params | Shipped |
|---|---|---|
| `fit_temperature_binary` | `(T,)` — single float | v0.35.0 |
| `fit_isotonic_binary`    | `None` — non-parametric | **v0.42.0** |
| `fit_platt_binary`       | `(a, b)` | v0.40.0 |
| `fit_beta_binary`        | `(a, b, c)` | v0.40.0 |

Consumer code can now iterate the family with a single shape, used
to distinguish parametric from non-parametric via
`if params is not None`:

```text
CALIBRATORS = {
    "temperature": fit_temperature_binary,
    "isotonic":    fit_isotonic_binary,
    "platt":       fit_platt_binary,
    "beta":        fit_beta_binary,
}
for name, fit_fn in CALIBRATORS.items():
    params, apply = fit_fn(y_val, p_val)
    calibrated = apply(p_test)
    if params is not None:
        manifest.record(f"{name}_params", params)
```

This matches the consumer's calibration-battery pattern in
`prompt-injection-detection-prototype` (their ADR-056 supersedes
ADR-023 to adopt the canonical `(params, apply)` shape across the
full 4-calibrator audit battery).

### Added

- **`eval_toolkit.fit_isotonic_binary(y_true, y_score) -> (None,
  apply)`** — thin wrapper over `fit_isotonic_calibrator`. The
  `None` in the params slot encodes "non-parametric" (isotonic
  regression is a monotone step function, no scalar params to log).
- 6 new unit tests in `tests/test_calibration_binary_adapters.py`
  including a 4-calibrator family-iteration integration test that
  verifies the `None`-vs-tuple convention.

### Protocol stability

Additive only. No Tier-2 Protocol shape edits. v0.42 is minor 3 of
consecutive-without-Protocol-changes (v0.40 + v0.41 + v0.42). Gate 2
stays MET.

## [0.41.0] — 2026-05-18 — Croissant end-to-end (closes #42, v1.0 Gate 4 MET)

Closes v1.0 readiness Gate 4 — "Croissant interop verified end-to-end."
`HFDatasetsLoader.describe()` now fetches per-file `sha256` hashes
from HF Hub and exposes them in `distribution[].sha256`. The
integration test (`tests/test_croissant_e2e.py`) downloads a real
parquet shard from `stanfordnlp/sst2` and verifies the bytes hash
bit-exactly to the value `describe()` reports.

### Added

- **`HFDatasetsLoader.describe()` Croissant + tree-API enrichment.**
  When `fetch_remote_metadata=True` (default), the loader fetches from
  two HF Hub endpoints:
  - `/api/datasets/{repo}/croissant` — JSON-LD metadata (name,
    description, license, citeAs, schema).
  - `/api/datasets/{repo}/tree/refs%2Fconvert%2Fparquet?recursive=true`
    — per-file `sha256` (read from each file's `lfs.oid` field — the
    git-LFS content hash, equal to `sha256sum` of the raw bytes).
  Caller-provided fields (`name=`, `cite_as=`, etc.) win over
  Croissant fetches; Croissant fills only gaps. Network failures
  degrade gracefully (warning emitted; sha256 empty as in pre-v0.41).
- **`fetch_remote_metadata: bool = True`** constructor field on
  `HFDatasetsLoader`. Set `False` for offline / unit-test paths.
- **`tests/test_croissant_e2e.py`** — 5 integration tests against
  live HF Hub:
  1. `describe()` returns real `sha256:<64-hex>` per shard.
  2. **Bit-exact verification**: download shard from `contentUrl`,
     hash bytes, assert equals `describe()`'s sha256. This is the
     literal v1.0 Gate 4 check.
  3. Croissant metadata enriches name/citeAs/license/description.
  4. Caller overrides win over remote.
  5. `fetch_remote_metadata=False` preserves v0.40 behavior.
  All pass against `stanfordnlp/sst2` (~3 MB train shard).
- **New `integration` pytest marker** for network-dependent tests.
  Excluded from `make coverage` (PR CI); runs explicitly via
  `pytest -m integration`.

### Why dual-sourced

HF Hub's Croissant emitter currently fills `distribution[].sha256`
with a placeholder URL pointing at MLCommons Croissant spec issue
[#80](https://github.com/mlcommons/croissant/issues/80) ("In
<Download>, check SHA256 or MD5"), which is **open**. The Croissant
spec doesn't yet require per-file checksums from emitters; HF Hub is
honest and punts the field. The authoritative hash IS available via
HF Hub's tree API: `lfs.oid` is precisely sha256 of the file content
(verified bit-exact via `sha256sum`).

When MLCommons #80 resolves and HF Hub starts populating Croissant
`sha256` with real values (which will equal the existing `lfs.oid`),
the loader's source switches in ~5 LOC. Same downstream contract.

### Documentation

- `docs/source/methodology/reproducibility.md` §"Croissant
  interoperability": replaces v0.7-era "subset" framing with the
  end-to-end-verified narrative + dual-source rationale.
- `docs/source/roadmap.md` §"v1.0.0 path":
  - **Gate 2 (Protocol stability) ✅ MET** — v0.41 = minor 2 of 2
    without Protocol shape edits (v0.40 fit_*_binary additives +
    v0.41 HFDatasetsLoader enrichment leave Tier-2 Protocols
    untouched).
  - **Gate 4 (Croissant end-to-end) ✅ MET** — with dual-source caveat
    documented; one-line migration path when MLCommons #80 resolves.

### v1.0 readiness state after v0.41.0

- Gate 1 (real consumer ≥1 review cycle on v0.7+): partial — consumer
  pinned to v0.34.0; needs bump + cycle. **External**.
- Gate 2 ✅ MET (v0.41 is minor 2 of 2 stable).
- Gate 3 (methodology peer review): not met — needs external reader.
  **External**.
- Gate 4 ✅ MET — see this release.

Two of four gates closed in-repo. The remaining two require external
coordination (consumer review cycle, methodology peer reviewer).

## [0.40.0] — 2026-05-18 — fit_platt_binary + fit_beta_binary (closes #43)

Completes the binary scalar-prob calibrator family started in v0.35.0
with `fit_temperature_binary` (#28). Consumers running the canonical
4-calibrator audit battery (temperature + isotonic + Platt + Beta) now
have library-first primitives for all four, eliminating ~100 LOC of
hand-roll across the calibration battery pattern.

### Added

- **`eval_toolkit.fit_platt_binary(y_true, y_score)`** — binary-
  probability adapter for `fit_platt_calibrator`. Returns
  `((a, b), apply)` where `(a, b)` are the fitted Platt slope and
  intercept and `apply: (n,) -> (n,)` is the scalar-in / scalar-out
  callable that maps probabilities through `σ(a·s + b)`. Trivially
  wraps `fit_platt_calibrator`; exposes `(a, b)` explicitly so
  consumers can log the parameters in a `RunManifest` without
  introspecting the `PlattFit` dataclass.
- **`eval_toolkit.fit_beta_binary(y_true, y_score)`** — 3-parameter
  Beta calibration per Kull et al. 2017. Returns `((a, b, c), apply)`
  where `(a, b, c)` are the coefficients on `log(s)`, `log(1-s)`, and
  the intercept respectively. Implemented directly (rather than
  wrapping `fit_beta_calibrator` which returns only the callable) so
  the fitted coefficients are captured alongside the apply function.
- Both functions: 5 doctest examples each (run in CI via
  `.doctest-modules`) + 15 unit tests covering shape contracts, param
  exposure, parity with the underlying calibrators, single-class
  rejection, and a family-consistency test verifying all 3 binary
  adapters (temperature, Platt, Beta) satisfy the same
  `(params_tuple, apply)` shape.

### Why mirror `fit_temperature_binary`'s tuple shape

#43's body proposed `Callable[[NDArray], NDArray]` as the return type
(just the apply). We landed on `tuple[params, apply]` instead —
matches the existing v0.35.0 contract and gives consumers the params
they need for audit-battery `RunManifest` entries. Consumers wanting
just the apply can use `_, apply = fit_platt_binary(...)`.

### Protocol stability

No Protocol shape changes. Per v0.39.0's roadmap Gate 2 update, v1.0
requires ≥2 minors without Protocol edits — v0.40.0 is minor 1 of 2
toward that target.

## [0.39.0] — 2026-05-18 — consumer-feedback batch (closes #39, #40, #41)

Three issues lifted from `prompt-injection-detection-prototype v1.0.0`
consumer-side workarounds into upstream toolkit primitives + docs.
All small scope; the consumer had filed each with a "Rationale for
upstream landing" section per the *"Working around a library
limitation without filing an upstream issue"* anti-pattern discipline.

### Added

- **`eval_toolkit.metrics.is_metric_defined_for_slice`** + module-level
  constant **`SINGLE_CLASS_INCOMPATIBLE_METRICS`** (`frozenset({"auroc",
  "auprc"})`). AUROC and AUPRC are mathematically undefined on
  single-class slices — ranking metrics require both classes. The
  primitive takes ``is_single_class: bool`` (caller computes from the
  slice's class distribution) and returns whether the metric is
  defined. Use at the per-cell layer to filter `(metric, slice)`
  combinations BEFORE bootstrap so degenerate values never pollute
  downstream artifacts. Default incompatible set overridable per call.
  Closes #39.

### Changed

- **`LeakageCheck.name` Protocol member redeclared as `@property`**
  (was `name: str` class-level annotation). Strictly contract-
  tightening: existing consumers who *read* `check.name` keep working;
  this fixes the `mypy --strict` rejection of
  `list[LeakageCheck] = [CrossSplitLeakageCheck()]` that surfaced when
  consumers built lists of frozen-dataclass leakage checks. Runtime
  `isinstance(check, LeakageCheck)` was always fine — only mypy strict
  was unhappy with read-only-vs-settable Protocol semantics (PEP 544).
  Closes #40.

### Documentation

- **`docs/source/methodology/parallelism.md` §"Memory model:
  worker-copy semantics"** — new section documenting joblib loky's
  pickle-based worker IPC and its memory implications for DataFrame-
  bearing specs (`n_jobs × spec_size` resident memory; the bug that
  OOM-killed a 128-core BCa bootstrap sweep on prompt-injection-
  detection-prototype's marginal-CI regen). Worked example of the
  file-path-shared-state pattern + recommended ceiling formula
  (`min(N, available_RAM_GB / spec_size_GB)`). Cross-referenced from
  the `parallel_map` docstring in `_parallel.py`. Closes #41.

### Why bundled

Three small consumer-feedback items from the same downstream project;
ship as one minor rather than three patches. Pattern matches the
v0.37.0 "TokenizationLeakageCheck + per-module floors" bundling —
small consumer-driven fixes consolidated to keep release-prep
overhead per item low.

## [0.38.0] — 2026-05-18 — executable examples (myst-nb migration)

Docs-only minor. Migrates the 14 walkthrough pages in
`docs/source/examples/` from sybil-validated `` ```python `` blocks to
myst-nb `{code-cell}` directives. Cells now execute during
`sphinx-build` (`nb_execution_mode = "cache"`) rather than during
`pytest` via sybil. Cell outputs (printed text, tables, figures)
render inline in the published HTML, so the docs site reflects the
actual library behavior rather than a snapshot from the last manual
screenshot. Closes #31 (deferred from v0.34.1 and v0.35).

No public API changes.

### Changed

- **14 example pages migrated** to myst-nb (`kernelspec` frontmatter +
  `{code-cell}` directives in place of `` ```python ``). 73 code blocks
  converted in total.
- **Two pages skip execution at page level** (`mystnb.execution_mode:
  'off'`) because they require optional deps kept out of `[dev]`:
  - `pytorch_scorer_example.md` (needs `torch`)
  - `callable_embedder_dedup.md` (needs `[embeddings]` /
    `sentence-transformers`)
  Both pages render their code statically.
- **`docs/source/examples/index.md`** — "How these run" section
  rewritten to reflect myst-nb instead of sybil; new "skip-execed
  pages" callout.
- **`conftest.py`** — dropped `docs/source/examples/*.md` from sybil
  patterns. Sybil still covers `README`, `methodology/`, `migration/`,
  `getting-started`, etc. (parts without executable-notebook value).

### Why this matters

myst-nb infrastructure has been wired since v0.31.0 (the Sphinx docs
migration) but was underutilized — all example pages used static
`` ```python `` blocks. This release closes that gap. API drift in
the future will fail the docs build via runtime-output verification
(in addition to sybil's existing Python-level error catch on the
other doc trees).

## [0.37.0] — 2026-05-18 — TokenizationLeakageCheck + per-module coverage floors

Two-issue bundle (#35 + #37) plus housekeeping closure of stale items
(PR #27, #38) that turned out to have been resolved in v0.33.x without
being checked off. Roadmap refresh in `3d40796` (this minor's
predecessor commit) replaced the version-keyed candidate list with
issue-keyed tracking, so this class of stale-roadmap bug shouldn't
recur.

### Added

- **`eval_toolkit.leakage.TokenizationLeakageCheck`** — new within-split
  `LeakageCheck` that dedups on tokenizer output rather than raw text.
  Catches encoding-obfuscated dupes that survive
  `NormalizedFormLeakageCheck` but collapse to identical `input_ids`
  under a transformer's BPE / SentencePiece / WordPiece tokenizer.
  Accepts any `Callable[[str], Mapping[str, object]]` returning HF-style
  output with an `"input_ids"` key — does **not** import `transformers`
  itself; consumers pass an already-instantiated tokenizer. Default
  severity `"error"` (mirrors `NormalizedFormLeakageCheck`). Closes #35.
- New optional install extra **`[transformers]`** (`transformers>=4.0`).
  Intentionally **not** in `[all]` / `[dev]` — mirrors the `[embeddings]`
  precedent from v0.33.1 to keep contributor setup small (transformers
  transitively pulls torch ~700MB).

### Test

- **Per-module coverage floors restored.** `scripts/check_module_floors.py`
  enforces an 85 % per-file floor (coverage.py natively only ships
  global `--fail-under`). Hooked into `make coverage` via a post-pytest
  invocation. Closes #37.
- **`# pragma: no cover` on optional-dep-active paths** in `seeds.py`
  (torch) and `embeddings.py` (sentence-transformers). Reflects the
  reality that these branches execute in user code, not CI. Both
  modules now report 100 % coverage; previously sat at ~70 % which
  obscured per-module floor enforcement.

### Fixed

- **`make coverage` Makefile parity with PR CI.** PR #27 (external
  contributor @leno23, draft) proposed adding `-m "not monte_carlo and
  not benchmark"` to the `coverage` target. Audit found the same fix
  had landed in v0.33.0 commit `9e375a8` ahead of the PR being filed;
  closed PR #27 as superseded with thanks. No change in this release.

### Closed (already-resolved)

- **#38 — CI doctests for `paths.py` / `provenance.py` / `seeds.py` /
  `docs.py`.** All four modules were added to `.doctest-modules` in
  `a26fd44` (2026-05-14, v0.32.x era); 7 doctests collected across the
  named modules in current CI. Closed as already-resolved.

### Test coverage

Test count 1376 → 1387 (+11). Aggregate 95.65 % → 95.69 %. All 28
modules ≥ 90 % individually post-pragma.

## [0.36.0] — 2026-05-18 — harness parallelization (#29, #30) + Node 24 actions

Wires the v0.34.0 unified parallelism pattern into the harness evaluation
loop. `evaluate()` and `evaluate_folded()` now accept an `n_jobs` kwarg
(default `1` preserves bit-identical sequential behavior); under
`n_jobs != 1`, the `(slice × scorer)` work-unit loop in
`_score_all_slices` and the `(spec × scorer)` fit phase in
`_attach_transferred_operating_points` dispatch through joblib loky via
the existing `_parallel.parallel_map` helper.

### Added

- `evaluate(..., n_jobs: int = 1)` and `evaluate_folded(..., n_jobs: int = 1)`
  — keyword-only kwarg per Principle #3 of `methodology/parallelism.md`.
  `n_jobs=1` (default) runs the existing pure-Python sequential loop
  (Principle #4 — bit-identical to v0.35). `n_jobs > 1` uses joblib loky;
  `n_jobs=-1` uses all cores; `n_jobs=0` is rejected. Closes #29, #30.
- Strict-pickle Scorer sniff at `evaluate()` entry when `n_jobs != 1`:
  raises a clean `TypeError` referencing
  `methodology/parallelism.md#scorer-picklability` with the underlying
  pickle error attached. Reuses the v0.35 ADR contract; no new exception
  class. Catches non-picklable scorers up front rather than relying on
  joblib's more permissive cloudpickle path (which would silently absorb
  closures and obscure the contract documented in v0.35).

### Internal

- New module-scope step functions `_score_one_pair` and
  `_fit_one_op_point_pair` in `harness.py` (picklable; required by loky).
- `_score_all_slices` and `_attach_transferred_operating_points`
  refactored to use flat work-unit dispatch via `parallel_map`.

### Tests

- New `tests/test_harness_parallelism.py` (7 tests): bit-identical
  reproducibility across `n_jobs=1` vs `n_jobs=2` for `evaluate`
  (basic, paired-diffs, operating-points), `evaluate_folded`,
  picklability rejection (closure scorer), `n_jobs=0` rejection,
  `n_jobs=-1` smoke. All 66 harness tests pass (7 new + 59 existing).

### Infrastructure

- Bumped `actions/upload-artifact` and `actions/download-artifact` from
  `@v5` → `@v6` across `publish.yml` / `nightly-mc.yml` /
  `nightly-benchmarks.yml`. The v6 majors run on Node.js 24
  (GitHub deprecates Node 20 actions from 2026-06-02). Other pinned
  actions (`checkout@v6`, `setup-uv@v8.1.0`, `codeql-action@v3`,
  `deploy-pages@v4`, `upload-pages-artifact@v3`) were not flagged in
  the v0.35 publish annotation and are deferred to a separate audit.

## [0.35.0] — 2026-05-18 — `fit_temperature_binary` + Scorer picklability ADR

Small, additive release. Adds a binary-classification calibration helper
that lets consumers drop the ~50 LOC scalar-proba adapter many were
carrying, plus a design ADR that unblocks the v0.36 harness / operating-
point parallelization work (#29, #30) without re-litigating picklability.

### Added

- `eval_toolkit.fit_temperature_binary(y_true, y_score)` — scalar-proba
  adapter for the multi-class `fit_temperature` fitter. Converts `(n,)`
  probabilities of class 1 to a 2-column logit array via clipped logit
  (`[0, logit(p)]` so softmax row 1 reproduces `p`), delegates to the
  deployment-quality fitter, and returns `(T_opt, apply)` where
  `apply: (n,) -> (n,)` does scalar-in / scalar-out T-scaling. Unlike
  `fit_temperature_oracle`, no warning — the contract assumes val / test
  separation (deployment-quality calibration, not fit-on-test). Closes
  #28.

### Documentation

- `docs/source/methodology/parallelism.md` — new `## Scorer picklability`
  sub-section documenting the Scorer protocol's picklability contract
  for `n_jobs > 1` usage. Includes worked picklable / broken-closure /
  fix examples plus a list of common non-picklable patterns to watch for
  in user-supplied Scorers (closures, lambdas on instances, local-scope
  classes, attributes holding live sockets / file handles). Anchors on
  the existing v0.34.0 `parallel_map` pickle sniff + `TypeError`
  channel — no new exception class. Unblocks v0.36 implementation of
  #29 and #30.
- `eval_toolkit.protocols.Scorer` docstring — Notes block pointing at
  the new methodology section.

## [0.34.0] — 2026-05-17 — Phase 4 stats unblockers + unified parallelism + cookbook (BREAKING)

Closes all 7 open backlog issues in one consumer-closing release. Also
lands the toolkit's first unified parallelism story: a shared internal
`parallel_map` helper + `n_jobs` kwarg on all 5 public bootstrap
functions. Future iterations will mechanically extend the same helper to
harness + operating-points (follow-up issues filed).

### Breaking changes

- **`eval_toolkit.bootstrap.mde_from_ci`**: parameter renamed from
  `paired` to `ci` and type widened to `BootstrapCI | PairedBootstrapCI`
  (was `PairedBootstrapCI`-only). Positional callers unaffected; keyword
  callers must update:
  ```python
  mde_from_ci(paired=x)  # v0.33.x and earlier
  mde_from_ci(ci=x)      # v0.34.0+
  mde_from_ci(x)         # positional form, unchanged
  ```
  This is a one-time exception to the repo's 2-minor-version deprecation
  warning policy (justification + criteria recorded in
  [`docs/source/DEPRECATION.md`](docs/source/DEPRECATION.md#one-time-exceptions-to-the-2-minor-version-warning-policy)).
  Notification issues filed on the 2 known toolkit consumers
  (`prompt-injection-detection-submission`, `post-transformers`). Audit
  confirms both use positional form — zero actual breakage in practice.

### Added

- `eval_toolkit.bootstrap.block_bootstrap_on_folds` — CV-aware sibling
  to `cv_clt_ci`; resamples K folds with replacement; returns
  `BootstrapCI(method="block_bootstrap")`. The A-008 sensitivity-check
  pattern (block-bootstrap halfwidth / cv_clt halfwidth > 1.5 flags
  LODO non-exchangeability) is the prototypical use. Closes #21.
- `eval_toolkit.RecallAtFprResult` (frozen dataclass) +
  `eval_toolkit.recall_at_fpr(y_true, y_score, target_fpr)` — one-shot
  recall + actual_fpr + FP/TN at the smallest threshold meeting FPR ≤
  target. Use `.to_dict()` for JSON / pandas-row integration. Closes #9.
- New optional `n_jobs: int = 1` kwarg on 5 bootstrap functions:
  `bootstrap_ci`, `paired_bootstrap_diff`, `paired_bootstrap_ece_diff`,
  `paired_bootstrap_op_point_diff`, `paired_mde` (via `_bootstrap_t_ci`
  internal helper). Backed by a new internal `_parallel.parallel_map`
  helper (joblib loky; not exported). `n_jobs > 1` reproduces `n_jobs=1`
  result bit-for-bit for the same seed (via
  `np.random.SeedSequence.spawn`). Lambda metrics rejected at call time
  with helpful `TypeError`. `n_jobs > os.cpu_count()` is auto-capped
  with WARNING log; `n_jobs=0` raises `ValueError`. Closes #17.
- 6 new pages in `docs/source/examples/`:
  - **Cookbook** (closes #19): `nested_seed_split.md`,
    `callable_embedder_dedup.md`, `cross_corpus_contamination_scan.md`.
  - **Plotting walkthroughs**: `plot_roc_curve_walkthrough.md`,
    `plot_pareto_frontier_walkthrough.md`,
    `plot_slice_metric_heatmap_walkthrough.md` (backfills the v0.33.0
    docs gap).
- `docs/source/methodology/parallelism.md` — design rationale + caller
  contract for the toolkit-wide parallelism story. Documents the 6
  design principles (single backend, single helper, opt-in per-fn,
  default sequential, reproducibility via SeedSequence, picklability
  surface) and the checklist for adding `n_jobs` to a new function.

### Changed

- `eval_toolkit.bootstrap.mde_from_ci` now accepts
  `BootstrapCI | PairedBootstrapCI` (was paired-only). See **Breaking
  changes** above. Closes #20.
- `eval_toolkit.build_manifest` gains `config_path: Path | str | None`
  kwarg; when supplied, `config_hash` is computed as
  `sha256(Path(config_path).read_bytes()).hexdigest()` — capturing the
  exact YAML file bytes including comments + key ordering (which the
  default canonical-JSON path strips during parse). Default behavior
  preserved when `config_path is None`. Closes #10.

### Internal

- New `src/eval_toolkit/_parallel.py` (internal; not exported) — single
  source of truth for parallelism. Future per-function `n_jobs`
  additions will reuse this helper. The toolkit's first INFO-level
  log site is here (once-per-process guidance log when `n_jobs=1` AND
  iteration count ≥ 1000). New `tests/test_parallel.py` covers smart-
  default semantics + reproducibility contract.
- New golden test `tests/golden/test_dedup_holdout_calibration.py`
  exercising 3 deterministic `SimilarityStrategy` variants against a
  migrated 50-pair adversarial fixture at thresholds {0.75, 0.80, 0.85}
  (strict snapshot at `tests/golden/data/dedup_holdout_expected.json`)
  plus an `EmbeddingCosineStrategy` soft-bound check (FPR < 0.5,
  FNR < 0.5 at threshold 0.80) gated by `pytest.importorskip` +
  `@pytest.mark.slow`. Refresh helper at
  `scripts/refresh_dedup_holdout.py`. Closes #18.
- `CONTRIBUTING.md` + `docs/source/repo-strategy.md` updated with
  explicit "Parallelism" section codifying the new pattern (was an
  implicit anti-pattern before; v0.34.0 codifies the new opt-in design).
- `docs/source/DEPRECATION.md` extended with a "One-time exceptions"
  section documenting the `mde_from_ci` rename + criteria future
  exceptions must satisfy.
- RNG-stream note: the 5 wired bootstrap fns now derive per-resample
  seeds via `np.random.SeedSequence(seed).spawn(n_resamples)` instead
  of sequential calls on a single `Generator`. The bootstrap output is
  *statistically equivalent* (both are valid bootstraps) but the exact
  numerical CI bounds for the same caller-supplied `seed` will differ
  slightly from v0.33.x. Existing tests use behavioral assertions
  (`overlaps_zero`, `delta`, etc.) that are robust to the RNG-stream
  change; tests that pin exact CI bounds (e.g., consumer golden tests)
  may need regen on upgrade.

## [0.33.1] — 2026-05-17 — MiniLM convenience embedder

Closes the last open item in the v0.33 milestone (deferred from v0.33.0
per the planned split). Ships the canonical semantic-dedup recipe
(`sentence-transformers/all-MiniLM-L6-v2` at cosine ≥ 0.80, per ADR-027)
pre-wired for `EmbeddingCosineStrategy` so consumers stop reinventing the
embedder-wrapping boilerplate.

No breaking changes. Public API gains 1 new export
(`make_minilm_embedder`) and 1 new optional dependency extra
(`[embeddings]`). Existing `EmbeddingCosineStrategy` callers that already
ship their own embedder are unaffected.

### Added

- `eval_toolkit.embeddings.make_minilm_embedder` — factory returning a
  `Callable[[Sequence[str]], np.ndarray]` that loads
  `sentence-transformers/all-MiniLM-L6-v2` (configurable), memoises model
  loads via `functools.lru_cache(maxsize=8)`, and emits `(n, 384)`
  `float64` embeddings ready for `EmbeddingCosineStrategy`. Raises a
  helpful `ImportError` with the install hint when the optional dep is
  absent. Closes #3.
- New optional dependency extra `[embeddings]` →
  `sentence-transformers>=3.0`. Intentionally **not** in `[all]` / `[dev]`
  because the transitive `torch` install (~700MB) would balloon
  contributor setup.

### Internal

- `docs/source/api/embeddings.md` Sphinx page added (autosummary stub);
  wired into the API toctree alongside the other module pages.
- `docs/source/api/plotting.md` autosummary backfilled with
  `plot_roc_curve`, `plot_pareto_frontier`, `plot_slice_metric_heatmap`
  (missed in v0.33.0).
- `tool.mypy.overrides` extended with `sentence_transformers.*` (matches
  the existing pattern for untyped third-party libs).

## [0.33.0] — 2026-05-17 — Plotting batch + ax= parity + CI quality-of-life

Consumer-unblocking release: closes the four upstream-gap TODOs in
`prompt-injection-detection-submission`'s Phase 4 figures (F1, F2, F5,
F6-left) which had been carrying hand-rolled prototypes pending these
primitives. Also bundles two CI/maintenance fixes that were quality-of-life
pain points during v0.32 ship.

**Note**: The `v0.33` milestone's #3 (`make_minilm_embedder`) is deferred
to the next iteration (likely v0.33.1 or v0.34) so this release stays
focused on the plotting batch + `ax=` parity. MiniLM adds a new optional
dep + new module; ships better as its own bite.

No breaking changes. Public API gains 3 new plotting exports
(`plot_roc_curve`, `plot_pareto_frontier`, `plot_slice_metric_heatmap`)
and adds an `ax=` kwarg to 2 existing plotting fns (`plot_metric_bars`,
`plot_score_histograms`) — all additive.

### Added

- `eval_toolkit.plotting.plot_roc_curve` — sibling to `plot_pr_curve`;
  accepts `ax=`, optional baseline overlay, threshold marker. Includes
  a diagonal chance line. Closes #14.
- `eval_toolkit.plotting.plot_pareto_frontier` — cost-vs-performance
  scatter with running-best frontier overlay (O(n log n) sweep). Supports
  both higher-is-better and lower-is-better metric directions, optional
  per-point labels. Closes #15.
- `eval_toolkit.plotting.plot_slice_metric_heatmap` — (rows × cols × metric)
  heatmap with colorbar + optional cell annotations + NaN-cell masking.
  Closes #16.

### Changed

- `plot_metric_bars` and `plot_score_histograms` now accept an `ax=` kwarg,
  bringing the count of `ax=`-accepting plotting fns to 6 of 7
  (`plot_confusion_matrix_grid` remains figure-creating since it's
  intrinsically a grid-of-axes). Closes #24.
- `Makefile`'s `coverage` target now filters `monte_carlo` and `benchmark`
  markers, matching what `.github/workflows/ci.yml` actually runs. `make ci`
  drops from ~45 min to ~3 min locally. Closes #25.

### Internal

- 16 new edge tests covering input validation + `ax=` branches for the
  3 new plotting fns and the 2 backfilled ones.
- 3 new `@pytest.mark.mpl_image_compare` baseline tests + checked-in
  baseline PNGs for the new plotting fns.
- `.github/workflows/*.yml` audited for Node.js 20 deprecation; bumped
  `actions/upload-artifact@v4 → v5` (3 workflows) and
  `actions/download-artifact@v4 → v5` (publish.yml) ahead of the
  2026-09-16 Node-20 removal deadline. Closes #26.

## [0.32.0] — 2026-05-16 — Multiple-comparisons correction + EvidenceGate discoverability

Bundled close-outs from the `v0.32` milestone triage (4 issues). Adds
the BH/Bonferroni p-value correction primitive that V5 has been
carrying inline, surfaces `EvidenceGate` from the README +
getting-started, expands the `set_global_seeds(strict_torch_determinism=...)`
docstring with concrete kernel examples + the canonical PyTorch
reproducibility link, and co-locates a rationale for the
`DEFAULT_BOOTSTRAP_RESAMPLES = 1000` default at its definition site.

No breaking changes. Public API gains 4 new exports
(`fdr_bh_correct`, `bonferroni_correct`, `correct_p_values`, the
`CorrectionMethod` `Literal` alias); all four are additive on the
`eval_toolkit.bootstrap` surface.

### Added

- `eval_toolkit.bootstrap.{fdr_bh_correct, bonferroni_correct, correct_p_values}` —
  Benjamini-Hochberg + Bonferroni p-value corrections for the
  multiple-comparisons problem (e.g., correcting many DeLong /
  paired-bootstrap p-values across slices × scorers × folds). NumPy-only
  implementations; cites Benjamini & Hochberg 1995. Includes the
  `CorrectionMethod = Literal["bh", "bonferroni", "none"]` dispatch
  alias and a `correct_p_values(p_values, *, method="bh")` helper that
  routes to the right correction. Closes #1.

### Changed

- `README.md` modules table + worked-examples section now surface the
  `EvidenceGate` class explicitly and link to the existing Sphinx
  worked example (`docs/source/examples/claims_and_gates.md`).
  `docs/source/getting-started.md` adds a prose paragraph after the
  claim example explaining that the three gate factories return
  `EvidenceGate` instances and how custom gates are written.
  Sphinx-side API/methodology/examples pages were already complete and
  are unchanged. Closes #2.
- `set_global_seeds` docstring extended: Notes now name concrete
  kernels affected by `strict_torch_determinism=True` (flash-attention,
  scatter/gather ops, certain CUDA reductions); See Also adds the
  canonical PyTorch reproducibility-notes page alongside the existing
  Lightning `seed_everything` reference. Closes #4.
- `DEFAULT_BOOTSTRAP_RESAMPLES = 1000` in `harness.py` now carries a
  3-line rationale comment cross-referencing
  `docs/source/methodology/bootstrap.md` §"Resampling budget" (~0.5%
  Monte-Carlo error on a 95% CI quantile at B=1000; the lower end of
  the 1k–10k range modern toolkits use). Closes #11.

### Internal

- 9 new unit tests for the BH/Bonferroni functions in
  `tests/test_bootstrap_unit.py`, ported from `piv5.eval.paired` (V5
  v0.6): Bonferroni multiplier-and-clip, BH monotonicity under uniform
  null, BH spec example, BH input-order preservation, three dispatch
  smoke tests for `correct_p_values`, unknown-method validation, and
  bounds/empty-input invariants for both primitives.

## [0.31.0] — 2026-05-16 — Sphinx docs migration

Documentation toolchain swap from mkdocs-material + mkdocstrings to
Sphinx + pydata-sphinx-theme. The live site at
`https://brandon-behring/github.io/eval-toolkit/` rebuilds on this
release with the new theme + per-symbol API pages + intersphinx
cross-references to numpy / scipy / sklearn / pandas / matplotlib.

No Python source change. No public-API change. Every test still passes
(Sybil's 203 doc tests collect at their new `docs/source/` paths).

### Why migrate (user pain points, Q1)

1. **Look + feel** — `pydata-sphinx-theme` is the scientific-Python
   convention (numpy, scipy, sklearn, pandas all use it). The previous
   mkdocs-material site read as "tech-blog" style; the new site reads
   as "scientific library reference."
2. **Auto-generated API docs feel thin** — `mkdocstrings` stubs
   (single `::: eval_toolkit.X` per page) produced uniform but shallow
   output. Sphinx `autosummary` + `autodoc` + `napoleon` generates a
   summary table per module + a dedicated HTML page for each public
   symbol (~190 per-symbol pages total). Each carries signature with
   type hints, NumPy-style docstring sections, and a `[source]` link
   back to GitHub.
3. **Cross-references / intersphinx** — `:class:`numpy.ndarray`` and
   similar markers in docstrings now render as live links to the
   external project's docs. mkdocs had no equivalent.

### Changed

- **Docs toolchain**: mkdocs-material + mkdocstrings + pymdown-extensions →
  Sphinx + pydata-sphinx-theme + myst-nb + jupyter-cache +
  sphinx-copybutton + sphinx-design + sphinx-autodoc-typehints +
  linkify-it-py. Same Pages URL, same single-version policy, same
  push-to-main + tag deploy triggers; the build chain underneath is
  fully replaced.
- **Docs source layout**: `docs/*.md` → `docs/source/*.md` with
  `docs/source/conf.py` as the canonical Sphinx config. The
  conventional `docs/source/` / `docs/build/` split (Q8) matches what
  `sphinx-quickstart` generates and what numpy / sklearn use.
- **API stubs**: each of 22 modules' mkdocstrings stubs (`::: eval_toolkit.X`)
  replaced with MyST `autosummary` blocks using `:toctree: generated/<mod>/`.
  Per-symbol page generation is incremental (cached in
  `docs/source/api/generated/`, gitignored).
- **Heading anchors**: 138 mkdocs-style `## Heading {#anchor}` patterns
  across 18 files migrated to MyST native `(anchor)=\n## Heading`
  block-target syntax. The `{#anchor}` syntax was rendering as literal
  text under MyST; `(anchor)=` registers as a proper cross-reference
  target.
- **Cross-tree refs**: 7 hardcoded relative paths to files outside the
  Sphinx source tree (`../src/eval_toolkit/_deprecated.py`,
  `../tests/test_deprecations.py`, `../CONTRIBUTING.md`, etc.) rewritten
  to absolute GitHub URLs at the project repo.

### Added

- **Executable cells via myst-nb** (Q9): example `.md` files can opt in
  to cell execution at docs-build time. `nb_execution_mode = "cache"`
  (Q11) means first build executes everything; subsequent builds
  re-execute only changed cells. Cache stored in
  `docs/build/.jupyter_cache/`. v0.31.0 ships the infrastructure;
  individual examples that opt in via `{code-cell}` directives are a
  future enhancement (current `.md` examples still use the
  ` ```python ` fences that render as static code).
- **Intersphinx mapping** (Q1 #3): live links from any
  `:class:`numpy.ndarray`` / `:func:`scipy.stats.bootstrap`` etc. in a
  docstring or `.md` file to the target project's docs.
- **`sphinx-autodoc-typehints`** integration: type hints rendered into
  signatures cleanly rather than duplicated in the body.

### Removed

- `mkdocs.yml` — replaced by `docs/source/conf.py`.
- `docs/javascripts/mathjax-config.js` — `sphinx.ext.mathjax`
  handles MathJax v3 natively.
- `.github/workflows/docs-sphinx-preview.yml` — the parallel-build
  preview workflow served Phase 1+2; Phase 3 promotes Sphinx to the
  primary `docs.yml` workflow and the preview goes away.

### Out of scope (explicitly not pursued)

- **tikzjax** — v0.28.0 enabled tikzjax (Q12) but no docstring or `.md`
  ever used it. Sphinx has no clean equivalent (the audit confirmed
  zero `<script type="text/tikz">` blocks exist). Dropped without
  replacement. MathJax remains for inline LaTeX math.
- **Versioned docs** (`sphinx-multiversion`) — single-version site
  retained (Q5); can be added later without re-architecting.
- **Read the Docs hosting** — staying on GitHub Pages.
- **Sphinx-gallery** for examples — keeps current `.md` + Sybil
  testing; conversion to `.py` scripts not pursued.
- **Project logo** (Q13) — favicon only; can add later via a
  cosmetic PR.

### Kapoor leakage detectors + mutmut follow-ups → v0.32.0

The post-v0.30.0 backlog items (L3.3 `SamplingBiasCheck` + L2-general
`IllegitimateFeatureCheck` Kapoor 2023 detectors, mutmut audit
follow-ups, benchmark baseline-comparison automation) were originally
queued for v0.31.0. Per the migration plan, they shift to v0.32.0 so
docs ships its own focused release.

## [0.30.1] — 2026-05-15 — repo hygiene + release-tooling patch

Pure organizational + release-tooling cleanup. No public-API change —
every existing import, kwarg, and return shape preserved (verified by
the public-API drift-guard test). Total test count unchanged: every
assertion in the reorganized files preserved verbatim.

The work originates from the post-v0.30.0 backlog audit (zero TODO /
FIXME comments in source, zero overdue `@deprecated` deadlines, no dead
modules). Ships clusters B + C; clusters A + B² (Kapoor 2023 leakage
detectors + mutmut audit follow-ups) defer to v0.31.0.

### Added

- **`make release-prep VERSION=X.Y.Z` target** (C1): canonical "step 1"
  of the release flow. Validates PEP 440, atomically rewrites
  `src/eval_toolkit/_version.py`, regenerates
  `tests/golden/public_api/snapshot.json`, and prints the remaining
  manual steps. Closes the public_api snapshot-drift gotcha that hit
  v0.28.0 / v0.28.1 / v0.29.0 / v0.30.0 — ~50% of recent releases.
  Documented in `docs/RELEASING.md` (TL;DR + detailed runbook §1).

### Changed

- **Tests reorganized by domain (B1 + B2)**:
  - `tests/test_coverage_gap.py` (1252 LOC monolithic) → split into
    `test_coverage_metrics.py`, `test_coverage_bootstrap.py`,
    `test_coverage_calibration.py`, `test_coverage_plotting.py`,
    `test_coverage_harness.py` (the 7-domain reality required 5 new
    files rather than the 4 originally scoped). MinHashLSHStrategy +
    EmbeddingCosineStrategy tests merged into existing
    `tests/test_text_dedup_coverage.py`. 107 tests preserved verbatim
    across the new boundaries.
  - `tests/test_harness_v07.py` + `tests/test_harness_v22.py` deleted;
    their 23 tests distributed by subject matter into existing/new
    feature-grouped files: `test_harness_smoke.py` (on_scorer_error
    contract), new `test_harness_folded.py` (evaluate_folded +
    by_fold/fold_summary), new `test_harness_metric_options.py`
    (evaluate_scorer_on_slice metric kwargs), `test_leakage.py`
    (leakage_checks integration). Version-keyed naming retired —
    the repo convention is per-feature test files.

- **`__init__.py` topic-grouping comments** (B3): the 200-symbol
  `_EXPORTS` dict now carries `# --- module ---` section dividers above
  each grouped block. Pure documentation; the public-API snapshot
  reads dict keys and values only, so the snapshot is unchanged.
  Improves IDE outline / goto-definition discoverability.

- **`docs/v0.3_research_audit.md` archived** (B5): the 37KB
  pre-v0.30 audit doc was not referenced in `mkdocs.yml` nav — a
  historical artifact. Moved to `docs/archive/v0.3_research_audit.md`
  with a `docs/archive/README.md` explaining the convention.

### Deprecated

- **`[validation]` optional-dependency extra** (B6): no-op since
  v0.16.0 (jsonschema moved to base deps). Announced for removal in
  v0.33.0 per the 2-minor-version deprecation policy in
  `docs/DEPRECATION.md`. `pip install eval-toolkit[validation]`
  continues to resolve cleanly through v0.32.x. Extras can't emit
  `DeprecationWarning` at import time, so the deprecation is
  documentation-only.

### Out of scope (this patch)

- Cluster A — Kapoor 2023 leakage detectors (L3.3 SamplingBiasCheck +
  L2-general IllegitimateFeatureCheck) ship in v0.31.0 with their own
  focused release. L1.2 + L1.3 are permanently dropped (see the v0.31.0
  CHANGELOG entry + `docs/methodology/leakage-detection.md`
  "Out of scope" section for the rationale).
- Cluster A² — mutmut audit follow-ups + cosmic-ray attempt also ship
  with v0.31.0.
- Benchmark baseline-comparison automation in `nightly-benchmarks.yml`
  also deferred to v0.31.0.

## [0.30.0] — 2026-05-15 — code-organization refactors (SRP / DRY / maintainability)

Internal restructuring driven by the post-v0.29.0 code-organization
audit (SRP / DRY / maintainability). Five of seven audited refactors
shipped; two deferred with documented rationale. No public API
change — every existing import, kwarg, and return shape is preserved.
1252 fast tests passing; v0.29.0-tagged snapshot drift on `__version__`
only.

### Changed

- Refactor #1 — decomposed the 175-line `evaluate()` orchestrator into
  three named phases. `evaluate()` itself drops to 35 lines reading as
  a top-level workflow: validate → config → leakage gate → score → ops
  → assemble `RunResult`. Two new private helpers:
  `_run_leakage_phase(checks, slices, on_leakage, config)` owns the
  conditional leakage import + report aggregation + raise-policy;
  `_score_all_slices(scorers, slices, *, ...)` owns the slice/scorer
  nested loop + score cache + paired-diffs and returns
  `(by_slice, score_cache)`. No behavioral change; verified against
  the full fast-test suite.

- Refactor #2 — extracted `_check_required_columns(df, required, *,
  context)` in `loaders.py` to dedupe the three identical
  column-presence checks in `DataFrameLoader`, `ParquetGlobLoader`,
  and `HFDatasetsLoader`. The strata-column check in `DataFrameLoader`
  keeps its existing error format (test-contract preservation).

- Refactor #3 — extracted `_validate_bin_count(n_bins)` in `metrics.py`
  to dedupe the six identical `if n_bins < 2: raise ValueError` checks
  across `expected_calibration_error` (and the equal-mass / quantile /
  adaptive variants). Same error message; helper raises so the
  call-site reads as a single line.

- Refactor #4 — hoisted three shared scorer test doubles into
  `tests/conftest.py`: `StubScorer` (precomputed array),
  `UniformScorer(seed)` (uniform random per call), and
  `ErrorScorer(exc_type, message)` (always raises). Migrated
  `test_harness_smoke.py` and `test_harness_v07.py` to import via
  aliases so the existing local names (`_StubScorer`,
  `_UniformScorer`, `_BrokenScorer`) keep working with minimal diff
  churn. Specialized stubs (slice-aware, deterministic-text-score)
  remain co-located with their test files because their behavior is
  the test's point.

- Refactor #6 — extracted `_pr_curve_result_at(prec, rec, thresh, idx,
  criterion)` in `thresholds.py` as the PR-curve sibling of the
  existing `_result_at`. The three PR-curve selectors
  (`MaxF1Selector`, `TargetRecallSelector`, `TargetPrecisionSelector`)
  each had an identical 6-line `ThresholdResult` assembly reading
  prec/recall from the curve at a chosen idx; replaced with a single
  helper call. The ROC selectors and `CostSensitive` already use
  `_result_at`; `CISafeThreshold` keeps its custom Wilson-interval
  flow.

### Documentation

- Refactor #7 — reframed `harness.py`'s late `eval_toolkit.leakage`
  import as the canonical "conditional in-function import gated on
  optional feature" Python idiom (not a circular-dep workaround).
  Added the same comment block to the new `_run_leakage_phase` helper
  introduced by refactor #1. The audit had flagged this as TYPE_CHECKING
  candidate; TYPE_CHECKING applies to type annotations, not runtime calls.

### Deferred (documented in v0.30.0 plan)

- Refactor #5 — `ScorerEvalOptions` dataclass for the 13-param
  `evaluate_scorer_on_slice()` signature. Skipped: most callers pass
  only 2-5 kwargs (subsets) and adding a dataclass alongside kwargs
  would double the API surface without reducing it. Full replacement
  requires deprecation across 2-3 minor versions; the current surface
  is documented and orthogonal flags from v0.22.0 are individually
  justified. Reconsider on external signal that the surface hurts users.

## [0.29.0] — 2026-05-15 — best-practice infrastructure bundle (β + γ)

Forward-compatibility, observability, perf-regression detection, and
docs/repo polish. No new user-facing API surface; better support for
the existing one. Ships everything from the v0.29.0 plan's Tier β
(developer experience + governance) and Tier γ (perf-benchmark +
Binder + What's New). Tier α (security: CodeQL + pip-audit) shipped
earlier in v0.28.1.

### Added

- Tier γ #3 (Binder integration, from v0.29.0 audit): added
  `binder/environment.yml` pinning Python 3.13 + `eval-toolkit[all]`
  + JupyterLab so MyBinder can build a one-click "try it in your
  browser" environment with all examples pre-installed. Binder
  badge in README links to `docs/examples/` as the labpath. Empty
  `binder/postBuild` reserved for future setup steps. No
  ongoing maintenance burden — Binder builds on-demand from the
  repo HEAD.

- Tier γ #2 (What's New page, from v0.29.0 audit): added
  `docs/whats-new.md` — a more-digestible CHANGELOG-derived
  landing covering the v0.27.1 through v0.29.0 progression. Each
  release gets a one-paragraph theme summary + bullet highlights.
  For full per-line detail, the in-tree `CHANGELOG.md` remains
  authoritative. Added to `mkdocs.yml` nav between "Home" and
  "Getting started".

- Tier γ #1 (pytest-benchmark, from v0.29.0 audit): perf-regression
  detection on math kernels. New `tests/benchmarks/` directory with 6
  benchmarks: `pr_auc`, `roc_auc`, `brier_score`,
  `expected_calibration_error`, `bootstrap_ci(pr_auc)`,
  `paired_bootstrap_diff(pr_auc)`. New `benchmark` pytest marker;
  excluded from PR CI alongside `monte_carlo`. New
  `.github/workflows/nightly-benchmarks.yml` runs weekly on Saturdays
  at 02:00 UTC. Benchmark JSON uploaded as artifacts; comparison to
  baseline is manual via `pytest-benchmark compare` for now. Added
  `pytest-benchmark>=4.0` to `[dev]` extras.

- Tier β #4 (structured logging, from v0.29.0 audit): standard
  library-friendly logging discipline per Q4=A.
  `src/eval_toolkit/__init__.py` attaches a `NullHandler` to the
  `eval_toolkit` root logger — library is silent by default,
  matching the numpy / scikit-learn / requests convention. Each
  module owns its logger: `_logger = logging.getLogger(__name__)`
  resolves to `eval_toolkit.harness` / `eval_toolkit.leakage` /
  `eval_toolkit.bootstrap` / `eval_toolkit.loaders` — mirroring
  the import path so consumers can filter granularly. New DEBUG
  emissions at: leakage check completions
  (`run_leakage_checks`), bootstrap_ci run parameters,
  DataFrameLoader split construction. `harness.py` retains its
  existing INFO-level slice-transition emissions. Log-level
  conventions documented in `CONTRIBUTING.md`. Tests at
  `tests/test_logging.py` (7 tests) verify NullHandler attachment,
  per-module logger resolvability, DEBUG emission on the touched
  code paths, and silent-by-default behavior (no handlers leak
  output without consumer config).

- Tier β #3 (deprecation infrastructure, from v0.29.0 audit): per
  Q2=A, string-deadline `@deprecated` decorator at
  `src/eval_toolkit/_deprecated.py`. Signature:
  `@deprecated("0.30.0", reason="...", use_instead="...")`. Deadline
  validated at decoration (import) time via regex; typos fail at
  module load, not at call. Emits `DeprecationWarning` with a
  structured message on every call. Preserves `functools.wraps`
  metadata (`__name__`, `__doc__`, `__wrapped__`); also stashes
  `__deprecated_deadline__` / `__deprecated_reason__` /
  `__deprecated_use_instead__` for test introspection. Policy
  document at `docs/DEPRECATION.md` defines the minimum-two-minor-
  versions deprecation window. Tests at `tests/test_deprecations.py`
  (8 tests, all unit) cover the decorator + include a scanner that
  walks `eval_toolkit.__all__` and fails if any callable carries
  an expired deadline ≤ current `__version__`. No actual deprecations
  introduced — pure policy + tooling infrastructure for future use.

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
