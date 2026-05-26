# Roadmap

Forward-looking tracker for `eval-toolkit`. Cross-links the consumer
gap docs that motivate upstream work and records the criteria for
v1.0.0.

This document is **descriptive of intent, not a commitment**. The
priorities reflect today's understanding of what consumers need;
order may change as feedback comes in.

## Currently shipped (as of v1.0.0)

See [`CHANGELOG.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md) for the full release history.
Highlights since v0.33:

- **v0.34.0** — Phase 4 stats unblockers (CV-aware block bootstrap,
  generalized `mde_from_ci`, multi-comparisons correction); unified
  internal `parallel_map` helper + `n_jobs` kwarg on all 5 public
  bootstrap functions; cookbook docs (3 compositional patterns).
  Closed the entire prior backlog in one release.
- **v0.35.0** — `fit_temperature_binary` (scalar-proba adapter for
  binary calibration; closes #28); Scorer picklability ADR in
  `methodology/parallelism.md` (unblocks v0.36 harness parallelism).
- **v0.36.0** — `evaluate(n_jobs=)` + `evaluate_folded(n_jobs=)` wire
  the unified parallelism pattern into the harness loop (closes #29,
  #30). CI actions bumped to Node 24 ahead of the 2026-06-02
  deprecation.
- **v0.37.0** — `TokenizationLeakageCheck` (HF-tokenizer-aware dedup
  leakage check; closes #35); restored per-module coverage floors
  (closes #37).
- **v0.38.0** — myst-nb migration of `docs/source/examples/`
  (closes #31); executable doc cells via Sybil.
- **v0.39.0** — consumer-feedback batch: `is_metric_defined_for_slice`
  primitive (closes #39), `LeakageCheck.name` relaxed to read-only
  `@property` (closes #40), `parallel_map` worker-copy memory docs
  (closes #41).
- **v0.40.0** — `fit_platt_binary` + `fit_beta_binary` calibrators
  (closes #43).
- **v0.41.0** — `HFDatasetsLoader` Croissant + tree-API hash
  provenance (closes #42 + v1.0 Gate 4 MET).
- **v0.42.0** — `fit_isotonic_binary` completes the 4-element binary
  calibrator family (`temperature` / `isotonic` / `platt` / `beta`
  all return `(params, apply)`; closes #44).
- **v0.43.0** — P1 batch: `ood_dataset_from_manifest` declarative OOD
  loader (closes #48), `character_injection` 6-core-technique
  adversarial suite + Scorer-Protocol matrix (closes #49 core-6;
  advanced-6 deferred), `ActivationDeltaProbe` TaskTracker-style
  linear activation probe (closes #53). New optional extra `[probes]
  = torch + transformers`.
- **v0.44.0** — Defenses + losses: `preprocessing` module with 3
  Spotlighting structural-defense variants (delimit / datamark /
  encode; closes #51), `RecallAtLowFPR` Meta Prompt Guard 2 loss
  recipe (closes #50). New optional extra `[losses] = torch>=2.0`
  (separate from `[probes]` to allow loss-only installs).
- **v0.45.0** — Stacking: `MetaLearner` Protocol + `LogisticStacker`
  reference impl (closes #52). Non-breaking; sklearn already core.
- **v0.46.0** — Scorecard primary metric surface (closes #36):
  `scorecard()` + `Scorecard` (`Mapping[str, MetricResult]`) +
  `metric_specs` namespace + `MetricSpec` Protocol (the 6th Tier-2
  Protocol). Soft-breaking — top-level scalar metric imports
  (`pr_auc`, `roc_auc`, `brier_score`, 5 ECE variants) emit
  `DeprecationWarning` via the `__getattr__` shim. ADR 0002 documents
  the scorecard-as-primary-metric-surface decision.
- **v0.46.1** — Round 6 audit hotfix per Decision R6-E: ECE strategy
  validation (`metric_specs.ece(strategy=...)` raises ValueError on
  invalid values; defence-in-depth at the `_EceSpec.compute()` boundary)
  + deprecation-warning snippet correctness for all 5 ECE variants.
- **v0.47.0** — Sweep unification + advanced-6 + cleanup (BREAKING):
  top-level `sweep()` accepting any `TextTransform` Protocol satisfier;
  `TextTransform` is the 9th strict Tier-2 Protocol (Decision K). 3
  preprocessing dataclasses (`DelimitVariant` / `DatamarkVariant` /
  `EncodeVariant`). 6 new advanced character-injection techniques
  (`BidiRTLInjection`, `TagStrippingInjection`, `SynonymSubstitution`,
  `TokenSplittingInjection`, `UnicodeNormalizationInjection`, `InvisibleCharsInjection`)
  → `ALL_TECHNIQUES` = 12. Removed: the v0.46 `__getattr__`
  deprecation shim (top-level scalars now `AttributeError`), module-
  level `adversarial.sweep` + `preprocessing.sweep`,
  `character_injection` + `spotlighting` `SimpleNamespace` shortcuts,
  `CharacterInjectionStrategy` per-module Protocol. Round 6 follow-on:
  R6-A docstring fix, R6-B duplicate `MetricSpec.name` guard, R6-C
  `to_pandas` schema gains `n_resamples` + `method` columns, R6-D
  Protocol method-shape drift guard, R6-F5 narrow exception catch in
  `_evaluate_spec()`, R6-H `metric_specs.make_spec_name()` helper.
  Migration guide: [`migration/v0.47.md`](migration/v0.47.md).
- **v0.48 → v0.51 → v1.0** — naming-standards sweep (v0.49), SPEC 7
  RNG convention (v0.50), Round 8 + Round 9 multi-LLM audit
  rectification (v0.51), and stability-contract activation (v1.0).
  Per-version migration guides:
  [`v0.49.md`](migration/v0.49.md), [`v0.50.md`](migration/v0.50.md),
  [`v0.51.md`](migration/v0.51.md), [`v1.0.md`](migration/v1.0.md).
  See [`audit_findings.md`](audit_findings.md) for the full Round
  5 → Round 9 ledger and `CHANGELOG.md` for per-release details.

State-of-the-toolkit:

- 10 strict Tier-2 Protocols (`Scorer`, `LeakageCheck`, `Splitter`,
  `ThresholdSelector`, `DatasetLoader`, `MetricSpec`, `MetaLearner`,
  `Probe`, `TextTransform`) + 1 opt-in (`Versioned`). The
  `tests/test_public_api.py` drift guard captures Protocol method
  signatures so changes to any of these trigger SemVer-major review.
- Reference impls: 6 selectors, 7 leakage checks (incl.
  `NormalizedFormLeakageCheck` for encoding-obfuscated dupes), 5
  splitters (incl. `SourceDisjointKFoldSplitter`), 4 loaders.
- `RunManifest` with NeurIPS Reproducibility Checklist alignment +
  Croissant-compatible loader metadata.
- Versioned JSON schemas at `src/eval_toolkit/schemas/`.
- Multi-file methodology curriculum:
  [16 chapters](methodology/README.md) covering leakage, splits,
  thresholds, calibration, comparison, fairness, reproducibility,
  testing, bootstrap, text dedup, versioning, length stratification,
  artifacts, claims, evidence, parallelism.
- Reference-equivalence tests against sklearn / scipy for the wrapped
  primitives (`pr_auc`, `roc_auc`, `brier_score`, `reliability_curve`,
  `bootstrap_ci`, `fit_isotonic_calibrator`, `fit_platt_calibrator`).
- 90 % global coverage gate; per-module breakdown in CI.
- Sybil-validated doc-blocks across `docs/source/methodology/`,
  `docs/source/extending.md`, `docs/source/migration/`, and `README.md`;
  `docs/source/examples/` is executed end-to-end via MyST-NB during the
  Sphinx build (a separate execution surface from Sybil's pytest
  collection).
- Per-version migration guides
  ([`migration/v0.7.md`](migration/v0.7.md),
  [`migration/v0.8.md`](migration/v0.8.md),
  [`migration/v0.9.md`](migration/v0.9.md),
  [`migration/v0.46.md`](migration/v0.46.md),
  [`migration/v0.47.md`](migration/v0.47.md),
  [`migration/v0.49.md`](migration/v0.49.md),
  [`migration/v0.50.md`](migration/v0.50.md),
  [`migration/v0.51.md`](migration/v0.51.md),
  [`migration/v1.0.md`](migration/v1.0.md))
  + general [`MIGRATION.md`](MIGRATION.md).

## Consumer gap docs (input)

If you maintain a downstream consumer of eval-toolkit and have an
upstream wish, the convention is to put a `docs/eval_toolkit_gaps.md`
in your repo, open an issue or PR against `eval-toolkit` linking it,
and we'll reconcile against the tracked-candidates list below.
Historical gap-closure status (Gaps 1–4 from the v0.7 era) is
preserved in CHANGELOG entries for v0.7.x / v0.8.0.

## Tracked candidates (see GitHub Issues)

Issue state is the source of truth; this section is a navigational gloss.
The May 2026 backlog burn closed 16 issues across v0.39–v0.44 (#30, #31,
#35, #37, #38, #39, #40, #41, #42, #43, #44, #48, #49 core-6, #50, #51,
#53). Remaining open:

All v0.45 / v0.46 / v0.46.1 / v0.47 tracked candidates closed:

- #36 (scorecard) — closed by v0.46.0.
- #52 (MetaLearner + LogisticStacker) — closed by v0.45.0.
- Advanced-6 character_injection (v0.43.0 forward-look) — shipped at
  v0.47.0 alongside the sweep consolidation per Decision Q11→11.3.

Post-v1.0 roadmap is v1.0.1 cleanup-only:

- **v1.0.1 (next minor)** — Round 9 deferred-minors batch: RC2
  `SimilarityStrategy` contract reconciliation, RC3 reseed_splitter
  test row-content hardening, RC4 audit-count-tally polish, and 3
  `brier_score` / ECE docstring precision items (F-metrics-1/3/4).
  All Tier-2 ADDITIVE or Tier-3 FREE — no Tier-1 changes.
  See the `v1.0.1 cleanup` GH issue for the live checklist.

The current planning document is
[`~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/.claude/plans/evaluate-all-the-work-twinkly-kite.md)
(local) — covers the staggered v0.45 → v0.46 → v0.47 → v0.48 → v1.0
sequence and the 17 design decisions locked across four `/exploring-options`
rounds.

Run `gh issue list -R brandon-behring/eval-toolkit --label P2` or
`--label P3` for live state.

## v1-prelude evidence core

The next stabilization step is the generic evidence layer now used by
consumer migrations:

- Validation-fit operating points can be applied to mixed-class,
  all-positive, or all-negative target slices with threshold provenance.
- `RunManifest` can carry optional source-role records and guardrails.
- Generic claim gates can fail missing headline comparisons, inadequate
  slice sizes, scorer/leakage errors, missing source roles, and metric
  caps such as hard-negative FPR.

These stay library-first: no prompt-injection datasets, presets, CLI,
or markdown report generator.

## v1.0.0 path (long-term, gated)

v1.0.0 signals API stability — breaking changes after v1.0 require
v2.0. Gated on:

1. **Real consumer running v0.7+ in production for ≥ 1 review cycle.**
   The canonical consumer is `prompt-injection-detection-submission`.
   Other `prompt_injection_*` / `prompt-injection-*` repos in the
   author's workspace are experiments, scaffolds, or earlier
   prototypes — only the detection-submission repo gates v1.0.
2. **Protocol shapes survive ≥ 1 "should we change this?" review
   cycle.** v0.7.x added 5 Tier-2 Protocols (`Scorer`, `LeakageCheck`,
   `Splitter`, `ThresholdSelector`, `DatasetLoader`) + 1 opt-in
   (`Versioned`). As of v0.44.0, all six have been stable across
   37 minor releases (v0.7 → v0.44) except for one contract-tightening
   edit to `LeakageCheck.name` in v0.39.0 (#40) — changing the
   Protocol declaration from a settable class-level attribute to a
   `@property` to align with the `@dataclass(frozen=True)`
   implementation pattern. The stability window now stands at
   **5 of 5 minors without Protocol edits** (v0.40 + v0.41 + v0.42 +
   v0.43 + v0.44 — all additive: new calibrators, Croissant
   enrichment, OOD loader, adversarial suite, structural defenses,
   losses, probes). Gate 2 ✅ **MET** and continues to track.
   The v1-prelude evidence APIs must also survive one real-consumer
   migration check (the `prompt-injection-detection-submission` repo).
3. **Methodology docs reviewed via multi-model cross-review** —
   redefined 2026-05-21 (see [`adr/0003-stability-contract-and-gate3-methodology.md`](adr/)).
   Original intent was external academic peer review, but for a
   single-author / single-consumer library that's a high-variance
   calendar dependency. Replaced by three independent reads:
   (a) manual review by author, (b) Codex independent report,
   (c) Gemini independent report. Different model training corpora
   provide the "outside eyes" value with predictable cycle time.
   Any reviewer-flagged blocker becomes a `p1-gate3`-labelled issue;
   must close before v1.0 tag. Gate 3 ✅ **MET** — closed at v1.0
   via Round 5 → Round 9 multi-LLM cross-review sequence. Round 8
   (verified at v0.51) confirmed 13 of 18 findings and rectified them;
   Round 9 (verified at v0.51 RC `edadddc`) confirmed 6 of 10 source
   items + 3 third-audit findings in modules neither auditor cited,
   with 2 candidate-blocker items fixed in-PR before tag. Full ledger
   at [`audit_findings.md`](audit_findings.md) Rounds 5–9.
4. **Croissant interop verified end-to-end** — ✅ **MET as of v0.41.0**
   (see `tests/test_croissant_e2e.py`). `HFDatasetsLoader.describe()`
   fetches Croissant metadata + per-file `sha256` from HF Hub; the
   integration test downloads a real parquet shard from
   `stanfordnlp/sst2` and verifies the bytes hash bit-exactly to the
   value `describe()` reports. **Caveat**: HF Hub's Croissant emitter
   currently punts `distribution[].sha256` (per the still-open
   MLCommons Croissant spec issue
   [#80](https://github.com/mlcommons/croissant/issues/80)), so
   `HFDatasetsLoader` reads sha256 from HF Hub's tree API (`lfs.oid`)
   today. When #80 resolves and HF Hub starts populating Croissant
   `sha256` with real values, the loader will pick up the new source
   automatically. See `methodology/reproducibility.md` §"Croissant
   interoperability" for the design.

When v1.0 ships:
- API surface freezes. Breaking changes require a v2.0 major bump.
- The five Tier-2 Protocols become contracts (no method-shape changes,
  only additive subprotocols).
- The JSON schemas (`schemas/*.v1.json`) become the canonical contract;
  any breaking change to a schema bumps to `*.v2.json`.

## Out of scope (deliberately)

These are valuable but **not** on the roadmap:

- **Native fairness metrics (demographic parity, equalized odds,
  calibration parity).** Consumer computes via [fairlearn](https://fairlearn.org/)
  + the toolkit's slicing primitives; eval-toolkit shouldn't duplicate.
- **McNemar's test.** Consumer computes via `scipy.stats.contingency`;
  the toolkit's bootstrap framework covers the same ground for
  arbitrary metrics. (DeLong's closed-form ROC-AUC variance is exported
  as {func}`~eval_toolkit.bootstrap.delong_roc_variance` +
  {class}`~eval_toolkit.bootstrap.DeLongResult` — it predates the
  comparison-curriculum write-up; bootstrap remains the documented
  default for general-purpose comparison.)
- **Common metrics (MCC, Cohen's kappa, balanced accuracy, log-loss).**
  Design intent keeps the metric set focused on the four headline
  primitives + ECE family + threshold selection. Consumers add what
  they need.
- **CLI.** The toolkit is a library; consumer projects build their
  own CLI (e.g., the `prompt_injection_*` repos' `evaluate.py` scripts).
- **A formal plugin registry / setuptools entry-points system.** The
  Protocol-based seam is sufficient.
- **Optional `fit_platt_calibrator(canonical: bool = True)` flag.**
  v0.3.0 already canonicalized the impl per Platt 1999 §2.2; the flag
  would re-introduce the non-canonical variant for backward
  compatibility but no consumer demand has surfaced. WONTFIX unless
  asked.

## How to file an upstream wish

1. Add a section in your project's `docs/eval_toolkit_gaps.md`
   describing the gap, severity, and a sketch of the patch (if known).
2. Open an issue or PR against `eval-toolkit`'s GitHub linking that
   gaps doc.
3. If you've done the work locally, the PR can be a draft with the
   suggested patch + tests; we'll reconcile against this roadmap.

## See also

- [`CHANGELOG.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md) — release history.
- [`docs/MIGRATION.md`](MIGRATION.md) — version-to-version migration
  guides.
- [`docs/methodology/reading_list.md`](methodology/reading_list.md) —
  citation-level "future work" pointers (statistical methods that
  could land if there's appetite).
