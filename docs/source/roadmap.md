# Roadmap

Forward-looking tracker for `eval-toolkit`. Cross-links the consumer
gap docs that motivate upstream work and records the criteria for
v1.0.0.

This document is **descriptive of intent, not a commitment**. The
priorities reflect today's understanding of what consumers need;
order may change as feedback comes in.

## Currently shipped (as of v0.36.0)

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

State-of-the-toolkit:

- 5 Tier-2 Protocols (`Scorer`, `LeakageCheck`, `Splitter`,
  `ThresholdSelector`, `DatasetLoader`) + 1 opt-in (`Versioned`).
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
  `docs/source/extending.md`, `docs/source/migration/`,
  `docs/source/examples/`, `README.md`.
- Per-version migration guides
  ([`migration/v0.7.md`](migration/v0.7.md),
  [`migration/v0.8.md`](migration/v0.8.md),
  [`migration/v0.9.md`](migration/v0.9.md))
  + general [`MIGRATION.md`](MIGRATION.md).

## Consumer gap docs (input)

If you maintain a downstream consumer of eval-toolkit and have an
upstream wish, the convention is to put a `docs/eval_toolkit_gaps.md`
in your repo, open an issue or PR against `eval-toolkit` linking it,
and we'll reconcile against the tracked-candidates list below.
Historical gap-closure status (Gaps 1–4 from the v0.7 era) is
preserved in CHANGELOG entries for v0.7.x / v0.8.0.

## Tracked candidates (see GitHub Issues)

The previously-untracked "v0.9 candidates" list has been filed as
GitHub Issues. Issue state is the source of truth; this section is a
navigational gloss.

- [#35](https://github.com/brandon-behring/eval-toolkit/issues/35) (P2) — `TokenizationLeakageCheck` (HF-tokenizer-aware dedup; complements `NormalizedFormLeakageCheck`).
- [#31](https://github.com/brandon-behring/eval-toolkit/issues/31) (P3) — Migrate `docs/source/examples/` from static MD to executable myst-nb cells.
- [#36](https://github.com/brandon-behring/eval-toolkit/issues/36) (P3) — Inline bootstrap CI on every metric (Inspect-AI / lm-eval scorecard pattern).
- [#37](https://github.com/brandon-behring/eval-toolkit/issues/37) (P3) — Restore per-module coverage floors (`seeds.py` 70 % due to optional torch path).
- [#38](https://github.com/brandon-behring/eval-toolkit/issues/38) (P3) — CI doctests for `paths.py` / `provenance.py` / `seeds.py` / `docs.py`.

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
   (`Versioned`). As of v0.41.0, all six have been stable across
   34 minor releases (v0.7 → v0.41) except for one contract-tightening
   edit to `LeakageCheck.name` in v0.39.0 (#40) — changing the
   Protocol declaration from a settable class-level attribute to a
   `@property` to align with the `@dataclass(frozen=True)`
   implementation pattern. v0.40.0 and v0.41.0 both shipped without
   Protocol shape edits (v0.40: `fit_platt_binary` + `fit_beta_binary`
   additions; v0.41: `HFDatasetsLoader` enrichment — neither touched
   Tier-2 Protocols). The stability window is now **2 of 2 minors
   without Protocol edits** as of v0.41.0 — Gate 2 ✅ **MET**.
   The v1-prelude evidence APIs must also survive one real-consumer
   migration check (the `prompt-injection-detection-submission` repo).
3. **Methodology docs peer-reviewed** by an external reader (statistics
   / methodology background, ideally not part of the
   `prompt_injection_*` core team).
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
- **McNemar / DeLong tests.** Consumer computes via `scipy.stats`;
  the toolkit's bootstrap framework covers the same ground for
  arbitrary metrics.
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
