# Roadmap

Forward-looking tracker for `eval-toolkit`. Cross-links the consumer
gap docs that motivate upstream work and records the criteria for
v1.0.0.

This document is **descriptive of intent, not a commitment**. The
priorities reflect today's understanding of what consumers need;
order may change as feedback comes in.

## Currently shipped (as of v0.8.0)

See [`CHANGELOG.md`](../CHANGELOG.md) for the full release history.
The state-of-the-toolkit summary:

- 5 Tier-2 Protocols (`Scorer`, `LeakageCheck`, `Splitter`,
  `ThresholdSelector`, `DatasetLoader`) + 1 opt-in (`Versioned`).
- Reference impls: 6 selectors, 7 leakage checks (incl.
  `NormalizedFormLeakageCheck` for encoding-obfuscated dupes), 5
  splitters (incl. `SourceDisjointKFoldSplitter`), 4 loaders.
- `RunManifest` with NeurIPS Reproducibility Checklist alignment +
  Croissant-compatible loader metadata.
- Versioned JSON schemas at `src/eval_toolkit/schemas/`.
- Multi-file methodology curriculum:
  [12 chapters](methodology/README.md) covering leakage, splits,
  thresholds, calibration, comparison, fairness, reproducibility,
  testing, bootstrap, text dedup, versioning, length stratification.
- Reference-equivalence tests against sklearn / scipy for the wrapped
  primitives (`pr_auc`, `roc_auc`, `brier_score`, `reliability_curve`,
  `bootstrap_ci`, `fit_isotonic_calibrator`, `fit_platt_calibrator`).
- 90 % global coverage gate; per-module breakdown in CI.
- Sybil-validated doc-blocks across `docs/methodology/`,
  `docs/extending.md`, `docs/migration/`, `docs/examples/`,
  `README.md`.
- Two release-vehicle migration guides
  ([`docs/MIGRATION.md`](MIGRATION.md)).

## Consumer gap docs (input)

External projects track upstream wishes in their own gap docs:

- **`prompt-injection-clean/docs/eval_toolkit_gaps.md`** — confirms
  the v0.7.0/0.7.1 release closed Gap 1 (`TargetPrecisionSelector`)
  and Gap 4 (`Versioned` adoption pattern); records Gap 2
  (`length_stratified_report` wrapper, **closed in v0.8.0** as
  `quantile_stratified_report`); flags Gap 3 (cost-matrix preset,
  YAGNI / probably WONTFIX).

If you maintain a downstream consumer of eval-toolkit and have an
upstream wish, the convention is to put a `docs/eval_toolkit_gaps.md`
in your repo and link it from this roadmap on PR.

## v0.9 candidates (next minor)

Items deferred from v0.8.0 or surfaced post-v0.8 release. None are
release-blockers; ship as feedback dictates.

- **Optional `fit_platt_calibrator(canonical: bool = True)` flag.** v0.8
  ships docstring-only acknowledgment of the (now non-divergent;
  v0.3.0 already canonicalized) Platt impl. The flag is now low-value;
  may be dropped entirely if no consumer demand surfaces.
- **Bootstrap CI inline on every metric.** Inspect-AI / lm-eval pattern.
  Useful for scorecard-oriented harnesses; not the toolkit's primary
  use case but worth surfacing if a consumer needs it.
- **Tokenizer-aware leakage check.** A
  `TokenizationLeakageCheck` that dedupes on a HuggingFace tokenizer's
  output rather than raw text. Requires the optional `transformers`
  install. Consumer-side stub today (see
  [`methodology/leakage.md` §"PyTorch & transformer-specific"](methodology/leakage.md#pytorch-pitfalls)).
- **Per-module coverage floors.** v0.8 restored the global 90 % gate
  but per-module floors (especially `seeds.py` at 70 % due to
  optional torch path) are uneven. v0.9 candidate: pragma the
  unreachable-without-torch lines OR add torch as a CI dep.
- **`paths.py` / `provenance.py` / `seeds.py` / `docs.py` doctests in
  CI.** Currently CI doctests only the math kernels.

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

1. **All four `prompt_injection_*` consumers fully migrated** to v0.7+
   and **running in production for ≥ 1 review cycle.** As of
   v0.8.0: 3 of 4 (`prompt_injection_detector`,
   `prompt_injection_classifier_showcase`, `prompt-injection-sdd`)
   are migrated and committed; `prompt-injection-clean` was
   scaffolded against v0.7.1 from the start.
2. **Protocol shapes survive ≥ 1 "should we change this?" review
   cycle.** v0.7.x added 5 new Protocols; v0.8.0 didn't change any.
   v0.9 might (e.g., the Versioned-canonical-impl shape if v0.9 ships
   the canonical Platt flag). v1.0 means we're *confident* the shapes
   are durable.
   The v1-prelude evidence APIs must also survive one V3-shaped and one
   SDD-shaped consumer migration check.
3. **Methodology docs peer-reviewed** by an external reader (statistics
   / methodology background, ideally not part of the
   `prompt_injection_*` core team).
4. **Croissant interop verified end-to-end** — a real Croissant-
   compliant dataset loaded via `HFDatasetsLoader`, scored, and the
   manifest's `data_hashes` matched against the Croissant
   `distribution.sha256` field.

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

## How to file an upstream wish

1. Add a section in your project's `docs/eval_toolkit_gaps.md`
   describing the gap, severity, and a sketch of the patch (if known).
2. Open an issue or PR against `eval-toolkit`'s GitHub linking that
   gaps doc.
3. If you've done the work locally, the PR can be a draft with the
   suggested patch + tests; we'll reconcile against this roadmap.

## See also

- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
- [`docs/MIGRATION.md`](MIGRATION.md) — version-to-version migration
  guides.
- [`docs/methodology/reading_list.md`](methodology/reading_list.md) —
  citation-level "future work" pointers (statistical methods that
  could land if there's appetite).
