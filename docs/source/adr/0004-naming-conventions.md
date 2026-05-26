# ADR 0004: Naming conventions

**Status:** Accepted at v0.49.0 (the final pre-v1.0 minor) — these
conventions are the v1.0 contract for naming.

**Date:** 2026-05-23

**Authors:** brandon-behring

**Context:** Codified out of an audit + industry-research pass run during
`~/.claude/plans/what-git-issues-are-bright-torvalds.md` planning.

## Context

eval-toolkit at v0.48.0 had 95–99% consistent naming across the public
surface — a real success, not an accident — but the conventions were
implicit rather than documented. v1.0 locks the Tier-1 API per
[ADR 0003](0003-stability-contract-and-gate3-methodology.md); any
inconsistency that ships at v1.0 lives forever (until v2.0). This ADR
both documents what's already true AND closes the small remaining gaps
flagged by the v0.49.0 audit.

The audit also surfaced a v1.0-critical RNG-parameter inconsistency
(`seed` in ~15 functions, `random_state` in `stacking.py`) that requires
adoption of [Scientific Python SPEC 7](https://scientific-python.org/specs/spec-0007/).
That adoption ships in v0.50.0 as a focused release; this ADR documents
the convention now so it locks the rule going forward.

## Decisions

### D1 — Module naming

Modules in `src/eval_toolkit/*.py` are flat per
[ADR 0001](0001-flat-module-layout.md). Within that constraint:

- **Plural noun for collection-of-types modules** — modules holding
  multiple related types of the same kind: `metrics`, `loaders`,
  `protocols`, `losses`, `probes`, `splits`, `paths`, `seeds`,
  `thresholds`, `artifacts`, `claims`, `embeddings`, `scorecards`
  (new at v0.49.0; was `_scorecard`).
- **Singular noun for domain-concept modules** — modules holding a
  single coherent concept: `harness`, `bootstrap`, `manifest`,
  `calibration`, `leakage`, `analysis`, `provenance`, `evidence`,
  `stacking`, `text_dedup`.
- **Gerund for process-domain modules** — modules describing an
  operation: `preprocessing`.
- **Private modules** carry a leading underscore: `_parallel.py`,
  `_deprecated.py`, `_version.py`, `_rng.py`, `_sweep.py`. These are
  not in `_EXPORTS`; their public symbols (if any) are accessed only
  via top-level `from eval_toolkit import X`.

The asymmetric-promotion sub-rule from
[ADR 0001](0001-flat-module-layout.md) controls when a private module
should be promoted to public: collection-of-types MAY promote;
single-function-only SHOULD stay underscore.

### D2 — Class naming

Classes follow `PascalCase` per PEP 8. Within that, eval-toolkit uses
**domain-specific suffixes** that map to Protocol contracts:

| Suffix | Domain | Examples | Protocol |
|---|---|---|---|
| `*Selector` | Threshold selection | `MaxF1Selector`, `CISafeThresholdSelector`, `YoudenJSelector` | `ThresholdSelector` |
| `*Splitter` | Cross-validation splits | `HoldoutSplitter`, `StratifiedKFoldSplitter`, `PurgedKFoldSplitter` | `Splitter` |
| `*Check` | Leakage detection | `ExactDuplicateCheck`, `TokenizationLeakageCheck`, `LabelConflictCheck` | `LeakageCheck` |
| `*Loader` | Dataset loading | `DataFrameLoader`, `HFDatasetsLoader`, `ParquetGlobLoader` | `DatasetLoader` |
| `*Reader` | Prediction artifact reading | `CsvPredictionReader`, `JsonlPredictionReader` | `PredictionReader` |
| `*Variant` | Preprocessing variant | `DelimitVariant`, `DatamarkVariant`, `EncodeVariant` | (functional API) |
| `*Strategy` | Dedup similarity backend | `TfidfCosineStrategy`, `EmbeddingCosineStrategy`, `MinHashLSHStrategy` | `SimilarityStrategy` |
| `*Injection` | Adversarial char-injection attack | `ZeroWidthSpaceInjection`, `BidiRTLInjection`, `TokenSplittingInjection`, `UnicodeNormalizationInjection`, `CaseInjection` (v0.49 renames) | `TextTransform` |
| `*Substitution` | Adversarial char-substitution attack | `HomoglyphSubstitution`, `SynonymSubstitution`, `DiacriticInjection` *(historical: `DiacriticInjection` is a substitution by mechanism; the name preserves continuity with the v0.43 release)* | `TextTransform` |

**Result/output dataclasses** use the suffix `*Result`, `*CI`,
`*Estimate`, or `*Report` where the type genuinely is a result object;
exception cases (`Scorecard`, `WilsonInterval`) are intentional —
`Scorecard` is the named domain concept, not a generic result, and
`WilsonInterval` is the math-term name. Document exceptions in the
class docstring.

**Config/metadata** dataclasses use `*Spec`, `*Metadata`, or
`*Manifest` per the domain.

### D3 — Function naming

Functions use `snake_case` per PEP 8. Prefer **verb-prefix for
factories** when the name is action-shaped:

| Prefix | Use | Examples |
|---|---|---|
| `make_*` | Construct an object | `make_minilm_embedder`, `make_palette`, `make_run_dir`, `make_manifest` (v0.49 rename) |
| `fit_*` | Fit a model/calibrator | `fit_temperature`, `fit_platt_binary`, `fit_beta_binary`, `fit_isotonic_binary`, `fit_operating_points` |
| `evaluate_*` | Run an evaluation harness | `evaluate`, `evaluate_folded`, `evaluate_claims` |
| `plot_*` | Render a figure | `plot_pr_curve`, `plot_roc_curve`, `plot_confusion_matrix_grid` |
| `write_*` | Serialize to disk | `write_manifest`, `write_run_result`, `write_json_strict` |
| `validate_*` | Check well-formedness | `validate_manifest`, `validate_payload`, `validate_source_roles` |
| `bootstrap_*` | Resample-based CI | `bootstrap_ci`, `bootstrap_metric_from_predictions` |
| `paired_*` | Paired-bootstrap difference | `paired_bootstrap_diff`, `paired_bootstrap_ece_diff`, `paired_mde` |
| `load_*` | I/O read | `load_prediction_arrays` |
| `compute_*` | Pure derivation | `compute_label_overlap`, `compute_file_hash` |

**Noun-form is OK** for scalar metric functions per
[ADR 0002](0002-scorecard-as-primary-metric-surface.md): `pr_auc`,
`roc_auc`, `brier_score`, `recall_at_fpr`, `wilson_interval`. These
are sklearn-aligned and would be awkward as `compute_pr_auc(...)`.

### D4 — Parameter naming (the canonical list)

Locked at v1.0 — these names mean these things, everywhere:

| Parameter | Meaning |
|---|---|
| `y_true` | Ground-truth labels (binary, shape `(n,)`) |
| `y_score` | Continuous score / probability (shape `(n,)`) |
| `y_pred` | Discrete prediction (when threshold-dependent) |
| `n_resamples` | Bootstrap iteration count |
| `confidence` | Two-sided confidence level (0.95 default) |
| `n_bins` | Binning count for calibration / ECE |
| `n_jobs` | Parallelism (joblib convention; sklearn-aligned) |
| `ax` | Matplotlib axis (matplotlib convention) |
| `metric` | Callable `(y_true, y_score) -> float` |
| `rng` | RNG argument per SPEC 7 — **canonical** convention (adopted v0.50.0). Accepts `int \| np.random.Generator \| BitGenerator \| SeedSequence \| None`. |
| `seed` | _legacy name_ used through v0.49 — replaced by `rng` at v0.50.0 across ~22 Tier-1 sites. EXCEPTIONS where `seed` is retained: `seeds.set_global_seeds(seed: int)` (global-state setter; SPEC 7 doesn't apply), adversarial dataclass fields (use Python `random.Random(seed)`; not NumPy-RNG), Splitter dataclass class-fields (configuration storage, not user-facing RNG parameter), `loaders.py` YAML config key. |

**Future functions MUST use these names.** A PR that introduces
`labels=` (instead of `y_true=`), `scores=` (instead of `y_score=`),
`alpha=` (instead of `confidence=`), or any deviation must justify it
in the PR description or rename to the canonical name.

### D5 — Constants

`UPPER_SNAKE_CASE` per PEP 8. Tier-1 constants in `_EXPORTS` include
`DEFAULT_SEED`, `DEFAULT_N_RESAMPLES`, `DEFAULT_CONFIDENCE`,
`DEFAULT_METHOD`, `MANIFEST_SCHEMA_VERSION`, `CORE_TECHNIQUES`,
`ADVANCED_TECHNIQUES`, `ALL_TECHNIQUES`.

### D6 — Protocol naming

Protocols follow `PascalCase` per PEP 8 and are **named semantically**
per [ADR 0003](0003-stability-contract-and-gate3-methodology.md) §1.
There is no forced uniform suffix — the *contract* is the method
shape, not the name shape. The 10 strict Tier-2 Protocols + 1 opt-in
named per their role: `Scorer`, `LeakageCheck`, `Splitter`,
`ThresholdSelector`, `DatasetLoader`, `MetricSpec`, `TextTransform`,
`MetaLearner`, `Probe`, `SimilarityStrategy`, `Versioned`. (Note:
`SimilarityStrategy` promoted from "pre-v0.7 internal" to strict
Tier-2 at v1.0.2 per RC2 reconciliation; #76.)

### D7 — TypeVars

Internal `TypeVar`s use a leading underscore per Google Python Style
Guide §3.19.10: `_T = TypeVar("_T")`. Public, constrained `TypeVar`s
without underscore are allowed when explicitly part of an exported API.

### D8 — Fitted estimator attributes (sklearn alignment)

Estimator-style classes (`fit`/`predict` pattern) that store
**learned-from-data attributes** use **trailing underscore** per
scikit-learn convention: `coef_`, `classes_`, `n_features_in_`,
`feature_importances_`. These attributes MUST NOT be set in
`__init__` — set them only in `fit()`. (See
[scikit-learn Developing estimators](https://scikit-learn.org/stable/developers/develop.html#fitted-attributes).)

Frozen reference-impl dataclasses (`@dataclass(frozen=True, slots=True)`)
are exempt — they hold config, not fitted state.

`stacking.LogisticStacker` is the current canonical example.

### D9 — Docstring style

NumPy docstring format per STYLE.md §12. **Prose wraps at 75 cols**
(numpydoc convention); doctest code blocks follow the 100-col Black
rule. The 75-col rule keeps docstrings readable in a terminal `help()`
call; the 100-col rule keeps code in docstrings readable in a normal
editor.

### D10 — Test naming

`tests/test_<module>.py` mirrors `src/eval_toolkit/<module>.py`. Test
functions are `test_<thing_under_test>_<scenario>`. No class-based
test grouping unless fixtures truly demand it (rare in this codebase).

## Industry alignment

The conventions above were verified against canonical sources during
the v0.49.0 audit:

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/) —
  module/class/function/variable/constant naming, leading-underscore
  privacy.
- [scikit-learn Developing estimators](https://scikit-learn.org/stable/developers/develop.html) —
  fit/predict contract, trailing-underscore for fitted attributes,
  parameter names `y_true` / `y_pred` / `y_score` / `n_jobs`.
- [numpydoc Format Spec](https://numpydoc.readthedocs.io/en/latest/format.html) —
  docstring sections + 75-col prose rule.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) —
  TypeVar leading-underscore (§3.19.10) for internal types.
- [Scientific Python SPEC 7 — Seeding pseudo-random number generation](https://scientific-python.org/specs/spec-0007/) —
  `rng: RNGLike | SeedLike | None` parameter convention (adopted in
  v0.50.0; documented here so the rule locks now).

Eval-toolkit deviates from industry conventions in three places, all
intentional and documented:

1. **Unicode math identifiers** (`π`, `θ`, `μ`, `σ`, `α`, `β`) are
   permitted in math kernels with required English-comment alias —
   per STYLE.md §3 and §16. PEP 8 forbids non-ASCII; eval-toolkit's
   math-paper-fidelity domain justifies the exception.
2. **`set_global_seeds(seed: int)`** keeps `seed` even after the
   v0.50.0 SPEC 7 adoption — this is a global-state setter, not a
   per-function RNG argument, so SPEC 7 doesn't apply.
3. **Adversarial dataclass fields** keep `seed: int = 42` because
   they use Python's stdlib `random.Random(seed)`, not NumPy's
   `Generator`. SPEC 7's typing (`RNGLike = np.random.Generator | ...`)
   is strictly NumPy-scoped.

## Forward enforcement

Every PR that adds a new public symbol MUST satisfy these conventions
or document the exception in the PR description. There is no automated
lint enforcement yet — that is a v1.x candidate (deferred N8 in the
v0.49.0 plan; tracked as a future-improvement issue once the patterns
have settled across a couple of v1.x minors).

## References

- [ADR 0001 — Flat module layout](0001-flat-module-layout.md) (cross-references the asymmetric-promotion principle).
- [ADR 0002 — Scorecard as primary metric surface](0002-scorecard-as-primary-metric-surface.md).
- [ADR 0003 — Stability contract + Gate 3 methodology](0003-stability-contract-and-gate3-methodology.md) (defines the Tier-1/2/3 framework these naming rules operate inside).
- `STYLE.md` (root) — contributor-facing daily reference; this ADR is the decision record.
- `~/.claude/plans/what-git-issues-are-bright-torvalds.md` — the v0.49.0 plan that produced this ADR.
