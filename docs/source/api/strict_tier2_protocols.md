# Strict Tier-2 Protocols at v1.0

This page enumerates the 10 strict Tier-2 Protocols + 1 opt-in Protocol
that make up the v1.0 stability contract per
[ADR 0003 — Stability contract and Gate 3 methodology](../adr/0003-stability-contract-and-gate3-methodology.md)
§1. Method-signature changes on any of these require a SemVer-major
(v2.0) bump.

The {mod}`eval_toolkit.protocols` module intentionally holds only the
**lightweight, low-dependency-surface** Protocols (`Scorer`,
`TextTransform`, `Versioned`, plus three additive helpers). The remaining
Tier-2 Protocols live in their topic modules to avoid pulling in
pandas / sklearn / matplotlib transitively from a "central protocol
module."

The **canonical import path** for every strict Tier-2 Protocol is the
top-level package — `from eval_toolkit import Scorer` (and so on for
each). The submodule paths in the table below show where the source
lives but are an internal detail; users should not depend on them
unless they explicitly need a typing-only import in a constrained
dependency-surface context.

## The 10 strict Tier-2 Protocols (+ 1 opt-in)

| Protocol | Canonical import | Source module | Concrete implementations |
|---|---|---|---|
| {class}`~eval_toolkit.Scorer` | `from eval_toolkit import Scorer` | {mod}`eval_toolkit.protocols` | Any object with `predict_proba(X) -> np.ndarray` |
| {class}`~eval_toolkit.LeakageCheck` | `from eval_toolkit import LeakageCheck` | {mod}`eval_toolkit.leakage` | `ExactDuplicateCheck`, `NearDuplicateCheck`, `NormalizedFormLeakageCheck`, `TokenizationLeakageCheck`, `LabelConflictCheck`, `CrossSplitLeakageCheck`, `GroupLeakageCheck`, `TemporalLeakageCheck` |
| {class}`~eval_toolkit.Splitter` | `from eval_toolkit import Splitter` | {mod}`eval_toolkit.splits` | `HoldoutSplitter`, `StratifiedKFoldSplitter`, `PurgedKFoldSplitter`, `SourceDisjointKFoldSplitter`, `TimeSeriesSplitter` |
| {class}`~eval_toolkit.ThresholdSelector` | `from eval_toolkit import ThresholdSelector` | {mod}`eval_toolkit.thresholds` | `MaxF1Selector`, `CISafeThresholdSelector`, `CostSensitiveSelector`, `TargetFPRSelector`, `TargetPrecisionSelector`, `TargetRecallSelector`, `YoudenJSelector` |
| {class}`~eval_toolkit.DatasetLoader` | `from eval_toolkit import DatasetLoader` | {mod}`eval_toolkit.loaders` | `DataFrameLoader`, `HFDatasetsLoader`, `SingleSliceLoader`, `ParquetGlobLoader`, `OodManifestLoader` |
| {class}`~eval_toolkit.MetricSpec` | `from eval_toolkit import MetricSpec` | {mod}`eval_toolkit.scorecards` | Anything in {mod}`eval_toolkit.metric_specs` (`pr_auc`, `roc_auc`, `brier`, `ece`); user-defined factory specs |
| {class}`~eval_toolkit.TextTransform` | `from eval_toolkit import TextTransform` | {mod}`eval_toolkit.protocols` | All 12 adversarial dataclasses (`ZeroWidthSpaceInjection`, `HomoglyphSubstitution`, `DiacriticInjection`, `WhitespaceInjection`, `CaseInjection`, `PunctuationInjection`, `BidiRTLInjection`, `TagStrippingInjection`, `SynonymSubstitution`, `TokenSplittingInjection`, `UnicodeNormalizationInjection`, `InvisibleCharsInjection`) + 3 preprocessing variants (`DelimitVariant`, `DatamarkVariant`, `EncodeVariant`) |
| {class}`~eval_toolkit.MetaLearner` | `from eval_toolkit import MetaLearner` | {mod}`eval_toolkit.stacking` | `LogisticStacker` |
| {class}`~eval_toolkit.Probe` | `from eval_toolkit import Probe` | {mod}`eval_toolkit.probes` | `ActivationDeltaProbe` |
| {class}`~eval_toolkit.SimilarityStrategy` | `from eval_toolkit import SimilarityStrategy` | {mod}`eval_toolkit.text_dedup` | `ExactNormalizedHashStrategy`, `EmbeddingCosineStrategy`, `JaccardNgramStrategy`, `MinHashLSHStrategy`, `TfidfCosineStrategy` |

**Opt-in Protocol** (additive on top of Tier-2):

| Protocol | Canonical import | Source module | Notes |
|---|---|---|---|
| {class}`~eval_toolkit.Versioned` | `from eval_toolkit import Versioned` | {mod}`eval_toolkit.protocols` | Any object exposing `version: str`. `RunManifest.versioned_objects` auto-collects implementations. Opt-in — no Tier-2 implementation is required to satisfy it. |

## Why no central re-export module?

The {mod}`eval_toolkit.protocols` module intentionally stays
lightweight — it imports nothing heavy (no pandas, sklearn,
matplotlib, or filesystem-oriented helpers), so consumers can type
adapters in a constrained dependency-surface context. If
{mod}`eval_toolkit.protocols` re-exported all 10 strict Tier-2
Protocols, importing it would transitively pull in every heavy
implementation module. The current design preserves the lightweight
intent.

For one-stop discovery, use this page or the table in ADR 0003 §1.
For type-only imports in your own code, the canonical
`from eval_toolkit import <Protocol>` form is always available and
stable through v1.x.

## See also

- [ADR 0003 — Stability contract and Gate 3 methodology](../adr/0003-stability-contract-and-gate3-methodology.md) — defines the Tier 1/2/3 framework these Protocols live in.
- {mod}`eval_toolkit.protocols` — the lightweight-Protocol module (`Scorer`, `TextTransform`, `Versioned`, `EvalSliceLike`, `PredictionReader`, `SliceAwareScorer`).
- [Migration guide](../migration/) — every breaking change to these Protocols would appear here as a SemVer-major bump.
