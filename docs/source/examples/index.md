# Examples

Minimal, focused worked examples — one concept per file. Each is
runnable end-to-end under myst-nb (cells execute during
`sphinx-build`; outputs render inline in the rendered HTML).

## By capability

| Example | Demonstrates | Minimum extras |
|---|---|---|
| [Metrics + bootstrap](metrics_and_bootstrap.md) | `pr_auc`, `roc_auc`, `brier_score`, `bootstrap_ci` (BCa / percentile) | none |
| [Evaluate harness](evaluate_harness.md) | `evaluate` orchestrator, `write_run_result`, schema validation | `[dataframe]` |
| [Calibration](calibration.md) | Platt + isotonic recalibration, ECE before/after | none |
| [Leakage detection](leakage_detection.md) | Exact dupe, normalized-form, label-conflict checks | `[dataframe]` |
| [Claims + gates](claims_and_gates.md) | `EvidenceGate` composition for release decisions | `[dataframe]` |
| [Paired comparison](paired_comparison.md) | `paired_bootstrap_diff`, MDE for two-scorer comparisons | none |
| [Prompt-injection walkthrough](prompt_injection_walkthrough.md) | Full pipeline on synthetic OWASP fixtures | `[dataframe]` |
| [PyTorch scorer](pytorch_scorer_example.md) | Wrapping a PyTorch model as a `Scorer` (skip-execed in CI) | `[dataframe]`, `torch` |
| [Nested seed-split](nested_seed_split.md) | LODO k-fold × multi-seed × stratified train/val composition | none |
| [Callable embedder for dedup](callable_embedder_dedup.md) | `EmbeddingCosineStrategy` with `make_minilm_embedder` / custom embedders | `[embeddings]` (optional) |
| [Cross-corpus contamination scan](cross_corpus_contamination_scan.md) | `pairs_across` for benign-vs-injection contamination flagging | none |
| [`plot_roc_curve` walkthrough](plot_roc_curve_walkthrough.md) | ROC rendering with threshold marker + baseline overlay | `[plotting]` |
| [`plot_pareto_frontier` walkthrough](plot_pareto_frontier_walkthrough.md) | Cost-vs-performance scatter with frontier overlay | `[plotting]` |
| [`plot_slice_metric_heatmap` walkthrough](plot_slice_metric_heatmap_walkthrough.md) | `(row × col → metric)` grid with colorbar + annotations | `[plotting]` |
| [OOD manifest loader](ood_dataset_from_manifest.md) | `ood_dataset_from_manifest` — declarative loader for multiple OOD slates with sha256 caching | `[dataframe]`, `[yaml]`, `[parquet]` |
| [Character-injection sweep](character_injection_sweep.md) | `eval_toolkit.adversarial` — six character-level techniques + Scorer-Protocol sweep for adversarial robustness | `[dataframe]` |
| [ActivationDeltaProbe](activation_delta_probe.md) | `eval_toolkit.probes.ActivationDeltaProbe` — TaskTracker-style linear probe on transformer activation deltas | `[probes]` for real backbones; mocked illustration here |
| [Spotlighting variants](spotlighting.md) | `eval_toolkit.preprocessing` — delimit / datamark / encode structural defenses + batch sweep | none |
| [RecallAtLowFPR loss](recall_at_low_fpr.md) | `eval_toolkit.losses.RecallAtLowFPR` — Meta Prompt Guard 2 training recipe (differentiable recall-at-fixed-FPR) | `[losses]`; static render in docs CI |
| [LogisticStacker](stacking.md) | `eval_toolkit.stacking.LogisticStacker` — combine multiple detector outputs into a calibrated meta-classifier via the `MetaLearner` Protocol | none |

## How these run

Since v0.38.0, examples are myst-nb notebooks (Markdown source with
`{code-cell}` directives). Cells execute during `sphinx-build` with
`nb_execution_mode = "cache"` — re-execution is triggered only when
the source page changes. Cell outputs (printed text, tables, figures)
render inline in the published HTML, so the docs site reflects the
actual library behavior.

Two pages have execution disabled at page level because they require
optional dependencies that aren't in `[dev]`:

- `pytorch_scorer_example.md` needs `torch` (~700MB transitive)
- `callable_embedder_dedup.md` needs `[embeddings]` (sentence-transformers)

These pages render their code statically.

```{toctree}
:hidden:
:maxdepth: 1

metrics_and_bootstrap
evaluate_harness
calibration
leakage_detection
claims_and_gates
paired_comparison
prompt_injection_walkthrough
pytorch_scorer_example
nested_seed_split
callable_embedder_dedup
cross_corpus_contamination_scan
plot_roc_curve_walkthrough
plot_pareto_frontier_walkthrough
plot_slice_metric_heatmap_walkthrough
ood_dataset_from_manifest
character_injection_sweep
activation_delta_probe
spotlighting
recall_at_low_fpr
stacking
```
