# Examples

Minimal, focused worked examples — one concept per file. Each is
runnable end-to-end under Sybil (every code block executes in CI).

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

## How these run

Sybil parses Python code blocks from each Markdown file and executes
them in CI (`make test-doctest`). Failures break the build —
documentation that doesn't run goes stale fast, so we don't ship it.

`<!-- skip: next -->` comments mark blocks that should NOT execute
(e.g., illustrative pseudocode, expensive examples, optional-dep
demos). Always test the runnable blocks against the latest installed
version.
