# eval-toolkit

A **methodology-aware evaluation harness for binary classification**:
metrics, bootstrap CIs, calibration, leakage detection, splitting,
threshold selection, dataset loading, reproducibility manifests, and a
slice-aware orchestrator that ties them together. Pure
numpy/scipy/sklearn core; pandas/matplotlib/hypothesis are optional
extras; PyTorch / HuggingFace / `datasets` are *consumer-side* (never
required).

Library-grade by design — every public function is type-annotated,
every math kernel is documented with LaTeX + literature references,
statistical validity (bootstrap CIs, MDE estimates, paired-difference
tests) is built in, and the JSON outputs (`results.json` /
`results_full.json` / `manifest.json`) ship with versioned [JSON
Schemas](src/eval_toolkit/schemas/) so downstream parsers can gate on
format changes.

## Three-tier architecture

```
┌─ Tier 3 ─ Reproducibility scaffolding ─────────────────┐
│  manifest.json + seeds + git_sha + data_hashes +       │
│  gpu_info + leakage_report (NeurIPS-aligned)           │
├─ Tier 2 ─ Protocol-based orchestration ────────────────┤
│  Scorer / SliceAwareScorer / LeakageCheck / Splitter   │
│  ThresholdSelector / DatasetLoader / SimilarityStrategy│
│  Versioned (opt-in: per-object versions in manifest)   │
├─ Tier 1 ─ Functional core ─────────────────────────────┤
│  pr_auc / roc_auc / ECE variants / Brier / bootstrap_ci│
│  paired_bootstrap_diff / cv_clt_ci / mde_from_ci       │
│  reliability_curve / fit_temperature / fit_isotonic    │
└────────────────────────────────────────────────────────┘
```

Pick the tier your task needs. Ad-hoc analysis: just call the
functional core. Full eval pipelines: implement the Protocols. Every
run: capture the manifest.

## Methodology

What good binary-classification evaluation looks like, with each
concern mapped to the toolkit primitive that operationalizes it.

- [`docs/methodology/`](docs/methodology/README.md) — the curriculum.
  Read [`leakage`](docs/methodology/leakage.md) →
  [`splits`](docs/methodology/splits.md) →
  [`thresholds`](docs/methodology/thresholds.md) →
  [`calibration`](docs/methodology/calibration.md) →
  [`comparison`](docs/methodology/comparison.md) →
  [`fairness`](docs/methodology/fairness.md) →
  [`reproducibility`](docs/methodology/reproducibility.md) →
  [`testing`](docs/methodology/testing.md) →
  [`reading_list`](docs/methodology/reading_list.md).

## Extending eval-toolkit

How to plug your own scorers / leakage checks / splitters / loaders /
threshold selectors into the harness.

- [`docs/extending.md`](docs/extending.md) — Protocol-by-Protocol
  guide, ~50-line full-harness recipe, project-layout pointer.

## Worked examples

- [`docs/examples/prompt_injection_walkthrough.md`](docs/examples/prompt_injection_walkthrough.md)
  — End-to-end prompt-injection eval on a synthetic OWASP LLM01:2025
  fixture; cross-links to the
  [showcase repo](https://github.com/brandon-behring/prompt_injection_classifier_showcase)
  for the real Lakera PINT walkthrough.
- [`docs/examples/pytorch_scorer_example.md`](docs/examples/pytorch_scorer_example.md)
  — HuggingFace transformer + LoRA `Scorer` adapter (batched inference,
  GPU/CPU placement, deterministic-mode setup).

## Install

```bash
uv venv
uv pip install -e .[dev]
```

For consumers who only need the math kernels (no plotting, no pandas):

```bash
pip install eval-toolkit                        # core only: numpy/scipy/sklearn
pip install "eval-toolkit[plotting]"            # adds matplotlib + pillow
pip install "eval-toolkit[dataframe]"           # adds pandas
pip install "eval-toolkit[all]"                 # everything
```

## Quick examples

### Metrics

```python
import numpy as np
from eval_toolkit import pr_auc, roc_auc, expected_calibration_error

rng = np.random.default_rng(42)
y = rng.integers(0, 2, size=200)
# Clip to [0, 1] — ECE only meaningful on calibrated probabilities.
s = np.clip(y + rng.normal(0, 0.3, size=200), 0, 1)

print(f"PR-AUC: {pr_auc(y, s):.3f}")
print(f"ROC-AUC: {roc_auc(y, s):.3f}")
print(f"ECE (10 bins): {expected_calibration_error(y, s, n_bins=10):.3f}")
```

### Bootstrap confidence intervals

```python
from eval_toolkit import bootstrap_ci, paired_bootstrap_diff, pr_auc

ci = bootstrap_ci(y, s, pr_auc, n_resamples=1000, seed=42)
print(f"PR-AUC: {ci.point_estimate:.3f}  95% CI: [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")

# Paired bootstrap on the lift between two scorers (s_baseline must be in [0, 1] too).
s_baseline = np.clip(rng.normal(0.5, 0.3, size=200), 0, 1)
diff = paired_bootstrap_diff(y, s_baseline, s, pr_auc, n_resamples=1000, seed=42)
print(f"Δ PR-AUC: {diff.delta:.3f}  overlaps zero: {diff.overlaps_zero}")
```

### Temperature scaling (Guo et al. 2017)

```python
from eval_toolkit import fit_temperature

logits = rng.normal(size=(500, 2))
labels = (logits[:, 1] > logits[:, 0]).astype(int)
result = fit_temperature(logits, labels)
print(f"Optimal T: {result['temperature']:.3f}")
print(f"NLL: {result['nll_pre']:.3f} -> {result['nll_post']:.3f}")
```

## Modules

| Module | Purpose |
|---|---|
| `eval_toolkit.metrics` | PR-AUC, ROC-AUC, ECE variants, Brier decomposition, prior-shift projection |
| `eval_toolkit.thresholds` | `ThresholdSelector` Protocol + 6 reference impls (max-F1, target-recall/precision/FPR, Youden-J, cost-sensitive) |
| `eval_toolkit.bootstrap` | BCa + paired bootstrap, MDE estimates, two-level operating-point bootstrap, K-fold CLT-corrected CI |
| `eval_toolkit.calibration` | Reliability curves, Bayes-optimal thresholds, isotonic/Platt/temperature scaling |
| `eval_toolkit.harness` | `Scorer` Protocol + `evaluate(...)` + `evaluate_folded(...)` slice-aware orchestrators |
| `eval_toolkit.leakage` | `LeakageCheck` Protocol + 7 reference impls (exact / near / encoding-obfuscated / cross-split / label-conflict / group / temporal); `Versioned` opt-in Protocol |
| `eval_toolkit.splits` | `Splitter` Protocol + 5 reference impls (holdout / stratified / group / source-disjoint / time-series) |
| `eval_toolkit.loaders` | `DatasetLoader` Protocol + 4 reference impls (DataFrame / SingleSlice / ParquetGlob / HF datasets) with Croissant-compatible `describe()` |
| `eval_toolkit.manifest` | `RunManifest` (NeurIPS-aligned) + `build_manifest` / `write_manifest` |
| `eval_toolkit.text_dedup` | `SimilarityStrategy` Protocol + 5 strategies (TF-IDF / hash / embedding / Jaccard / MinHash-LSH); `near_dedup` / `cross_dedup` orchestrators |
| `eval_toolkit.plotting` | PR curves, reliability diagrams, confusion matrices, score histograms, lift CIs |
| `eval_toolkit.provenance` | File hashing, run-directory layout, figure metadata sidecar |
| `eval_toolkit.schemas` | Versioned JSON Schemas (`results.v1.json`, `results_full.v1.json`, `manifest.v1.json`) |
| `eval_toolkit.paths` | Repo-relative path normalization |
| `eval_toolkit.seeds` | `set_global_seeds` (random + numpy + optional torch) |
| `eval_toolkit.config` | `frozen_config` decorator + `from_yaml` loader |
| `eval_toolkit.docs` | Anchor-based markdown rendering with formatter registry |

## Standards

See [`STYLE.md`](STYLE.md) for the full reconciled coding standards (formatting,
naming, errors, docstrings, tests, packaging).

## Versioning

Semver from v0.1.0. See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT — see [`LICENSE`](LICENSE).
