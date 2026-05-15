# eval-toolkit

A **methodology-aware evaluation harness for binary classification**:
metrics, bootstrap CIs, calibration, leakage detection, splitting,
threshold selection, dataset loading, reproducibility manifests, and a
slice-aware orchestrator. Pure numpy/scipy/sklearn core;
pandas/matplotlib/hypothesis are optional extras.

## Install

```bash
pip install eval-toolkit
# or with DataFrame loaders + plotting:
pip install 'eval-toolkit[dataframe,plotting]'
# or everything:
pip install 'eval-toolkit[all]'
```

## Three-tier architecture

```
┌─ Tier 3 ─ Reproducibility scaffolding ─────────────────┐
│  manifest.json + seeds + git_sha + data_hashes +       │
│  gpu_info + leakage_report (NeurIPS-aligned)           │
├─ Tier 2 ─ Protocol-based orchestration ────────────────┤
│  Scorer / SliceAwareScorer / LeakageCheck / Splitter   │
│  ThresholdSelector / DatasetLoader / SimilarityStrategy│
├─ Tier 1 ─ Functional core ─────────────────────────────┤
│  pr_auc / roc_auc / ECE variants / Brier / bootstrap_ci│
│  paired_bootstrap_diff / cv_clt_ci / mde_from_ci       │
│  reliability_curve / fit_temperature / fit_isotonic    │
└────────────────────────────────────────────────────────┘
```

## Quick links

- **[Getting started](getting-started.md)** — end-to-end walkthrough
  for new users.
- **[Examples](examples/index.md)** — minimal worked examples for each
  major capability (metrics + bootstrap, harness, calibration,
  leakage, claims/gates, paired comparison).
- **[Methodology](methodology/README.md)** — chapters on splits,
  metrics, calibration, evidence gates, prediction artifacts.
- **[API reference](api/index.md)** — auto-generated from NumPy-style
  docstrings; organized by the three-tier architecture above.
- **[Schemas](schemas.md)** — field-by-field semantics for
  `results.v1.json`, `manifest.v3.json`.
- **[Extending](extending.md)** — Protocol-by-Protocol guide for
  custom Scorers, Splitters, LeakageChecks, ThresholdSelectors,
  DatasetLoaders, EvidenceGates.

## Minimal example

```python
import numpy as np
from eval_toolkit import pr_auc, bootstrap_ci

y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1] * 20)
rng = np.random.default_rng(42)
y_score = np.clip(0.5 + 0.3 * (y_true - 0.5) + rng.normal(0, 0.2, size=len(y_true)), 0, 1)

ci = bootstrap_ci(y_true, y_score, metric=pr_auc, n_resamples=500, seed=42)
print(f"pr_auc = {ci.point_estimate:.3f}  [95% CI: {ci.ci_low:.3f}, {ci.ci_high:.3f}]")
```

→ see [`examples/metrics_and_bootstrap.md`](examples/metrics_and_bootstrap.md)
for the full walkthrough.
