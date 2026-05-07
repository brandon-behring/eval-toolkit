# eval-toolkit

Reusable evaluation toolkit for binary classification: metrics, bootstrap CIs,
calibration, plotting, and lightweight orchestration. Pure numpy/scipy/sklearn
core; pandas/matplotlib/hypothesis are optional extras.

Library-grade by design — every public function is type-annotated, every math
kernel is documented with LaTeX + literature references, and statistical
validity (bootstrap CIs, MDE estimates, paired-difference tests) is built in.

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
s = y + rng.normal(0, 0.3, size=200)

print(f"PR-AUC: {pr_auc(y, s):.3f}")
print(f"ROC-AUC: {roc_auc(y, s):.3f}")
print(f"ECE (10 bins): {expected_calibration_error(y, s, n_bins=10):.3f}")
```

### Bootstrap confidence intervals

```python
from eval_toolkit import bootstrap_ci, paired_bootstrap_diff, pr_auc

ci = bootstrap_ci(y, s, metric_fn=pr_auc, n_resamples=1000, seed=42)
print(f"PR-AUC: {ci.point_estimate:.3f}  95% CI: [{ci.ci_low:.3f}, {ci.ci_high:.3f}]")

# Paired bootstrap on the lift between two scorers
s_baseline = rng.normal(0, 1, size=200)
diff = paired_bootstrap_diff(y, s_baseline, s, metric_fn=pr_auc, n_resamples=1000, seed=42)
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
| `eval_toolkit.metrics` | PR-AUC, ROC-AUC, ECE, threshold selection, prior-shift projection |
| `eval_toolkit.bootstrap` | BCa + paired bootstrap, MDE estimates, two-level operating-point bootstrap |
| `eval_toolkit.calibration` | Reliability curves, Bayes-optimal thresholds, isotonic/Platt/temperature scaling |
| `eval_toolkit.plotting` | PR curves, reliability diagrams, confusion matrices, score histograms, lift CIs |
| `eval_toolkit.harness` | `Scorer` Protocol + slice-aware evaluation orchestrator |
| `eval_toolkit.text_dedup` | Near-duplicate + cross-source leakage scrubbing |
| `eval_toolkit.provenance` | File hashing, run-directory layout, figure metadata sidecar |
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
