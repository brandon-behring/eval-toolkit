# Reproducibility

> **Background** *(skip if you've internalized this)*. A "reproducible"
> result is one that re-runs to bit-identical numbers given the same
> code, data, and configuration. ML reproducibility usually settles for
> *statistical* reproducibility: re-runs land in the same CI you
> originally reported. Both fail in different ways — bit-identity fails
> on cross-architecture GPU runs and mixed precision; statistical
> reproducibility fails on undocumented seeds, mutated input data,
> dependency drift. The
> [NeurIPS Reproducibility Checklist](https://neurips.cc/public/guides/PaperChecklist)
> formalizes the minimum information needed to re-run a result; this
> chapter walks through how eval-toolkit captures it.

eval-toolkit's reproducibility primitives:

- [`set_global_seeds`](../api/seeds.md) — seeds
  numpy, random, and (optional) torch in one call.
- [`provenance.file_sha256`](../api/provenance.md) —
  hash any input artifact for the manifest.
- [`provenance.capture_git_sha`](../api/provenance.md)
  + [`provenance.make_run_dir`](../api/provenance.md) —
  per-run directory + git provenance.
- [`build_manifest`](../api/manifest.md) +
  [`write_manifest`](../api/manifest.md) — aggregates
  all of the above into one `manifest.json` per run, NeurIPS-aligned.

## Setup

```python
import json
import tempfile
from pathlib import Path
from eval_toolkit import build_manifest, write_manifest, MANIFEST_SCHEMA_VERSION
```

(end-to-end)=
## A reproducible run, end-to-end
The minimum-viable reproducible run captures: code version, git SHA,
dirty-flag, seeds, data hashes, env, GPU info, wall-clock time, and any
inline leakage report.

```python
m = build_manifest(
    run_id="2026-05-08T15:00",
    config={"model": "deberta-lora", "k_folds": 5, "seed": 42},
    seeds={"global": 42, "bootstrap": 42},
    extra_code_versions={"my_app": "0.1.0"},
)
print(f"schema: {m.schema_version}")
print(f"git_sha: {m.git_sha[:10] if m.git_sha else '<not in git repo>'}")
print(f"dirty_flag: {m.dirty_flag}")
print(f"config_hash: {m.config_hash[:30]}...")
print(f"env keys: {sorted(m.env.keys())}")
```

Write it to a run directory next to `results.json`:

```python
with tempfile.TemporaryDirectory() as d:
    path = write_manifest(m, d)
    loaded = json.loads(path.read_text())
    print(f"manifest fields: {sorted(loaded.keys())}")
```

(neurips-mapping)=
## Manifest fields ↔ NeurIPS Reproducibility Checklist
The [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist)
demands ten artifacts. Eight map directly to manifest fields:

| Checklist item | Manifest field | Source |
|---|---|---|
| Code version | `git_sha`, `code_versions` | `provenance.capture_git_sha`, package metadata |
| Working-tree state | `dirty_flag` | `git status --porcelain` |
| Random seed | `seeds` | caller-passed |
| Data version | `data_hashes` | `provenance.file_sha256` (sha256 of every input) |
| Software env | `env` | `sys.version`, `platform.platform()`, importable lib `__version__`s |
| Compute resources | `gpu_info`, `cuda_version` | `nvidia-smi --query-gpu` (graceful fallback) |
| Wall-clock time | `wall_clock_seconds` | caller-passed (timed externally) |
| Eval config | `config_hash` | sha256 of canonical-JSON config |

The two checklist items the toolkit can't capture mechanically:

- **Hyperparameter search ranges.** Belongs in your config — the
  manifest's `config_hash` will catch any change to it.
- **Number of training runs / variation across seeds.** Captured by
  running [`evaluate_folded(... seeds=(1, 2, 3))`](../api/harness.md)
  and reporting `RunResult.fold_summary`'s `n_folds`.

(croissant)=
## Croissant interoperability
The [Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)
metadata format (MLCommons, 2024) is the de-facto standard for ML
dataset metadata, integrated with HuggingFace, Kaggle, and OpenML
covering 400 k+ datasets. eval-toolkit's
[`DatasetLoader.describe()`](../api/loaders.md) emits a
Croissant-compatible subset:

```python
from eval_toolkit import DataFrameLoader
import pandas as pd

df = pd.DataFrame({"split": ["train", "test"], "text": ["a", "b"], "label": [0, 1]})
loader = DataFrameLoader(
    df=df, split_col="split",
    name="example",
    cite_as="arXiv:0000.0000",
    license="MIT",
    url="https://example.com/dataset",
)
desc = loader.describe()
print(f"Croissant subset: {sorted(desc.keys())}")
```

The fields `name / description / citeAs / license / url / distribution`
match Croissant's vocabulary. `distribution` carries
`{name, contentUrl, sha256, contentSize}` per file. Consumers who need
*full* Croissant production wrap eval-toolkit's `describe()` output in
their own publishing pipeline.

(pytorch-determinism)=
## PyTorch determinism — the sharp edges
Bitwise reproducibility on GPU is harder than on CPU. The
[PyTorch 2.8 reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
document four sharp edges that every PyTorch eval pipeline silently
hits unless explicitly addressed.

### 1. DataLoader worker seeding

PyTorch `DataLoader` workers seed themselves *independently of your
global seed* unless you pass `worker_init_fn=` AND `generator=`. Default
behavior is silently non-deterministic.

<!-- skip: next -->
```python
# Sketch — requires torch, marked skip for Sybil.
import torch  # noqa
from torch.utils.data import DataLoader  # noqa

def seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

# loader = DataLoader(dataset, batch_size=32, shuffle=True,
#                     worker_init_fn=seed_worker, generator=g)
```

### 2. `CUBLAS_WORKSPACE_CONFIG` must be set in the environment, BEFORE CUDA init

This is the trap. CUBLAS allocates a workspace once per CUDA context;
once the context is created, setting the env var has no effect. Set it
in your shell or *before* any `import torch.cuda`-equivalent.

<!-- skip: next -->
```python
# Sketch — must run BEFORE any torch.cuda usage.
import os  # noqa
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# import torch  # ← only AFTER setting the env var
```

The recommended setting is `:4096:8`. The variant `:16:8` reduces
memory but degrades throughput; only use it if you're memory-bound.

### 3. `torch.use_deterministic_algorithms(True, warn_only=True)`

Some PyTorch ops have no deterministic kernel. `True` (without
`warn_only`) raises on the first one — not graceful for production.
`warn_only=True` falls back to non-deterministic for those ops with a
warning, while keeping everything else deterministic.

<!-- skip: next -->
```python
# import torch  # noqa
# torch.use_deterministic_algorithms(True, warn_only=True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False  # disables autotuner; deterministic but slower
```

### 4. Mixed precision is NOT bitwise reproducible across GPU architectures

Even with all flags set, bf16 / fp16 inference produces *different*
logits on V100 vs A100 vs H100. This is the precision-vs-determinism
trade-off and there's no fix. Two implications:

1. **Bootstrap CIs absorb the noise.** A 5e-4 logit difference produces
   a metric difference well below the BCa CI width. "Statistical
   reproducibility" still holds; bit-identity does not.
2. **Calibrate at inference precision.** Don't fit temperature on fp32
   and deploy in bf16; the calibration drifts. See
   [calibration.md §"PyTorch & transformer specifics"](calibration.md#pytorch).

(replay-recipe)=
## Replay recipe
To re-run a result from its manifest:

1. Check `git_sha` and `dirty_flag`. Hard-fail if `dirty_flag=True` —
   the original run wasn't reproducible to start with.
2. `git checkout <git_sha>`.
3. Recreate the env: `pip install eval-toolkit==<code_versions['eval_toolkit']>`
   plus pinned versions of any other libraries listed in `code_versions`
   / `env`.
4. Verify input data: hash every file with
   `provenance.file_sha256` and confirm it matches `data_hashes`.
5. Set seeds via `set_global_seeds(manifest['seeds']['global'])`.
6. Re-run the eval entry point. The output `manifest.json`'s
   `config_hash` should match the original.

If you're on a different GPU architecture than the original (CUDA major
version mismatch in `cuda_version`), expect statistical-but-not-bit
reproducibility — metrics within the BCa CI width.

(reproducibility-pitfalls)=
## Pitfalls / Common mistakes
- **Setting seed AFTER imports.** Some libraries (e.g., `transformers`,
  `tokenizers`) seed their RNG at import time. Call `set_global_seeds`
  *first*, before any other ML library imports.
- **Forgetting the dataloader workers.** A perfectly seeded model still
  shuffles data non-deterministically without `worker_init_fn` +
  `generator=`.
- **Trusting `dirty_flag=False` to mean "perfectly reproducible".** It
  means "no uncommitted changes" — submodule state, lockfile drift, OS
  package versions are not in git. Combine with `pyproject.toml` /
  `uv.lock` pinning and `env` snapshot.
- **Hashing data after preprocessing.** Hash the *raw* inputs, not the
  output of your preprocessing pipeline. Otherwise a preprocessing bug
  is invisible to downstream replay.
- **Comparing manifests across CUDA versions.** Different
  `cuda_version` → expect logit-level differences below CI width.
  Compare metrics, not raw scores.
- **Logging seeds verbatim into a public artifact.** Seeds are usually
  fine to share, but if your eval involves sensitive synthetic data
  generation, the seed reveals the data.

## Putting it all together

A full reproducible-run skeleton:

```python
from eval_toolkit import set_global_seeds

# 1. Seeds first (before any heavy imports).
set_global_seeds(42)

# 2. Build manifest with seeds + config + data hashes.
m = build_manifest(
    run_id="reproducibility-demo",
    config={"k_folds": 5, "splitter": "StratifiedKFoldSplitter"},
    seeds={"global": 42, "bootstrap": 42},
    extra_code_versions={"my_app": "0.1.0"},
    wall_clock_seconds=12.3,  # measured externally
)

# 3. Emit alongside results.
with tempfile.TemporaryDirectory() as d:
    manifest_path = write_manifest(m, d)
    # results.json / results_full.json are written separately by
    # eval_toolkit.harness.write_run_result(...)
    print(f"manifest written: {manifest_path.name}")
    print(f"  schema_version: {m.schema_version}")
    print(f"  recorded fields: {len(m.to_dict())}")
```

## Further reading

- *NeurIPS Paper Checklist.* https://neurips.cc/public/guides/PaperChecklist
- *PyTorch 2.8 reproducibility notes.* https://docs.pytorch.org/docs/stable/notes/randomness.html
- *Croissant: A Metadata Format for ML-Ready Datasets.* MLCommons, 2024.
  [arXiv:2403.19546](https://arxiv.org/abs/2403.19546).
- *MLCommons Croissant spec.* https://docs.mlcommons.org/croissant/
- Pineau, J. et al. *Improving reproducibility in machine learning
  research.* JMLR 22, 2021.

See also: [comparison.md](comparison.md) (CIs absorb sub-CI-width
noise), [testing.md](testing.md) (golden tests for bit-identical
reproducibility on CPU).
