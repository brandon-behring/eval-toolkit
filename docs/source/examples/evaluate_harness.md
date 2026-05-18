---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Worked example: slice-aware `evaluate` harness

> **What this shows.** Run two scorers across two slices via
> `evaluate(...)`; persist the result with `write_run_result(...)`; load
> the JSON back; verify it conforms to the `results.v1.json` schema.
>
> **Runtime:** ~2 s. Requires `pandas` for `EvalSlice`'s DataFrame
> wrapper — install via `pip install 'eval-toolkit[dataframe]'`.

## Setup

```{code-cell}
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from eval_toolkit import (
    EvalSlice, evaluate, write_run_result, set_global_seeds,
)
from eval_toolkit.artifacts import validate_results
set_global_seeds(42)
```

## Build two slices

A "validation" slice (in-distribution) and an "ood" slice (lower-signal
out-of-distribution). The harness scores each slice independently:

```{code-cell}
rng = np.random.default_rng(42)

def _make_slice(name: str, n: int, signal: float) -> EvalSlice:
    """Synthetic slice: balanced labels + discriminative-but-noisy scores."""
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    rng.shuffle(y)
    df = pd.DataFrame({
        "text": [f"{name}_row_{i}" for i in range(n)],
        "label": y,
    })
    return EvalSlice(name=name, df=df)

val_slice = _make_slice("validation", n=100, signal=0.4)
ood_slice = _make_slice("ood", n=80, signal=0.2)
print(f"slices: {val_slice.name} (n={len(val_slice.df)}), {ood_slice.name} (n={len(ood_slice.df)})")
```

## Define two `Scorer` Protocols

Any object with `predict_proba(X) -> np.ndarray` satisfies the
`Scorer` Protocol. Toolkit consumers wire their real models here
(sklearn estimators, PyTorch transformers, LLM judges); for this example
we use two minimal stubs:

```{code-cell}
class _DiscriminativeStub:
    """Returns scores correlated with label + Gaussian noise."""
    def __init__(self, signal: float, noise: float, seed: int) -> None:
        self._signal = signal
        self._noise = noise
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        n = len(X)
        # Recover label from the synthetic text suffix
        labels = np.array([int(x.split("_")[-1]) % 2 for x in X])
        return np.clip(
            0.5 + self._signal * (labels - 0.5) + self._rng.normal(0, self._noise, size=n),
            0.0, 1.0,
        )

baseline = _DiscriminativeStub(signal=0.3, noise=0.2, seed=42)
challenger = _DiscriminativeStub(signal=0.4, noise=0.15, seed=43)
```

## Run `evaluate`

`evaluate` is the pure (no IO) orchestrator: scorers × slices →
`RunResult`. Bootstrap CIs on each (slice, scorer) cell:

```{code-cell}
result = evaluate(
    scorers={"baseline": baseline, "challenger": challenger},
    slices=[val_slice, ood_slice],
    run_id="example_run",
    n_resamples=50,  # smaller for the example — production: 1000+
    seed=42,
)
print(f"run_id: {result.run_id}")
print(f"slices in result: {list(result.by_slice.keys())}")
print(f"scorers per slice: {list(result.by_slice['validation']['by_scorer'].keys())}")
```

## Persist + validate the JSON contract

`write_run_result` writes both a compact and a full JSON. The compact
one strips per-row prediction arrays so it's small enough to git-commit;
the full one keeps everything for offline analysis:

```{code-cell}
with TemporaryDirectory() as tmpdir:
    run_dir = Path(tmpdir) / "example_run"
    compact_path, full_path = write_run_result(result, run_dir)
    assert compact_path.exists()
    assert full_path.exists()
    payload = json.loads(compact_path.read_text())

    # The JSON contract: must validate against schemas/results.v1.json
    validate_results(payload)

    # Schema-required fields
    assert payload["schema_version"] == "v1"
    assert payload["run_id"] == "example_run"
    assert "validation" in payload["by_slice"]
    assert "baseline" in payload["by_slice"]["validation"]["by_scorer"]
    print("compact JSON validated against results.v1.json ✓")
```

## Pre-1.0 design note

`evaluate(...)` is *pure*: no filesystem touched. `write_run_result(...)`
is the only IO sink. This split lets you test the harness logic
deterministically (no `tmp_path` fixture needed for in-process verification)
and keeps the durable on-disk artifact a separate, schema-validated layer.

## See also

- [`harness.py` reference](../api/harness.md) — `evaluate`,
  `evaluate_folded`, `EvalSlice`, `RunResult`, `Scorer` Protocol.
- [`artifacts.py` reference](../api/artifacts.md) — `validate_results`,
  `validate_manifest`, JSON-schema dispatcher.
- [Calibration example](calibration.md) — apply `fit_platt_calibrator`
  to the scorer outputs before evaluating.
- [Leakage detection example](leakage_detection.md) — gate the harness
  with `LeakageCheck`s.
