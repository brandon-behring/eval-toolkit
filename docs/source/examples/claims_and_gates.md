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

# Worked example: claims + evidence gates

> **What this shows.** Compose `EvidenceGate`s to drive a binary
> release decision ("ship this model" vs "block on insufficient
> evidence"). Each gate is a single check; `evaluate_claims` runs them
> against a `RunResult` and emits a structured `ClaimReport`.
>
> **Runtime:** ~1 s. Pure-numpy core; no optional deps.

## Setup

```{code-cell}
import numpy as np
import pandas as pd
from eval_toolkit import (
    EvalSlice, evaluate, set_global_seeds,
    ClaimSpec, evaluate_claims,
    metric_threshold_gate,
    minimum_slice_size_gate,
)
set_global_seeds(42)
```

## Build a small `RunResult` to test against

(See [`evaluate_harness.md`](evaluate_harness.md) for the full pipeline;
here we just need a `RunResult` to run claims on.)

```{code-cell}
rng = np.random.default_rng(42)
n = 100
y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
rng.shuffle(y)
df = pd.DataFrame({"text": [f"r{i}" for i in range(n)], "label": y})
parent = EvalSlice(name="test", df=df)

class _Stub:
    def predict_proba(self, X):
        # Discriminative-ish: scores correlated with the synthetic label
        labels = np.array([int(x[1:]) % 2 for x in X])
        return np.clip(
            0.5 + 0.3 * (labels - 0.5) + rng.normal(0, 0.1, size=len(X)),
            0.0, 1.0,
        )

result = evaluate(
    scorers={"model_a": _Stub()},
    slices=[parent],
    run_id="claim_example",
    n_resamples=50,
    seed=42,
)
```

## Compose a `ClaimSpec` from gates

A `ClaimSpec` is a named bundle of gates. All gates must pass for the
claim to pass. The toolkit ships a library of reusable gates; you can
also write custom ones (any callable matching the `EvidenceGate`
Protocol). Two common ones:

```{code-cell}
claim = ClaimSpec(
    name="model_a_releasable",
    gates=(
        # Quality bar: PR-AUC on the test slice must be ≥ 0.60
        metric_threshold_gate(
            slice_name="test",
            scorer_name="model_a",
            metric_path="pr_auc",
            op=">=",
            threshold=0.60,
        ),
        # Statistical-power floor: don't claim anything on a tiny slice
        minimum_slice_size_gate(slice_name="test", min_n=30),
    ),
)
```

## Run the claim

`evaluate_claims(result, [claim])` runs every gate against the
`RunResult` and produces a `ClaimReport` keyed by claim name → list of
`GateResult`:

```{code-cell}
report = evaluate_claims(result, [claim])
for gate_result in report.claims["model_a_releasable"]:
    status = "PASS" if gate_result.passed else "FAIL"
    print(f"  [{status}] {gate_result.name}: {gate_result.message}")
print(f"has_failures: {report.has_failures()}")
```

In this synthetic example the noisy stub's `pr_auc` is around 0.51 (not
significantly better than random for this small slice), so the
threshold gate FAILS. The minimum-size gate PASSES (n=100 ≥ 30). The
claim as a whole fails.

## Release decision: every claim must pass

The pattern for shipping a release: every `ClaimSpec` must pass.
`report.has_failures()` returns `True` if any gate failed:

```{code-cell}
if report.has_failures():
    print("BLOCK: at least one gate failed — see evidence above")
else:
    print("RELEASE: all gates passed")
```

`GateResult.evidence` carries structured detail (the actual value, the
threshold, the metric path) so a downstream tool can render the failure
breakdown in CI logs or a release dashboard.

## Pre-1.0 design note

The `EvidenceGate` Protocol is intentionally simple: a callable that
takes `(RunResult, ...) -> GateResult`. Custom gates plug into
`ClaimSpec.gates` alongside the toolkit-provided ones. This matches
the
[NeurIPS Reproducibility Checklist](https://aclrollingreview.org/responsibleNLPresearch/)
pattern of structured evidence: each claim is gated on specific,
auditable conditions.

For multi-comparison correction (Bonferroni / BH-FDR) when running
many gates over many slices, see issue
[#1](https://github.com/brandon-behring/eval-toolkit/issues/1) (planned).

## See also

- [`claims.py` reference](../api/claims.md) — full gate library
  (`required_scorer_gate`, `required_slice_gate`,
  `paired_diff_present_gate`, `no_leakage_errors_gate`,
  `headline_present_gate`, `low_fpr_feasibility_gate`, etc.).
- [`evidence.py` reference](../api/evidence.md) — `EvidenceAxis`,
  `AggregateEvidence` for typed claim aggregation.
- [Evaluate harness example](evaluate_harness.md) — the upstream
  `RunResult` that gets fed into `evaluate_claims`.
