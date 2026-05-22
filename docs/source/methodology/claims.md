# Claims and Gates

This chapter covers `eval_toolkit.claims` — the layer that turns a
`RunResult` into a **release-time go/no-go decision** by evaluating
named gates against the result payload and (optionally) the manifest.
The toolkit ships generic gates; consumers write their own when domain
logic requires it.

> **Background.** This chapter assumes you've already produced a
> `RunResult` (see [getting-started](../getting-started.md)) and read
> the broader framing in [evidence.md](evidence.md) (source roles,
> threshold transfer, claim mode vs exploratory mode). The piece this
> chapter pins down is the *contract* between a gate and a
> `ClaimReport` — exactly what passing / failing means, how exceptions
> are normalized, and what a `has_failures()` verdict commits to.

(when-to-use)=
## When to reach for claims
Use claims when you need a verdict, not a diagnostic:

- "Is this release ready to ship?"
- "Does this run satisfy the preregistered evidence requirements?"
- "Should the report renderer print 'we claim X' or 'we cannot claim X'?"

Don't use claims for exploratory metrics ("how did model A do?"),
ablations, or score-distribution audits. Those produce numbers; claims
produce booleans.

(claims-data-model)=
## The data model
A `ClaimSpec` bundles a claim's name with one or more `EvidenceGate`s.
Each gate runs a `check(result, manifest) -> GateResult`. The full
`ClaimReport` collects every gate's verdict and exposes
`has_failures()` for the go/no-go decision.

```python
from eval_toolkit.claims import (
    ClaimSpec,
    EvidenceGate,
    GateResult,
    evaluate_claims,
    required_metric_gate,
    minimum_slice_size_gate,
)

result = {
    "run_id": "demo-run",
    "by_slice": {
        "dev": {
            "n": 200,
            "n_positive": 100,
            "by_scorer": {"my_model": {"pr_auc": 0.82}},
        },
    },
}

spec = ClaimSpec(
    name="dev_pr_auc_supported",
    gates=(
        required_metric_gate("dev", "my_model", "pr_auc"),
        minimum_slice_size_gate("dev", min_n=100, min_positive=20, min_negative=20),
    ),
)

report = evaluate_claims(result, [spec])
assert report.has_failures() is False
assert all(g.passed for g in report.claims["dev_pr_auc_supported"])
```

`GateResult` is JSON-serializable and carries the gate's `name`, a
`passed: bool`, the `severity` (`"error"`, `"warning"`, or `"info"`), a
human-readable `message`, and a free-form `evidence` dict useful for
report-renderer post-processing.

`ClaimReport.to_dict()` writes the full per-gate tree plus the
aggregated `has_failures` field — drop that into your `results.json`
under `claim_report` (the v0.9-added `RunResult` field) or pass it
through `with_claim_report(...)`.

(claims-worked-walkthrough)=
## Worked walkthrough
Start from a real `RunResult` shape: two scorers, two slices, mixed
metrics. The claim is "the new model beats the baseline on the dev
slice with statistically supported PR-AUC."

```python
from eval_toolkit.claims import (
    ClaimSpec,
    evaluate_claims,
    metric_threshold_gate,
    no_scorer_errors_gate,
    paired_diff_present_gate,
    required_scorer_gate,
)

result = {
    "by_slice": {
        "dev": {
            "n": 500,
            "n_positive": 250,
            "by_scorer": {
                "baseline": {"pr_auc": 0.65},
                "candidate": {"pr_auc": 0.82},
            },
            "paired_diffs": {
                "candidate_minus_baseline": {
                    "delta": 0.17,
                    # v0.48+ schema: separate `low` + `high` keys (was `ci_95: [l, h]`)
                    "low": 0.08,
                    "high": 0.26,
                }
            },
        }
    }
}

spec = ClaimSpec(
    name="candidate_beats_baseline_on_dev",
    gates=(
        required_scorer_gate("dev", "candidate"),
        required_scorer_gate("dev", "baseline"),
        metric_threshold_gate(
            "dev", "candidate", "pr_auc", op=">=", threshold=0.80
        ),
        paired_diff_present_gate("dev", "candidate_minus_baseline"),
        no_scorer_errors_gate(),
    ),
)

report = evaluate_claims(result, [spec])
assert report.has_failures() is False
gate_names = [g.name for g in report.claims["candidate_beats_baseline_on_dev"]]
assert "metric_threshold:dev:candidate:pr_auc" in gate_names
```

Each gate is independent; failures don't short-circuit. That's
deliberate: the report carries every verdict so the renderer can show
"3 of 5 gates passed; here is what's missing" rather than "stopped at
gate 2."

(exception-contract)=
## Exception handling contract
`EvidenceGate.evaluate` catches a *specific* set of runtime/data
exceptions and converts them to typed failures:

`KeyError`, `ValueError`, `TypeError`, `RuntimeError`,
`AttributeError`, `LookupError`.

Any of those raised inside a gate's `check` becomes a
`GateResult(passed=False, message=f"{type(exc).__name__}: {exc}")`.
Other exceptions — `NameError`, `AssertionError`, `ImportError`,
`KeyboardInterrupt` — propagate. The rationale: data-shape errors are
expected (a gate looks up a metric path that doesn't exist) and should
record cleanly; implementer bugs are unexpected and should crash
loudly so they don't get silently coerced into "gate failed" noise.

```python
from collections.abc import Mapping
from typing import Any
from eval_toolkit.claims import ClaimSpec, EvidenceGate, GateResult, evaluate_claims


def _buggy_check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
    # Typo: meant `result["by_slice"]`. KeyError is normalized to a typed failure.
    _ = result["by_slise"]
    return GateResult(name="buggy", passed=True)


gate = EvidenceGate(name="buggy", check=_buggy_check)
spec = ClaimSpec(name="demo", gates=(gate,))
report = evaluate_claims({"by_slice": {}}, [spec])
assert report.has_failures() is True
assert report.claims["demo"][0].message.startswith("KeyError:")
```

Compare to an `AssertionError` (an implementer-bug class), which is
*not* caught and propagates:

```python
import pytest
from collections.abc import Mapping
from typing import Any
from eval_toolkit.claims import ClaimSpec, EvidenceGate, GateResult, evaluate_claims


def _asserts(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
    assert False, "this is a bug, not a missing metric"


gate = EvidenceGate(name="asserts", check=_asserts)
spec = ClaimSpec(name="demo", gates=(gate,))
with pytest.raises(AssertionError):
    evaluate_claims({}, [spec])
```

(severity-policy)=
## Severity policy
A gate's `severity` controls whether failure flips `has_failures()`:

- `error` (default) — failure flips `has_failures()` to True.
- `warning` — failure does **not** flip `has_failures()` unless
  `report.has_failures(include_warnings=True)` is explicitly requested.
- `info` — never flips `has_failures()`. Use for purely informational
  gates that produce evidence the renderer reads but don't gate
  release.

Use `warning` for soft gates whose failure should *surface* in the
report but not block release. Reserve `error` for the hard
preconditions of the claim.

```python
from eval_toolkit.claims import (
    ClaimReport,
    GateResult,
)

# A claim with one error-passing and one warning-failing gate.
report = ClaimReport(
    claims={
        "demo": [
            GateResult(name="hard_gate", passed=True, severity="error"),
            GateResult(name="soft_gate", passed=False, severity="warning"),
        ]
    }
)
assert report.has_failures() is False
assert report.has_failures(include_warnings=True) is True
```

(claims-pitfalls)=
## Pitfalls / Common mistakes
**Do attach the report.** `evaluate_claims` doesn't mutate the
`RunResult` for you. Either set `RunResult.claim_report =
report.to_dict()` directly or use `eval_toolkit.harness.with_claim_report`
which returns a new `RunResult` with the field populated.

**Don't write overly broad gates.** A gate that always returns
`passed=True` (e.g., `lambda r, m: GateResult(name="g", passed=True)`)
satisfies `has_failures()` trivially. Each gate should encode a
checkable precondition — "this metric exists," "this slice is at least
N rows," "this diff CI excludes zero." Walk through every gate and
ask: *what would make this fail?* If the answer is "nothing
realistic," delete it.

**Don't lean on warnings for go/no-go.** `warning` severity is exactly
that — surfaced in the report but not a fail. If the absence of
something should *block* release, use `error`.

**Do separate exploratory metrics from claim gates.** `RunResult`
carries all the metrics — bootstrap CIs, slice-stratified PR-AUC,
calibration error. Only the *subset* of those numbers that's
preregistered as a claim precondition belongs in a `ClaimSpec`. The
rest is exploratory and lives in the result payload, not in the
gates.

**Watch for KeyError-as-failure masking real bugs.** Because
`KeyError` is in the catch list, a gate that fails because *you typed
a metric path wrong* looks identical to a gate that fails because the
metric is genuinely missing. Log gate failures during development and
read the `message` field — it includes the exception type prefix.

## See also

- [evidence.md](evidence.md) — broader framing of source roles,
  threshold transfer, and the claim-mode contract.
- [artifacts.md](artifacts.md) — the `PredictionArtifactRef` /
  `MetricState` data layer that claim gates read from.
- [versioning.md](versioning.md#schema-evolution) — schema-evolution
  policy for the `claim_report` field on `RunResult`.
- [../extending.md](../extending.md) — how to write a custom gate that
  follows the `EvidenceGate` contract.
