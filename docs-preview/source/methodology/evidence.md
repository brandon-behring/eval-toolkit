# Evidence And Claims

This chapter covers the generic evidence layer added for v1-prelude:
source-role metadata, validation-fit operating points applied across
slices, and claim gates. The goal is not to render a report. The goal is
to make it hard for a consumer project to publish a claim whose required
evidence is missing.

(claim-mode)=
## Claim Mode vs Exploratory Mode
Exploratory runs may record missing evidence as ordinary context. Claim
mode should fail closed: if the declared headline comparison, source
roles, sample size, leakage checks, or diagnostic gates are missing, the
consumer report should say "no-go" or refuse to render claim language.

The toolkit supplies machine-readable gates only. Consumer projects own
their domain claim text and any report renderer.

(source-roles)=
## Source Roles
Use source roles when datasets have different evidentiary weight. The
toolkit does not enforce a taxonomy, but the recommended starting
vocabulary is:

- `train`
- `validation`
- `development_eval`
- `external_diagnostic`
- `final_holdout_candidate`
- `locked_final_holdout`
- `excluded`

These roles are optional manifest metadata. They are useful when a
claim gate needs to ask whether a run actually included the evidence
class it says it used.

```python
from eval_toolkit import SourceRoleRecord, build_manifest

manifest = build_manifest(
    run_id="demo",
    config={"seed": 42},
    source_roles=[
        SourceRoleRecord(source="main_train", role="train", n_rows=1000),
        SourceRoleRecord(source="hard_negatives", role="external_diagnostic", n_rows=200),
    ],
    required_source_roles=("train", "external_diagnostic"),
    guardrails=["do not tune thresholds on external diagnostics"],
)
assert manifest.source_roles[0]["role"] == "train"
```

(threshold-transfer)=
## Threshold Transfer
For operating-point evidence, fit the threshold on a mixed-class
validation slice and apply that exact threshold elsewhere. This is the
right primitive for OOD slices that are all-positive or all-negative:
the target slice cannot fit a threshold itself, but it can still answer
recall or false-positive-rate questions at a threshold chosen upstream.

```python
import numpy as np
from eval_toolkit import MaxF1Selector, apply_operating_points, fit_operating_points

y_val = np.array([0, 0, 1, 1])
s_val = np.array([0.1, 0.3, 0.7, 0.9])
y_ood = np.array([1, 1, 1])
s_ood = np.array([0.8, 0.4, 0.95])

fitted = fit_operating_points(
    y_val,
    s_val,
    [MaxF1Selector()],
    fitted_on_slice="validation",
    scorer_name="model",
)
applied = apply_operating_points(
    y_ood,
    s_ood,
    fitted,
    applied_to_slice="ood_positive",
    scorer_name="model",
)
assert applied["max_f1"]["slice_class"] == "all_positive"
assert "threshold_provenance" in applied["max_f1"]
```

When using the harness, pass `OperatingPointSpec` to `evaluate`. The
result is attached under each target scorer block as
`transferred_operating_points`, preserving where the threshold was fit
and where it was applied.

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, MaxF1Selector, OperatingPointSpec, evaluate

class ScoreByText:
    def predict_proba(self, X):
        table = {"v0": 0.1, "v1": 0.2, "v2": 0.8, "v3": 0.9, "h0": 0.1, "h1": 0.9}
        return np.array([table[str(x)] for x in X])

validation = EvalSlice(
    name="validation",
    df=pd.DataFrame({"text": ["v0", "v1", "v2", "v3"], "label": [0, 0, 1, 1]}),
)
hard_negative = EvalSlice(
    name="hard_negative",
    df=pd.DataFrame({"text": ["h0", "h1"], "label": [0, 0]}),
)
run = evaluate(
    {"model": ScoreByText()},
    [validation, hard_negative],
    run_id="threshold-transfer",
    n_resamples=10,
    operating_point_specs=[
        OperatingPointSpec(
            name="validation_fit",
            fit_slice="validation",
            apply_slices=("hard_negative",),
            selectors=(MaxF1Selector(),),
        )
    ],
)
fpr = run.by_slice["hard_negative"]["by_scorer"]["model"][
    "transferred_operating_points"
]["validation_fit"]["max_f1"]["fpr@threshold"]
assert fpr == 0.5
```

(claim-gates)=
## Claim Gates
Claim gates are small checks over a result payload and optional manifest.
They are intentionally generic: the consumer names the claim, chooses
the gates, and renders any report text. Gates can be used for
claim-bearing runs, while exploratory runs can persist the same report as
context without treating failures as publish blockers.

```python
from eval_toolkit import (
    ClaimSpec,
    RunResult,
    evaluate_claims,
    low_fpr_feasibility_gate,
    metric_threshold_gate,
    minimum_slice_size_gate,
    no_scorer_errors_gate,
    source_role_gate,
    with_claim_report,
)

result = {
    "by_slice": {
        "test": {"n": 200, "n_positive": 80, "by_scorer": {}},
        "hard_negative": {
            "n": 200,
            "n_positive": 0,
            "by_scorer": {
                "model": {
                    "transferred_operating_points": {
                        "calib": {"max_f1": {"fpr@threshold": 0.01}}
                    }
                }
            },
        },
    }
}
manifest_payload = {
    "source_roles": [
        {"source": "main", "role": "train"},
        {"source": "diag", "role": "external_diagnostic"},
    ]
}
claim = ClaimSpec(
    name="example claim",
    gates=(
        minimum_slice_size_gate("test", min_n=100, min_positive=40, min_negative=40),
        source_role_gate(("train", "external_diagnostic")),
        metric_threshold_gate(
            "hard_negative",
            "model",
            "transferred_operating_points.calib.max_f1.fpr@threshold",
            op="<=",
            threshold=0.05,
        ),
        no_scorer_errors_gate(),
    ),
)
report = evaluate_claims(result, [claim], manifest=manifest_payload)
assert not report.has_failures()

run_result = RunResult(run_id="claim-demo", git_sha=None, config={}, by_slice=result["by_slice"])
stored = with_claim_report(run_result, report)
assert stored.claim_report["has_failures"] is False
```

(low-fpr-feasibility)=
## Low-FPR Feasibility
A tiny holdout cannot support a low-FPR claim even if it observes zero
false positives. `low_fpr_feasibility_gate` computes the best-case
Wilson upper bound for `0 / n_negative` and requires that upper bound to
be no larger than the requested FPR.

```python
from eval_toolkit import ClaimSpec, evaluate_claims, low_fpr_feasibility_gate

tiny_result = {"by_slice": {"holdout": {"n": 48, "n_positive": 23}}}
tiny_claim = ClaimSpec(
    name="low-FPR claim",
    gates=(low_fpr_feasibility_gate("holdout", max_fpr=0.05),),
)
tiny_report = evaluate_claims(tiny_result, [tiny_claim])
gate = tiny_report.claims["low-FPR claim"][0]
assert tiny_report.has_failures()
assert gate.evidence["n_negative"] == 25
assert gate.evidence["best_case_fpr_ci_high"] > 0.05
```

(pitfalls)=
## Pitfalls / Common mistakes
- **Fitting a threshold on the target OOD slice.** If that slice is part
  of claim evidence, this leaks target information into the operating
  point.
- **Treating single-class OOD recall as a full classifier claim.**
  All-positive slices cannot estimate precision, FPR, or calibration.
- **Letting diagnostics stay advisory in claim mode.** If a hard-negative
  diagnostic is required for a claim, encode its FPR cap as a gate.
- **Claiming low FPR from too few negatives.** A zero-FP result on a
  small negative slice can still have a large upper confidence bound.
- **Putting domain policy into the toolkit.** Domain labels, claim text,
  and deployment policy belong in the consumer project.
