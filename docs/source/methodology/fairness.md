# Fairness & subgroup slicing

> **Background** *(skip if you've internalized this)*. Aggregate metrics
> hide subgroup performance. A model with 90 % overall recall can have
> 95 % recall on the majority subgroup and 60 % on a minority. "Fairness
> evaluation" — at the metric level, separately from upstream
> debiasing — means computing your headline metrics *per subgroup* and
> looking at the gap. Different fairness criteria operationalize "the
> gap shouldn't be too large" differently (demographic parity,
> equalized odds, calibration parity, ...) — they are mutually
> incompatible in general (Kleinberg et al. 2017), so picking one is a
> domain-specific call.

eval-toolkit ships subgroup *slicing* infrastructure
([`EvalSlice`](../api/harness.md),
[`SliceAwareScorer`](../api/harness.md)) but
*deliberately* does not implement fairness metrics — they're consumer-
side concerns sensitive to domain semantics, and good libraries already
exist (`fairlearn`, `aequitas`).

## Setup

```python
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, metrics_at_threshold, MaxF1Selector
```

A 300-row fixture with a categorical sensitive attribute:

```python
rng = np.random.default_rng(42)
n = 300
group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])  # majority/minority
y = (rng.uniform(0, 1, size=n) < (0.4 if "A" else 0.4)).astype(int)
# B subgroup has subtly noisier scores than A
noise = np.where(group == "A", 0.20, 0.30)
s = np.clip(0.6 * y + rng.normal(0, noise, size=n), 0, 1)
df = pd.DataFrame({"text": [f"row_{i}" for i in range(n)], "label": y, "group": group})
slice_ = EvalSlice(name="all", df=df, strata_col="group")
```

(per-subgroup)=
## Per-subgroup metrics via slicing
The simplest fairness audit: split the eval set by subgroup, compute
the headline metrics independently, look at the gap. eval-toolkit
already supports this via the `strata_col` field on `EvalSlice` (used
by `headline_metrics(... strata=...)` for stratified recall).

For arbitrary metrics, just iterate manually:

```python
y_arr = slice_.y_true
groups = slice_.df["group"].to_numpy()
result = MaxF1Selector().select(y_arr, s)
threshold = result.threshold

per_group = {}
for g in np.unique(groups):
    mask = groups == g
    if mask.sum() == 0:
        continue
    m = metrics_at_threshold(y_arr[mask], s[mask], threshold)
    per_group[g] = {"n": int(mask.sum()), "f1": m["f1"],
                    "precision": m["precision"], "recall": m["recall"]}

for g, m in per_group.items():
    print(f"  {g}: n={m['n']:3d}  F1={m['f1']:.3f}  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}")
```

The threshold is selected on the *aggregate* slice — a single common
operating point. Per-subgroup thresholds (i.e., one threshold per group
to equalize some criterion) is a different, opt-in choice; see
[fairlearn's `ThresholdOptimizer`](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html).

(fairness-criteria)=
## Common fairness criteria
These compute on top of `metrics_at_threshold` per subgroup. None ship
in eval-toolkit; the formulas below are how you'd express each on the
toolkit's primitives.

### Demographic parity (statistical parity)

$P(\hat y = 1 | g)$ is roughly equal across groups $g$. Computed as
*positive prediction rate* per subgroup — the column-marginal of the
prediction, ignoring the true label.

```python
ppr = {}
y_pred = (s >= threshold).astype(int)
for g in np.unique(groups):
    mask = groups == g
    ppr[g] = float(y_pred[mask].mean())
print(f"Positive prediction rate: {ppr}")
print(f"Demographic parity gap: {max(ppr.values()) - min(ppr.values()):.3f}")
```

### Equalized odds (Hardt et al. 2016)

$P(\hat y = 1 | y, g)$ is roughly equal across $g$ for both $y = 0$
(equal FPR) and $y = 1$ (equal TPR / recall).

```python
tpr_per_group = {}
fpr_per_group = {}
for g in np.unique(groups):
    mask = groups == g
    m = metrics_at_threshold(y_arr[mask], s[mask], threshold)
    tpr_per_group[g] = m["recall"]
    fpr_per_group[g] = m["fpr"]
gap_tpr = max(tpr_per_group.values()) - min(tpr_per_group.values())
gap_fpr = max(fpr_per_group.values()) - min(fpr_per_group.values())
print(f"TPR gap: {gap_tpr:.3f}   FPR gap: {gap_fpr:.3f}")
```

### Calibration parity

ECE per subgroup should be roughly equal. Useful when downstream
decisions interpret the score as P(y=1) and unequal calibration creates
unequal trust.

```python
from eval_toolkit import expected_calibration_error_l2_debiased
ece_per_group = {}
for g in np.unique(groups):
    mask = groups == g
    if mask.sum() < 30:
        continue
    if y_arr[mask].sum() in (0, mask.sum()):
        continue  # single-class
    ece_per_group[g] = expected_calibration_error_l2_debiased(y_arr[mask], s[mask])
print(f"ECE per group: { {k: round(v, 4) for k, v in ece_per_group.items()} }")
```

> **Pitfall.** Each fairness criterion above can be optimized
> independently, but Kleinberg et al. ([2017](https://arxiv.org/abs/1609.05807))
> show that calibration parity, equal FPR, and equal FNR are mutually
> incompatible whenever base rates differ across groups. Pick the
> criterion that maps to your decision-making cost structure.

(sliceaware)=
## Cost-controlled subgroup auditing
When subgroup analysis involves running many slices and the scorer is
expensive (LLM judge, large transformer), the
[`SliceAwareScorer`](../api/harness.md) Protocol lets
the scorer skip slices it's not relevant to. The harness honors this
automatically — see the existing
[`evaluate(...)`](../api/harness.md) machinery and the
`should_score_slice` hook.

Concrete example: an LLM-judge scorer that costs $0.001 per call might
be configured to only run on the headline `test` slice and skip the
8 OOD subgroup slices, while a free regex scorer runs on all of them:

```python
class _DummyExpensiveScorer:
    """Stand-in showing the SliceAwareScorer hook (see harness.py)."""
    def predict_proba(self, X):
        return np.full(len(X), 0.5)

    def should_score_slice(self, slice_name: str) -> bool:
        # Only score the headline slice; skip subgroup slices.
        return slice_name == "test"
```

The harness records `{"skipped": "<reason>"}` in `RunResult.by_slice`
for slices the scorer opted out of, so the audit trail is complete.

(fairness-out-of-scope)=
## What's NOT in eval-toolkit (and why)
- **Demographic parity / equalized odds metrics as named functions.**
  They're trivial one-liners on top of `metrics_at_threshold`; baking
  them in would force opinions about which fairness definition to
  privilege.
- **`ThresholdOptimizer`-style post-hoc fairness fitting.** Use
  [fairlearn](https://fairlearn.org/) — it has the canonical
  implementations.
- **Subgroup discovery.** Finding *which* subgroups have the largest
  gaps is a separate problem (slice-discovery / data-debugging).
  See [Snorkel Sliceline](https://github.com/HazyResearch/snorkel)
  and [DOMINO](https://github.com/HazyResearch/domino) for that.

(fairness-pitfalls)=
## Pitfalls / Common mistakes
- **Using one threshold but reporting subgroup metrics as if it were
  the per-group operating point.** Acceptable, but document explicitly:
  "F1 / precision / recall reported at the aggregate-slice max-F1
  threshold". Per-group thresholds give different numbers.
- **Bootstrap CIs on subgroup metrics without per-group resampling.**
  The toolkit's `bootstrap_ci` resamples row-wise globally; for
  per-group CIs you need to resample *within* each group. Slice the
  EvalSlice first, then `bootstrap_ci` on the slice.
- **Comparing fairness gaps across runs without CIs.** A 0.02 TPR gap
  on n=300 has a wide CI; treat single gap numbers cautiously.
- **Ignoring small subgroups.** A subgroup with n < 30 has unstable
  metrics (and `bootstrap_ci` may emit a warning). Either accept the
  uncertainty or aggregate small subgroups into "other".
- **Using ECE-equality to claim calibration parity at small n.** ECE is
  binned and noisy; the toolkit emits NaN for single-class subgroups.

## Putting it all together

A complete subgroup audit on the fixture:

```python
print(f"Aggregate F1 at threshold={threshold:.3f}: "
      f"{metrics_at_threshold(y_arr, s, threshold)['f1']:.3f}")

print("Per-subgroup:")
for g, m in per_group.items():
    print(f"  {g}: F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

print(f"Demographic-parity gap: "
      f"{max(ppr.values()) - min(ppr.values()):.3f}")
print(f"Equalized-odds gap: TPR={gap_tpr:.3f}  FPR={gap_fpr:.3f}")
```

For end-to-end production fairness eval (bias mitigation, post-hoc
threshold optimization, group-aware CI computation), use
[`fairlearn`](https://fairlearn.org/) on top of eval-toolkit's outputs
— neither library duplicates the other.

## Further reading

- Hardt, M., Price, E., & Srebro, N. *Equality of Opportunity in
  Supervised Learning.* NeurIPS 2016 — equalized odds.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. *Inherent
  Trade-offs in the Fair Determination of Risk Scores.* ITCS 2017.
  [arXiv:1609.05807](https://arxiv.org/abs/1609.05807) — incompatibility
  of fairness criteria.
- Mitchell, M. et al. *Model Cards for Model Reporting.* FAccT 2019 —
  documentation pattern that consumes per-subgroup metrics.
- [fairlearn](https://fairlearn.org/) and
  [aequitas](http://aequitas.dssg.io/) — production-grade fairness
  libraries built on top of sklearn-shaped predictions.
- Hooker, S. *The hardware lottery.* CACM 2021 — a reminder that
  algorithm choices have downstream subgroup consequences.

See also: [thresholds.md](thresholds.md) (per-group threshold
selection), [calibration.md](calibration.md) (calibration parity),
[reproducibility.md](reproducibility.md) (manifest captures slice list).
