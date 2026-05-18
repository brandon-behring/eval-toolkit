# Threshold selection

> **Background** *(skip if you've internalized this)*. A binary classifier
> outputs a real-valued score per row; a *threshold* turns that score
> into a 0/1 prediction. The threshold is a free parameter — different
> thresholds give different (precision, recall, FPR) trade-offs along
> the model's PR / ROC curves. Picking the threshold is therefore an
> *operational* choice (downstream cost of FP vs FN, target rate, etc.)
> that the model itself doesn't determine. The classic mistake is
> reporting metrics at the toolkit's default threshold (often 0.5) when
> the deployment will use a tuned operating point.

This chapter covers the six [`ThresholdSelector`
](../api/thresholds.md) reference impls, the Bayes-
optimal derivation, and the subtle question of when to refit the
threshold per bootstrap resample.

## Setup

```python
import numpy as np
from eval_toolkit import metrics_at_threshold, select_threshold
```

A 200-row informative fixture used throughout:

```python
rng = np.random.default_rng(42)
y = rng.binomial(1, 0.3, size=200).astype(int)
s = np.clip(y * 0.6 + rng.normal(0, 0.25, 200), 0, 1)
```

(migration)=
## v0.7.0 BREAKING — `select_threshold` now takes a Selector instance
The v0.6 string form is **removed**. Every call site passes a
[`ThresholdSelector`](../api/thresholds.md) instance.

| v0.6 | v0.7 |
|---|---|
| `criterion="max_f1"` | `MaxF1Selector()` |
| `criterion="recall_0.90"` | `TargetRecallSelector(0.90)` |
| `criterion="recall_0.95"` | `TargetRecallSelector(0.95)` |
| `criterion="precision@0.90"` | `TargetPrecisionSelector(0.90)` *(new)* |
| `criterion="recall@0.90"` | `TargetRecallSelector(0.90)` |

Passing a string raises `TypeError` with the migration mapping in the
message. See `CHANGELOG.md` for the rationale.

(selectors)=
## Selectors
(max-f1)=
### Max-F1
**When to use.** Default for binary classification when costs are
unknown / symmetric / "we want a balanced operating point". Especially
common as a *reporting* number — pair the model's max-F1 threshold with
its threshold-free metrics (PR-AUC, ROC-AUC) for a complete picture.

**Optimality.** Lipton, Elkan & Naryanaswamy
([2014](https://arxiv.org/abs/1402.1892)) prove that the max-F1
threshold over the empirical PR curve is the F1-optimal among all
constant decision rules — under the labeling distribution observed in
the eval slice.

```python
from eval_toolkit import MaxF1Selector

result = MaxF1Selector().select(y, s)
print(f"max-F1 threshold={result.threshold:.3f}  F1={result.f1:.3f}")
```

(target-recall)=
### Target recall
**When to use.** Operationally, "find ≥X % of positives". Replaces the
`recall_0.90` / `recall_0.95` strings from v0.6.

**Semantics.** Picks the **highest threshold** meeting recall ≥ target —
i.e., the *most-precise* operating point still satisfying the floor.
Recall is monotonically non-increasing in threshold, so among the
contiguous block of thresholds that meet the floor we pick the rightmost
(lowest recall, highest precision).

```python
from eval_toolkit import TargetRecallSelector

result = TargetRecallSelector(0.90).select(y, s)
print(f"recall>=0.90 threshold={result.threshold:.3f}  recall={result.recall:.3f}")
```

(target-precision)=
### Target precision (NEW in v0.7.0)
**When to use.** Operationally, "every positive prediction must be ≥X %
likely true" — e.g., a moderation queue where each FP costs human
attention. Required by `prompt-injection-sdd`'s `"precision@0.90"`
workflow which previously forced a local re-implementation of
`select_threshold`.

**Semantics.** Picks the **smallest threshold** meeting precision ≥
target — the *highest-recall* operating point still satisfying the
floor. Precision is *not* monotonic in threshold, so the full PR curve
is scanned.

```python
from eval_toolkit import TargetPrecisionSelector

result = TargetPrecisionSelector(0.50).select(y, s)
print(f"precision>=0.50 threshold={result.threshold:.3f}  precision={result.precision:.3f}")
```

(target-fpr)=
### Target FPR
**When to use.** Operationally, "alarm-rate must be ≤X %" — e.g., a
classifier feeding a paged queue where false alarms cost on-call
minutes.

**Semantics.** Smallest threshold meeting FPR ≤ target on the ROC
curve — i.e., the *highest-TPR* feasible point.

```python
from eval_toolkit import TargetFPRSelector

result = TargetFPRSelector(0.10).select(y, s)
fpr = metrics_at_threshold(y, s, result.threshold)["fpr"]
print(f"FPR<=0.10 threshold={result.threshold:.3f}  fpr={fpr:.3f}")
```

(youden)=
### Youden's J
**When to use.** When you want a single "balanced" operating point and
costs are unknown. Threshold-free criterion that maximizes
TPR − FPR.

**Reference.** Youden, W. J. *Index for rating diagnostic tests.*
Cancer 3(1), 1950.

```python
from eval_toolkit import YoudenJSelector

result = YoudenJSelector().select(y, s)
print(f"Youden-J threshold={result.threshold:.3f}")
```

(cost-sensitive)=
### Cost-sensitive (Bayes-optimal)
**When to use.** When you have explicit FP / FN costs and a deployment
prior π = P(y=1). The closed-form Bayes-optimal threshold from Elkan
([2001](https://www.cs.iastate.edu/~honavar/elkan.pdf)) §4 is

$$
t^* = \frac{c_\text{fp}\,(1-\pi)}{c_\text{fp}\,(1-\pi) + c_\text{fn}\,\pi}.
$$

**Important caveat.** Bayes-optimal assumes **calibrated probability
scores** in [0, 1]. For raw logits, apply a calibrator first
([calibration.md](calibration.md): `fit_temperature`,
`fit_isotonic_calibrator`, `fit_platt_calibrator`).

```python
from eval_toolkit import CostMatrix, CostSensitiveSelector

cm = CostMatrix(prior=0.30, fp_cost=1.0, fn_cost=2.0)
result = CostSensitiveSelector(cm).select(y, s)
print(f"cost-sensitive threshold={result.threshold:.3f}  (Bayes-optimal)")
```

The threshold is determined entirely by `(prior, fp_cost, fn_cost)` —
the data is only used to compute precision / recall / F1 *at* that
threshold for reporting. This is intentional: Bayes-optimal is a
*derivation*, not an empirical search.

(bootstrap-refit)=
## When to refit threshold per bootstrap resample
A subtle methodological choice: when reporting a CI on F1 (or any
threshold-dependent metric), should the threshold be fixed once on the
full data, or refit on each bootstrap resample?

**Fix once (single-level bootstrap).** Use
[`bootstrap_ci`](../api/bootstrap.md) wrapping a
function that takes `(y_true, y_score)` and uses a pre-fit threshold.
The CI captures uncertainty in the *metric at this fixed threshold*.

**Refit per resample (two-level bootstrap).** Use
[`paired_bootstrap_op_point_diff`](../api/bootstrap.md)
which re-runs the threshold selection on each resample, then computes
the metric. The CI captures *both* the metric uncertainty and the
threshold-selection uncertainty.

**Which to use.** Refit per resample when the threshold-selection rule
has meaningful variance (small slices, noisy PR curves) AND you're
reporting *operating-point* metrics (F1, precision, recall) AND the
operational deployment will re-run threshold selection on each new data
batch. Otherwise fix once — simpler, narrower CI, easier to interpret.

The two-level bootstrap CI is wider (often 1.5–2× wider) than the
fixed-threshold CI because it includes selection variance. Reporting the
narrower one when the deployment really refits is overconfident.

```python
from eval_toolkit import paired_bootstrap_op_point_diff, pr_auc, MaxF1Selector

s_a = np.clip(rng.normal(0.5, 0.3, size=200), 0, 1)
s_b = np.clip(y * 0.5 + rng.normal(0.2, 0.3, size=200), 0, 1)

def threshold_fn(yt, ys):
    return MaxF1Selector().select(yt, ys).threshold

def f1_at(yt, ys, t):
    from eval_toolkit import metrics_at_threshold
    return float(metrics_at_threshold(yt, ys, t)["f1"])

# Two-level: refits the max-F1 threshold per resample on the val side,
# applies it on the test side, computes paired F1 difference.
diff = paired_bootstrap_op_point_diff(
    val_y=y, val_score_a=s_a, val_score_b=s_b,
    test_y=y, test_score_a=s_a, test_score_b=s_b,
    threshold_fn=threshold_fn, metric_fn=f1_at,
    n_resamples=200, seed=42,
)
print(f"Δ F1: {diff.delta:.3f}  CI [{diff.ci_low:.3f}, {diff.ci_high:.3f}]")
```

(thresholds-threshold-transfer)=
## Applying validation thresholds to other slices
For OOD or diagnostic slices, the common pattern is to fit the
threshold on a mixed-class validation slice and apply it elsewhere.
Use [`operating_points.py`](../api/operating_points.md)
for this instead of hand-rolling post-processing. Mixed-class targets
report `metrics_at_threshold`; all-positive targets report
`recall@threshold`; all-negative targets report `fpr@threshold` and
specificity.

See [evidence.md](evidence.md#threshold-transfer) for the full
claim-evidence framing.

(thresholds-pitfalls)=
## Pitfalls / Common mistakes
- **Reporting metrics at threshold = 0.5 by default.** sklearn defaults
  to 0.5 for `predict()`. For an imbalanced classifier with a calibrated
  output, the Bayes-optimal threshold is rarely 0.5 — it's
  `(1−π) / (1+(c_fn / c_fp − 1)·π)`. Pick a selector deliberately.
- **Picking the threshold on the test set.** This is selection on the
  same data you'll evaluate on — straightforward leakage. Pick the
  threshold on a *validation* slice carved off the train fold.
- **Fitting `CostSensitiveSelector` on uncalibrated scores.** The
  Bayes-optimal formula assumes the scores are P(y=1 | x). Raw logits
  or pre-sigmoid model outputs aren't probabilities. Calibrate first.
- **Treating `criterion="max_f1"` as universally best.** Max-F1 is
  appropriate when (a) F1 is the right metric and (b) costs are roughly
  symmetric. For asymmetric-cost deployments, use
  `CostSensitiveSelector` or `TargetFPRSelector`.
- **Reporting fixed-threshold CI when deployment refits.** Use the
  two-level bootstrap if production retunes the threshold on each new
  batch. The narrower fixed-threshold CI is overconfident.

- **`recall@p` semantics divergence on migration.** If you migrated
  from a code base where `recall@p` (or `"recall_0.90"` in the v0.6
  string API) picked the *smallest* threshold meeting the recall floor
  — i.e., the *highest-recall* feasible point — eval-toolkit's
  [`TargetRecallSelector(p)`](../api/thresholds.md)
  picks the *highest* threshold meeting the floor — i.e., the
  *most-precise* feasible point. Both are valid operating points; we
  standardize on the most-precise convention because it matches
  Lipton-Elkan 2014 §3 and the canonical "least-aggressive predictor
  that still meets the recall floor" framing. *Numerical effect*: same
  recall (≥ p), but the toolkit's threshold is ≥ the legacy
  threshold — and therefore precision is ≥ the legacy precision. If
  you have a golden fixture pinned to the legacy convention, you'll
  see threshold differences but recall ≥ p still holds. See the
  `prompt-injection-sdd` v0.7.0 migration commit for a worked example
  (its `recall@0.90` golden case is now skipped with this rationale).

## Putting it all together

Compare four selectors side-by-side:

```python
from eval_toolkit import (
    MaxF1Selector, TargetRecallSelector, TargetPrecisionSelector,
    YoudenJSelector,
)

selectors = {
    "max_f1":          MaxF1Selector(),
    "recall_0.90":     TargetRecallSelector(0.90),
    "precision_0.50":  TargetPrecisionSelector(0.50),
    "youden_j":        YoudenJSelector(),
}
for name, sel in selectors.items():
    r = sel.select(y, s)
    print(f"  {name:18s}  thr={r.threshold:.3f}  P={r.precision:.3f}  "
          f"R={r.recall:.3f}  F1={r.f1:.3f}")
```

## Further reading

- Lipton, Z., Elkan, C., & Naryanaswamy, B. *Optimal thresholding of
  classifiers to maximize F1 measure.* ECML PKDD 2014.
  [arXiv:1402.1892](https://arxiv.org/abs/1402.1892).
- Elkan, C. *The foundations of cost-sensitive learning.* IJCAI 2001.
- Youden, W. J. *Index for rating diagnostic tests.* Cancer 3(1), 1950.
- sklearn docs:
  [`precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html),
  [`roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).

See also: [calibration.md](calibration.md) (calibrate before cost-
sensitive selection), [comparison.md](comparison.md) (single-level vs
two-level bootstrap).
