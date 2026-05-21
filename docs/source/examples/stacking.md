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

# Worked example: detector stacking with `LogisticStacker`

> **What this shows.** Combine three synthetic per-sample detectors into a
> calibrated ensemble using `LogisticStacker`, the v0.45.0 reference impl of
> the `MetaLearner` protocol. Demonstrates the canonical pipeline: per-detector
> scores → stacker fit → stacker `predict_proba` → optional second-stage
> calibration via `fit_platt_binary`.
>
> **Runtime:** <1 s. Pure sklearn + numpy; no optional dependencies.
> Closes [eval-toolkit#52](https://github.com/brandon-behring/eval-toolkit/issues/52).

## Why stack?

A single binary detector — a fine-tuned classifier, an activation probe,
an LLM-judge — captures one kind of signal. Stacking learns the best
(regularized) combination of multiple detectors' outputs on a held-out
*stacking* set, producing a meta-classifier that is at least as good as
the best base detector and usually better.

The toolkit ships one reference stacker, `LogisticStacker`, which wraps
`sklearn.linear_model.LogisticRegression` and exposes the v1.0 Tier-2
`MetaLearner` Protocol. Custom stackers (random forests, gradient boosting,
calibrated ensembles) can implement the same Protocol structurally.

## Setup

```{code-cell}
import numpy as np

from eval_toolkit import (
    LogisticStacker,
    MetaLearner,
    fit_platt_binary,
    pr_auc,
    roc_auc,
)

rng = np.random.default_rng(0)
N = 800
```

## Synthetic three-detector data

We simulate three binary detectors with descending signal-to-noise on the
same `N`-sample stacking set.

```{code-cell}
y = rng.binomial(1, 0.3, size=N).astype(int)

# Detector 0: strong signal, low noise
det0 = np.clip(y * 0.75 + rng.normal(0, 0.15, N), 0, 1)
# Detector 1: medium signal, more noise
det1 = np.clip(y * 0.55 + rng.normal(0, 0.25, N), 0, 1)
# Detector 2: weak signal, near-random
det2 = np.clip(y * 0.30 + rng.normal(0, 0.40, N), 0, 1)

score_matrix = np.column_stack([det0, det1, det2])
score_matrix.shape
```

Baseline performance per detector:

```{code-cell}
for i, name in enumerate(["det0", "det1", "det2"]):
    pr = pr_auc(y, score_matrix[:, i])
    roc = roc_auc(y, score_matrix[:, i])
    print(f"{name}: PR-AUC={pr:.3f}, ROC-AUC={roc:.3f}")
```

## Fit the stacker

`LogisticStacker(C=1.0, class_weight='balanced')` is a reasonable default
for imbalanced binary detection. Smaller `C` (e.g. `0.1`) regularizes more
strongly, useful when the stacking set is small or detectors are noisy.

```{code-cell}
stacker = LogisticStacker(C=1.0, class_weight="balanced", random_state=0)
stacker.fit(score_matrix, y)
print("coef_:", stacker.coef_)
print("intercept_:", stacker.intercept_)
print("classes_:", stacker.classes_.tolist())
```

The fitted weights reflect the signal ordering: `det0` (strongest) gets the
largest weight, `det2` (weakest) gets the smallest.

## Stacked predictions

```{code-cell}
stacked_proba = stacker.predict_proba(score_matrix)[:, 1]
stacked_pr = pr_auc(y, stacked_proba)
stacked_roc = roc_auc(y, stacked_proba)
print(f"Stacked: PR-AUC={stacked_pr:.3f}, ROC-AUC={stacked_roc:.3f}")
```

The stacker outperforms every base detector — that's the whole point.

## Second-stage calibration

Logistic regression's sigmoid output is well-calibrated on the *training*
distribution but can drift on a held-out test set. For downstream calibration
metrics (ECE, Brier), chain through `fit_platt_binary` on a disjoint
calibration set. Here we use the same data for illustration:

```{code-cell}
(a, b), apply_platt = fit_platt_binary(y, stacked_proba)
calibrated_proba = apply_platt(stacked_proba)

# Discrimination is unchanged (Platt is a monotone transform):
print(f"Pre-calibration  PR-AUC={pr_auc(y, stacked_proba):.4f}")
print(f"Post-calibration PR-AUC={pr_auc(y, calibrated_proba):.4f}")
print(f"Platt (a, b): ({a:.3f}, {b:.3f})")
```

For a full 4-calibrator audit (temperature / isotonic / Platt / Beta), see
the [calibration chapter](../methodology/calibration.md).

## Custom `MetaLearner` impls

The `MetaLearner` Protocol is structural — any class exposing `coef_`,
`classes_`, `intercept_`, `fit(score_matrix, y)`, `predict(score_matrix)`,
and `predict_proba(score_matrix)` satisfies it. Drop-in alternatives to
`LogisticStacker` include sklearn `GradientBoostingClassifier`, `RandomForest`,
or domain-specific ensemblers — wrap them with a thin Protocol-conformant
adapter and the rest of the harness works without changes.

```{code-cell}
# Verify structural Protocol satisfaction
assert isinstance(stacker, MetaLearner)
print("LogisticStacker satisfies MetaLearner ✓")
```

## References

- Wolpert, D. H. 1992. "Stacked generalization." *Neural Networks* 5(2),
  241–259. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
- Breiman, L. 1996. "Stacked regressions." *Machine Learning* 24(1),
  49–64.
