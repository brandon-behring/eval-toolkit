# Worked example: calibration with Platt + isotonic

> **What this shows.** A miscalibrated scorer (uncalibrated logits-like
> values), the resulting ECE, then Platt + isotonic recalibration
> reducing ECE. The two-step "fit on dev / apply on test" pattern.
>
> **Runtime:** ~1 s. Pure-numpy core, no optional dependencies.

## Setup

```python
import numpy as np
from eval_toolkit import (
    fit_platt_calibrator,
    fit_isotonic_calibrator,
    expected_calibration_error,
    set_global_seeds,
)
set_global_seeds(42)
```

## Synthetic miscalibrated scorer

Build a scenario where the scorer's outputs are *informative* (high AUC)
but *miscalibrated* — e.g., the scorer outputs values shifted toward
0.7 even on negatives. Real-world causes: SVMs producing decision-margin
values, neural networks before sigmoid calibration, etc.

```python
rng = np.random.default_rng(42)
n_dev, n_test = 500, 500

# True probability is 0.3 (mostly negatives); scorer outputs are skewed high
y_dev = (rng.uniform(0, 1, n_dev) < 0.3).astype(int)
y_test = (rng.uniform(0, 1, n_test) < 0.3).astype(int)

def _miscalibrated(y, rng):
    # Discriminative (informative for ranking) but biased high
    return np.clip(0.7 + 0.2 * (y - 0.5) + rng.normal(0, 0.1, size=len(y)), 0.0, 1.0)

s_dev = _miscalibrated(y_dev, rng)
s_test = _miscalibrated(y_test, rng)
```

## Before calibration: high ECE

`expected_calibration_error` bins predictions and measures
``|accuracy(bin) - confidence(bin)|`` averaged across bins. Miscalibrated
scores produce high ECE despite potentially good ranking metrics:

```python
ece_uncal = expected_calibration_error(y_test, s_test, n_bins=10)
print(f"ECE (uncalibrated): {ece_uncal:.3f}")
assert ece_uncal > 0.2, "expected the synthetic scorer to be visibly miscalibrated"
```

## Platt calibration (parametric, fast)

Platt fits a sigmoid ``σ(a·s + b)`` to the labels via maximum likelihood
with Lin's 2007 Laplace-smoothed targets. Two scalar parameters → fast,
robust on small dev sets:

```python
platt = fit_platt_calibrator(y_dev, s_dev)
s_test_platt = platt(s_test)
ece_platt = expected_calibration_error(y_test, s_test_platt, n_bins=10)
print(f"ECE (Platt):        {ece_platt:.3f}  (a={platt.a:.3f}, b={platt.b:.3f})")
assert ece_platt < ece_uncal, "Platt calibration should reduce ECE"
```

## Isotonic regression (non-parametric, flexible)

Isotonic fits a monotone step function via PAVA (pool-adjacent-violators).
More flexible than Platt but needs more dev data to avoid overfitting:

```python
isotonic = fit_isotonic_calibrator(y_dev, s_dev)
s_test_iso = isotonic(s_test)
ece_iso = expected_calibration_error(y_test, s_test_iso, n_bins=10)
print(f"ECE (isotonic):     {ece_iso:.3f}")
assert ece_iso < ece_uncal, "isotonic calibration should reduce ECE"
```

## Picking between Platt and isotonic

A practical rule of thumb (Niculescu-Mizil & Caruana 2005):

- **Platt** when the dev set is small (<1000) or the miscalibration is
  approximately monotone-sigmoid (common for SVMs, log-reg with mild
  shift, NNs with temperature drift).
- **Isotonic** when the dev set is large (>1000) and the miscalibration
  is non-monotone (e.g., random forests with bimodal output).

Both produce a `__call__`-able fit object so they're interchangeable in
calibration pipelines. The toolkit also ships
`fit_beta_calibrator` (Kull & Filho 2017) for the parametric beta-family
when sigmoid is too restrictive.

## See also

- [`calibration.py` reference](../api/calibration.md) — full list:
  Platt, isotonic, beta, temperature scaling.
- [`metrics.py` reference](../api/metrics.md) — ECE variants
  (`expected_calibration_error_debiased`,
  `expected_calibration_error_l2`, etc.) and `brier_decomposition`.
- [Metrics + bootstrap example](metrics_and_bootstrap.md) — wrap any of
  these calibrated outputs in a CI.
