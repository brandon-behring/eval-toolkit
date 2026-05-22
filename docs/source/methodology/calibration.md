# Calibration

> **Background** *(skip if you've internalized this)*. A classifier is
> *calibrated* if its predicted probabilities match observed
> frequencies — among rows where the model says "70 % positive", roughly
> 70 % are positive. Modern neural networks are routinely
> *miscalibrated*: confidence systematically diverges from accuracy
> (Guo et al., [2017](https://arxiv.org/abs/1706.04599)). Miscalibrated
> probabilities aren't just an aesthetic problem — they break
> cost-sensitive thresholding ([thresholds.md](thresholds.md)),
> Bayes-optimal decision rules, and any downstream system that treats
> the score as a probability (selective prediction, ensembling,
> abstain-cost analysis).

This chapter covers how to *measure* calibration (ECE variants, Brier
decomposition, reliability diagrams), how to *fix* it (temperature,
isotonic, Platt, and Beta — all four shipped as binary-adapter pairs
returning `(params, apply)`: {func}`~eval_toolkit.calibration.fit_temperature_binary`
[v0.35], {func}`~eval_toolkit.calibration.fit_isotonic_binary` [v0.42],
{func}`~eval_toolkit.calibration.fit_platt_binary` [v0.40], and
{func}`~eval_toolkit.calibration.fit_beta_binary` [v0.40]), and when
*not* to. The 4-binary-adapter family is the v1.0 calibrator surface;
cross-reference [`roadmap.md`](../roadmap.md) "Currently shipped" for the
shipped baseline.

## Setup

```python
import numpy as np
from eval_toolkit import reliability_curve
from eval_toolkit.metrics import (
    expected_calibration_error,
    expected_calibration_error_l2,
    expected_calibration_error_debiased,
    expected_calibration_error_l2_debiased,
    expected_calibration_error_equal_mass,
    brier_score, brier_decomposition,
)
```

A 500-row miscalibrated fixture (overconfident scores: shifted away
from 0.5):

```python
rng = np.random.default_rng(42)
y = rng.binomial(1, 0.4, size=500).astype(int)
# Model output: correct on the rank, but overconfident.
linear = 0.7 * y + 0.3 * rng.normal(0, 0.5, size=500)
s_overconfident = np.clip(0.5 + np.tanh(linear * 2.5) * 0.45, 0, 1)
```

(reliability)=
## Reliability diagram
The visual canonical: bin the predictions, plot mean predicted probability
vs observed positive rate per bin. Diagonal = perfect calibration.

```python
curve = reliability_curve(y, s_overconfident, n_bins=10, strategy="quantile")
print(f"n_bins={curve['n_bins']}  ece_equal_mass={curve['ece_equal_mass']:.3f}")
# `prob_true` / `prob_pred` arrays plot as the reliability diagram;
# eval_toolkit.plotting.plot_reliability_diagram(...) renders it.
```

`strategy="quantile"` (equal-mass binning) is preferred over
`"uniform"` (equal-width) under class imbalance — equal-width
concentrates most mass in 1–2 bins and the calibration signal collapses.

(ece-variants)=
## ECE variants
Expected Calibration Error: a single-number summary of the reliability
diagram. Four variants ship with eval-toolkit, differing on (a) L1 vs L2
norm and (b) plug-in vs debiased.

| Variant | Function | Norm | Debiased |
|---|---|---|---|
| L1 plug-in | `expected_calibration_error` | L1 | no |
| L1 debiased | `expected_calibration_error_debiased` | L1 | yes |
| L2 plug-in | `expected_calibration_error_l2` | L2 | no |
| L2 debiased | `expected_calibration_error_l2_debiased` | L2 | yes |

```python
e1   = expected_calibration_error(y, s_overconfident, n_bins=10)
e1_d = expected_calibration_error_debiased(y, s_overconfident, n_bins=10)
e2   = expected_calibration_error_l2(y, s_overconfident, n_bins=10)
e2_d = expected_calibration_error_l2_debiased(y, s_overconfident, n_bins=10)
print(f"L1: {e1:.4f} (plug-in)  {e1_d:.4f} (debiased)")
print(f"L2: {e2:.4f} (plug-in)  {e2_d:.4f} (debiased)")
```

**Which to use.**

- **L2-debiased** is the toolkit default for *reporting* — preserves
  rank ordering across bin counts (Naeini et al.,
  [2015](https://arxiv.org/abs/1411.0760)) and the debiasing correction
  removes the small-sample inflation Kumar et al.
  ([2019](https://arxiv.org/abs/1909.10155)) document. **Pitfall**: L1
  plug-in can swap rank when bin count changes.
- **L1 plug-in** matches sklearn's calibration error and many published
  results — use it for *comparison* with prior work, not for
  decision-making.
- **Equal-mass** (quantile) ECE is more robust to imbalance than
  equal-width:

```python
e_eqmass = expected_calibration_error_equal_mass(y, s_overconfident, n_bins=10)
print(f"L1 equal-mass: {e_eqmass:.4f}")
```

> **What NOT to do.** Don't compare ECE across two models computed with
> different bin counts. ECE is a *binned* estimator — small bin counts
> understate, large bin counts overstate, the bias direction depends on
> sample size. Pin n_bins per project and document it.

(brier)=
## Brier score & decomposition
The Brier score (mean squared probability error) is the
*proper-scoring-rule* analogue of ECE — it's threshold-free, sensitive
to calibration AND ranking, and decomposes additively into three
interpretable components (Murphy, 1973).

$$\text{BS} = \text{REL} - \text{RES} + \text{UNC}$$

- **REL** (reliability, lower better): squared distance between
  predicted probability and empirical positive rate per bin.
- **RES** (resolution, higher better): variance of bin rates around the
  marginal — how much the model *separates* outcomes.
- **UNC** (uncertainty, irreducible): the marginal Bernoulli variance
  $\bar y (1-\bar y)$.

```python
bs = brier_score(y, s_overconfident)
parts = brier_decomposition(y, s_overconfident, n_bins=10)
print(f"Brier: {bs:.4f}")
print(f"  reliability={parts['reliability']:.4f}  "
      f"resolution={parts['resolution']:.4f}  "
      f"uncertainty={parts['uncertainty']:.4f}")
print(f"  identity check: REL - RES + UNC = "
      f"{parts['reliability'] - parts['resolution'] + parts['uncertainty']:.4f}")
```

A model can have low Brier from *either* good calibration (low REL) *or*
strong separation (high RES). The decomposition makes this trade-off
visible — two models with the same Brier may have very different
operational profiles.

(recalibration)=
## Recalibration
When ECE / REL is high, the model can often be *recalibrated*
post-hoc — fit a 1-D function on validation data that maps raw scores to
calibrated probabilities, leaving the rank unchanged.

### Temperature scaling (Guo et al. 2017)

Single-parameter: divide logits by T before softmax. Preserves accuracy
exactly (argmax is invariant to monotone scaling). Simplest and most
common for transformer outputs.

```python
from eval_toolkit import fit_temperature

# fit_temperature wants logits as shape (n, 2): col 0 = neg, col 1 = pos.
val_logits = np.column_stack([1 - linear, linear])
val_labels = y
result = fit_temperature(val_logits, val_labels)
print(f"T*={result['temperature']:.3f}  NLL: {result['nll_pre']:.3f} -> {result['nll_post']:.3f}")
```

### Isotonic regression

Non-parametric monotone fit. More flexible than temperature, requires
more validation data (sklearn rule of thumb: ≥ 1000 rows). Doesn't
necessarily preserve smoothness of the score distribution.

```python
from eval_toolkit import fit_isotonic_calibrator

apply = fit_isotonic_calibrator(y, s_overconfident)
s_calibrated = apply(s_overconfident)
print(f"ECE before: {expected_calibration_error_l2_debiased(y, s_overconfident):.4f}")
print(f"ECE after:  {expected_calibration_error_l2_debiased(y, s_calibrated):.4f}")
```

### Platt scaling

Sigmoid fit (logistic regression with a single scaled+shifted feature).
Two parameters; more flexible than temperature, less than isotonic.

```python
from eval_toolkit import fit_platt_calibrator

apply = fit_platt_calibrator(y, s_overconfident)
s_calibrated = apply(s_overconfident)
print(f"ECE after Platt: {expected_calibration_error_l2_debiased(y, s_calibrated):.4f}")
```

(do-not-recalibrate)=
### When NOT to recalibrate
- **You have <500 validation rows.** Recalibrators overfit small
  samples; the post-cal ECE on a fresh test set can be *worse* than
  uncalibrated.
- **The score distribution is bimodal at 0/1.** Many production
  classifiers output near-deterministic predictions. Recalibration
  can't add information; it just smooths the distribution. ECE will
  improve mechanically but the decision rule is unchanged.
- **You're comparing two raw models.** ECE comparison must be on the
  *raw* outputs unless you explicitly note "after recalibration on
  shared val set". Recalibrating one but not the other is unfair.
- **Production won't apply the calibrator.** If the deployment ships
  raw model outputs, your calibrated ECE is fiction.

(pytorch)=
## PyTorch & transformer specifics
### Logit-domain calibration

Temperature scaling is computed on **logits**, not probabilities. With
HuggingFace transformers, this is `model(...).logits` *before* applying
softmax. Calibrating after softmax loses information (the post-softmax
distribution has already saturated).

<!-- skip: next -->
```python
# Sketch — requires torch / transformers, marked skip for Sybil.
import torch  # noqa
# logits = model(input_ids).logits  # shape (batch, 2) for binary
# T = fit_temperature(logits.cpu().numpy(), labels.cpu().numpy())["temperature"]
# probs = torch.softmax(logits / T, dim=-1)
```

### fp16 / bf16 numerics

Mixed-precision inference (fp16, bf16) introduces small numerical noise
in logits, which softmax amplifies in the tail. Empirically: ECE under
bf16 is ~0.001–0.005 higher than fp32 on the same model, depending on
batch size. Two implications:

1. **Calibrate at inference precision.** If production uses bf16, fit
   temperature on bf16 logits — not on fp32 logits cast back.
2. **Don't compare ECE across precision levels.** A 1 % ECE delta is
   well within fp16/bf16 noise for moderate-size eval sets.

### Calibration drift across checkpoints

A transformer's calibration changes during fine-tuning even when its
accuracy plateaus. Fit temperature *on the same checkpoint* you'll
deploy; don't reuse a temperature from a previous epoch.
[Reproducibility.md](reproducibility.md) discusses why the checkpoint
hash should land in the manifest's `code_versions`.

(calibration-pitfalls)=
## Pitfalls / Common mistakes
- **Reporting ECE on uncalibrated logits.** ECE is only meaningful when
  scores are in [0, 1] and interpretable as P(y=1 | x). The toolkit's
  `expected_calibration_error*` functions raise `ValueError` if scores
  fall outside [0, 1] — apply softmax / sigmoid first.
- **Picking n_bins arbitrarily.** Both equal-width and equal-mass ECE
  are sensitive to bin count. The toolkit defaults to 10; document and
  pin whatever you choose. Cross-paper comparisons require the same
  n_bins.
- **Comparing L1 and L2 ECE numerically.** They're on different scales
  (L1 is bounded by 1; L2 by 1 too but typically smaller). Pick one,
  document the choice.
- **Recalibrating on the test set.** Use a *validation* slice carved
  off the train fold. Recalibrating on test is a direct leakage of the
  metric you're about to report.
- **Ignoring single-class slices.** ECE is degenerate when y is
  all-positive or all-negative. The toolkit's reliability_curve flags
  these with a `"skipped"` marker.

## Putting it all together

```python
# Full calibration audit on a single slice.
bs = brier_score(y, s_overconfident)
ece = expected_calibration_error_l2_debiased(y, s_overconfident, n_bins=10)
parts = brier_decomposition(y, s_overconfident, n_bins=10)
curve = reliability_curve(y, s_overconfident, n_bins=10, strategy="quantile")

print(f"Brier: {bs:.4f}")
print(f"  reliability={parts['reliability']:.4f}  resolution={parts['resolution']:.4f}")
print(f"L2-debiased ECE: {ece:.4f}")
print(f"Equal-mass L1 ECE: {curve['ece_equal_mass']:.4f}")
print(f"  (quantile bins; n_bins={curve['n_bins']})")
```

## Further reading

- Guo, C. et al. *On Calibration of Modern Neural Networks.* ICML 2017.
  [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) — temperature
  scaling, the canonical post-hoc method.
- Naeini, M. P. et al. *Obtaining Well Calibrated Probabilities Using
  Bayesian Binning.* AAAI 2015.
  [arXiv:1411.0760](https://arxiv.org/abs/1411.0760) — ECE definition
  and the equal-mass-binning rationale.
- Kumar, A., Liang, P., & Ma, T. *Verified Uncertainty Calibration.*
  NeurIPS 2019. [arXiv:1909.10155](https://arxiv.org/abs/1909.10155)
  — debiased ECE estimators.
- Murphy, A. H. *A new vector partition of the probability score.* J.
  Appl. Meteorology 12, 1973 — the Brier decomposition.
- Nixon, J. et al. *Measuring Calibration in Deep Learning.* CVPRW 2019.
- sklearn: [`calibration_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html),
  [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

See also: [thresholds.md](thresholds.md) (calibrate before
`CostSensitiveSelector`), [comparison.md](comparison.md) (paired
bootstrap on ECE differences).
