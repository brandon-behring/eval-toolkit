# Stratified PR-AUC & the gap-flag report

> **Background** *(skip if you've internalized this)*. A binary
> classifier's headline PR-AUC can mask domain-specific failure: the
> model may exploit a *confounder* correlated with the label
> distribution rather than the underlying signal. Examples: text length
> (long inputs are more common for one class than the other),
> time-of-day (label distribution drifts), document source (one source
> is 90 % positive, another 10 %). A *stratified* PR-AUC, computed on
> the central window of the confounder, removes the tails where one
> class dominates and isolates whether the model still discriminates
> when the confounder is held roughly constant. McClish 1989 framed
> this as "partial AUC" for ROC curves; the same idea applies to PR
> space.

This chapter covers `quantile_stratified_pr_auc` (the trimmed
primitive) and `quantile_stratified_report` (the four-field reporting
wrapper that became canonical in the SDD `REPORT.md` convention).

## Setup

```python
import numpy as np
from eval_toolkit import (
    quantile_stratified_pr_auc, quantile_stratified_report,
)
from eval_toolkit.metrics import pr_auc
```

A 500-row fixture where the model's signal is genuine but a
*confounder* (text length) is correlated with the label distribution:

```python
rng = np.random.default_rng(42)
n = 500
y = rng.binomial(1, 0.3, size=n)
# Length is correlated with label: positives tend to be longer.
length = rng.integers(20, 100, size=n) + (y * 80)
# Score has two components: (a) genuine signal, (b) length confound.
score_signal = y * 0.5 + rng.normal(0, 0.25, n)
score_confound = (length - length.mean()) / length.std() * 0.2
s = np.clip(score_signal + score_confound, 0, 1)
```

The full PR-AUC will look great because the model "sees" both the
signal and the confound. Stratifying on length removes the
confound-tails:

```python
full = pr_auc(y, s)
print(f"Full PR-AUC: {full:.3f}")

trimmed_block = quantile_stratified_pr_auc(y, s, length, q_low=0.25, q_high=0.75)
print(f"Trimmed PR-AUC (central 50 %): {trimmed_block['pr_auc']:.3f}")
print(f"Window: lengths in [{trimmed_block['stratifier_low']:.0f}, "
      f"{trimmed_block['stratifier_high']:.0f}]")
```

If the trimmed metric is much lower than the full metric, the model
was riding the confound. The four-field report makes this auditable in
one shot:

```python
report = quantile_stratified_report(y, s, length, gap_threshold=0.05)
print(report)
```

(gap-flag)=
## The gap-flag convention
`quantile_stratified_report` returns:

```text
{"full": ..., "trimmed": ..., "gap": full - trimmed, "gap_flag": gap > threshold}
```

- **`full`** — PR-AUC over all rows. The headline metric.
- **`trimmed`** — PR-AUC over the central `[q_low, q_high]` quantile
  window of the stratifier. Default `[0.25, 0.75]` keeps the middle 50 %.
- **`gap`** — `full - trimmed`. Positive ⇒ the tails inflate the
  headline. Negative ⇒ the tails are *harder* than the central window
  (rarer; usually a sample-size artifact when one tail is small).
- **`gap_flag`** — `gap > gap_threshold` (default 0.05). A single bit
  designed for at-a-glance reviewer-friendly tables.

The 0.05 threshold is the SDD reporting convention; tune for your
domain. A gap of 0.02 is usually noise; 0.10+ is almost always real
confound exploitation.

(length_stratification-when)=
## When to use
Use `quantile_stratified_report` whenever you suspect the score
correlates with a continuous covariate that's *also* correlated with
the label. Common stratifiers:

- **Text length** — for prompt-injection / safety eval, length is
  notoriously confounded with the label distribution (long jailbreak
  prompts, short benign queries). The gap reveals whether the model
  learned the attack pattern or just "long text → flag it".
- **Time** — for any drift-prone task (recommender systems, fraud
  detection), the gap on a temporal stratifier shows whether the model
  generalizes across time periods.
- **Source** — when training pools mix sources with different label
  priors (Lakera + LLMail + OASST in PI eval), the source-stratified
  gap indicates whether the model is per-source-overfit.
- **The score itself** — passing the score as the stratifier is
  equivalent to McClish 1989's partial-AUC over a score-quantile
  window. Useful when you only care about the operating range
  `[q_low, q_high]` of scores (e.g., "the part of the PR curve where
  precision is between 0.7 and 0.9").

(when-not)=
## When NOT to use
- **The stratifier is the label** — degenerate; you'd be measuring
  PR-AUC on a single class.
- **The stratifier is unrelated to the label** — the gap will be
  noise. Verify upstream (e.g., `np.corrcoef(label, stratifier)`)
  before reading the gap as evidence of confound exploitation.
- **n is too small in the central window** — the toolkit raises
  `ValueError` if either class has fewer than 10 rows in the trimmed
  subset. A failing gap report on small slices is usually just
  insufficient power.

## Putting it all together

A complete length-stratified audit:

```python
from eval_toolkit.metrics import expected_calibration_error_l2_debiased

print("=" * 60)
print(f"Headline PR-AUC: {pr_auc(y, s):.3f}")
print(f"Headline ECE:    {expected_calibration_error_l2_debiased(y, s):.4f}")
print("-" * 60)
report = quantile_stratified_report(y, s, length, gap_threshold=0.05)
print(f"Length-stratified report:")
for k, v in report.items():
    print(f"  {k}: {v}")
if report["gap_flag"]:
    print("⚠  GAP > 0.05: model may exploit length confound.")
```

If `gap_flag` fires, the right next step is *not* "drop the model"
— it's "investigate the confound". Maybe length actually IS a
legitimate signal in your domain (longer documents really are more
likely to be one class). The gap-flag is a *prompt for inspection*,
not a verdict.

(length_stratification-pitfalls)=
## Pitfalls / Common mistakes
- **Reporting `full` without `trimmed`.** A single PR-AUC number is
  uninspectable. Always pair with the stratified version when a
  plausible confounder exists.
- **Picking q_low / q_high arbitrarily.** The default `[0.25, 0.75]`
  is the SDD convention but may be wrong for your distribution. If
  the stratifier is bimodal (e.g., very-short and very-long with
  little in between), the central 50 % window is mostly noise; use a
  narrower window or a different stratifier.
- **Treating the gap as a calibrated probability.** It's a
  diff-of-PR-AUCs, bounded in `[-1, 1]` but not a fraction of any
  meaningful quantity. Don't say "the model is 8 % more confounded".
- **Reporting gap without CIs.** Both `full` and `trimmed` have
  bootstrap variance; the gap inherits both. For statistically-rigorous
  reporting, bootstrap each, then bootstrap the difference. (See
  [comparison.md](comparison.md).)
- **Stacking many stratifications.** If you run gap-reports on length,
  source, time, and 5 other covariates, you're multiple-testing
  yourself into noise. Pick the 1–2 stratifiers your domain knowledge
  flags as plausible.

## Further reading

- McClish, D. K. *Analyzing a portion of the ROC curve.* Medical
  Decision Making 9(3), 1989. — partial-AUC framework, the conceptual
  ancestor of stratified PR-AUC.
- Saito, T. & Rehmsmeier, M. *The precision-recall plot is more
  informative than the ROC plot when evaluating binary classifiers
  on imbalanced datasets.* PLOS ONE 10(3), 2015. — why PR-AUC over
  ROC-AUC on imbalanced data.
- Recht, B. et al. *Do ImageNet classifiers generalize to ImageNet?*
  ICML 2019. — empirical case study of distribution-shift gaps that
  the headline metric hides.

See also: [calibration.md](calibration.md) (decomposing a metric into
reliability + resolution + uncertainty),
[comparison.md](comparison.md) (bootstrap CIs on differences).
