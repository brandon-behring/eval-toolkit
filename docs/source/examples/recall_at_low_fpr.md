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

# Worked example: RecallAtLowFPR loss training

> **What this shows.** Train a tiny logistic regressor using
> `RecallAtLowFPR` — the Meta Prompt Guard 2 (PG2) training recipe.
> Demonstrates that the loss descends under SGD on a well-separated
> toy problem (the headline correctness property from the
> acceptance criteria).
>
> **Runtime:** <2 s. Requires `pip install eval-toolkit[losses]`
> (torch only — no transformers).
> Closes [eval-toolkit#50](https://github.com/brandon-behring/eval-toolkit/issues/50).

## Setup

This page is rendered statically because the docs CI environment
doesn't include `[losses]`. Real usage looks like:

```text
pip install eval-toolkit[losses]
```

```{code-cell}
:tags: [skip-execution]

import torch

from eval_toolkit.losses import RecallAtLowFPR

torch.manual_seed(42)
```

## Construct the loss

```{code-cell}
:tags: [skip-execution]

loss = RecallAtLowFPR(
    fpr_target=0.01,           # operating point: keep FPR ≤ 1%
    fpr_smoothing_beta=10.0,   # soft-indicator temperature
    pos_weight=1.0,             # unweighted (use >1 for imbalanced sets)
    reduction="mean",
)

print(f"loss.fpr_target = {loss.fpr_target}")
print(f"loss.fpr_smoothing_beta = {loss.fpr_smoothing_beta}")
```

## Single training step

```{code-cell}
:tags: [skip-execution]

# 32 negatives + 8 positives; positives initialized BELOW the threshold
logits = torch.cat([torch.randn(32) - 0.5, torch.randn(8) - 0.5])
logits = logits.detach().clone().requires_grad_(True)
labels = torch.cat([torch.zeros(32), torch.ones(8)]).int()

out = loss(logits, labels)
out.backward()

print(f"loss = {out.item():.4f}")
print(f"gradient norm = {logits.grad.norm().item():.4f}")
```

The gradient shifts positive logits upward (since increasing TP score
reduces `1 - Recall@FPR`) and tugs FP scores downward at the
threshold.

## Multi-step convergence

```{code-cell}
:tags: [skip-execution]

optimizer = torch.optim.SGD([logits], lr=1.0)

losses_over_time = []
for step in range(50):
    optimizer.zero_grad()
    out = loss(logits, labels)
    out.backward()
    optimizer.step()
    losses_over_time.append(out.item())

print(f"Initial loss: {losses_over_time[0]:.4f}")
print(f"Final loss:   {losses_over_time[-1]:.4f}")
print(f"Reduction:    {(1 - losses_over_time[-1] / losses_over_time[0]) * 100:.1f}%")
```

## Comparison vs. fpr_target

```{code-cell}
:tags: [skip-execution]

# Stricter FPR constraint → higher loss (fewer positives clear the threshold)
for fpr in [0.5, 0.1, 0.05, 0.01]:
    l = RecallAtLowFPR(fpr_target=fpr)
    val = l(logits.detach(), labels).item()
    print(f"  fpr_target={fpr:.2f}  →  loss={val:.4f}")
```

As `fpr_target` shrinks, the bar for "TP above threshold" rises and
the loss increases — exactly what you want for a detector that must
operate at a strict low-FPR point.

## Reduction modes

```{code-cell}
:tags: [skip-execution]

logits_d = logits.detach().clone().requires_grad_(True)

for reduction in ("mean", "sum", "none"):
    l = RecallAtLowFPR(fpr_target=0.1, reduction=reduction)
    out = l(logits_d, labels)
    print(f"  reduction={reduction:6s}  shape={tuple(out.shape) if out.dim() else '()'}")
```

## Production usage

Drop into a standard PyTorch training loop:

```text
model = MyDetector()
loss_fn = RecallAtLowFPR(fpr_target=0.01)
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch["input_ids"])
    loss = loss_fn(logits, batch["labels"])
    loss.backward()
    optimizer.step()
```

The standard training-loop shape, drop-in replacement for
`nn.BCEWithLogitsLoss` when you need FPR-constrained training.
