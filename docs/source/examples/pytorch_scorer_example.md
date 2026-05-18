---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: 'off'
---

# Worked example: PyTorch + LoRA `Scorer` adapter

> **What this shows.** How to wrap a PyTorch transformer with a LoRA
> adapter as a `Scorer` for eval-toolkit's harness — batched inference,
> GPU/CPU placement, deterministic-mode setup, returning a numpy array.
>
> **Optional dependencies.** This example requires
> [`torch`](https://pytorch.org/) ≥ 2.5 and
> [`transformers`](https://huggingface.co/docs/transformers/) (and
> [`peft`](https://huggingface.co/docs/peft/) if you actually fine-tune
> with LoRA). The toolkit's core does *not* depend on any of these —
> they're consumer-side concerns.
>
> Code blocks below are marked `<!-- skip: next -->` so Sybil doesn't
> try to execute them in CI without torch installed. The toolkit's own
> `tests/` includes a guarded smoke test
> (`pytest.importorskip("torch")`) that does run end-to-end if torch
> is available — see verification step in the v0.7.0 plan.

## Setup (CPU baseline; runs in CI)

```{code-cell}
import numpy as np
import pandas as pd
from eval_toolkit import EvalSlice, evaluate
```

## Minimal `Scorer` Protocol shape

The Protocol is just `predict_proba(X) -> np.ndarray`:

```{code-cell}
class _UniformBaseline:
    """Reference shape — not a real model, just shows the Protocol."""
    version = "0.0.0"

    def predict_proba(self, X: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.uniform(0, 1, size=len(X))


df = pd.DataFrame({"text": [f"row_{i}" for i in range(40)],
                   "label": [i % 2 for i in range(40)]})
slice_ = EvalSlice(name="test", df=df)
result = evaluate({"u": _UniformBaseline()}, [slice_], run_id="proto-shape")
print(f"PR-AUC: {result.by_slice['test']['by_scorer']['u']['pr_auc']:.3f}")
```

Anything implementing this is a valid `Scorer`. The PyTorch adapter
below is the same shape, with the work happening inside
`predict_proba`.

## Transformer `Scorer` skeleton

The pattern: a class wrapping `(tokenizer, model, device, batch_size)`,
with `predict_proba` doing tokenize → forward → softmax → numpy in
batches.

<!-- skip: next -->
```{code-cell}
# Requires: torch, transformers. Marked skip for Sybil.
from __future__ import annotations

import os
# IMPORTANT: set CUBLAS_WORKSPACE_CONFIG BEFORE importing torch.cuda.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# One-time, BEFORE first CUDA op:
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TransformerScorer:
    """eval_toolkit.Scorer wrapping a HuggingFace classification model.

    Run with:
        scorer = TransformerScorer(
            "protectai/deberta-v3-base-prompt-injection-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16,
        )
    """
    version = "v2-2025-q4"  # bump when the underlying checkpoint changes

    def __init__(self, model_name: str, *, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Return P(positive) for each text. Batched + numpy-out."""
        out = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits  # shape (batch, n_classes)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(injection)
            out.append(probs.cpu().numpy())
        return np.concatenate(out)
```

## With LoRA adapters

If you fine-tuned with [PEFT](https://huggingface.co/docs/peft/), load
the base model + adapter and otherwise reuse the wrapper above:

<!-- skip: next -->
```{code-cell}
# Requires: torch, transformers, peft. Marked skip for Sybil.
from peft import PeftModel  # noqa


class LoRATransformerScorer(TransformerScorer):
    """Loads base model + LoRA adapter; predict_proba reused unchanged."""
    version = "lora-2026-q1"

    def __init__(self, base_model_name: str, lora_path: str, **kwargs) -> None:
        super().__init__(base_model_name, **kwargs)
        self.model = PeftModel.from_pretrained(self.model, lora_path).eval()
```

## With `SliceAwareScorer` for cost control

LLM-judge and large-transformer scorers are expensive — typical
production runs skip them on inexpensive subgroup slices and only run
them on the headline `test` slice. Implement
[`SliceAwareScorer`](../api/harness.md)'s
`should_score_slice` hook:

<!-- skip: next -->
```{code-cell}
# Requires: torch, transformers. Marked skip for Sybil.
class CostControlledTransformerScorer(TransformerScorer):
    """Skip subgroup slices to save GPU minutes."""
    version = "v2-cost-controlled"

    SLICE_ALLOW_LIST: frozenset[str] = frozenset({
        "test", "ood_lakera", "ood_llmail",  # full-cost slices
    })

    def should_score_slice(self, slice_name: str) -> bool:
        return slice_name in self.SLICE_ALLOW_LIST
```

The harness honors `should_score_slice` automatically — see
[`evaluate(...)`](../api/harness.md); skipped slices
land in `RunResult.by_slice[name].by_scorer[scorer_name] =
{"skipped": "..."}`.

## DataLoader worker seeding (when training, not just inference)

If your `Scorer.fit(...)` trains a transformer, the DataLoader worker
seeding rules in
[reproducibility.md §"DataLoader worker seeding"](../methodology/reproducibility.md#pytorch-determinism)
apply:

<!-- skip: next -->
```{code-cell}
# Requires: torch. Marked skip for Sybil.
import random  # noqa
import numpy as np  # noqa
import torch  # noqa
from torch.utils.data import DataLoader  # noqa

def seed_worker(worker_id: int) -> None:
    """Per-worker reseed used as DataLoader.worker_init_fn."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# generator = torch.Generator()
# generator.manual_seed(42)
# loader = DataLoader(
#     dataset, batch_size=32, shuffle=True,
#     worker_init_fn=seed_worker,
#     generator=generator,
# )
```

Without these two arguments, every fold's data shuffle is
non-deterministic regardless of `set_global_seeds(42)`.

## fp16 / bf16: trade-offs

Mixed-precision speeds up inference 1.5–3× on modern GPUs but
introduces small numerical noise that bootstrap CIs absorb but bit-
identity does not. Two implications:

- **Calibrate at inference precision.** If you'll deploy in bf16, fit
  [`fit_temperature`](../api/calibration.md) on bf16
  logits, not fp32 logits cast to bf16.
- **Don't compare ECE across precision levels.** A 0.001–0.005 ECE
  delta is well within fp16/bf16 noise on moderate-size eval sets.

<!-- skip: next -->
```{code-cell}
# Requires: torch. Marked skip for Sybil.
# bf16 inference:
# self.model = self.model.to(dtype=torch.bfloat16)
# logits.float()  # cast back to fp32 BEFORE softmax for numerical stability
```

## Putting the LoRA scorer into the harness

Drop-in replacement for `_UniformBaseline` in the [PI
walkthrough](prompt_injection_walkthrough.md):

<!-- skip: next -->
```{code-cell}
# Requires: torch, transformers, peft. Marked skip for Sybil.
# from eval_toolkit import evaluate_folded, SourceDisjointKFoldSplitter
#
# scorer = LoRATransformerScorer(
#     base_model_name="microsoft/deberta-v3-base",
#     lora_path="my/adapter",
#     device="cuda",
#     batch_size=32,
# )
#
# result = evaluate_folded(
#     {"deberta_lora": scorer},
#     SourceDisjointKFoldSplitter(source_col="source", k=3, seed=42),
#     parent_slice,
#     run_id="lora-v1",
#     leakage_checks=[NormalizedFormLeakageCheck(), CrossSplitLeakageCheck()],
#     on_leakage="record",
# )
```

## Copy-paste minimal scorer (CPU, no LoRA, no PEFT)

If you just want to start scoring with a public injection-detector
checkpoint:

<!-- skip: next -->
```{code-cell}
# Requires: torch, transformers. Marked skip for Sybil.
# import os; os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# import torch
# torch.use_deterministic_algorithms(True, warn_only=True)
#
# scorer = TransformerScorer(
#     "protectai/deberta-v3-base-prompt-injection-v2",
#     device="cpu",
#     batch_size=8,
# )
# # Now plug into evaluate(...) as in the PI walkthrough.
```

## Pitfalls / Common mistakes

- **`CUBLAS_WORKSPACE_CONFIG` set after import.** Has no effect once
  the CUDA context exists. Set in `os.environ` BEFORE `import torch`.
- **Calling `predict_proba` outside `torch.no_grad()`.** Builds the
  autograd graph; OOMs fast on long inputs. The decorator above
  prevents this.
- **`model.train()` left on during eval.** Dropout is still active;
  every `predict_proba` call returns slightly different scores.
  `model.eval()` is mandatory.
- **Returning a torch tensor from `predict_proba`.** The harness
  expects `np.ndarray`. Always `.cpu().numpy()` at the end.
- **fp16 inference with `torch.softmax` on small logits.** Underflows
  to 0 in the tail; you lose calibration in the [0, 0.01] regime.
  Cast to fp32 *before* softmax.

## See also

- [extending.md §"Implementing a Scorer"](../extending.md#scorer)
- [methodology/calibration.md §"PyTorch & transformer specifics"
  ](../methodology/calibration.md#pytorch)
- [methodology/reproducibility.md §"PyTorch determinism"
  ](../methodology/reproducibility.md#pytorch-determinism)
- [PyTorch 2.8 reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- HuggingFace
  [`AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification)
  / [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
