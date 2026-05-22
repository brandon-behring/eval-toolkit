# `eval_toolkit.losses`

Differentiable losses for prompt-injection detector training.
{class}`RecallAtLowFPR` is the Meta Prompt Guard 2 training recipe: a
differentiable approximation of recall-at-fixed-FPR that optimizes
detector ranking at a constrained operating point.

Optional dependency: `pip install eval-toolkit[losses]` (installs torch).

```{eval-rst}
.. currentmodule:: eval_toolkit.losses

.. autosummary::
   :toctree: generated/losses/
   :nosignatures:

   RecallAtLowFPR
```
