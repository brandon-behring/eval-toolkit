# `eval_toolkit.probes`

Activation probes — linear classifiers over transformer hidden states.
{class}`ActivationDeltaProbe` is a port of the TaskTracker methodology
(Abdelnabi et al. 2024, arXiv:2406.00799) for prompt-injection detection
via linear probing of activation deltas at a chosen transformer layer.

Optional dependency: `pip install eval-toolkit[probes]` (installs torch
+ transformers).

```{eval-rst}
.. currentmodule:: eval_toolkit.probes

.. autosummary::
   :toctree: generated/probes/
   :nosignatures:

   ActivationDeltaProbe
   ActivationExtractor
   Probe
```
