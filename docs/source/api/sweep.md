# `sweep` — unified text-transform enumeration (v0.47)

The v0.47 sweep consolidation (Decision K + Decision D) replaces the two
per-module ``sweep`` functions (``adversarial.sweep`` and
``preprocessing.sweep``) with a single top-level entry point that accepts
any object satisfying the {class}`~eval_toolkit.TextTransform` Protocol.
Defence strategies ({class}`~eval_toolkit.DelimitVariant`,
{class}`~eval_toolkit.DatamarkVariant`, {class}`~eval_toolkit.EncodeVariant`)
and attack strategies (all 12 character-injection dataclasses in
{mod}`eval_toolkit.adversarial`) compose freely in the same call.

```{eval-rst}
.. currentmodule:: eval_toolkit

.. autosummary::
   :toctree: generated/sweep/
   :nosignatures:

   sweep
```
