# `eval_toolkit.protocols`

Lightweight, low-dependency-surface Protocols intentionally kept here
so adapter authors can type-check against them without importing pandas,
matplotlib, or filesystem-oriented helpers. The 10 strict Tier-2
Protocols of the v1.x stability contract live in their topic modules
(e.g., `LeakageCheck` in `leakage.py`, `MetricSpec` in `scorecards.py`,
`SimilarityStrategy` in `text_dedup.py`)
to avoid pulling heavy dependencies into this module. See
[Strict Tier-2 Protocols at v1.0](strict_tier2_protocols.md) for the
canonical enumeration + import paths.

```{eval-rst}
.. currentmodule:: eval_toolkit.protocols

.. autosummary::
   :toctree: generated/protocols/
   :nosignatures:

   EvalSliceLike
   PredictionReader
   Scorer
   SliceAwareScorer
   TextTransform
   Versioned
```
