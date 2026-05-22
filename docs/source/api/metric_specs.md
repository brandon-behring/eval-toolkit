# `eval_toolkit.metric_specs`

First-party `MetricSpec` instances consumed by
{func}`~eval_toolkit.scorecard`. The namespace ships **threshold-free**
specs only (Decision R). Each unparameterized metric is a module-level
singleton; parameterized metrics (currently just the ECE family) are
factory callables with LRU-cached identity.

```{eval-rst}
.. currentmodule:: eval_toolkit.metric_specs

.. autosummary::
   :toctree: generated/metric_specs/
   :nosignatures:

   brier
   ece
   make_spec_name
   pr_auc
   roc_auc
```
