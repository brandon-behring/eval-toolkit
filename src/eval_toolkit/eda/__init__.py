"""``eval_toolkit.eda`` — EDA-first dataset integrity gating (Tier-2 surface).

This subpackage is the **Job-1 integrity gate** of an EDA-first research
program: thin, composable, torch-free per-split profiling + dataset-soundness
gates, built by reusing the v1.4.0 :mod:`eval_toolkit.leakage`,
:mod:`~eval_toolkit.text_dedup`, :mod:`~eval_toolkit.claims`, and
:mod:`~eval_toolkit.artifacts` primitives.

Stability tier
--------------
Public access is ``eval_toolkit.eda.*`` — **Tier-2** per ADR 0003. This layer
is intentionally evolvable and is **not** part of the v2.0-frozen top-level
:mod:`eval_toolkit` surface; nothing here is added to the package-root
``_EXPORTS`` / ``__all__``. Import explicitly::

    from eval_toolkit.eda import audit_dataset, DataAudit, SplitSummary

Scope (deliberately narrow)
--------------------------
Integrity gating only: row counts, class balance, text-length quantiles,
dedup / cross-split leakage. **No** embeddings, semantic similarity,
contamination scoring, or UMAP — those distribution-shift concerns are
deferred to a future ``distribution_shift`` module.
"""

from __future__ import annotations

from eval_toolkit.eda.data_audit import (
    DEFAULT_MAX_NEG_POS_RATIO,
    DEFAULT_MIN_NEG_POS_RATIO,
    DEFAULT_PCT_OVER_CONTEXT_THRESHOLD,
    EDA_AUDIT_SCHEMA_VERSION,
    DataAudit,
    SplitSummary,
    Tokenizer,
    audit_dataset,
    class_balance,
    length_quantiles,
    summarize_split,
)

__all__ = [
    "DEFAULT_MAX_NEG_POS_RATIO",
    "DEFAULT_MIN_NEG_POS_RATIO",
    "DEFAULT_PCT_OVER_CONTEXT_THRESHOLD",
    "EDA_AUDIT_SCHEMA_VERSION",
    "DataAudit",
    "SplitSummary",
    "Tokenizer",
    "audit_dataset",
    "class_balance",
    "length_quantiles",
    "summarize_split",
]
