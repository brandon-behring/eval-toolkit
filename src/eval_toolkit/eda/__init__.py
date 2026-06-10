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

Scope
-----
- **Job-1 integrity gate** (``data_audit`` + ``obfuscation``): row counts, class
  balance, text-length quantiles, dedup / cross-split leakage, obfuscation
  prevalence.
- **Job-2 lexical shortcut diagnostics** (``lexical_association``): weighted
  log-odds + PMI (C1) and partial-input / competency baselines (C2) — torch-free
  (NumPy + scikit-learn).
- **Job-3 distribution shift** (``distribution_shift``): proxy-A-distance, MMD
  (permutation-tested), and kNN purity (E1) — operates on feature matrices, so
  still base-install-safe (NumPy + SciPy + scikit-learn).

The shift functions take **feature matrices**, not text — embed first with
:func:`eval_toolkit.embeddings.make_minilm_embedder` (the optional
``[embeddings]`` extra) or any vectorizer. UMAP / 2-D projections stay caller-side.
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
from eval_toolkit.eda.distribution_shift import (
    DEFAULT_KNN_K,
    DEFAULT_MMD_PERMUTATIONS,
    DEFAULT_PAD_C,
    DEFAULT_PAD_FOLDS,
    DistributionShiftResult,
    KnnPurityResult,
    MmdResult,
    PadResult,
    distribution_shift,
    knn_purity,
    maximum_mean_discrepancy,
    median_bandwidth,
    proxy_a_distance,
)
from eval_toolkit.eda.lexical_association import (
    DEFAULT_CHAR_NGRAM_RANGE,
    DEFAULT_MIN_COUNT,
    DEFAULT_PRIOR_SCALE,
    BaselineScore,
    CompetencyResult,
    LexicalAssociationResult,
    StrTokenizer,
    class_lexical_association,
    competency_baselines,
    default_tokenizer,
    weighted_log_odds,
)
from eval_toolkit.eda.obfuscation import (
    BASE64_ENTROPY_THRESHOLD,
    HEX_ENTROPY_THRESHOLD,
    ObfuscationProfile,
    analyze_obfuscation,
    count_invisible_chars,
    has_high_entropy_alnum_run,
    has_rot13_marker,
    is_leeted_token,
    leetspeak_counts,
    nfkc_changed,
    nfkc_char_delta,
    shannon_entropy,
)

__all__ = [
    # --- constants ---
    "BASE64_ENTROPY_THRESHOLD",
    "DEFAULT_CHAR_NGRAM_RANGE",
    "DEFAULT_KNN_K",
    "DEFAULT_MAX_NEG_POS_RATIO",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_MIN_NEG_POS_RATIO",
    "DEFAULT_MMD_PERMUTATIONS",
    "DEFAULT_PAD_C",
    "DEFAULT_PAD_FOLDS",
    "DEFAULT_PCT_OVER_CONTEXT_THRESHOLD",
    "DEFAULT_PRIOR_SCALE",
    "EDA_AUDIT_SCHEMA_VERSION",
    "HEX_ENTROPY_THRESHOLD",
    # --- classes / type aliases ---
    "BaselineScore",
    "CompetencyResult",
    "DataAudit",
    "DistributionShiftResult",
    "KnnPurityResult",
    "LexicalAssociationResult",
    "MmdResult",
    "ObfuscationProfile",
    "PadResult",
    "SplitSummary",
    "StrTokenizer",
    "Tokenizer",
    # --- functions ---
    "analyze_obfuscation",
    "audit_dataset",
    "class_balance",
    "class_lexical_association",
    "competency_baselines",
    "count_invisible_chars",
    "default_tokenizer",
    "distribution_shift",
    "has_high_entropy_alnum_run",
    "has_rot13_marker",
    "is_leeted_token",
    "knn_purity",
    "leetspeak_counts",
    "length_quantiles",
    "maximum_mean_discrepancy",
    "median_bandwidth",
    "nfkc_char_delta",
    "nfkc_changed",
    "proxy_a_distance",
    "shannon_entropy",
    "summarize_split",
    "weighted_log_odds",
]
