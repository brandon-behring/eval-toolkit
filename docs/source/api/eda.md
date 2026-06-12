# `eval_toolkit.eda`

EDA-first dataset integrity gating — thin, composable, torch-free
per-split profiling and dataset-soundness diagnostics. **Tier-2 surface
per ADR 0003**: import explicitly (`from eval_toolkit.eda import
audit_dataset`); nothing here is in the package-root `__all__`, and the
layer is intentionally evolvable within v1.x.

Three jobs, one module each (plus obfuscation profiling):

- **Job-1 integrity gate** (`data_audit` + `obfuscation`): row counts,
  class balance, text-length quantiles, dedup / cross-split leakage,
  obfuscation prevalence.
- **Job-2 lexical shortcut diagnostics** (`lexical_association`):
  weighted log-odds + PMI and partial-input / competency baselines —
  NumPy + scikit-learn only.
- **Job-3 distribution shift** (`distribution_shift`):
  proxy-A-distance, MMD (permutation-tested), and kNN purity over
  **feature matrices** — embed text first with
  `eval_toolkit.embeddings.make_minilm_embedder` (the optional
  `[embeddings]` extra) or any vectorizer.

## Job-1: data audit

```{eval-rst}
.. currentmodule:: eval_toolkit.eda

.. autosummary::
   :toctree: generated/eda/
   :nosignatures:

   audit_dataset
   class_balance
   length_quantiles
   summarize_split
   DataAudit
   SplitSummary
   Tokenizer
```

## Job-1: obfuscation profiling

```{eval-rst}
.. autosummary::
   :toctree: generated/eda/
   :nosignatures:

   analyze_obfuscation
   count_invisible_chars
   has_high_entropy_alnum_run
   has_rot13_marker
   is_leeted_token
   leetspeak_counts
   nfkc_changed
   nfkc_char_delta
   shannon_entropy
   ObfuscationProfile
```

## Job-2: lexical association

```{eval-rst}
.. autosummary::
   :toctree: generated/eda/
   :nosignatures:

   class_lexical_association
   competency_baselines
   default_tokenizer
   weighted_log_odds
   BaselineScore
   CompetencyResult
   LexicalAssociationResult
   StrTokenizer
```

## Job-3: distribution shift

```{eval-rst}
.. autosummary::
   :toctree: generated/eda/
   :nosignatures:

   distribution_shift
   knn_purity
   maximum_mean_discrepancy
   median_bandwidth
   proxy_a_distance
   DistributionShiftResult
   KnnPurityResult
   MmdResult
   PadResult
```

## Constants

```{eval-rst}
.. autosummary::
   :toctree: generated/eda/
   :nosignatures:

   BASE64_ENTROPY_THRESHOLD
   DEFAULT_CHAR_NGRAM_RANGE
   DEFAULT_KNN_K
   DEFAULT_MAX_NEG_POS_RATIO
   DEFAULT_MIN_COUNT
   DEFAULT_MIN_NEG_POS_RATIO
   DEFAULT_MMD_PERMUTATIONS
   DEFAULT_PAD_C
   DEFAULT_PAD_FOLDS
   DEFAULT_PCT_OVER_CONTEXT_THRESHOLD
   DEFAULT_PRIOR_SCALE
   EDA_AUDIT_SCHEMA_VERSION
   HEX_ENTROPY_THRESHOLD
```
