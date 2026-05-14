# Pre-training Corpus Deduplication and Downstream Effects — Synthesis

This file synthesizes D1 (the measured effects of applying dedup to LLM pre-training corpora). Companion raw-table dossier: `_dossier/04_pretrain_dedup_effects.md`. The dedup algorithms themselves are in `03_text_dedup_methods.md`.

---

## D1. Pre-training dedup and memorization

- **Extracting Training Data from Large Language Models** — Carlini et al. (USENIX Security 2021, Distinguished Paper).
  - **Source:** https://arxiv.org/abs/2012.07805
  - **Code:** —
  - **Mechanism:** Demonstrates training-data extraction attacks against GPT-2; recovers hundreds of verbatim sequences from the training set under adversarial prompting, including PII.
  - **Result:** First systematic empirical demonstration that LLMs memorize and emit training data verbatim; motivates pre-training dedup as a memorization-mitigation strategy. Foundational for the "dedup helps privacy" argument.
  - **Status:** Verified (no widely-known repo).

- **Deduplicating Training Data Makes Language Models Better** — Lee et al. (ACL 2022).
  - **Source:** https://arxiv.org/abs/2107.06499
  - **Code:** —
  - **Mechanism:** Develops two dedup tools (ExactSubstr for substring-level exact and NearDup using MinHash for fuzzy) and applies them to C4. Finds substantial duplication.
  - **Result:** Empirically shows deduplicated training reduces verbatim emission roughly tenfold and reduces training compute for equivalent eval performance. Canonical modern reference for "dedup helps LLM training quality."
  - **Status:** Verified (no widely-known repo).

- **Quantifying Memorization Across Neural Language Models** — Carlini et al. (ICLR 2023).
  - **Source:** https://arxiv.org/abs/2202.07646
  - **Code:** https://github.com/ethz-spylab/lm_memorization_data
  - **Mechanism:** Measures verbatim-memorization rate as a function of model capacity, example duplication count in training, and prompt-context length.
  - **Result:** Establishes three log-linear scaling laws: memorization grows log-linearly with (1) model capacity, (2) duplication count, and (3) context length. Directly motivates aggressive dedup — duplication is a primary driver of memorization.
  - **Status:** Verified.

- **The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only** — Penedo et al. (NeurIPS 2023 Datasets and Benchmarks).
  - **Source:** https://arxiv.org/abs/2306.01116
  - **Code:** —
  - **Mechanism:** Builds a 5-trillion-token web-only corpus with aggressive fuzzy + exact dedup pipelines; releases a 500B-token public subset and trains 1.3B/7.5B-parameter LLMs on it.
  - **Result:** Production-scale demonstration that properly filtered + deduplicated web data alone trains models competitive with curated mixtures (such as The Pile); methodology reference for large-scale dedup pipelines.
  - **Status:** Verified (no widely-known repo).
