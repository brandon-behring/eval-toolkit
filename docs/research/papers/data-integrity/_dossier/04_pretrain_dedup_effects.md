# Pre-training Corpus Deduplication and Downstream Effects

This file covers D1 (the measured effects of applying dedup to LLM pre-training corpora — Lee et al., Penedo et al., Carlini et al.). The dedup algorithms themselves are in `03_text_dedup_methods.md`.

---

## D1. Pre-training dedup and memorization

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Extracting Training Data from Large Language Models | Carlini et al. (2021) | USENIX Security 2021 (Distinguished Paper) | arXiv:2012.07805 | — | Demonstrates training-data extraction attacks against GPT-2; recovers hundreds of verbatim sequences including PII | First systematic empirical demonstration that LLMs memorize and emit training data verbatim under adversarial prompting; motivates pre-training dedup as a memorization-mitigation strategy |
| Deduplicating Training Data Makes Language Models Better | Lee et al. (2022) | ACL 2022 | arXiv:2107.06499 | — | Develops two dedup tools (ExactSubstr and NearDup/MinHash); applies to C4 and finds substantial duplication | Empirically shows that deduplicated training reduces verbatim emission by ~10x and reduces training compute for equivalent eval performance. Canonical reference for "dedup helps LLM training" |
| Quantifying Memorization Across Neural Language Models | Carlini et al. (2023) | ICLR 2023 | arXiv:2202.07646 | ethz-spylab/lm_memorization_data | Three log-linear scaling laws for memorization: with model capacity, with example duplication count, and with prompt-context length | Establishes that example duplication count is a primary driver of memorization — directly motivates aggressive dedup. Most-cited modern reference for the dedup → memorization-reduction causal link |
| The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only | Penedo et al. (2023) | NeurIPS 2023 Datasets and Benchmarks | arXiv:2306.01116 | — | Builds a 5-trillion-token web-only corpus with aggressive fuzzy + exact dedup; releases a 500B-token public subset | Production-scale demonstration that properly filtered + deduplicated web data alone trains models competitive with curated mixtures (The Pile); methodology reference for large-scale dedup pipelines |
