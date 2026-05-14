# Research Plan: Data integrity for ML evaluation

This research grounds the `eval-toolkit` library's data-integrity machinery — splits/CV strategies, leakage taxonomy, text deduplication, train-test contamination detection — in primary literature so future Claude agents can reason about why each guardrail exists and how it should be applied. Narrow scope: ~16–20 papers across 6 sub-areas. Covers tabular and text data; LLM-specific eval methodology is partially covered (the contamination sub-area) but the broader LLM-eval ecosystem is out of scope.

## Sub-areas

- A1. Splits and CV strategies (k-fold, stratified, group, time-series, source-disjoint)
  - Source types: monograph, journal (JMLR, JASA), conference (NeurIPS, KDD), arXiv
  - Notes: General CV strategies for binary classification. Excludes bootstrap-CV hybrids (covered in `../inference/` § A1). The Bates et al. 2024 paper on CV variance is also in inference; not duplicated here.

- A2. Nested CV and hyperparameter selection
  - Source types: journal (JMLR), arXiv
  - Notes: Why HP tuning on the same fold as evaluation biases performance estimates. Cawley & Talbot 2010 is foundational. Excludes Bayesian optimization details.

- A3. Data leakage taxonomy and detection
  - Source types: arXiv, journal (Patterns, Nature Machine Intelligence)
  - Notes: Kapoor & Narayanan 2023 catalog of leakage modes. Includes target leakage, train-test contamination, group leakage. Excludes adversarial-eval-specific leakage (in `../prompt-injection/`).

- A4. Benchmark contamination measurement
  - Source types: arXiv, vendor blog (OpenAI, Meta), conference (ACL, NeurIPS)
  - Notes: LLM-specific train-test contamination detection. Sainz et al. 2023, Roberts et al. 2023, OpenAI's GPT contamination disclosures. Excludes general LLM eval methodology (in `../eval-ecosystem/`).

- A5. Text deduplication methods
  - Source types: conference (STOC, SIGMOD, ACL), arXiv
  - Notes: Algorithms — MinHash (Broder 1997), LSH (Indyk & Motwani 1998), SemDeDup (Abbas et al. 2023), TF-IDF cosine, exact-hash. Foundation for the eval-toolkit `text_dedup` module.

- A6. Pre-training corpus deduplication and its effect on downstream models
  - Source types: arXiv, conference (ACL, ICLR, NeurIPS)
  - Notes: Lee et al. ACL 2022 on Deduplicating Training Data; Penedo et al. 2023 RefinedWeb; Tirumala et al. SemDeDup paper. Excludes general pre-training methodology.

## Out-of-scope

- General LLM evaluation harness patterns (HELM, lm-eval, Inspect AI) — covered in `../eval-ecosystem/`.
- Prompt-injection-specific contamination (adversarial-eval slice taxonomy) — covered in `../prompt-injection/`.
- Bootstrap CIs and ROC variance — covered in `../inference/`.
- Privacy-preserving deduplication / differential privacy — different concern (privacy vs eval integrity).
- Data-quality cleaning beyond dedup (typo correction, schema normalization) — out of scope.
- Active learning / curriculum learning split design — different optimization.
- Federated-learning data-isolation — different threat model.

## Claim family taxonomy

- `splits_methodology` — strategies for partitioning data into train/val/test (stratified, group, time-series, source-disjoint, holdout)
- `nested_cv` — nested cross-validation for unbiased HP-tuned model evaluation
- `leakage_taxonomy` — taxonomy of leakage modes; survey-style or methodology-of-detection papers
- `benchmark_contamination` — detection or quantification of train-test contamination on standard benchmarks
- `text_dedup_method` — algorithms for detecting near-duplicate text (MinHash, LSH, SemDeDup, TF-IDF, embedding)
- `pretrain_dedup` — application of dedup to LLM pre-training corpora and its measured effect
- `foundational_text` — textbooks and monographs spanning the cluster (Hastie et al. ESL ch. 7 etc.)

## Known landmark papers

- `broder1997minhash` — Broder 1997 "On the resemblance and containment of documents" — MinHash foundational paper.
- `indyk1998lsh` — Indyk & Motwani 1998 "Approximate nearest neighbors: towards removing the curse of dimensionality" — locality-sensitive hashing.
- `cawley2010overfitting` — Cawley & Talbot 2010 "On over-fitting in model selection and subsequent selection bias in performance evaluation" — nested CV foundational.
- `kapoor2023leakage` — Kapoor & Narayanan 2023 "Leakage and the reproducibility crisis in machine-learning-based science" — leakage taxonomy.
- `lee2022deduplicating` — Lee et al. 2022 "Deduplicating Training Data Makes Language Models Better" — empirical dedup effect on LLM training.
- `penedo2023refinedweb` — Penedo et al. 2023 "The RefinedWeb Dataset for Falcon LLM" — large-scale dedup methodology for pre-training.
- `abbas2023semdedup` — Abbas et al. 2023 "SemDeDup: Data-efficient learning at web-scale through semantic deduplication" — semantic dedup.
- `sainz2023nlp` — Sainz et al. 2023 "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark" — contamination measurement.
