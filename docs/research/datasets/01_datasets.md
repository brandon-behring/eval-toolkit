# Datasets — Detail (5-bullet entries)

5-bullet schema for datasets: Source / Access / Schema / Size+License / Tasks. Companion ledger: `dataset_ledger.yml`. Cross-cluster context in `README.md`.

---

## A1. Prompt-injection eval datasets

- **PINT Benchmark (Prompt Injection Test)** — Lakera AI (2024).
  - **Source:** https://github.com/lakeraai/pint-benchmark
  - **Access:** Public code on GitHub; eval set is private (held by Lakera to prevent overfitting).
  - **Schema:** Inputs labeled by attack technique category + false-positive controls + large-doc handling tests.
  - **Size+License:** ~3,007 English inputs; benchmark code MIT-licensed.
  - **Tasks:** Prompt-injection detection / classifier evaluation; not for training.

- **AdvBench (Harmful Behaviors)** — Zou et al. (2023).
  - **Source:** https://github.com/llm-attacks/llm-attacks
  - **Access:** Public CSV bundled with the GCG paper code.
  - **Schema:** CSV with `goal` (harmful prompt) + `target` (affirmative response) columns.
  - **Size+License:** 520 behaviors; MIT-licensed.
  - **Tasks:** Jailbreak attack-success-rate evaluation; used as the standard test set in the GCG paper and many follow-ups.

- **HarmBench Behaviors and Targets** — Mazeika et al. (2024).
  - **Source:** https://github.com/centerforaisafety/HarmBench
  - **Access:** Public code + behaviors on GitHub; HuggingFace mirror at `walledai/HarmBench`.
  - **Schema:** Behaviors organized into four functional groups: standard, contextual, copyright, multimodal.
  - **Size+License:** Standardized harmful-behavior set; MIT-licensed.
  - **Tasks:** Automated red-teaming method comparison (18 methods × 33 LLMs in original eval); robust-refusal evaluation.

- **JBB-Behaviors (JailbreakBench)** — Chao et al. (2024).
  - **Source:** https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
  - **Access:** Public on HuggingFace and GitHub (`JailbreakBench/jailbreakbench`).
  - **Schema:** 100 jailbreak behaviors aligned to OpenAI usage policies; both original and sourced from prior work.
  - **Size+License:** 100 behaviors; permissive license (per JailbreakBench repo).
  - **Tasks:** Standardized jailbreak attack/defense evaluation with leaderboard tracking.

## A2. Classical binary-classification benchmarks

- **Adult (Census Income)** — Becker & Kohavi (UCI 1996).
  - **Source:** https://archive.ics.uci.edu/dataset/2/adult
  - **Access:** Public on UCI Machine Learning Repository.
  - **Schema:** 14 features (categorical + continuous); binary label: income > $50K/yr.
  - **Size+License:** 48,842 instances; CC BY 4.0.
  - **Tasks:** Binary classification (income prediction); standard demo for fairness-aware ML (protected attributes: race, sex).

- **Breast Cancer Wisconsin (Diagnostic)** — Wolberg, Mangasarian, Street & Street (UCI 1995).
  - **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  - **Access:** Public on UCI Machine Learning Repository.
  - **Schema:** 30 numerical features extracted from FNA images; binary label: malignant (M) vs benign (B).
  - **Size+License:** 569 instances; CC BY 4.0.
  - **Tasks:** Binary classification (diagnosis); classic harness demo for calibration + threshold-selection examples.

- **OpenML — Open Machine Learning Platform** — Vanschoren et al. (2014).
  - **Source:** https://www.openml.org/
  - **Access:** Public web platform; Python API (`openml` package); REST API.
  - **Schema:** Datasets + tasks + runs with standardized splits and baselines; Croissant-compatible metadata.
  - **Size+License:** 20,000+ datasets indexed; per-dataset licenses vary.
  - **Tasks:** Meta-platform — use to source binary classification benchmarks with standardized CV splits.

## A3. Pre-training corpora (dedup / contamination context)

- **C4 (Colossal Clean Crawled Corpus)** — Raffel et al. (T5, 2020); HuggingFace mirror.
  - **Source:** https://huggingface.co/datasets/allenai/c4
  - **Access:** Public on HuggingFace (multiple variants: `en`, `multilingual`, `realnewslike`, `webtextlike`).
  - **Schema:** Cleaned, sentence-split web pages from Common Crawl.
  - **Size+License:** ~750GB English text; ODC-BY license.
  - **Tasks:** Pretraining corpus; subject of Lee et al. 2022 ACL dedup study (61-word sentence repeated 60,000+ times pre-dedup).

- **RefinedWeb (Falcon LLM pretraining corpus, public subset)** — Penedo et al. (2023).
  - **Source:** https://huggingface.co/datasets/tiiuae/falcon-refinedweb
  - **Access:** Public on HuggingFace.
  - **Schema:** Plain text from Common Crawl after heavy fuzzy + exact dedup pipeline.
  - **Size+License:** ~600B-token public subset (5T-token corpus is internal); ODC-BY license.
  - **Tasks:** Pretraining; demonstrates that web-only deduplicated data can match curated mixtures (per Penedo et al. 2023 paper).

## A4. Contamination-target benchmarks

- **MMLU (Massive Multitask Language Understanding)** — Hendrycks et al. (ICLR 2021).
  - **Source:** https://huggingface.co/datasets/cais/mmlu
  - **Access:** Public on HuggingFace.
  - **Schema:** 57 subjects, 4-way multiple choice across humanities, STEM, social sciences.
  - **Size+License:** ~15,908 questions; Apache-2.0.
  - **Tasks:** Standard LLM knowledge eval. Frequently flagged as contaminated in modern LLMs (per Sainz et al. 2023 — see `../papers/data-integrity/02_leakage_and_contamination.md` § B2).

- **HellaSwag** — Zellers et al. (ACL 2019).
  - **Source:** https://huggingface.co/datasets/Rowan/hellaswag
  - **Access:** Public on HuggingFace.
  - **Schema:** Commonsense NLI with adversarially generated wrong endings.
  - **Size+License:** ~70K examples; MIT.
  - **Tasks:** Commonsense reasoning eval; standard in HELM and lm-evaluation-harness. Frequently flagged as contaminated.
