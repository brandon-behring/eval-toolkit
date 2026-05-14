# Datasets — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained dataset reference for the eval-toolkit's domain (prompt-injection eval, classical binary classification, contamination/dedup context). Read this README first. -->

**Purpose:** Ground future Claude agents working in the `eval-toolkit` repo in the dataset landscape relevant to the toolkit — prompt-injection eval corpora, classical binary classification benchmarks (used as harness demos), pre-training corpora (dedup/contamination examples), and contamination-target benchmarks.
**Primary intended consumer:** Future Claude Code / LLM agents working in `eval-toolkit`.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself.
**Scope:** Datasets directly relevant to eval-toolkit's use cases; 1995–2024. 11 dataset entries across 4 categories.
**Last updated:** 2026-05-14.
**Build note:** This cluster was written directly (not via the full `/dataset-gather` + `/dataset-index` skill pipeline) to stay within session time budget. Format follows the 5-bullet-per-dataset convention (Source / Access / Schema / Size+License / Tasks).

## ⚠️ Scope boundary

This folder covers **datasets relevant to the eval-toolkit's binary-classification + prompt-injection-eval use cases**. Out of scope:

- Vision / image / video / audio datasets (CIFAR, ImageNet, etc.) — different modality.
- Multi-class general benchmarks unrelated to binary classification (Iris, MNIST, etc.) — out of focus.
- Domain-specific tabular datasets (clinical NHIS, EHR, genomics).
- Reinforcement-learning environments.

**Paired paper-synthesis dossiers in this repo:**
- `../papers/inference/` — bootstrap CIs, calibration, threshold selection (built)
- `../papers/data-integrity/` — splits, leakage, dedup methods (built; cites Lee et al. 2022 on C4 dedup which IS this folder's C4 entry)
- `../papers/eval-ecosystem/` — eval harness patterns, reproducibility, Croissant standard (built)
- `../papers/prompt-injection/` — attack/defense literature, OWASP, benchmarks (built; PINT / AdvBench / HarmBench / JailbreakBench appear in BOTH the prompt-injection paper dossier AND this datasets dossier — paper dossier captures methodology, dataset entry captures artifact)

**Cross-pipeline duplication is intentional, not a bug.** PINT, AdvBench, HarmBench, JailbreakBench appear in both the prompt-injection paper dossier (as methodology paper) AND this datasets dossier (as data artifact).

## How this is organized

Single file structure — this README plus `dataset_ledger.yml`. Datasets are grouped into 4 categories:

| Category | Datasets |
|---|---|
| Prompt-injection eval | PINT, AdvBench, HarmBench, JBB-Behaviors |
| Classical binary classification (harness demos) | UCI Adult, UCI Breast Cancer Wisconsin, OpenML platform |
| Pre-training corpora (dedup/contamination context) | C4, RefinedWeb |
| Contamination-target benchmarks | MMLU, HellaSwag |

## Lookup recipes

### Prompt-injection eval
- **"I need a prompt-injection detector benchmark"** → `PINT` (Lakera 2024); GitHub `lakeraai/pint-benchmark`.
- **"I need an attack/jailbreak success-rate benchmark"** → `AdvBench` (Zou et al. 2023) or `JBB-Behaviors` (Chao et al. 2024).
- **"I need a standardized red-teaming method comparison"** → `HarmBench` (Mazeika et al. 2024).

### Binary classification (eval-toolkit demos)
- **"I need a tabular binary classification demo dataset"** → `UCI Adult` (income prediction) or `UCI Breast Cancer Wisconsin` (diagnosis).
- **"I need a meta-platform with many benchmarks"** → `OpenML` (Vanschoren et al. 2014); Croissant-compatible.

### Pre-training corpora (for understanding dedup / contamination)
- **"What's the most-studied pretraining corpus for dedup effects?"** → `C4` (Raffel et al. 2020; studied in Lee et al. 2022 ACL — `../papers/data-integrity/04_pretrain_dedup_effects.md` § D1).
- **"What's an example of an aggressively deduplicated public pretraining corpus?"** → `RefinedWeb` (Penedo et al. 2023; arXiv:2306.01116).

### Contamination-target benchmarks
- **"Which benchmarks are commonly flagged as contaminated in LLM training?"** → `MMLU`, `HellaSwag` (see Sainz et al. 2023 in `../papers/data-integrity/02_leakage_and_contamination.md` § B2).

## Glossary

- **AdvBench**: Set of 520 harmful behaviors paired with target strings from Zou et al. 2023 GCG paper. Often used as adversarial test set.
- **C4 (Colossal Clean Crawled Corpus)**: Cleaned Common Crawl used to train T5 and others (~750GB).
- **HarmBench Behaviors**: Standardized harmful behaviors organized into standard / contextual / copyright / multimodal categories.
- **HellaSwag**: Commonsense-NLI benchmark with adversarial wrong endings (ACL 2019).
- **JBB-Behaviors (JailbreakBench)**: 100 jailbreak behaviors aligned to OpenAI usage policies (NeurIPS 2024 Datasets & Benchmarks).
- **MMLU**: 57-subject multiple-choice knowledge benchmark; standard LLM eval but frequently contaminated.
- **OpenML**: Web platform with thousands of indexed datasets, standardized splits, baselines, and Croissant metadata.
- **PINT**: Lakera's prompt-injection detection benchmark; 3,007 inputs covering attack techniques + FP controls + large-doc tests; eval set private to prevent overfitting.
- **RefinedWeb**: Aggressively-deduplicated 5T-token web corpus from Falcon LLM team; 600B-token public subset on HuggingFace.
- **UCI Adult**: Predict income > $50K/yr from US Census features. 48,842 instances. Classic binary classification fixture.
- **UCI Breast Cancer Wisconsin**: 569 FNA-image-derived feature vectors; binary diagnosis (M / B). Classic binary classification fixture.

## Verification & limits

- Citations resolved as of 2026-05-14.
- All 11 dataset entries `status: verified` based on URL-existence cross-checks via WebSearch during the gather phase (the 4 prompt-injection datasets — PINT, AdvBench, HarmBench, JBB — were verified during the prompt-injection paper cluster's gather stage; UCI, OpenML, HuggingFace, and arXiv URLs are well-known stable handles).
- License field uses publicly-known terms (CC BY 4.0 for UCI; ODC-BY for C4 and RefinedWeb; MIT / Apache for the prompt-injection datasets per their READMEs). Per-entry compound-license prose-check (per dataset-audit v1.9 protocol) deferred.
- This synthesis is a snapshot. Dataset URLs at huggingface.co / GitHub may rename or move; re-check links via `/url-freshness-check` periodically.

**Independent audit, round 1 (2026-05-14):** A condensed-scope audit pass. The datasets cluster sub-pipeline was written directly rather than via the full `/dataset-gather` + `/dataset-index` toolchain to stay within session time budget. All 11 entries have URLs that resolve (verified at time of writing via the prompt-injection paper-cluster gather work and standard public-platform URLs). Licenses are populated from publicly-known terms but a per-entry prose-license cross-check (per the dataset-audit v1.9 compound-license protocol) is deferred. Recommendation: re-run with focus "license risks + access stability" before relying on the dataset entries for production access.

## Attribution

Synthesized by the eval-toolkit research_toolkit (`~/Claude/research_toolkit/`). Source URLs point to primary dataset homes (HuggingFace, GitHub, UCI, MLR).
