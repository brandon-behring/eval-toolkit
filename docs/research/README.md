# eval-toolkit Research Dossier

<!-- AGENT-INDEX: top-level cross-cluster index for the eval-toolkit research dossier. Read this README first. -->

**Purpose:** Provide future Claude agents working in `eval-toolkit` with a verified, primary-source research dossier covering the library's full methodological footprint — statistical inference, data integrity, eval-ecosystem positioning, prompt-injection consumer-domain methodology, and datasets. Designed for dual consumption (humans + LLM agents).

**Last updated:** 2026-05-14.
**Total entries across all clusters:** 74 (24 inference + 17 data-integrity + 9 eval-ecosystem + 12 prompt-injection + 12 datasets).
**Audit coverage:** All 5 clusters have completed their planned audit rounds (inference: 2, data-integrity: 2, eval-ecosystem: 1, prompt-injection: 1, datasets: condensed).
**URL freshness:** 80 unique URLs across the dossier; 64 OK, 16 bot-blocked (paywall publishers), 0 broken — see `url-freshness-report.md`.

## How this dossier is organized

```
docs/research/
├── README.md                          # this file (cross-cluster index)
├── url-freshness-report.md            # URL liveness categorization
├── papers/
│   ├── inference/                     # 24 entries: bootstrap CIs, ROC variance, calibration, thresholds
│   ├── data-integrity/                # 17 entries: splits, leakage, dedup, contamination
│   ├── eval-ecosystem/                # 9 entries:  HELM, lm-eval, Inspect AI, NeurIPS checklist, Croissant
│   └── prompt-injection/              # 12 entries: attacks, defenses, OWASP, eval benchmarks
└── datasets/                          # 12 entries: PINT, AdvBench, HarmBench, UCI, C4, MMLU, Open-Prompt-Injection, etc.
```

Each `papers/<cluster>/` folder contains:
- `README.md` — agent-readable index with scope boundary, lookup recipes (~15–25 per cluster), glossary, audit-trail
- `01_<topic>.md` … `0K_<topic>.md` — 5-bullet-per-entry synthesis files (Source / Code / Mechanism / Result / Status)
- `bib_ledger.yml` — source-of-truth ledger
- `research_plan.md` — original scope + claim_family taxonomy
- `_dossier/` — raw 7-column dossier tables (for human review and audit trail)

The `datasets/` folder uses a 5-bullet-per-dataset variant (Source / Access / Schema / Size+License / Tasks).

## Scope: which clusters inform code vs. consumer repos

The dossier covers **the full methodological footprint that motivates this library and its consumers**, but not every cluster maps to code in `src/eval_toolkit/`. The split:

| Cluster | Maps to | Justification |
|---|---|---|
| **inference/** | `src/eval_toolkit/bootstrap.py`, `calibration.py`, `metrics.py`, `thresholds.py` | Core library: BCa/paired/CLT-corrected bootstrap, Platt/isotonic/temperature/beta calibration, ECE variants, threshold selectors. |
| **data-integrity/** | `src/eval_toolkit/splits.py`, `leakage.py`, `text_dedup.py`, `manifest.py` | Core library: stratified/time-series/group/nested splits, leakage taxonomy detection, MinHash/LSH/TF-IDF/embedding/Jaccard dedup, provenance tracking (including `RunManifest.contamination_flags` per `data-integrity/` § B2). |
| **eval-ecosystem/** | `src/eval_toolkit/manifest.py` + `README.md` positioning | Positioning material: NeurIPS reproducibility checklist alignment, Croissant metadata standard awareness. Some research informs design choices rather than translating to specific functions. |
| **prompt-injection/** | *Consumer repos (e.g., `prompt-injection-v4`)* | **Out of toolkit scope.** Documents PI attacks (GCG, PAIR), defenses (Self-Reminder, DataSentinel + PromptLocate), benchmarks (PINT/HarmBench/JailbreakBench/HackAPrompt), OWASP LLM01:2025 as reference material for downstream consumers. eval-toolkit provides only generic primitives (harness slicing, `attack_style` pass-through label); PI-specific loaders, refusal metrics, and threat-model selectors live in consumer repos. |
| **datasets/** | Mixed | Generic schemas (UCI Adult, Wisconsin Breast Cancer, OpenML, MMLU, C4) inform `loaders.py` shape recognition. PI dataset entries (PINT, AdvBench, HarmBench, JBB, Open-Prompt-Injection) are reference material for consumer repos. |

This split is intentional: eval-toolkit's value proposition is *reusable, domain-agnostic eval primitives*. Domain-specific code (prompt-injection eval methodology, fairness audits, conformal prediction, etc.) lives in consumer repos that depend on this library.

## Cluster index

| Cluster | What it covers | When to read |
|---|---|---|
| **[inference/](papers/inference/)** | Bootstrap CI methods (BCa, paired, DeLong correlated-ROC, K-fold CLT-corrected); probability calibration (Platt, isotonic, temperature, beta); ECE variants and debiasing; threshold selection theory (Lipton F1, Youden J, cost-sensitive Bayes-optimal); power analysis | You're working on the eval-toolkit's `bootstrap`, `calibration`, `metrics`, or `thresholds` modules. Foundational stats-inference math grounding. |
| **[data-integrity/](papers/data-integrity/)** | Splits / CV strategies (time-series, block, group, nested); leakage taxonomy (Kapoor & Narayanan); benchmark contamination detection; text deduplication algorithms (MinHash, LSH, SemDeDup); pre-training corpus dedup and memorization | You're working on the eval-toolkit's `splits`, `leakage`, or `text_dedup` modules. Or you need to justify why a particular guardrail exists. |
| **[eval-ecosystem/](papers/eval-ecosystem/)** | LLM eval harness frameworks (HELM, EleutherAI lm-evaluation-harness, AISI Inspect AI); NeurIPS reproducibility checklist; metadata standards (Datasheets, Model Cards, Croissant); benchmarking-methodology critiques | You're positioning eval-toolkit relative to ecosystem peers, aligning RunManifest to NeurIPS checklist, or implementing Croissant-compatible metadata. |
| **[prompt-injection/](papers/prompt-injection/)** | Prompt-injection attack taxonomy (direct/indirect/jailbreak); automated attacks (GCG, PAIR); defenses (Self-Reminder); eval benchmarks (PINT, HarmBench, JailbreakBench); OWASP LLM01:2025 industry standard | You're working on prompt-injection eval as a consumer-domain application of eval-toolkit. |
| **[datasets/](datasets/)** | Prompt-injection eval datasets (PINT, AdvBench, HarmBench Behaviors, JBB-Behaviors); classical binary-classification benchmarks (UCI Adult, Wisconsin Breast Cancer, OpenML); pretraining corpora examples (C4, RefinedWeb); contamination-target benchmarks (MMLU, HellaSwag) | You need a specific dataset — concrete URLs, schemas, licenses, sizes. |

## Cross-cluster glossary

Terms that appear in ≥2 clusters. Per-cluster glossaries cover terms specific to that cluster.

- **AUC / ROC AUC**: Area under the ROC curve. Foundational in `inference/` (variance, paired-difference tests, calibration metrics).
- **Bootstrap**: Resampling-based CI construction. Foundational in `inference/` § A1.
- **Calibration**: Predicted probabilities match observed frequencies. Defined in `inference/` § C1; ECE metrics in `inference/` § D1–D2.
- **Contamination (benchmark contamination)**: Train-test overlap where the model saw the eval set during training. `data-integrity/` § B2 covers detection methodology; `eval-ecosystem/` § C1 (Sainz et al. 2023) covers the per-benchmark reporting norm.
- **Cross-validation (CV)**: K-fold and variants. Strategies in `data-integrity/` § A1; variance theory in `inference/` § A2 (Bates et al. 2024).
- **Croissant**: ML-dataset metadata standard. Defined in `eval-ecosystem/` § B2; eval-toolkit's `loaders` aligns to it.
- **DeLong test**: Nonparametric paired test for correlated ROC AUCs. `inference/` § B1.
- **Dedup (deduplication)**: `data-integrity/` § C1 (algorithms) and § D1 (effects on LLM training). `datasets/` C4 and RefinedWeb are real-world examples.
- **ECE (Expected Calibration Error)**: Average gap between confidence and accuracy. Defined in `inference/` § C1 (Naeini 2015 modern form); debiasing in `inference/` § D2.
- **GCG (Greedy Coordinate Gradient)**: White-box transferable jailbreak attack. `prompt-injection/` § A2.
- **Holistic Evaluation (HELM)**: Multi-metric eval paradigm. `eval-ecosystem/` § A1.
- **Jailbreak**: Adversarial prompt bypassing LLM safety training. `prompt-injection/`.
- **Leakage**: Train-test information bleed. `data-integrity/` § B1 (Kapoor & Narayanan 2023 8-type taxonomy with L1.1–L3.3 codes).
- **MinHash / MinHash-LSH**: Constant-size fingerprint + locality-sensitive hashing for near-duplicate detection. `data-integrity/` § C1.
- **NeurIPS reproducibility checklist**: Submission-time reproducibility checklist. `eval-ecosystem/` § B1 (Pineau et al. 2021); eval-toolkit's `manifest` aligns to it.
- **OWASP LLM01:2025**: Top-ranked vulnerability in OWASP Top 10 for LLM Apps — prompt injection. `prompt-injection/` § C2.
- **PINT (Prompt Injection Test)**: Lakera's prompt-injection-detector benchmark. `prompt-injection/` § C1 (methodology) + `datasets/` (artifact).
- **Platt scaling**: 2-parameter sigmoid calibrator. `inference/` § C1.
- **Temperature scaling**: Single-parameter logit divisor for NN calibration. `inference/` § C1 (Guo et al. 2017).

## Lookup recipes (cross-cluster routing)

When a question crosses cluster boundaries:

- **"How do I report CIs for an eval-toolkit metric and avoid leakage?"** → `inference/` (CI methods) + `data-integrity/` (leakage taxonomy + nested CV).
- **"My eval harness reports per-benchmark contamination — what's the methodological basis?"** → `data-integrity/02_leakage_and_contamination.md` § B2 (Sainz et al. 2023) + `eval-ecosystem/03_methodology_critiques.md` (benchmark lottery).
- **"I'm writing a model card for a binary classifier with calibration and threshold results — primary refs?"** → `eval-ecosystem/02_reproducibility_and_metadata.md` § B2 (Mitchell et al. 2019 Model Cards) + `inference/04_calibration_metrics.md` § D1 (Brier decomposition, Murphy reliability/resolution) + `inference/05_thresholds_power_foundations.md` § E1 (threshold selection).
- **"How does prompt-injection eval relate to the toolkit's general binary classification eval?"** → `prompt-injection/README.md` (the domain layer) + `inference/` (CI machinery applied to safety/refusal rates) + `data-integrity/02_leakage_and_contamination.md` § B2 (benchmark contamination is the data-integrity guardrail for prompt-injection benchmarks).
- **"Why does my K-fold CV CI under-cover?"** → `inference/01_bootstrap_and_cv_variance.md` § A2 (Bates et al. 2024; Bengio & Grandvalet 2004).
- **"I'm using HarmBench / JailbreakBench — what's their methodological positioning?"** → `prompt-injection/03_benchmarks_and_standards.md` § C1 + `eval-ecosystem/03_methodology_critiques.md` (benchmark lottery — Dehghani et al. 2021).

## Companion lighter-weight reference

`docs/methodology/reading_list.md` is a 40+ reference list maintained as a lighter-weight companion. This dossier was built independently of it to serve as a cross-check; reconciliation is in `RECONCILIATION.md` (sibling file).

## Verification

- All 69 paper entries have `status: verified` in their cluster's `bib_ledger.yml`.
- All 11 dataset entries have URL-existence verification; per-entry license prose-check is the recommended next audit focus.
- 80 unique URLs HEAD-checked on 2026-05-14: 0 broken, 16 bot-blocked (paywall publishers — expected), 64 OK. See `url-freshness-report.md`.
- 6 total `/dossier-audit` rounds run across the 4 paper clusters (inference: 2; data-integrity: 2; eval-ecosystem: 1; prompt-injection: 1). Each round's findings are documented in the respective cluster's `README.md` under the `## Verification & limits` section.

## Attribution

Synthesized by the research_toolkit (`~/Claude/research_toolkit/`). Source URLs link to primary sources (arXiv, journal DOIs, conference proceedings, GitHub, HuggingFace, vendor blogs). Each cluster's `README.md` is the canonical entry point for that area.
