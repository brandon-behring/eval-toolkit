# Data Integrity for ML Evaluation — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained reference for data-integrity guardrails (splits / CV / leakage / dedup / contamination) used by the eval-toolkit library. Read this README first. -->

**Purpose:** Ground future Claude agents working in the `eval-toolkit` repo in the primary literature for splits/CV strategies, leakage taxonomy, benchmark contamination, text deduplication, and pre-training corpus dedup. Designed for dual consumption — humans and future LLM agents.
**Primary intended consumer:** Future Claude Code / LLM agents working in `eval-toolkit` or adjacent eval projects who need detailed primary-source context for data-integrity machinery. Secondary: humans.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself. Move it elsewhere and it still works.
**Scope:** Data integrity for binary classification and (where overlapping) LLM eval; 1997–2024. 6 sub-areas (A1, A2, B1, B2, C1, D1) across 4 topic files. 15 primary-source entries.
**Coverage:** 15 entries across 4 topic files; structured 5-bullet entries (Source / Code / Mechanism / Result / Status).
**Last updated:** 2026-05-14.

## ⚠️ Scope boundary

This cluster covers **data-integrity guardrails** — making sure train and eval sets stay properly separated, with proper accounting for structure (group/temporal/spatial), proper handling of hyperparameter tuning, and proper deduplication. The following are explicitly out of scope:

- **Bootstrap CIs, ROC variance, K-fold variance estimation** — see `../inference/` (Bates et al. 2024 CV-variance paper is there; this cluster only references it).
- **Prompt-injection-specific contamination / slice taxonomy** — see `../prompt-injection/`.
- **General LLM eval harness patterns** (HELM, lm-eval, Inspect AI) — see `../eval-ecosystem/`.
- **Privacy-preserving deduplication / differential privacy** — different concern (privacy vs eval integrity).
- **Active learning / curriculum learning** — split design but a different optimization framing.
- **Federated-learning data-isolation** — different threat model.

**Adjacent dossiers in this repo:**
- `../inference/` — bootstrap CIs, ROC variance, calibration, threshold selection (built)
- `../eval-ecosystem/` — eval harness patterns, reproducibility, Croissant (not yet built)
- `../prompt-injection/` — OWASP LLM01, PINT, attack/defense literature (not yet built)
- `../../datasets/` — eval benchmarks and corpora (not yet built)

## How this is organized

Sub-section anchors use a per-file letter prefix (`## A1.` in file 01, `## B1.` in file 02, etc.).

| File | Topic | When to read |
|---|---|---|
| `01_splits_and_nested_cv.md` | Splits (A1) + nested CV (A2) | You're picking a CV scheme for structured/time-series data, or implementing nested CV for HP-tuned evaluation |
| `02_leakage_and_contamination.md` | Leakage taxonomy (B1) + benchmark contamination (B2) | You're auditing eval pipelines for leakage modes or measuring LLM contamination on a public benchmark |
| `03_text_dedup_methods.md` | Text dedup algorithms (C1) | You're implementing or selecting a near-duplicate detection algorithm (MinHash-LSH, SemDeDup, etc.) |
| `04_pretrain_dedup_effects.md` | Pre-training dedup effects (D1) | You need to justify the importance of dedup in an LLM-pretraining or eval-leakage context |

Raw 7-column dossier tables live in `_dossier/01_*.md` … `_dossier/04_*.md`. Source ledger: `bib_ledger.yml`. Research plan: `research_plan.md`.

## Lookup recipes

### Foundational papers
- **"What's the standard leakage taxonomy?"** → `02_leakage_and_contamination.md` § B1 (Kapoor & Narayanan 2023 — 8-type taxonomy).
- **"What's the foundational paper for MinHash?"** → `03_text_dedup_methods.md` § C1 (Broder 1997).
- **"What's the foundational paper for LSH?"** → `03_text_dedup_methods.md` § C1 (Indyk & Motwani 1998).
- **"What's the foundational paper for nested CV?"** → `01_splits_and_nested_cv.md` § A2 (Varma & Simon 2006; Cawley & Talbot 2010).
- **"Why does dedup matter for LLM training?"** → `04_pretrain_dedup_effects.md` § D1 (Lee et al. 2022).

### Method / technique by name
- **"How do I detect near-duplicate documents at scale?"** → `03_text_dedup_methods.md` § C1 (MinHash-LSH = Broder 1997 + Indyk & Motwani 1998; semantic = Abbas et al. 2023 SemDeDup).
- **"How should I split time-series data for CV?"** → `01_splits_and_nested_cv.md` § A1 (Bergmeir & Benitez 2012; Roberts et al. 2017 block CV).
- **"How should I split spatially/hierarchically structured data?"** → `01_splits_and_nested_cv.md` § A1 (Roberts et al. 2017 block CV).
- **"How do I avoid optimistic bias when CV is used for both HP-tuning and evaluation?"** → `01_splits_and_nested_cv.md` § A2 (Varma & Simon 2006 nested CV).
- **"What's the difference between memorization and exploitation of contaminated data?"** → `02_leakage_and_contamination.md` § B2 (Magar & Schwartz 2022).
- **"What are the scaling laws for memorization?"** → `04_pretrain_dedup_effects.md` § D1 (Carlini et al. 2023 — three log-linear laws).

### eval-toolkit code-mapping
- **"I'm working on `splits.TimeSeriesSplit` — primary refs?"** → `01_splits_and_nested_cv.md` § A1 (Bergmeir & Benitez 2012).
- **"I'm working on `splits.GroupKFold` / source-disjoint splits — primary refs?"** → `01_splits_and_nested_cv.md` § A1 (Roberts et al. 2017 block CV).
- **"I'm working on `splits.NestedCV` / PoolBuilder — primary refs?"** → `01_splits_and_nested_cv.md` § A2 (Varma & Simon 2006; Cawley & Talbot 2010).
- **"I'm working on `leakage` module checks — primary refs?"** → `02_leakage_and_contamination.md` § B1 (Kapoor & Narayanan 2023 — 8-type taxonomy maps to the eval-toolkit's 7-check taxonomy).
- **"I'm working on `text_dedup.minhash_lsh` — primary refs?"** → `03_text_dedup_methods.md` § C1 (Broder 1997 + Indyk & Motwani 1998).
- **"I'm working on `text_dedup.semantic` / embedding-cosine — primary refs?"** → `03_text_dedup_methods.md` § C1 (Abbas et al. 2023 SemDeDup).
- **"Why does the eval-toolkit's contamination check exist?"** → `02_leakage_and_contamination.md` § B2 + `04_pretrain_dedup_effects.md` § D1 (the contamination-and-memorization literature jointly motivates the test-set leakage check).

### Concept / glossary
- **"What's MinHash-LSH?"** → README glossary; primary sources Broder 1997 + Indyk & Motwani 1998.
- **"What's group / block CV?"** → README glossary; primary source Roberts et al. 2017.
- **"What's the difference between exact, near-duplicate, and semantic dedup?"** → `03_text_dedup_methods.md` (all three covered).

### Out of scope routing
- **"Where's the Bates et al. 2024 paper on CV variance?"** → `../inference/01_bootstrap_and_cv_variance.md` § A2 (it's in the inference cluster; this cluster covers split *strategies*, not variance theory).
- **"Where are prompt-injection slice contamination patterns?"** → `../prompt-injection/` (not built yet).
- **"Where's HELM / lm-eval ecosystem?"** → `../eval-ecosystem/` (not built yet).

## Glossary

- **Block CV (spatial / temporal / hierarchical block CV)**: CV variant where folds are constructed from contiguous blocks (in time, space, or group structure) rather than randomly sampled. Reduces optimism when data points are not i.i.d. Source: Roberts et al. 2017 (`01_splits_and_nested_cv.md` § A1).
- **Contamination (benchmark contamination)**: Train-test overlap where the model has seen the eval set during training. For LLMs, often measured via memorization signals or time-stratified pass-rate analysis. Source: Sainz et al. 2023; Roberts et al. 2023; Magar & Schwartz 2022 (`02_leakage_and_contamination.md` § B2).
- **Group K-fold**: K-fold CV where examples sharing a group identifier (subject, source) are always in the same fold; prevents group leakage. Implemented in scikit-learn's `GroupKFold`.
- **Hierarchical / source-disjoint split**: Block CV variant where train and test are disjoint at the source level (e.g., different patients, different domains, different prompts).
- **Jaccard similarity**: |A ∩ B| / |A ∪ B| for two sets. Used to define text similarity over shingle sets. Foundation for MinHash.
- **Leakage**: Any way information from the test set influences the model's training or hyperparameter selection. The Kapoor-Narayanan hierarchical taxonomy's 8 leaf types: **L1.1** no test set; **L1.2** preprocessing on combined train+test; **L1.3** feature selection on combined train+test; **L1.4** train-test duplicates; **L2** illegitimate features (target leakage in vernacular); **L3.1** temporal leakage; **L3.2** train-test nonindependence (group / spatial / hierarchical); **L3.3** sampling bias. Source: Kapoor & Narayanan 2023 (`02_leakage_and_contamination.md` § B1).
- **LSH (Locality-Sensitive Hashing)**: Hash family where similar inputs collide with high probability; enables sublinear-time approximate nearest-neighbor search. Source: Indyk & Motwani 1998 (`03_text_dedup_methods.md` § C1).
- **Memorization**: An LLM's ability to emit verbatim sequences from its training data. Distinguished from exploitation (using contaminated data to improve eval performance). Source: Carlini et al. 2021 / 2023; Magar & Schwartz 2022.
- **MinHash**: Constant-size fingerprint for a set that allows Jaccard-similarity estimation in O(1) per pair. Source: Broder 1997 (`03_text_dedup_methods.md` § C1).
- **MinHash-LSH**: Combination of MinHash signatures with LSH bucketing; scalable approximate-Jaccard near-duplicate detection. Used in eval-toolkit's `text_dedup.minhash_lsh`.
- **Nested CV**: Outer CV loop estimates the generalization error; inner loop tunes hyperparameters within each outer fold. Avoids the optimistic bias of using the same fold for both. Source: Varma & Simon 2006; Cawley & Talbot 2010 (`01_splits_and_nested_cv.md` § A2).
- **Semantic dedup**: Dedup based on embedding-space proximity rather than lexical overlap; finds paraphrases and near-duplicates that exact-match and MinHash miss. Source: Abbas et al. 2023 SemDeDup.
- **Shingle (k-gram)**: A contiguous sequence of k tokens or characters from a document. Set of shingles defines a document for Jaccard / MinHash purposes.
- **Temporal leakage / look-ahead bias**: Using future information to predict past outcomes, often by random K-fold on time-ordered data. Source: Bergmeir & Benitez 2012; Kapoor & Narayanan 2023.

## Verification & limits

- Citations resolved as of 2026-05-14.
- All 15 entries are `status: verified` in `bib_ledger.yml` after round 1 audit (see audit-trail note below).
- This synthesis is a snapshot. The dedup, contamination, and memorization literature is currently moving fast (post-2022 LLM-scale studies); re-check after 6 months.
- 2 audit rounds are scheduled (data-integrity is a load-bearing cluster per the dossier roadmap). Round 1 complete; round 2 pending.

**Independent audit, round 1 (2026-05-14):** A standard complementary-scope review pass focused on attribution correctness for all 15 entries marked `status: unverified` after the gather stage. Prior rounds covered: (none). Findings: 0 dropped, 0 corrected, 0 flagged (the audit surfaced one minor venue-year question for `sainz2023nlp` but Findings of EMNLP 2023 is the correct venue per the ACL Anthology; kept as-is). Several primary URLs were paywall-blocked (Elsevier, Wiley Ecography, Springer, IEEE, ACM DL) and required WebSearch cross-validation; this is normal for older / journal-published references and not URL-rot worth flagging. All 15 entries promoted to `verified`. Recommendation: re-run with focus "Mechanism / Result bullet claim accuracy, plus lookup-recipe sanity in the README".

**Independent audit, round 2 (2026-05-14):** A standard complementary-scope review pass focused on Mechanism / Result bullet claim accuracy in the 4 synthesis files, plus a check that every lookup recipe in this README points to a valid file-anchor. Prior rounds covered: round 1 (attribution correctness). Findings: 0 dropped, 2 corrected, 0 flagged (both applied). The corrections were (1) the `kapoor2023leakage` § B1 Result-bullet enumeration of the 8 leakage types used common-vernacular labels (target / data-augmentation / group / train-test overlap) that don't match the paper's actual taxonomy (L1.1–L3.3 hierarchical codes); corrected to use the paper's labels with vernacular annotations in parens, and the README glossary entry for "Leakage" updated to match; (2) `abbas2023semdedup` § C1 Result-bullet "substantially reducing training compute" tightened to "removing ~50% of examples preserves performance, halving training time" per the paper's abstract. SPOT-CHECKS passed for: carlini2022memorization (3 scaling laws), lee2022deduplicating (~10× verbatim emission reduction), bergmeir2012cv, roberts2017spatial, varma2006bias, cawley2010overfitting, broder1997minhash, indyk1998lsh, carlini2021extracting, penedo2023refinedweb, magar2022contamination, sainz2023nlp, roberts2023contamination. All 13 lookup recipes in the README point to valid file-anchor combinations. Recommendation: Clean — stop here.

## Attribution

Synthesized from a research dossier maintained by the research_toolkit (`~/Claude/research_toolkit/`). Source URLs link to primary sources. Companion raw dossier tables in `_dossier/`. Ledger: `bib_ledger.yml`. Plan: `research_plan.md`.
