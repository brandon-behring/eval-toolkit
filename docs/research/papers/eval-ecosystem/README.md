# ML Evaluation Ecosystem and Reproducibility Standards — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained reference for the ML evaluation ecosystem (HELM, EleutherAI lm-eval, AISI Inspect AI) and reproducibility / metadata standards (NeurIPS checklist, Datasheets, Model Cards, Croissant). Read this README first. -->

**Purpose:** Ground future Claude agents working in the `eval-toolkit` repo in the broader ML-eval ecosystem — eval-harness frameworks, reproducibility checklists, and dataset / model metadata standards. Designed for dual consumption.
**Primary intended consumer:** Future Claude Code / LLM agents working in `eval-toolkit` or adjacent eval projects who need positioning context.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself.
**Scope:** ML evaluation ecosystem; 2018–2024. 4 sub-areas (A1, B1, B2, C1) across 3 topic files. 9 primary-source entries.
**Last updated:** 2026-05-14.

## ⚠️ Scope boundary

This cluster covers **eval-ecosystem positioning** — eval-harness frameworks, reproducibility checklists, and metadata standards. Out of scope:

- Statistical inference theory (bootstrap, calibration, ROC) — see `../inference/`.
- Data integrity (splits, leakage, dedup) — see `../data-integrity/`.
- Prompt-injection-specific eval methodology — see `../prompt-injection/`.
- Specific eval results or leaderboards.
- Domain-specific eval harnesses (vision, RL benchmarks).

**Adjacent dossiers in this repo:** `../inference/` (built), `../data-integrity/` (built), `../prompt-injection/` (pending), `../../datasets/` (pending).

## How this is organized

| File | Topic | When to read |
|---|---|---|
| `01_eval_harness_frameworks.md` | HELM, EleutherAI lm-eval, AISI Inspect AI (A1) | You're positioning the eval-toolkit relative to existing LLM-eval harnesses |
| `02_reproducibility_and_metadata.md` | NeurIPS reproducibility checklist (B1) + dataset / model metadata standards (B2) | You're aligning RunManifest to NeurIPS checklist or implementing Croissant / Datasheets / Model Cards support |
| `03_methodology_critiques.md` | Benchmarking critiques — Bowman & Dahl, Dehghani (C1) | You need an argument that benchmark choice matters, or you're justifying multi-benchmark evaluation |

Raw 7-column dossier tables in `_dossier/`. Source ledger: `bib_ledger.yml`. Research plan: `research_plan.md`.

## Lookup recipes

### Eval harness positioning
- **"What's HELM?"** → `01_eval_harness_frameworks.md` § A1 (Liang et al. 2022).
- **"What's the EleutherAI lm-evaluation-harness?"** → `01_eval_harness_frameworks.md` § A1 (Gao et al. 2024).
- **"What's AISI Inspect AI?"** → `01_eval_harness_frameworks.md` § A1 (UK AISI 2024).
- **"What does 'holistic evaluation' mean?"** → `01_eval_harness_frameworks.md` § A1 (HELM): measure accuracy + calibration + robustness + fairness + bias + toxicity + efficiency per scenario.

### Reproducibility / metadata
- **"What's the NeurIPS reproducibility checklist?"** → `02_reproducibility_and_metadata.md` § B1 (Pineau et al. 2021).
- **"What's a datasheet for a dataset?"** → `02_reproducibility_and_metadata.md` § B2 (Gebru et al. 2021).
- **"What's a model card?"** → `02_reproducibility_and_metadata.md` § B2 (Mitchell et al. 2019).
- **"What's Croissant metadata?"** → `02_reproducibility_and_metadata.md` § B2 (Akhtar et al. 2024).

### Methodology critiques
- **"What's the benchmark lottery?"** → `03_methodology_critiques.md` § C1 (Dehghani et al. 2021): method rankings are fragile to benchmark choice.
- **"What's the foundational critique of NLU benchmarking?"** → `03_methodology_critiques.md` § C1 (Bowman & Dahl 2021): four criteria — validity, metric reliability, statistical significance, annotator-divergence inclusion.

### eval-toolkit code-mapping
- **"I'm aligning RunManifest to NeurIPS checklist — primary refs?"** → `02_reproducibility_and_metadata.md` § B1 (Pineau et al. 2021).
- **"I'm implementing Croissant-compatible metadata in `loaders` — primary refs?"** → `02_reproducibility_and_metadata.md` § B2 (Akhtar et al. 2024).
- **"I'm implementing model-card-style reporting — primary refs?"** → `02_reproducibility_and_metadata.md` § B2 (Mitchell et al. 2019).
- **"I'm positioning eval-toolkit vs HELM / Inspect AI in docs — primary refs?"** → `01_eval_harness_frameworks.md` § A1.

### Out of scope routing
- **"Where are bootstrap CI / calibration / ROC variance refs?"** → `../inference/`.
- **"Where's leakage / contamination / dedup?"** → `../data-integrity/`.
- **"Where's prompt-injection eval methodology?"** → `../prompt-injection/`.

## Glossary

- **Croissant**: ML-ready dataset metadata format from ML Commons (2024); schema.org extension; adopted by Kaggle, HuggingFace, OpenML. Source: Akhtar et al. 2024 (`02_reproducibility_and_metadata.md` § B2).
- **Datasheet (for datasets)**: Document accompanying a dataset describing motivation, composition, collection, recommended uses, distribution, maintenance. Source: Gebru et al. 2021 (`02_reproducibility_and_metadata.md` § B2).
- **HELM (Holistic Evaluation of Language Models)**: Stanford CRFM's LLM eval framework measuring accuracy + calibration + robustness + fairness + bias + toxicity + efficiency per scenario. Source: Liang et al. 2022 (`01_eval_harness_frameworks.md` § A1).
- **Inspect AI**: UK AISI's open-source LLM-eval framework, with built-in tool-use / multi-turn / model-graded primitives. Source: UK AISI 2024 (`01_eval_harness_frameworks.md` § A1).
- **lm-evaluation-harness (EleutherAI)**: De facto standard open-source LLM eval harness; 60+ benchmarks, tokenization-agnostic API. Source: Gao et al. 2024 (`01_eval_harness_frameworks.md` § A1).
- **Model card**: Short document accompanying a trained model with benchmarked eval across cultural / demographic / phenotypic / intersectional groups. Source: Mitchell et al. 2019 (`02_reproducibility_and_metadata.md` § B2).
- **NeurIPS reproducibility checklist**: Submission-time checklist covering experimental detail, datasets, code, theoretical results, ethics. Standard component of NeurIPS / ICML / ICLR submissions since 2019. Source: Pineau et al. 2021 (`02_reproducibility_and_metadata.md` § B1).
- **Benchmark lottery**: Phenomenon where method rankings are substantially altered by benchmark choice, even within the same domain. Source: Dehghani et al. 2021 (`03_methodology_critiques.md` § C1).

## Verification & limits

- Citations resolved as of 2026-05-14.
- All 9 entries `status: verified` in `bib_ledger.yml` after round 1 audit (see audit-trail note below).
- This synthesis is a snapshot. The eval-harness ecosystem (HELM, Inspect AI, lm-eval) is evolving — re-check after 6 months.
- 1 audit round completed.

**Independent audit, round 1 (2026-05-14):** A standard combined-scope review pass focused on attribution correctness, Mechanism / Result claim accuracy, and README lookup-recipe sanity. Prior rounds covered: (none). Findings: 0 dropped, 0 corrected, 4 flagged (3 applied, 1 left as-is). The flags were (1) `aisi2024inspect` Mechanism/Result claim "inline CIs" not supported by primary source — corrected to remove specific claim and soften the eval-toolkit-roadmap attribution; (2) `gao2024lmevalharness` Zenodo v0.4.3 first creator is Sutawika not Gao — added clarifying parenthetical preserving the "Gao et al." canonical attribution; (3) `akhtar2024croissant` Mechanism claims about Kaggle/HuggingFace/OpenML adoption are factually correct but not in arXiv abstract — added "per mlcommons docs" qualifier; (4) `pineau2021reproducibility` year is arXiv 2020 / JMLR 2021 — kept as 2021 since JMLR is the canonical venue (no edit). All README lookup recipes (A1, B1, B2, C1) point to valid anchors. All 9 entries promoted to `verified`. Recommendation: Clean — stop here.

## Attribution

Synthesized from a research dossier maintained by the research_toolkit (`~/Claude/research_toolkit/`). Source URLs link to primary sources. Companion raw dossier tables in `_dossier/`.
