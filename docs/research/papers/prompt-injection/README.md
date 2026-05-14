# Prompt-Injection Evaluation Methodology — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained reference for prompt-injection eval methodology used in the eval-toolkit. Read this README first. -->

**Purpose:** Ground future Claude agents working on prompt-injection consumer-domain evals in eval-toolkit in the primary literature for attack taxonomy, defense methods, evaluation benchmarks, and industry standards. Designed for dual consumption.
**Primary intended consumer:** Future Claude Code / LLM agents working on prompt-injection eval in eval-toolkit or similar projects.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself.
**Scope:** Prompt-injection / jailbreak / LLM safety eval; 2022–2025. 5 sub-areas (A1, A2, B1, C1, C2) across 3 topic files. 10 primary-source entries.
**Last updated:** 2026-05-14.

## ⚠️ Scope boundary

This cluster covers **prompt-injection consumer-domain eval methodology**. Out of scope:

- General alignment / RLHF training methodology.
- Privacy-extraction / training-data-extraction attacks — see `../data-integrity/` (Carlini et al. 2021 there is the foundational reference).
- Adversarial examples in vision or non-LLM modalities.
- General eval harness patterns (HELM, lm-eval, Inspect AI) — see `../eval-ecosystem/`.
- Data integrity (splits / leakage / dedup) — see `../data-integrity/`.
- Statistical inference for safety-eval metrics — see `../inference/`.

**Adjacent dossiers in this repo:** `../inference/` (built), `../data-integrity/` (built), `../eval-ecosystem/` (built), `../../datasets/` (pending).

## How this is organized

| File | Topic | When to read |
|---|---|---|
| `01_attacks.md` | Attack taxonomy (A1) + automated attack methods (A2) | You're picking an attack benchmark, understanding threat models, or implementing red-teaming |
| `02_defenses.md` | Defense methods (B1) | You're implementing a system-prompt defense or comparing defense classes |
| `03_benchmarks_and_standards.md` | Eval benchmarks (C1) + OWASP standards (C2) | You're choosing an eval benchmark, or aligning the eval-toolkit's prompt-injection slice taxonomy to OWASP |

Raw 7-column dossier tables in `_dossier/`.

## Lookup recipes

### Foundational attack literature
- **"What's the first prompt-injection paper?"** → `01_attacks.md` § A1 (Perez & Ribeiro 2022).
- **"What's indirect prompt injection?"** → `01_attacks.md` § A1 (Greshake et al. 2023).
- **"Why do jailbreaks work? Failure-mode taxonomy?"** → `01_attacks.md` § A1 (Wei, Haghtalab & Steinhardt 2023 — competing objectives + mismatched generalization).
- **"What's GCG?"** → `01_attacks.md` § A2 (Zou et al. 2023 — Greedy Coordinate Gradient).
- **"What's PAIR?"** → `01_attacks.md` § A2 (Chao et al. 2023 — Prompt Automatic Iterative Refinement).

### Defenses
- **"What's the foundational system-prompt defense reference?"** → `02_defenses.md` § B1 (Xie et al. 2023, Self-Reminder, Nature MI).

### Benchmarks
- **"Which jailbreak benchmark should I use for comprehensive red-teaming eval?"** → `03_benchmarks_and_standards.md` § C1 (HarmBench — Mazeika et al. 2024 — 18 methods × 33 LLMs).
- **"Which jailbreak benchmark for head-to-head reproducible attack/defense comparison?"** → `03_benchmarks_and_standards.md` § C1 (JailbreakBench — Chao et al. 2024 — standardized threat model + scoring).
- **"Which benchmark for evaluating prompt-injection *detectors* (classifiers)?"** → `03_benchmarks_and_standards.md` § C1 (PINT — Lakera 2024).

### Standards
- **"What's OWASP LLM01:2025?"** → `03_benchmarks_and_standards.md` § C2 (OWASP Top 10 for LLM Apps 2025; LLM01 = prompt injection).

### eval-toolkit code-mapping
- **"I'm aligning the eval-toolkit prompt-injection slice taxonomy to OWASP — primary refs?"** → `03_benchmarks_and_standards.md` § C2 (OWASP LLM01:2025).
- **"I'm using PINT as a benchmark in the eval-toolkit — primary refs?"** → `03_benchmarks_and_standards.md` § C1 (Lakera 2024).
- **"I'm implementing red-teaming attacks for testing — primary refs?"** → `01_attacks.md` § A2 (GCG for white-box / transfer; PAIR for black-box).
- **"I'm benchmarking defenses — primary refs?"** → `02_defenses.md` § B1 + `03_benchmarks_and_standards.md` § C1.

### Out of scope routing
- **"Where are training-data-extraction attacks?"** → `../data-integrity/04_pretrain_dedup_effects.md` § D1 (Carlini et al. 2021).
- **"Where are general LLM eval harnesses (HELM, lm-eval)?"** → `../eval-ecosystem/`.

## Glossary

- **Direct prompt injection**: Adversarial input from the user that causes the LLM to deviate from its system prompt. Source: Perez & Ribeiro 2022 (`01_attacks.md` § A1).
- **Indirect prompt injection**: Adversarial instructions embedded in data the LLM retrieves (web pages, documents, emails). Source: Greshake et al. 2023 (`01_attacks.md` § A1).
- **GCG (Greedy Coordinate Gradient)**: White-box attack that optimizes adversarial suffixes against open-source LLMs and transfers them to closed-source models. Source: Zou et al. 2023 (`01_attacks.md` § A2).
- **PAIR (Prompt Automatic Iterative Refinement)**: Black-box attack where an attacker LLM iteratively refines jailbreaks. Source: Chao et al. 2023 (`01_attacks.md` § A2).
- **Jailbreak**: Adversarial prompt that bypasses an LLM's safety / refusal training. Closely related to but conceptually distinct from prompt injection.
- **Self-reminder**: System-prompt-based defense wrapping user queries with a reminder to respond responsibly. Source: Xie et al. 2023 (`02_defenses.md` § B1).
- **HarmBench**: Standardized red-teaming eval framework covering many attack methods and target LLMs. Source: Mazeika et al. 2024 (`03_benchmarks_and_standards.md` § C1).
- **JailbreakBench**: Open-source jailbreak benchmark with standardized threat model. Source: Chao et al. 2024 (`03_benchmarks_and_standards.md` § C1).
- **PINT (Prompt Injection Test)**: Lakera's benchmark for evaluating prompt-injection *detectors*; private eval set to prevent overfitting. Source: Lakera 2024 (`03_benchmarks_and_standards.md` § C1).
- **OWASP LLM01:2025**: First entry in OWASP Top 10 for LLM Applications (2025 edition) — prompt injection. Source: OWASP 2025 (`03_benchmarks_and_standards.md` § C2).
- **Goal hijacking**: Sub-type of prompt injection where the attacker redirects the LLM to a different task. Source: Perez & Ribeiro 2022.
- **Prompt leaking**: Sub-type of prompt injection where the attacker extracts the system prompt. Source: Perez & Ribeiro 2022.

## Verification & limits

- Citations resolved as of 2026-05-14.
- All 10 entries `status: verified` in `bib_ledger.yml` after round 1 audit (see audit-trail note below).
- This synthesis is a snapshot. Prompt-injection is a fast-moving field — new attacks/defenses appear monthly; re-check after 3 months.
- 1 audit round completed.

**Independent audit, round 1 (2026-05-14):** A standard combined-scope review pass focused on attribution correctness, Mechanism / Result claim accuracy, and README lookup-recipe sanity. Prior rounds covered: (none). Findings: 0 dropped, 1 corrected, 0 flagged-with-action. The correction was `chao2023pair` venue: "later NeurIPS 2024 R0-FoMo Workshop" → "NeurIPS 2023 R0-FoMo Workshop" (the workshop was actually at NeurIPS 2023). All other 9 entries spot-checked clean against primary sources. All README lookup recipes (A1, A2, B1, C1, C2) point to valid file-anchor combinations. All 10 entries promoted to `verified`. Recommendation: Clean — stop here.

## Attribution

Synthesized from a research dossier maintained by the research_toolkit (`~/Claude/research_toolkit/`). Source URLs link to primary sources. Companion raw dossier tables in `_dossier/`.
