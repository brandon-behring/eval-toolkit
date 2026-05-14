# Research Plan: Prompt-injection evaluation methodology

This research grounds the `eval-toolkit` library's prompt-injection consumer-domain methodology in primary literature — attack taxonomy, defense methods, evaluation benchmarks, and industry standards (OWASP LLM01:2025). Future Claude agents working on prompt-injection eval inside eval-toolkit can ground in this dossier. Narrow scope: ~12–16 entries across 4 sub-areas.

## Sub-areas

- A1. Prompt-injection attack taxonomy and foundational attacks
  - Source types: arXiv, vendor blog (Greshake, Lakera), USENIX/IEEE S&P
  - Notes: Direct vs indirect injection, jailbreak techniques (Perez & Ribeiro 2022; Greshake et al. 2023; Zou et al. 2023 GCG; Wei et al. 2023). Excludes specific prompt-extraction attacks (separate).

- A2. Defense methods
  - Source types: arXiv, vendor blog (Anthropic, OpenAI, Lakera, Hidden Layer)
  - Notes: System-prompt hardening, classifier-based filters, training-time defenses (Kim et al. 2023 Self-Reminder; Phute et al. 2023; Jain et al. 2023). Excludes general alignment training.

- A3. Evaluation benchmarks and methodology
  - Source types: arXiv, vendor blog (Lakera PINT), HuggingFace
  - Notes: PINT (Lakera 2024), AdvBench (Zou et al. 2023), HarmBench (Mazeika et al. 2024), JailbreakBench (Chao et al. 2024). Eval-methodology questions specific to safety / refusal eval.

- A4. Industry standards and threat modeling
  - Source types: OWASP, NIST, vendor publications
  - Notes: OWASP Top 10 for LLM Applications (2025 update); NIST AI RMF where relevant. Slice taxonomy for prompt-injection safety eval.

## Out-of-scope

- General alignment / RLHF training — out of scope.
- Privacy-extraction or training-data-extraction attacks — different threat model.
- Adversarial examples in vision / non-LLM modalities — different domain.
- General eval harness patterns (HELM, lm-eval, Inspect AI) — see `../eval-ecosystem/`.
- Data integrity (splits, leakage, dedup) — see `../data-integrity/`.
- Statistical inference for safety-eval metrics — see `../inference/`.

## Claim family taxonomy

- `attack_taxonomy` — taxonomies and foundational papers on attack types (direct/indirect injection, jailbreaks)
- `attack_method` — specific attack techniques (GCG, AutoDAN, PAIR, etc.)
- `defense_method` — defense techniques (system prompts, filters, training-time)
- `benchmark` — standardized eval benchmarks (PINT, AdvBench, HarmBench, JailbreakBench)
- `industry_standard` — OWASP / NIST / vendor threat models

## Known landmark papers

- `perez2022ignore` — Perez & Ribeiro 2022 "Ignore Previous Prompt: Attack Techniques for Language Models" — first systematic prompt-injection paper.
- `greshake2023indirect` — Greshake et al. 2023 "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications" — indirect prompt injection.
- `zou2023gcg` — Zou et al. 2023 "Universal and Transferable Adversarial Attacks on Aligned Language Models" — GCG.
- `wei2023jailbroken` — Wei, Haghtalab, Steinhardt 2023 "Jailbroken: How Does LLM Safety Training Fail?" — failure-mode taxonomy.
- `mazeika2024harmbench` — Mazeika et al. 2024 "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal" — HarmBench benchmark.
- `chao2024jailbreakbench` — Chao et al. 2024 "JailbreakBench" — open red-teaming benchmark.
- `owasp2025llmtop10` — OWASP Top 10 for LLM Applications (2025 update).
