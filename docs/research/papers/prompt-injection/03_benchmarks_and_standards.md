# Evaluation Benchmarks and Industry Standards — Synthesis

This file synthesizes C1 (eval benchmarks for prompt injection / jailbreaks) and C2 (industry standards — OWASP). Companion raw-table dossier: `_dossier/03_benchmarks_and_standards.md`.

---

## C1. Evaluation benchmarks

- **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal** — Mazeika et al. (ICML 2024).
  - **Source:** https://arxiv.org/abs/2402.04249
  - **Code:** https://github.com/centerforaisafety/HarmBench
  - **Mechanism:** Standardized evaluation framework with 18 red-teaming methods × 33 target LLMs; harmful behaviors grouped into four functional categories: standard, contextual, copyright, multimodal.
  - **Result:** Standard reference benchmark for automated red-teaming evaluation; enables head-to-head method comparison and proposes an adversarial-training defense (R2D2) that improves robustness across the suite.
  - **Status:** Verified.

- **JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models** — Chao et al. (NeurIPS 2024 Datasets and Benchmarks).
  - **Source:** https://arxiv.org/abs/2404.01318
  - **Code:** https://github.com/JailbreakBench/jailbreakbench
  - **Mechanism:** Open-source benchmark with 100 jailbreak behaviors aligned to OpenAI usage policies; provides standardized threat model, system prompts, chat templates, scoring functions, and a leaderboard tracking attack/defense progress.
  - **Result:** Establishes the modern reproducibility standard for jailbreak evaluation. Addresses prior-benchmark incomparability issues (different cost/success-rate computations, withheld adversarial prompts, evolving proprietary APIs).
  - **Status:** Verified.

- **Lakera's Prompt Injection Test (PINT) — A New Benchmark for Evaluating Prompt Injection Solutions** — Lakera (2024).
  - **Source:** https://www.lakera.ai/blog/lakera-pint-benchmark
  - **Code:** https://github.com/lakeraai/pint-benchmark
  - **Mechanism:** Evaluation benchmark for prompt-injection *detection systems* (classifiers); dataset of public + proprietary attack techniques plus false-positive controls and large-document handling tests. The eval set is private to prevent overfitting.
  - **Result:** Industry-side reference benchmark for prompt-injection detectors. Used by Lakera Guard, Azure AI Prompt Shield, and other commercial / open-source detection systems for head-to-head comparison.
  - **Status:** Verified ((vendor blog) — primary source is a vendor publication, treat any specific numerical PINT scores with skepticism).

## C2. Industry standards and threat modeling

- **LLM01:2025 Prompt Injection — OWASP Top 10 for LLM Applications (2025)** — OWASP Gen AI Security Project (2025).
  - **Source:** https://genai.owasp.org/llmrisk/llm01-prompt-injection/
  - **Code:** —
  - **Mechanism:** Industry-standard threat-modeling document describing prompt injection as the top vulnerability for LLM applications; covers attack vectors (direct, indirect, multi-modal) and standardized mitigation patterns (input filtering, output validation, privilege separation).
  - **Result:** Authoritative industry reference for prompt-injection threat modeling. LLM01 held the top spot in both the 2023 and 2025 editions of OWASP Top 10 for LLM Applications — community consensus on its severity. Slice taxonomy for safety eval in the eval-toolkit aligns to OWASP categories.
  - **Status:** Verified.
