# Evaluation Benchmarks and Industry Standards

This file covers C1 (evaluation benchmarks for prompt injection / jailbreaks) and C2 (industry standards — OWASP). Attack and defense literature live in `01_attacks.md` and `02_defenses.md`.

---

## C1. Evaluation benchmarks

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal | Mazeika et al. (2024) | ICML 2024 | arXiv:2402.04249 | centerforaisafety/HarmBench | Standardized evaluation framework covering 18 red-teaming methods × 33 target LLMs, with harmful behaviors grouped into standard / contextual / copyright / multimodal | Standard reference benchmark for automated red-teaming evaluation; supports head-to-head method comparison and proposes an adversarial-training defense |
| JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models | Chao et al. (2024) | NeurIPS 2024 Datasets and Benchmarks | arXiv:2404.01318 | JailbreakBench/jailbreakbench | Open-source benchmark with 100 jailbreak behaviors, standardized threat model + chat templates + scoring, plus a leaderboard tracking attack/defense progress | Establishes the modern reproducibility standard for jailbreak evaluation — addresses incomparability issues across prior benchmarks |
| Lakera's Prompt Injection Test (PINT) — A New Benchmark for Evaluating Prompt Injection Solutions | Lakera (2024) | Vendor blog (Lakera AI) | (no arXiv) | lakeraai/pint-benchmark | Benchmark for prompt-injection detection systems; evaluates classifier-based defenses across a dataset of public + proprietary attack techniques, false-positive controls, and large-document handling tests | Industry-side reference benchmark designed to test prompt-injection detectors without leaking the eval set (private dataset prevents overfitting) |

## C2. Industry standards and threat modeling

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| LLM01:2025 Prompt Injection — OWASP Top 10 for LLM Applications (2025) | OWASP Gen AI Security Project (2025) | OWASP Foundation (2025 edition) | (no arXiv) | — | Industry-standard threat model for prompt injection; describes the vulnerability, attack vectors (direct/indirect), and mitigation strategies | Authoritative industry reference for prompt-injection threat modeling; LLM01 held the top spot in both 2023 and 2025 editions of the OWASP Top 10 for LLM Apps |
