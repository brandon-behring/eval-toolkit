# LLM Evaluation Harness Frameworks — Synthesis

This file synthesizes A1 (eval harness frameworks — HELM, EleutherAI lm-eval, AISI Inspect AI). Companion raw-table dossier: `_dossier/01_eval_harness_frameworks.md`. Reproducibility / metadata standards live in `02_reproducibility_and_metadata.md`.

---

## A1. Eval harness frameworks

- **Holistic Evaluation of Language Models** — Liang et al. (Stanford CRFM, arXiv 2022; later TMLR 2023).
  - **Source:** https://arxiv.org/abs/2211.09110
  - **Code:** https://github.com/stanford-crfm/helm
  - **Mechanism:** Large-scale eval framework covering 30 prominent LLMs across 42 scenarios; reports 7 metric categories per scenario (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency).
  - **Result:** Establishes the "holistic evaluation" paradigm — measure multiple dimensions per scenario rather than a single accuracy number. Reference design for multi-metric eval harnesses including eval-toolkit's slice-aware metric composition.
  - **Status:** Verified.

- **The Language Model Evaluation Harness** — Gao et al. (EleutherAI lm-evaluation-harness contributors, Zenodo 2024). Note: the v0.4.3 Zenodo record lists Lintang Sutawika as first creator; "Gao et al." is the canonical attribution honoring Leo Gao as the project's original lead.
  - **Source:** https://zenodo.org/records/12608602
  - **Code:** https://github.com/EleutherAI/lm-evaluation-harness
  - **Mechanism:** Unified Python framework for evaluating generative LLMs across 60+ standard academic benchmarks. Tokenization-agnostic API; multiple model backends (transformers, vLLM, OpenAI / TextSynth APIs); quantization support (GPTQ).
  - **Result:** De facto standard open-source LLM eval harness; cited by virtually every modern LLM paper that reports benchmark scores. Architectural reference for the eval-toolkit `harness` module's Scorer / evaluate / evaluate_folded pattern.
  - **Status:** Verified.

- **Inspect: A framework for large language model evaluations** — UK AI Security Institute (GitHub 2024).
  - **Source:** https://github.com/UKGovernmentBEIS/inspect_ai
  - **Code:** https://github.com/UKGovernmentBEIS/inspect_ai
  - **Mechanism:** LLM eval framework from AISI / UK government with 200+ pre-built evals; built-in primitives for prompt engineering, tool-use, multi-turn dialog, and model-graded evaluation.
  - **Result:** Frontier-safety-oriented harness. The eval-toolkit's v0.9 roadmap item for inline bootstrap CIs on every metric is inspired by the broader Inspect-AI design philosophy of richer metric reporting (specific implementation details vary; verify against Inspect docs before further code-level claims).
  - **Status:** Verified.
