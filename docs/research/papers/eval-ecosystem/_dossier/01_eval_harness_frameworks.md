# LLM Evaluation Harness Frameworks

This file covers A1 (eval harness frameworks — HELM, EleutherAI lm-evaluation-harness, AISI Inspect AI). Reproducibility and metadata standards live in `02_reproducibility_and_metadata.md`. Methodology critiques live in `03_methodology_critiques.md`.

---

## A1. Eval harness frameworks

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Holistic Evaluation of Language Models | Liang et al. (2022) | arXiv preprint (Stanford CRFM); later TMLR 2023 | arXiv:2211.09110 | stanford-crfm/helm | Large-scale evaluation framework covering 30 prominent LLMs across 42 scenarios with 7 metric categories | Establishes the "holistic" evaluation paradigm — measure multiple dimensions (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) per scenario rather than a single accuracy number |
| The Language Model Evaluation Harness | Gao et al. (2024) | Zenodo (DOI 10.5281/zenodo.12608602) | DOI:10.5281/zenodo.12608602 | EleutherAI/lm-evaluation-harness | Unified framework for testing generative LLMs across 60+ standard academic benchmarks with hundreds of subtasks | De facto standard open-source LLM eval harness; tokenization-agnostic API; multiple backends (transformers, vLLM, OpenAI / TextSynth APIs) |
| Inspect: A framework for large language model evaluations | UK AI Security Institute (2024) | GitHub (open-source framework) | (no arXiv) | UKGovernmentBEIS/inspect_ai | LLM eval framework from AISI / UK government, with built-in prompt engineering, tool-use, multi-turn dialog, and model-graded eval primitives | Frontier-safety-oriented harness with 200+ pre-built evals covering coding, agentic tasks, reasoning, multi-modal; supports VS Code extension + web Inspect View for monitoring |
