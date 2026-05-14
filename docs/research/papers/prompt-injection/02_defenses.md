# Prompt-Injection Defenses — Synthesis

This file synthesizes B1 (defense methods). Companion raw-table dossier: `_dossier/02_defenses.md`. Attacks live in `01_attacks.md`. Benchmarks for measuring defense efficacy live in `03_benchmarks_and_standards.md`.

---

## B1. Defense methods

- **Defending ChatGPT against jailbreak attack via self-reminders** — Xie et al. (Nature Machine Intelligence 2023).
  - **Source:** https://www.nature.com/articles/s42256-023-00765-8
  - **Code:** —
  - **Mechanism:** Wraps the user's query inside a system prompt that explicitly reminds the model to respond responsibly. Training-free; works by exploiting the model's existing safety training rather than modifying weights.
  - **Result:** Foundational system-prompt-hardening defense; demonstrates that simple system-prompt scaffolding can substantially reduce jailbreak success rates against ChatGPT. Widely cited as the canonical reference for "system prompt defense" as a baseline.
  - **Status:** Verified (no widely-known repo).

- **DataSentinel + PromptLocate — Strict Normalization for Prompt-Injection Defense** — (arXiv 2025).
  - **Source:** https://arxiv.org/abs/2511.15759
  - **Code:** —
  - **Mechanism:** Two-stage defense pipeline: DataSentinel applies input-side strict normalization (Unicode / whitespace / control-char canonicalization) to detect prompt-injection attempts that exploit encoding obfuscation; PromptLocate locates the injected segment within longer prompts to support targeted scrubbing.
  - **Result:** Complement to Xie 2023 self-reminder defenses by attacking the input-normalization layer rather than the system-prompt layer. Useful for PI-benchmark contamination detection and for pre-processing pipelines feeding downstream classifiers. Surfaced via the v0.24.1 RECONCILIATION pass.
  - **Status:** Verified (arXiv preprint).
