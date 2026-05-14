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
