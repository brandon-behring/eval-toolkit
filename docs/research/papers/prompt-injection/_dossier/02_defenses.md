# Prompt-Injection Defenses

This file covers B1 (defense methods). Attack taxonomy and attack methods live in `01_attacks.md`. Evaluation benchmarks live in `03_benchmarks_and_standards.md`.

---

## B1. Defense methods

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Defending ChatGPT against jailbreak attack via self-reminders | Xie et al. (2023) | Nature Machine Intelligence 5 | DOI:10.1038/s42256-023-00765-8 | — | System-prompt-based defense that wraps user queries with a "self-reminder" instructing the model to respond responsibly | Demonstrates a simple training-free defense substantially reducing jailbreak success rates; widely cited as the foundational system-prompt-hardening reference |
| DataSentinel + PromptLocate — Strict Normalization for Prompt-Injection Defense | (2025) | arXiv preprint | arXiv:2511.15759 | — | Two-stage defense: DataSentinel (input-side strict normalization to detect PI attempts via Unicode / whitespace / control-char canonicalization) + PromptLocate (locates injected segments within longer prompts) | Provides a complement to Xie 2023 self-reminder defenses by attacking the input-normalization layer rather than the system-prompt layer; useful for PI-benchmark contamination detection |
