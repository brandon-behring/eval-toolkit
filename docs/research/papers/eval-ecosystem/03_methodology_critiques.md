# Methodology Surveys and Best-Practices Critiques — Synthesis

This file synthesizes C1 (papers critiquing or surveying ML evaluation methodology broadly). Companion raw-table dossier: `_dossier/03_methodology_critiques.md`.

---

## C1. ML evaluation methodology critiques

- **What Will it Take to Fix Benchmarking in Natural Language Understanding?** — Bowman & Dahl (NAACL 2021).
  - **Source:** https://arxiv.org/abs/2104.02145
  - **Code:** —
  - **Mechanism:** Position paper laying out four criteria NLU benchmarks should meet: validity, reliability of metrics, statistical significance, and inclusion of annotator-divergence / disagreement.
  - **Result:** Foundational modern critique of benchmarking conventions in NLU; argues adversarial data collection alone does not address the underlying validity and reliability failures. Widely cited in eval-methodology literature.
  - **Status:** Verified (no widely-known repo).

- **The Benchmark Lottery** — Dehghani et al. (arXiv 2021).
  - **Source:** https://arxiv.org/abs/2107.07002
  - **Code:** —
  - **Mechanism:** Empirical analysis showing the relative ranking of ML methods can be substantially altered by the choice of benchmark tasks, even within the same domain (NLP, computer vision, IR, recommender systems, RL).
  - **Result:** Names and characterizes the "benchmark lottery" — the fragility of method rankings to benchmark choice. Justifies multi-benchmark evaluation and discourages single-benchmark conclusions about method superiority.
  - **Status:** Verified (no widely-known repo).
