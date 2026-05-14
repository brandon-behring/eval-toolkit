# Methodology Surveys and Best-Practices Critiques

This file covers C1 (papers critiquing or surveying ML evaluation methodology broadly). Eval harness frameworks live in `01_eval_harness_frameworks.md`. Reproducibility / metadata standards live in `02_reproducibility_and_metadata.md`.

---

## C1. ML evaluation methodology critiques

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| What Will it Take to Fix Benchmarking in Natural Language Understanding? | Bowman & Dahl (2021) | NAACL 2021 | arXiv:2104.02145 | — | Lays out four criteria for NLU benchmarks: validity, reliability of metrics, statistical significance, and inclusion of disagreement / annotator-divergence | Foundational critique of benchmarking conventions in NLU; argues that adversarial data collection alone does not address the underlying validity and reliability failures |
| The Benchmark Lottery | Dehghani et al. (2021) | arXiv preprint | arXiv:2107.07002 | — | Shows that the relative ranking of ML methods can be substantially altered by the choice of benchmark tasks, even within the same domain | Names and characterizes "benchmark lottery" — the fragility of method rankings to benchmark choice; spans NLP, vision, IR, recommender systems, RL |
