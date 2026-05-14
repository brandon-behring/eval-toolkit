# Data Leakage Taxonomy and Benchmark Contamination — Synthesis

This file synthesizes B1 (the general leakage taxonomy) and B2 (LLM-specific benchmark contamination). Companion raw-table dossier: `_dossier/02_leakage_and_contamination.md`. Adversarial-eval slice contamination (prompt-injection) lives in `../prompt-injection/`.

---

## B1. Leakage taxonomy

- **Leakage and the reproducibility crisis in machine-learning-based science** — Kapoor & Narayanan (Patterns 2023).
  - **Source:** https://arxiv.org/abs/2207.07048
  - **Code:** —
  - **Mechanism:** Systematic review of 294 ML-based-science papers across 17 fields with documented leakage; introduces an 8-type leakage taxonomy and a model-info-sheet checklist for self-reporting.
  - **Result:** Standard modern reference for the leakage taxonomy. The paper's 8 leaf-level types (with common-vernacular annotations in parens): **L1.1** no test set; **L1.2** pre-processing on combined train+test data; **L1.3** feature selection on combined train+test; **L1.4** duplicates between train and test (i.e., train-test overlap); **L2** model uses features that are not legitimate (sometimes called "target leakage" in practitioner vernacular); **L3.1** temporal leakage; **L3.2** train-test nonindependence (e.g., group / source / spatial / hierarchical leakage); **L3.3** sampling bias. Maps to the eval-toolkit's 7-check `leakage` module taxonomy (exact-dup, near-dup, encoding-obfuscated, cross-split, label-conflict, group, temporal) — primarily covers L1.4, L3.1, and L3.2.
  - **Status:** Verified (no widely-known repo).

- **Don't push the button! Data leakage risks in ML and transfer learning** — Pellizzoni et al. (AI Review / Springer 2025).
  - **Source:** https://link.springer.com/article/10.1007/s10462-025-11326-3
  - **Code:** —
  - **Mechanism:** Extends the Kapoor & Narayanan 2023 leakage taxonomy to cover transfer-learning and pre-training scenarios — pretraining-corpus overlap with downstream eval, fine-tuning contamination, foundation-model evaluation pitfalls.
  - **Result:** Modern leakage taxonomy update that bridges the classical Kapoor framework with the LLM / transfer-learning era. Recommended companion to Kapoor 2023 for any modern eval pipeline. Surfaced via the v0.24.1 RECONCILIATION pass.
  - **Status:** Verified (Springer AI Review, peer-reviewed).

## B2. Benchmark contamination measurement

- **Data Contamination: From Memorization to Exploitation** — Magar & Schwartz (ACL 2022 Short Papers).
  - **Source:** https://arxiv.org/abs/2203.08242
  - **Code:** https://github.com/schwartz-lab-NLP/data_contamination
  - **Mechanism:** Pre-trains BERT on joint Wikipedia + labeled-downstream corpora; compares performance on memorized-vs-unseen examples to operationalize the distinction between memorization (can recall) and exploitation (uses to improve eval).
  - **Result:** First systematic framework separating memorization from exploitation; shows contamination does not always lead to exploitation, complicating naive contamination-as-cheating arguments.
  - **Status:** Verified.

- **NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark** — Sainz et al. (Findings of EMNLP 2023).
  - **Source:** https://arxiv.org/abs/2310.18018
  - **Code:** —
  - **Mechanism:** Position paper surveying contamination detection techniques and arguing for per-benchmark contamination reporting as a community norm.
  - **Result:** Establishes the community expectation that every NLP benchmark used to evaluate an LLM must report a contamination measurement; widely cited in modern LLM-eval methodology.
  - **eval-toolkit code:** `RunManifest.contamination_flags` (manifest.v3 schema, shipped in v0.24.0) — required per-scorer contamination posture as one of `{verified_disjoint, suspected_contamination, vendor_black_box, unknown}`. Closes V4 audit issue A6 (contamination flag was previously docstring-only).
  - **Status:** Verified (no widely-known repo).

- **Data Contamination Through the Lens of Time** — Roberts et al. (arXiv 2023).
  - **Source:** https://arxiv.org/abs/2310.10628
  - **Code:** —
  - **Mechanism:** Longitudinal analysis of LLM contamination using GPT-model training-cutoff dates as a natural experiment; studies Codeforces and Project Euler problem-solving rates as a function of release date.
  - **Result:** First time-stratified contamination study; finds statistically significant pass-rate vs release-date trends consistent with contamination. Methodology reference for time-aware contamination detection.
  - **Status:** Verified (no widely-known repo).
