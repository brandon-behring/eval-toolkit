# Reproducibility Checklists and Metadata Standards — Synthesis

This file synthesizes B1 (reproducibility checklists) and B2 (dataset / model metadata standards). Companion raw-table dossier: `_dossier/02_reproducibility_and_metadata.md`.

---

## B1. Reproducibility checklists

- **Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)** — Pineau et al. (JMLR 2021).
  - **Source:** https://arxiv.org/abs/2003.12206
  - **Code:** —
  - **Mechanism:** Reports on the NeurIPS 2019 reproducibility program: (1) code-submission policy, (2) community-wide reproducibility challenge, (3) ML reproducibility checklist integrated into paper submission.
  - **Result:** Standard reference for the NeurIPS reproducibility checklist now required by NeurIPS, ICML, ICLR, and adopted as the template by other conferences. Documents empirical evidence that mandatory code submission and checklist completion improve downstream reproducibility outcomes. Underlies the eval-toolkit's RunManifest schema design.
  - **Status:** Verified (no widely-known repo).

## B2. Dataset and model metadata standards

- **Datasheets for Datasets** — Gebru et al. (Communications of the ACM 2021).
  - **Source:** https://arxiv.org/abs/1803.09010
  - **Code:** —
  - **Mechanism:** Proposes a structured datasheet template documenting: motivation, composition, collection process, preprocessing / labeling, uses, distribution, maintenance.
  - **Result:** Standard reference for dataset documentation; influenced HuggingFace dataset cards and the eval-toolkit `manifest` module's dataset-provenance fields.
  - **Status:** Verified (no widely-known repo).

- **Model Cards for Model Reporting** — Mitchell et al. (FAT* 2019).
  - **Source:** https://arxiv.org/abs/1810.03993
  - **Code:** —
  - **Mechanism:** Proposes "model cards" — short structured documents accompanying trained ML models with benchmarked evaluation across cultural, demographic, phenotypic, and intersectional groups.
  - **Result:** Standard reference for model documentation; foundational for model-reporting fields in eval-toolkit's RunManifest (especially the slice-aware reporting requirement).
  - **Status:** Verified (no widely-known repo).

- **Croissant: A Metadata Format for ML-Ready Datasets** — Akhtar et al. (NeurIPS 2024 Datasets and Benchmarks; also DEEM 2024).
  - **Source:** https://arxiv.org/abs/2403.19546
  - **Code:** https://github.com/mlcommons/croissant
  - **Mechanism:** Standardized dataset metadata format; per the mlcommons/croissant docs it extends schema.org and is adopted by Kaggle, HuggingFace, and OpenML, with loaders available for TensorFlow Datasets, PyTorch, and JAX (specific platform claims confirmed against mlcommons docs, not in arXiv abstract).
  - **Result:** Industry-standard ML-dataset interchange format that the eval-toolkit `loaders` module is being aligned to (Croissant-compatible metadata is a v1.0 gating criterion in the eval-toolkit roadmap).
  - **Status:** Verified.
