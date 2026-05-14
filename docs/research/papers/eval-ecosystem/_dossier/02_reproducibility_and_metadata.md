# Reproducibility Checklists and Metadata Standards

This file covers B1 (reproducibility checklists — Pineau et al. 2021 / NeurIPS checklist) and B2 (dataset / model metadata standards — Datasheets, Model Cards, Croissant). Eval harness frameworks live in `01_eval_harness_frameworks.md`.

---

## B1. Reproducibility checklists

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program) | Pineau et al. (2021) | JMLR 22 | arXiv:2003.12206 | — | Reports on the three-pronged NeurIPS 2019 reproducibility program (code-submission policy + reproducibility challenge + ML reproducibility checklist) | Standard reference for the NeurIPS reproducibility checklist; documents the empirical effect of mandatory code submission and checklist completion on reproducibility outcomes |

## B2. Dataset and model metadata standards

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Datasheets for Datasets | Gebru et al. (2021) | Communications of the ACM 64(12) | arXiv:1803.09010 | — | Proposes that every dataset be accompanied by a "datasheet" documenting motivation, composition, collection process, recommended uses, distribution, and maintenance | Standard reference for dataset documentation; influenced the dataset cards format adopted by HuggingFace and the dataset documentation in eval-toolkit's `manifest` module |
| Model Cards for Model Reporting | Mitchell et al. (2019) | FAT* 2019 | arXiv:1810.03993 | — | Proposes "model cards" — short documents accompanying trained models with benchmarked evaluation across cultural, demographic, phenotypic, and intersectional groups | Standard reference for model documentation; foundational for the model-reporting fields in eval-toolkit's `manifest` (RunManifest aligned to NeurIPS checklist) |
| Croissant: A Metadata Format for ML-Ready Datasets | Akhtar et al. (2024) | NeurIPS 2024 Datasets and Benchmarks (also DEEM 2024) | arXiv:2403.19546 | mlcommons/croissant | Standardized metadata format (schema.org extension) for ML-ready datasets adopted by Kaggle, HuggingFace, OpenML; loadable via TensorFlow / PyTorch / JAX | Industry-standard dataset interchange format that the eval-toolkit `loaders` module is being aligned to (Croissant-compatible metadata) |
