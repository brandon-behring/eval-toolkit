# Data Leakage Taxonomy and Benchmark Contamination

This file covers B1 (the general leakage taxonomy across ML applications) and B2 (LLM-specific benchmark contamination). Adversarial-eval slice leakage (prompt-injection) lives in `../../prompt-injection/`.

---

## B1. Leakage taxonomy

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Leakage and the reproducibility crisis in machine-learning-based science | Kapoor & Narayanan (2023) | Patterns 4(9) | arXiv:2207.07048 | — | Systematic review of 294 ML-based-science papers across 17 fields; introduces an 8-type hierarchical leakage taxonomy and a model-info-sheet checklist for self-reporting | Standard reference for the modern leakage taxonomy. Paper's 8 leaf types: L1.1 no test set; L1.2 preprocessing on train+test; L1.3 feature selection on train+test; L1.4 train-test duplicates; L2 illegitimate features (target leakage); L3.1 temporal; L3.2 train-test nonindependence (group/spatial); L3.3 sampling bias |

## B2. Benchmark contamination measurement

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Data Contamination: From Memorization to Exploitation | Magar & Schwartz (2022) | ACL 2022 Short Papers | arXiv:2203.08242 | schwartz-lab-NLP/data_contamination | Distinguishes memorization (model can recall contaminated data) from exploitation (model uses it to improve task performance); studies BERT pre-trained on joint Wikipedia + downstream-test corpora | First systematic framework separating memorization from exploitation; shows that contamination does not always lead to exploitation, complicating naive contamination-as-cheating arguments |
| NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark | Sainz et al. (2023) | Findings of EMNLP 2023 | arXiv:2310.18018 | — | Position paper arguing every NLP benchmark must report a contamination measurement when used with LLMs | Establishes the community norm that per-benchmark contamination disclosure is required; surveys detection techniques |
| Data Contamination Through the Lens of Time | Roberts et al. (2023) | arXiv preprint | arXiv:2310.10628 | — | Longitudinal analysis of LLM contamination using training-cutoff dates as a natural experiment; studies Codeforces and Project Euler | First time-stratified contamination study; finds statistically significant pass-rate vs release-date trends consistent with contamination |
