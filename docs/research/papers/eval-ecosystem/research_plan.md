# Research Plan: ML evaluation ecosystem and reproducibility standards

This research grounds the `eval-toolkit` library's positioning relative to the broader ML-eval ecosystem — eval-harness frameworks (EleutherAI lm-eval, HELM, AISI Inspect AI), reproducibility standards (NeurIPS reproducibility checklist), and dataset / model metadata standards (Croissant, datasheets, model cards). Narrow scope: ~10–14 papers and ecosystem references across 4 sub-areas. Focus is positioning and methodological standards, not implementation details of each harness.

## Sub-areas

- A1. LLM evaluation harness frameworks
  - Source types: arXiv, GitHub README, vendor blog (Stanford CRFM, EleutherAI, AISI)
  - Notes: Coverage of HELM (Liang et al. 2022), EleutherAI lm-evaluation-harness, AISI Inspect AI. Includes architectural patterns (test runners, score aggregation, slicing). Excludes implementation details of individual evals.

- A2. Reproducibility checklists and standards
  - Source types: vendor blog, conference paper, NeurIPS / ACL official sites
  - Notes: NeurIPS reproducibility checklist (Pineau et al. 2021). ML reproducibility primer papers. Excludes general scientific reproducibility outside ML.

- A3. Dataset and model metadata standards
  - Source types: vendor blog (Google, ML Commons), arXiv
  - Notes: Croissant metadata format (ML Commons 2024). Datasheets for Datasets (Gebru et al. 2021). Model cards (Mitchell et al. 2019). Excludes domain-specific schemas (HL7 / FHIR / etc.).

- A4. Methodology survey and best-practices papers
  - Source types: arXiv, journal
  - Notes: Empirical-methodology papers on ML evaluation pitfalls (Dehghani et al. 2021 "The Benchmark Lottery"; Bowman & Dahl 2021 "What Will it Take to Fix Benchmarking?"). Excludes purely application-specific surveys.

## Out-of-scope

- Specific eval results / leaderboards — out of scope; this cluster covers eval *methodology*, not specific scores.
- Domain-specific eval harnesses (image classification, RL benchmarks) — covered if a foundational paper, otherwise out.
- Statistical inference (bootstrap, calibration, ROC) — see `../inference/`.
- Data integrity (splits, leakage, dedup) — see `../data-integrity/`.
- Prompt-injection-specific eval methodology — see `../prompt-injection/`.

## Claim family taxonomy

- `eval_harness` — LLM / ML eval harness frameworks (HELM, lm-eval, Inspect AI, etc.)
- `reproducibility_standard` — checklists and conventions for reproducible reporting
- `metadata_standard` — schemas for dataset / model / experiment metadata (Croissant, datasheets, model cards)
- `methodology_survey` — papers critiquing or surveying ML eval methodology broadly
- `foundational_text` — textbooks and monographs (cross-referenced, e.g., Mitchell ML book)

## Known landmark papers

- `liang2022helm` — Liang et al. 2022 "Holistic Evaluation of Language Models" — HELM framework.
- `pineau2021reproducibility` — Pineau et al. 2021 "Improving Reproducibility in Machine Learning Research" — NeurIPS checklist.
- `gebru2021datasheets` — Gebru et al. 2021 "Datasheets for Datasets" — dataset documentation standard.
- `mitchell2019modelcards` — Mitchell et al. 2019 "Model Cards for Model Reporting" — model documentation standard.
- `bowman2021benchmarking` — Bowman & Dahl 2021 "What Will it Take to Fix Benchmarking in Natural Language Understanding?" — benchmarking critique.
- `dehghani2021benchmarklottery` — Dehghani et al. 2021 "The Benchmark Lottery" — eval-methodology critique.
