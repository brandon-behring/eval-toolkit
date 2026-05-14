# Reconciliation: dossier ↔ docs/methodology/reading_list.md

**Generated:** 2026-05-14
**Compared:**
- `docs/research/` (new dossier, 69 entries across 5 clusters — built independently from `reading_list.md`)
- `docs/methodology/reading_list.md` (existing 40+ references, NOT modified by the dossier build)

Per the dossier plan, this reconciliation surfaces gaps in either direction without editing `reading_list.md`.

## Shared entries (in both)

Both the dossier and `reading_list.md` cite these references. The dossier's bibkey is on the left; `reading_list.md`'s section/citation on the right.

| Dossier bibkey | Cluster | reading_list.md section |
|---|---|---|
| `kapoor2023leakage` | data-integrity/ B1 | Core methodology |
| `guo2017calibration` | inference/ C1 | Core methodology |
| `naeini2015bbq` | inference/ C1 | Core methodology (cites arXiv:1411.0760 preprint; dossier uses AAAI URL — same paper, different source) |
| `kumar2019verified` | inference/ D2 | Core methodology |
| `lipton2014thresholding` | inference/ E1 | Threshold selection |
| `elkan2001foundations` | inference/ E1 | Threshold selection |
| `youden1950index` | inference/ E1 | Threshold selection |
| `bates2024crossvalidation` | inference/ A2 | Splits & cross-validation |
| `hastie2009elements` | inference/ E3 | Splits & cross-validation |
| `pineau2021reproducibility` | eval-ecosystem/ B1 | Reproducibility |
| `akhtar2024croissant` | eval-ecosystem/ B2 | Reproducibility (cites the same arXiv:2403.19546) |
| `mitchell2019modelcards` | eval-ecosystem/ B2 | Fairness (note: reading_list places under Fairness; dossier under metadata_standard) |
| `gao2024lmevalharness` | eval-ecosystem/ A1 | Eval harness ecosystem |
| `aisi2024inspect` | eval-ecosystem/ A1 | Eval harness ecosystem |
| `liang2022helm` | eval-ecosystem/ A1 | Eval harness ecosystem (reading_list cites GitHub; dossier cites the arXiv paper) |
| `owasp2025llmtop10` | prompt-injection/ C2 | Prompt-injection eval |
| `lakera2024pint` | prompt-injection/ C1 + datasets/ | Prompt-injection eval |
| `delong1988auc` | inference/ B1 | Statistical comparison |
| `efron1996bootstrap` | inference/ A1 | Statistical comparison |

**Shared count: 19** (counting `naeini2015bbq` despite citing different source URLs for the same paper).

## In reading_list.md but NOT in the dossier (10)

These references appear in `reading_list.md` but were not picked up during the independent dossier build. Categorized by reason:

### Genuine gap — could be added to the dossier in a follow-up

- **Yan, X. et al. (2025) — *Hidden Leaks in Time Series Forecasting*.** [arXiv:2512.06932](https://arxiv.org/html/2512.06932v1). Relevant to `data-integrity/` § A1 (time-series leakage) — could be added next round.
- **Pellizzoni, S. et al. (2025) — *Don't push the button! Data leakage risks in ML and transfer learning*.** [Springer AI Review DOI](https://link.springer.com/article/10.1007/s10462-025-11326-3). Modern leakage taxonomy extending Kapoor & Narayanan. Mentioned in the data-integrity plan but not added; should be in `data-integrity/` § B1 next round.
- **PI_HackAPrompt_SQuAD analysis (2025).** [arXiv:2505.04806](https://arxiv.org/html/2505.04806v1). Naive-dedup detection vs attack-success-rate finding for prompt-injection benchmarks. Should be in `prompt-injection/` § C1 or `data-integrity/` § B2 next round.
- **DataSentinel + PromptLocate.** [arXiv:2511.15759](https://arxiv.org/abs/2511.15759). Strict-normalization contamination checks for prompt-injection benchmarks. Should be in `prompt-injection/` § B1 (defense) or `data-integrity/` § B2 next round.
- **Open-Prompt-Injection (Liu et al.).** [github.com/liu00222/Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection). Reference dataset of attack prompts. Should be in `datasets/` next round (sibling to AdvBench / HarmBench).

### Deliberately out of dossier scope

- **Efron & Tibshirani (1993) — *An Introduction to the Bootstrap* (book).** Skipped from the dossier because the book has no clean canonical URL (publisher pages drift). DiCiccio & Efron 1996 (Statistical Science) is in the dossier as the bootstrap-CI reference. `reading_list.md` correctly cites the book as the foundational text.
- **Recht, Roelofs, Schmidt & Shankar (2019) — *Do ImageNet classifiers generalize to ImageNet?* (ICML 2019).** Vision benchmark; out of scope for the binary-classification + LLM-eval focus of this dossier.
- **PyTorch 2.8 reproducibility notes.** Framework-specific reproducibility guide; out of scope for the dossier (which covers methodology, not framework determinism).
- **Hardt, Price & Srebro (2016) — *Equality of Opportunity in Supervised Learning*.** Fairness reference; fairness was explicitly out of scope in the dossier's `inference/research_plan.md`.
- **Kleinberg, Mullainathan & Raghavan (2017) — *Inherent Trade-offs in the Fair Determination of Risk Scores*.** Fairness reference; same exclusion as above.

## In the dossier but NOT in reading_list.md (50)

These references were discovered during the independent dossier build and do not appear in `reading_list.md`. They constitute the bulk of the dossier's added value. Counts by cluster:

### inference/ (15 entries not in reading_list)
- **A1 bootstrap_ci**: (none — `efron1996bootstrap` is the lone A1 entry, also in `reading_list.md`)
- **A2 cv_variance**: `nadeau2003inference`, `bengio2004nounbiased` (Bates 2024 is shared)
- **B1 roc_variance**: `hanley1982meaning`, `hanley1983method`, `sun2014fastdelong` (DeLong 1988 is shared)
- **C1 calibration_method**: `platt1999probabilistic`, `zadrozny2002transforming`, `niculescumizil2005calibration`, `kull2017beta` (Guo 2017 + Naeini 2015 are shared)
- **D1 calibration_metric (proper scoring rules)**: `brier1950verification`, `murphy1973vector`
- **D2 calibration_metric (debiased ECE)**: `roelofs2022mitigating` (Kumar 2019 is shared)
- **E1 threshold_decision**: `saerens2002adjusting` (Youden 1950 + Elkan 2001 + Lipton 2014 are shared)
- **E2 power_analysis**: `obuchowski1998sample`
- **E3 foundational_text**: (none — Hastie ESL is shared)

### data-integrity/ (14 entries not in reading_list)
- **A1 splits_methodology**: `bergmeir2012timeseries`, `roberts2017blockcv`
- **A2 nested_cv**: `varma2006bias`, `cawley2010overfitting`
- **B1 leakage_taxonomy**: (none — Kapoor 2023 is shared)
- **B2 benchmark_contamination**: `magar2022contamination`, `sainz2023nlp`, `roberts2023contamination`
- **C1 text_dedup_method**: `broder1997minhash`, `indyk1998lsh`, `abbas2023semdedup`
- **D1 pretrain_dedup**: `carlini2021extracting`, `lee2022deduplicating`, `carlini2022memorization`, `penedo2023refinedweb`

### eval-ecosystem/ (4 entries not in reading_list)
- `gebru2021datasheets` — Datasheets for Datasets
- `bowman2021benchmarking` — What Will it Take to Fix Benchmarking
- `dehghani2021benchmarklottery` — The Benchmark Lottery
- (HELM / lm-eval / Inspect AI / Pineau / Croissant / Mitchell Model Cards are shared)

### prompt-injection/ (8 entries not in reading_list)
- **A1 attack_taxonomy**: `perez2022ignore`, `greshake2023indirect`, `wei2023jailbroken`
- **A2 attack_method**: `zou2023gcg`, `chao2023pair`
- **B1 defense_method**: `xie2023selfreminder`
- **C1 benchmark**: `mazeika2024harmbench`, `chao2024jailbreakbench`
- (OWASP + PINT are shared)

### datasets/ (9 entries not in reading_list — `reading_list.md` doesn't separately list datasets)
- All 11 dataset entries are dossier-specific except for `pint_benchmark_lakera` which has a counterpart in `reading_list.md` (under prompt-injection eval) and `advbench_zou` which appears via its parent paper `zou2023gcg`.

## Citation-detail disagreements (1)

| Reference | reading_list.md | dossier | Resolution |
|---|---|---|---|
| Naeini, Cooper & Hauskrecht 2015 BBQ | Cites `arXiv:1411.0760` (preprint) | Cites `https://ojs.aaai.org/index.php/AAAI/article/view/9602` (AAAI canonical) | Both URLs point to the same paper. AAAI URL is the published venue; arXiv URL is the preprint. The dossier's choice (published venue) is canonical per `citation_rules.md` "DOI / canonical URL preferred over preprint". |

## Summary

- **19 shared entries** between dossier and `reading_list.md`.
- **5 genuine gaps** where `reading_list.md` has entries the dossier should add in a follow-up round (Yan 2025, Pellizzoni 2025, PI_HackAPrompt_SQuAD, DataSentinel/PromptLocate, Open-Prompt-Injection).
- **5 deliberate exclusions** from the dossier (Efron-Tibshirani book, Recht ImageNet, PyTorch notes, two fairness papers).
- **50 dossier-only entries** — the dossier substantially extends the reading list's coverage in inference (calibration methods, ROC variance, CV variance), data-integrity (dedup, contamination, splits), eval-ecosystem (HELM paper, benchmarking critiques, Datasheets), and prompt-injection (attacks, defenses, benchmarks).
- **1 citation-detail nuance** (Naeini 2015 BBQ — both URLs are correct sources for the same paper).

## Recommended follow-up

If you want the dossier to absorb the 5 gap-entries from `reading_list.md`, the natural placements are:
- Yan 2025 → `papers/data-integrity/_dossier/01_splits_and_nested_cv.md` § A1 + synthesis update
- Pellizzoni 2025 → `papers/data-integrity/_dossier/02_leakage_and_contamination.md` § B1 + synthesis update
- PI_HackAPrompt_SQuAD 2025 → either `papers/prompt-injection/_dossier/03_benchmarks_and_standards.md` § C1 or `papers/data-integrity/_dossier/02_leakage_and_contamination.md` § B2
- DataSentinel + PromptLocate → `papers/prompt-injection/_dossier/02_defenses.md` § B1
- Open-Prompt-Injection (Liu et al.) → `datasets/dataset_ledger.yml`

`reading_list.md` itself was left untouched per the dossier-build plan.
