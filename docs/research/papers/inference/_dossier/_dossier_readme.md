# Dossier — Inference cluster (eval-toolkit)

**Compiled**: 2026-05-14
**Source ledger**: `../bib_ledger.yml` (24 entries)
**Research plan**: `../research_plan.md`
**Agent-readable index**: `../README.md` + `../01_*.md` … `../05_*.md` (5-bullet synthesis; preferred entry point for LLM agents)

## Stats

| File | Topic | Entries |
|---|---|---|
| `01_bootstrap_and_cv_variance.md` | Bootstrap CI foundations (A1) + CV variance (A2) | 4 |
| `02_roc_variance.md` | ROC / AUC variance and comparison tests (B1) | 4 |
| `03_calibration_methods.md` | Post-hoc probability calibration methods (C1) | 6 |
| `04_calibration_metrics.md` | Proper scoring rules (D1) + ECE estimator bias (D2) | 4 |
| `05_thresholds_power_foundations.md` | Threshold selection (E1) + power analysis (E2) + foundational text (E3) | 6 |
| **Total** | | **24** |

## Entries per claim_family

| claim_family | Count |
|---|---|
| `bootstrap_ci` | 1 |
| `roc_variance` | 4 |
| `cv_variance` | 3 |
| `calibration_method` | 6 |
| `calibration_metric` | 4 |
| `threshold_decision` | 4 |
| `power_analysis` | 1 |
| `foundational_text` | 1 |

## Status field summary

- `verified` (WebFetch/WebSearch-confirmed title + first-author + year, post round-1 audit): **24** entries (all).
- `unverified`: 0.
- `mismatched`: 0.

## Audit history

Two independent audit rounds run on 2026-05-14 (full details in `../README.md` § Verification & limits):
- Round 1 — attribution correctness for 17 unverified entries + 2 spot-checks. 0 DROP / 0 CORRECT / 2 FLAGs applied (one generalization, one bibkey-naming note kept as-is).
- Round 2 — Mechanism / Result bullet claim accuracy + lookup-recipe sanity. 0 DROP / 0 CORRECT / 2 FLAGs applied (Elkan formula generalized to full four-cell form; Kumar "upward-biased" softened to "biased"). Lookup recipes all point to valid anchors.

## Notes

- PDF caching was deferred — none of the entries have a local PDF. The user can re-run `/research-gather --cache-pdfs` if local copies become needed.
- Raw 7-column dossier tables (this folder) are kept alongside the agent-index synthesis (one folder up) for human review and audit trail purposes; the synthesis files are the preferred entry point for LLM agents.
