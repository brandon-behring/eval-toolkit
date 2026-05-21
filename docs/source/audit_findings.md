# Audit findings ledger

This ledger tracks findings from each cross-model methodology audit (Gate 3 per
[ADR 0003](adr/) — to be drafted at v0.48). Each row records the finding ID,
severity, finding summary, disposition (how it was resolved or scheduled), and
a link to the tracked issue (where applicable).

**Convention**: blocker-severity findings get a `p1-gate3`-labelled GitHub issue
for fix-tracking. Lower-severity findings are recorded here only.

**Cross-references**:
- The audit prompt template is at
  [`gate3-audit-prompt.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/.claude/plans/gate3-audit-prompt.md)
  (local — not in published docs).
- The v1.0 plan that drives audit cadence is at
  `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` (local).
- Audit re-run schedule: after each breaking minor (v0.46, v0.47, v0.48) plus
  the original Round 5 pre-implementation pass. 7-day audit-completion timeout
  per gate.

---

## Round 5 (2026-05-21) — Codex + Gemini pre-implementation audit

**Reviewers**: author (manual) + Codex (independent report) + Gemini
(independent report).

**Packet**: v0.44.0 code state + the v1.0 release plan
(`~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`) +
`docs/source/methodology/` (16 chapters) + `docs/source/roadmap.md` +
`CHANGELOG.md` + `src/eval_toolkit/*.py` + existing migration guides
(`migration/v0.7.md`, `v0.8.md`, `v0.9.md`).

**Audit prompt**:
[`~/.claude/plans/gate3-audit-prompt.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/.claude/plans/gate3-audit-prompt.md).

| ID    | Severity              | Finding                                                                                              | Disposition                                                                                            | Issue |
|-------|-----------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-------|
| R5-F1 | blocker before v0.46  | `scorecard()` first-party metric list mixes threshold-free + threshold-dependent metrics; `MetricSpec.compute(y, s) -> float` has no threshold input | **Decision R**: drop F1/accuracy/precision/recall from v0.46 specs; keep `metrics_at_threshold` + `ThresholdSelector` as separate path | _(in plan)_ |
| R5-F2 | blocker before v0.46  | `Scorecard` result type has no contract for undefined/skipped/error cells; existing `MetricState` infra unused | **Decision S**: reuse `MetricState` (`ok/skipped/error`) vocabulary from `artifacts.py:30-61`           | _(in plan)_ |
| R5-F3 | blocker before v0.47  | Sweep unification plan assumes `DelimitVariant`/`DatamarkVariant` classes exist; `preprocessing.py` exports only functions | Plan revised: create 3 preprocessing dataclasses at v0.47 as part of sweep work; sweep contract clarified (neutral default; explicit `attack_threshold` required for ASR) | _(in plan)_ |
| R5-F4 | blocker before v0.46  | Plan's scalar-deprecation directive would replace the lazy export resolver (`__init__.py:302-312`), shattering all root imports | Plan corrected: extend existing `__getattr__` with a transitional deprecation branch (BEGIN/END markers); v0.47 removes only that branch, base resolver intact | _(in plan)_ |
| R5-F5 | blocker before v1.0   | DeLong (`DeLongResult`, `delong_roc_variance`) is publicly exported + in API docs, but methodology + roadmap docs say "out of scope" | **Decision U**: keep public; update `methodology/comparison.md`, `methodology/reading_list.md`, `roadmap.md` to align with shipped state. Bundled into v0.48 §5E-prep | _(in plan)_ |
| R5-F6a | packet drift          | `cv_clt_ci` docstring claims "Bayle et al. 2020 prove a CV-CLT with a correction factor"; code does naive sample variance (no scalar correction) | Docstring fix bundled into v0.48 §5E-prep. Code is correct per Bayle (2020) Thm 3.1; docstring oversells | _(in plan)_ |
| R5-F6b | packet drift          | `methodology/parallelism.md:143-181` says "as of v0.34, harness not yet parallelized" + "once #29/#30 land" — but v0.36 wired `evaluate(n_jobs=)` + `evaluate_folded(n_jobs=)` | Doc fix bundled into v0.48 §5E-prep. Also clarify `bootstrap_ci`'s `n_jobs` is studentized-only       | _(in plan)_ |
| R5-F6c | packet drift          | `methodology/testing.md:108-136` says reference-equivalence gap "closing in PR 1.5"; roadmap state shows it's closed | Doc fix bundled into v0.48 §5E-prep                                                                    | _(in plan)_ |
| R5-F6d | packet drift          | `methodology/calibration.md:15-18` lists only temperature/isotonic/Platt; Beta + 4-binary-adapter family also shipped | Doc fix bundled into v0.48 §5E-prep                                                                    | _(in plan)_ |
| R5-F6e | packet drift + code   | `methodology/bootstrap.md` two-level example uses same array for val + test, causing ~63.2% overlap when `paired_bootstrap_op_point_diff` resamples val/test independently | Doc fix + defensive code guard bundled into v0.48 §5E-prep: rewrite example with disjoint split + raise `ValueError` in `paired_bootstrap_op_point_diff` if `val_y is test_y` | _(in plan)_ |
| R5-F6f | partial verified      | `CostSensitiveSelector` formula `t* = c_FP·(1-π) / (c_FP·(1-π) + c_FN·π)` is the prior-corrected form; existing docstring already cites Elkan's prior-independent variant — intentional design, but easy to misuse on already-calibrated probabilities | Docstring sharpening (add `Warning` admonition) bundled into v0.48 §5E-prep. Math itself is correct per the documented intent | _(in plan)_ |
| R5-F7 | governance risk       | Gate 3 redefinition (multi-model cross-review) is useful but not the same evidence class as external academic peer review; the plan understated this | **Decision O revised**: ADR 0003 explicitly states Gate 3 at v1.0 is internal model-assisted cross-review, NOT external academic peer review; documents what it catches and doesn't | _(in plan)_ |
| R5-Px | plan internal inconsistency | v0.46 verification smoke imports `TextTransform`, but Decision K introduces it at v0.47 | Plan smoke test corrected — `TextTransform` import removed from v0.46 smoke | _(in plan)_ |

**Round 5 totals**: 12 findings (5 blockers, 1 high-pre-v1, 5 packet drift, 1 governance, 1 plan inconsistency). All resolved or scheduled before any v0.46 code shipped.

**Key follow-on decisions** (driven by Round 5):
- **Decision W**: packet-drift fixes (F6a–F6f) bundle into v0.48 polish (not earlier hotfix).
- **Decision X**: scorecard skipped-status detection reuses `is_metric_defined_for_slice` from v0.39.0.
- **Decision Y**: Round 6 / 7 / 8 audits run as formal stop-gates between v0.46 / v0.47 / v0.48 / v1.0; 7-day timeout per gate.
- **Decision Z**: this ledger file.

---

## Round 6 (active: post-v0.46 ship — pending Codex + Gemini reports)

**Ship date**: v0.46.0 tagged + published to PyPI 2026-05-21. STOP-GATE per
Decision Y.2 — `release/v0.47.0` cannot open until this audit completes (or
the 7-day timeout from 2026-05-21 expires).

**Reviewers**: author (manual) + Codex (independent) + Gemini (independent).

**Packet to send to Codex + Gemini**:

- This plan file (`~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`)
- This audit ledger (`docs/source/audit_findings.md`)
- `docs/source/methodology/` (16 chapters — unchanged from Round 5)
- `docs/source/roadmap.md`
- `docs/source/migration/v0.46.md` (**NEW** — consumer migration guide)
- `docs/source/adr/0002-scorecard-as-primary-metric-surface.md` (**NEW**)
- `CHANGELOG.md` (new v0.45.0 + v0.46.0 entries since Round 5)
- Source tree, with focus on the v0.45 + v0.46 additions:
  - `src/eval_toolkit/stacking.py` (v0.45, ~370 lines)
  - `src/eval_toolkit/_scorecard.py` (v0.46, ~430 lines)
  - `src/eval_toolkit/metric_specs.py` (v0.46, ~180 lines)
  - `src/eval_toolkit/__init__.py` — `__getattr__` deprecation branch +
    `_DEPRECATED_SCALARS` (Audit F4 invariant region)
  - `src/eval_toolkit/metrics.py:120-200` —
    `SINGLE_CLASS_INCOMPATIBLE_METRICS` extension (Round-5 X.2 precondition)

**Audit prompt**:
[`~/.claude/plans/gate3-audit-prompt.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/.claude/plans/gate3-audit-prompt.md)
(local). The "Known issues already in the v0.48 backlog (skip re-reporting)"
section already lists drift items scheduled for v0.48 polish — Round 6
reviewers should skip those and surface only NEW findings against the v0.46
state.

**Focus areas** for Round 6 review:

- **scorecard surface design lock-in.** The Tier-2 `MetricSpec` Protocol
  freezes at v1.0 — method-signature changes require a v2.0 major bump.
  Last cheap chance to catch contract gaps.
- **MetricResult cell-state contract** — does the `ok` / `skipped` /
  `error` vocabulary cover every relevant failure mode? Are the reason
  strings useful for triage?
- **Per-cell error isolation** — confirm that catching all exceptions in
  `_evaluate_spec` doesn't hide important failures the user should see.
- **`__getattr__` deprecation shim** — Audit F4 invariant: does the branch
  correctly route deprecated names, NOT break non-deprecated resolution,
  and cleanly delete at v0.47?
- **Spec name encoding for parameterized metrics** — is
  `"ece_n_bins_15_strategy_uniform"` a stable v1.0 commitment, or does the
  alphabetize-kwargs rule create surprise keys for custom user specs with
  multi-kwarg signatures?
- **`Scorecard.to_pandas()` MultiIndex schema** — first-time-public; any
  shape lock-in concerns?

**Triage on findings**: each blocker → `p1-gate3`-labelled GitHub issue +
a row in this ledger. Either fix-as-v0.46.1-hotfix or fold into v0.47
design (per Decision Q severity-tiered hotfix policy).

| ID | Severity | Finding | Disposition | Issue |
|----|----------|---------|-------------|-------|
| _pending Codex + Gemini reports_ | | | | |

---

## Round 7 (planned: post-v0.47 ship) — STOP-GATE before v0.48 release branch

_To be populated after v0.47 ships. Focus: sweep + `TextTransform` Protocol
lock-in; advanced-6 character_injection structural satisfaction; SimpleNamespace
removal acceptance._

| ID | Severity | Finding | Disposition | Issue |
|----|----------|---------|-------------|-------|
| _pending_ | | | | |

---

## Round 8 (planned: post-v0.48 ship) — STOP-GATE before v1.0 tag

_To be populated after v0.48 ships. Focus: final pre-v1.0 packet, including the
7 packet-drift fixes from §5E-prep, ADRs 0001 + 0002 + 0003, all locked
decisions A–Z reflected in shipped state._

| ID | Severity | Finding | Disposition | Issue |
|----|----------|---------|-------------|-------|
| _pending_ | | | | |
