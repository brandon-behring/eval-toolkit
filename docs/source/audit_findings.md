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

## Round 6 (complete: 2026-05-21 — Codex + Gemini + manual; 11 findings)

**Ship date**: v0.46.0 tagged + published to PyPI 2026-05-21. STOP-GATE per
Decision Y.2 — `release/v0.47.0` opens once R6-F1 (BLOCKER) + R6-F2 (HIGH)
are fixed via v0.46.1 hotfix and the remaining items are scheduled.

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
  - `src/eval_toolkit/scorecards.py` (renamed from `_scorecard.py` at v0.49.0; ~585 lines)
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

| ID | Reviewer | Severity | Finding | Disposition | Lands |
|----|----------|----------|---------|-------------|-------|
| R6-F1 | Codex | **BLOCKER** before v0.47 opens | `metric_specs.ece(strategy="typo")` silently dispatches to quantile ECE and returns scorecard cell with `status="ok"` under invalid key (`"ece_n_bins_15_strategy_typo"`). Wrong-by-design data correctness path. Verified via Codex runtime probe. | Add strategy validation in `ece()` factory + `_EceSpec.compute()`; raise `ValueError("ECE strategy must be 'uniform' or 'quantile'; got {strategy!r}")` (plan §2.5A). | **RESOLVED v0.46.1** (commit `7a4bb14`, tag `v0.46.1` 2026-05-21; consumer pin bumped same day) |
| R6-F2 | Codex + Gemini | HIGH before v0.47 scalar hard-removal | ECE deprecation warnings in `__init__.py:_scorecard_spec_for()` emit broken migration snippets for all 5 ECE variants. Two-part bug: (a) for the 2 variants in `metric_specs`, the suggested scorecard key uses the factory-call expression (`"ece(n_bins=10)"`) instead of the encoded spec name (`"ece_n_bins_10_strategy_uniform"`); (b) for the 3 variants NOT in `metric_specs` (`_debiased`, `_l2`, `_l2_debiased`), the fallback name isn't an importable spec. Gemini claimed pre-v0.46 default was `n_bins=15` (verified incorrect — code at `metrics.py:730-734` shows `n_bins=10`); Decision R6-F resolves: warning uses `n_bins=10` to preserve pre-v0.46 math + adds migration note about new factory default. | Restructure `_scorecard_spec_for()` to return `(factory_expr, scorecard_key, has_first_party)` tuple; correct snippets for first-party variants with `n_bins=10`; submodule-path template for 3 non-first-party variants per Decision R6-G (plan §2.5B). | **RESOLVED v0.46.1** (commit `7a4bb14`, tag `v0.46.1` 2026-05-21; consumer-side smoke verified all 5 ECE-variant warnings + submodule-path routing) |
| R6-F3 | Codex | HIGH before scorecard freeze | Duplicate `MetricSpec.name` values in the same `scorecard()` call silently overwrite earlier cells (last-wins). Not a documented contract. | Decision R6-B (locked): reject in `scorecard()` with `ValueError("Duplicate MetricSpec name 'X' at index N; ...")`. Forces caller to disambiguate; no silent data loss. (Plan §4G.) | **v0.47** |
| R6-F4 (= Gemini R6-F1) | Codex + Gemini | HIGH before v1.0 | `scorecard(seed=None)` documented as non-deterministic; implementation coerces `None → 0`. Doc/impl contradiction. Verified by Codex via bit-for-bit equality test. | Decision R6-A (locked): deterministic-by-default; fix docs only. No behavior change. Plan §4G-prep. (Decision R6-E: rolls to v0.47 — R6-A is non-blocker per Decision Q's "docstring" category.) | **v0.47** |
| R6-F5 | Codex | Contract-enforcement gap before v1.0 | ADR 0003 promises strict Tier-2 Protocol method-shape stability; current public-API drift guard only snapshots `(*args, **kwargs)` for Protocol classes, not method signatures. The guard does not see changes to `MetricSpec.compute`, `MetaLearner.fit`, etc. | Decision R6-D (locked): extend `tests/test_public_api.py` snapshot to capture Protocol method signatures via `inspect.signature` + `typing.get_type_hints` for the 9 Tier-2 Protocols. (Plan §4I.) | **v0.47** |
| R6-F6 | Codex | Packet drift | v1.0 plan + roadmap still describe pre-v0.46 scorecard shapes that didn't ship: `ece_n_bins_15` without strategy in plan, `ece_quantile()` factory listed (shipped as `ece(strategy='quantile')`), `MetricUndefinedError` mentioned (ADR 0002 chose no new public exception), `n_resamples >= 100` floor (shipped is `>= 1`). Roadmap "Currently shipped" still says v0.44. | Plan §4L: refresh plan §3A scorecard examples + roadmap shipped-state section. Doc-only commit on v0.47 release branch. | **v0.47** |
| R6-F3 (Gemini) | Gemini | MEDIUM (schema lock-in before v1.0) | `Scorecard.to_pandas()` MultiIndex columns expose `value, status, reason, ci_low, ci_high, confidence` but drop `n_resamples` + `method` from `BootstrapCI`. Provenance loss compared to `to_dict()`. v1.0 is about to lock the schema. | Decision R6-C (locked): add `n_resamples` + `method` columns at v0.47 (additive). Schema becomes lossless against `to_dict()`. (Plan §4H.) | **v0.47** |
| R6-F4 (Gemini) | Gemini | LOW | `MetricSpec` Protocol doesn't enforce stable parameterized-spec naming. Custom users implementing multi-kwarg parameterized specs can silently spawn distinct dict keys if constructor arg order varies. | Decision R6-H (locked): add `make_spec_name(prefix, **kwargs)` canonicalization helper in `metric_specs.__all__` only (NOT top-level `_EXPORTS` — Tier-2 additive contract). Alphabetized kwargs, snake_cased, joined by underscore. (Plan §4J.) | **v0.47** |
| R6-F5 (Gemini) | Gemini | LOW | `_evaluate_spec()` wraps `spec.compute()` in broad `except Exception`. Swallows `MemoryError`, `RecursionError`, `KeyboardInterrupt`, `SystemExit` into cell state — process exhaustion / user-interrupt signals get hidden as metric errors. | Narrow exception catch: `except (MemoryError, RecursionError, KeyboardInterrupt, SystemExit): raise` first, then existing broad catch. (Plan §4K.) | **v0.47** |

**Round 6 totals**: 11 findings (Codex 6 + Gemini 5; 2 overlap on `seed=None` + ECE deprecation snippets but with different reasoning angles). 1 BLOCKER (R6-F1) + 5 HIGH + 2 MEDIUM/contract + 3 LOW. All dispositioned to either v0.46.1 (2 fixes) or v0.47 (9 fixes).

**Key follow-on decisions** (driven by Round 6 — locked in plan):

- **Decision R6-A**: `seed=None` deterministic-by-default; fix docs only.
- **Decision R6-B**: Reject duplicate `MetricSpec.name` with `ValueError`.
- **Decision R6-C**: Add `n_resamples` + `method` to `to_pandas()` schema.
- **Decision R6-D**: Extend public-API snapshot to cover Protocol method signatures.
- **Decision R6-E**: v0.46.1 scope = R6-F1 + R6-F2 only; R6-A rolls to v0.47 (non-blocker per Decision Q's "docstring" category).
- **Decision R6-F**: Use `n_bins=10` (pre-v0.46 default) in deprecation warnings + migration note about new v0.46+ factory default of `n_bins=15`. Corrects Gemini's misverified pre-v0.46 default claim.
- **Decision R6-G**: 3 ECE variants without `metric_specs` (debiased, l2, l2_debiased) route deprecation warnings to submodule path; do NOT add to `metric_specs` at v0.47.
- **Decision R6-H**: `make_spec_name()` helper in `metric_specs` submodule only; not top-level.

### Round 6 v0.46.1 ship status (2026-05-21)

- **R6-F1** ✅ SHIPPED in v0.46.1 (PR #67, squash `7a4bb14`). End-to-end verified
  in consumer: `ms.ece(strategy="typo")` raises `ValueError`; direct
  `_EceSpec(strategy=...)` construction also raises (defence-in-depth).
- **R6-F2** ✅ SHIPPED in v0.46.1 (PR #67, squash `7a4bb14`). End-to-end verified
  in consumer: `eval_toolkit.expected_calibration_error` warning carries
  `ece(n_bins=10)` + key `ece_n_bins_10_strategy_uniform` + migration note about
  v0.46+ `n_bins=15` default. All 3 non-first-party variants (debiased, l2,
  l2_debiased) route to `from eval_toolkit.metrics import …` submodule path.
- **9 other Round 6 items** (R6-A docstring, R6-B duplicate-name guard, R6-C
  to_pandas schema, R6-D Protocol method-shape snapshot, R6-F4-Gemini
  `make_spec_name`, R6-F5-Gemini narrow `except`, R6-F6 plan/roadmap state-drift)
  → folded into `release/v0.47.0` per Decision R6-E.
- **Round 6 STOP-GATE status**: CLOSED. `release/v0.47.0` can open after the
  v0.46.1 consumer cycle observation completes (1 cycle).

---

## Round 7 (complete: 2026-05-21 — Codex + Gemini; 3 substantive findings)

**Reviewers**: author (manual) + Codex (independent report) + Gemini
(independent report).

**Packet**: v0.47.0 code state + the v1.0 plan + `docs/source/methodology/`
(16 chapters) + ADRs 0001/0002/0003 + `docs/source/migration/v0.46.md` +
`docs/source/migration/v0.47.md` + Round 5/6 ledger.

**Round-7 briefing**: `gate3-audit-round-7.md` (committed `a9e1114`).

**Reports**: `gate3-audit-round-7-codex-report.md` + `gate3-audit-round-7-gemini-report.md` (untracked per `.gitignore`).

**Headline**: Codex 3 substantive findings; Gemini 0. Overlap was zero
between the two reports — the most consequential finding (R7-F1
doc-migration boundary gap between Sybil-tested fences and
MyST-NB-executed example notebooks) was Codex-only. Reinforces the
Round 6 pattern (do not use overlap as a confidence floor; single-reviewer
findings can be the most critical).

| ID | Reviewer | Severity (their words) | Finding | Disposition | Lands |
|----|----------|------------------------|---------|-------------|-------|
| R7-F1 | Codex | high before v0.48 | v0.47 doc migration missed MyST-NB executable example notebooks (separate from Sybil-collected `.md` fences). 6 example pages + 4 module-level docstrings + `protocols.md` autosummary + roadmap wording still reference removed APIs. Docs CI runs `sphinx-build` without `-W`, so notebook execution failures pass as advisory warnings. Verified via `sphinx-build` runtime probe — 6 execution failures buried in the warning stream. | Decision R7-A (locked at /exploring-options Q3): bundle into v0.48 §5G/§5H. §5G migrates the 6 notebooks + 4 docstrings + autosummary + roadmap; §5H enables `nb_execution_raise_on_error = True` in `conf.py`. Audit-as-seed expansion (Q2 locked full sweep) covered ALL module docstrings + drift in 5 existing `api/*.md` autosummary lists + 8 missing `api/*.md` pages. | **RESOLVED v0.48.0** (§5G commit `e07db16` + §5H commit `6349472` on `release/v0.48.0`) |
| R7-F2 | Codex | high before sweep freezes | `sweep()` records only `strategy.name` per row; two configured instances of same dataclass (e.g., `DelimitVariant(delimiter="<<")` + `DelimitVariant(delimiter="[[")`) silently merge under `groupby("variant")`. Style-coherent defect class with Round 6 R6-F3 (scorecard duplicate name) but with different semantics (row container vs. Mapping). | Decision R7-B option C (locked): emit `strategy_id` canonical column AND reject duplicate `strategy_id` at sweep boundary. Style invariants 1 (no silent failures) + 2 (natural call pattern is right) + 4 (canonical identifier + reject in canonical dimension) read together. | **RESOLVED v0.48.0** (§5I commit `f454afe`) |
| R7-F3 | Codex | worth fixing before v1.0 | `sweep()` doesn't validate scorer output cardinality. Three failure modes via runtime probe: overlong 1-D → silent truncation (worst); short 1-D → IndexError later; (n,2) matrix → TypeError when `float()` applied. | Decision R7-C (locked): API-level `ValueError` with contextual label at the sweep boundary; replaces all three low-level failure modes. Style invariants 1 + 3. | **RESOLVED v0.48.0** (§5J commit `fcf99f0`) |

### Gemini observations (Round 7)

Gemini's report verdict was "highly stable; release/v0.48.0 is safe to open." Six minor observations / validations; nothing critical that Codex hadn't covered. The actionable items folded into v0.48:

- §1-3 + 5-7: VALIDATIONS of v0.47 shipped state (`TextTransform` shape, shim removal, sweep design, R6-D Protocol method-shape snapshot, ADR 0003 tiers). No action needed.
- §4 (pedagogical drift): Gemini noted "from eval_toolkit.metrics import pr_auc" is syntactically green but slightly undermines ADR 0002. v0.48 §5G migration explicitly chose `scorecard()` for example notebooks teaching METRIC USAGE; submodule path only where teaching the underlying math.
- §4 (Makefile pre-push): Gemini recommended hardening to prevent the `pytest tests/` path-override trap. Landed as v0.48 §5L (`make pre-push` target running all 3 doc-execution surfaces; commit `9878a54`).
- §5 (R6-C dtype coercion): Gemini noted `n_resamples` (int + NaN) → `float64` is an accepted tradeoff. Landed as v0.48 §5K (Notes section on `Scorecard.to_pandas()` docstring; commit `6304cea`).
- §6 (SynonymSubstitution whitelist): Gemini recommended adding a docstring note about the hardcoded 6-entry whitelist. Landed as v0.48 §5K (`adversarial.py` Notes section; commit `6304cea`).

### Audit-as-seed extensions (v0.48)

Per user direction during plan refinement ("use the audits as seeds for things to reconsider"), the Round 7 findings + style-invariants framing surfaced additional v0.48 scope beyond Codex's explicit list:

- **§5G expansion**: from 4 Codex-flagged module docstrings to full sweep across `src/eval_toolkit/` module docstrings + audit of all `docs/source/api/*.md` autosummary pages. Found 8 missing API pages + 5 drifted autosummary lists.
- **§5M new**: in-source docstring drift audit (third doc-execution surface). Result: 82 PASS / 1 skipped / 0 fail; expanded `.doctest-modules` from 11 → 21 modules so CI catches future drift.
- **§5N comprehensive**: cross-API shape-validation consistency sweep beyond Codex's R7-F3 target. Audited `metrics_at_threshold`, `paired_bootstrap_op_point_diff`, `bootstrap_metric_from_predictions`, `metrics.py` scalars, `fit_*_binary` / `fit_*_calibrator`. Tightening commit landed for `metrics_at_threshold` silent threshold semantics (commit `76773dc`); `paired_bootstrap_op_point_diff` `val_y is test_y` guard landed as part of §5E-prep code-side fix (commit `5c8e68d`).

### Round 7 ship status

- **3 substantive Codex findings**: all RESOLVED in v0.48.0 via §5G + §5H + §5I + §5J.
- **6 Gemini observations**: all RESOLVED in v0.48.0 via §5G + §5K + §5L.
- **Audit-as-seed extensions** (§5G expanded, §5M new, §5N comprehensive): all RESOLVED in v0.48.0.
- **Round 7 STOP-GATE status**: CLOSED via v0.48.0 release. Round 8 audit STOP-GATE per Decision Y.2 opens against the v0.48.0 state before `v1.0.0` tag can land.

---

## Round 8 (2026-05-24) — STOP-GATE before v1.0 tag, rectified in v0.51.0

Multi-LLM cross-review fired against the v0.50.0 state. **Codex** returned
10 findings (3 with reproducible probes: C3, C4(a)/(b), C9); **Gemini**
returned 5 findings + 3 positive validations (V1 style, V2 methodology,
V3 scorer_error safety). Independent verification by Claude
(`audit-verification-codex-gemini-v0.50.0.md`, 510 lines, untracked
per `.gitignore`) confirmed 13 of 18 items, refuted 3, left 2 undecided.
The v0.51.0 release ships fixes for all 13 confirmed items; 2 (G3, G4)
are deferred to v1.x as Tier-2 additive; 3 refuted entries are recorded
below for the audit trail.

Round 9 multi-LLM cross-review runs against the v0.51 RC before v1.0
tags.

### Confirmed (13) — fixed in v0.51.0

| ID | Severity | Finding | Disposition | Commit |
|----|----------|---------|-------------|--------|
| R8-C3 | blocker (escalated from Codex fix-rec) | `recall_at_fpr` fallback violated FPR ceiling (`actual_fpr=1.0` when `target_fpr=0.0`); probe `y=[0,1], scores=[1.0,1.0]`. | v0.51 sentinel `threshold=np.inf, actual_fpr=0.0, fp=0`. | `672d45f` |
| R8-C4(a) | blocker (escalated) | Generator-rng not parallel-stable in `_score_all_slices`; n_jobs=1 vs n_jobs=2 produced different CIs; SPEC 7 contract violation. | v0.51 spawns child SeedSequences at the dispatch boundary. | `61964f6` |
| R8-C4(b) | blocker (escalated) | `spawn_seed_sequences` ignored Generator state (extracted `bit_generator.seed_seq`). | v0.51 draws fresh entropy via `rng.integers(...)`. | `87453f6` |
| R8-C1 | fix-recommended | `evaluate_folded(seeds=)` reused the splitter across the seed loop — partitions never varied. | v0.51 adds `reseed_splitter` callback + DeprecationWarning when multi-seed + None. Warning persists past v1.0. | `f60d43c` |
| R8-C2 | fix-recommended | `SourceDisjointKFoldSplitter.iter_folds` yielded empty test partitions when k > n_sources; `get_n_splits` returned `min(k, n_sources)`. | v0.51 caps iter_folds at `min(k, n_sources)` + UserWarning. | `f60d43c` |
| R8-C5 | fix-recommended | README links to `docs/...` (broken); migration toctree listed only v0.7–v0.9 despite v0.49–v0.51 BREAKING releases. | v0.51 repoints all README links to `docs/source/...`; toctree + MIGRATION.md index extended through v0.51; v0.49/v0.50/v0.51 migration guides authored. | `c206b54` |
| R8-C6 | fix-recommended | `calibration.reliability_curve`, `maximum_calibration_error` did not validate `y_score ∈ [0,1]`; `fit_temperature` did not validate `bounds`. | v0.51 Tier-2 additive validation matching `metrics.py`-side ECE rigor. | `4c5e140` |
| R8-C7 | fix-recommended | Tests covered counts where semantics matter: `test_harness_folded.py:92-107` no partition-content check; `test_harness_parallelism.py:77-122` no Generator-rng; `test_protocol_conformance.py:228-240` no k>n_sources. | v0.51 adds semantic-property regression tests bundled with C1/C2/C4 fixes. | `87453f6` + `61964f6` + `f60d43c` |
| R8-C8 | minor | `SimilarityStrategy` listed as Tier-2 in README + `extending.md` but absent from `strict_tier2_protocols.md`'s 9-strict list. | v0.51 demotes SimilarityStrategy to "pre-v0.7 internal interface" in README + extending.md (aligns to v1.0 contract per ADR 0003). | `c206b54` |
| R8-C9 | minor | `GateResult.to_dict()` returned numpy + NaN unchanged; `json.dumps(..., allow_nan=False)` raised TypeError on the result. | v0.51 docstring documents the JSON-safety contract — strict-JSON requires `artifacts.write_json_strict` or `sanitize_for_json`. | `c206b54` |
| R8-C10 | minor | `.gitignore` patterns covered only `gate3-audit-*-report.md`; new `codex-comprehensive-audit-*-report.md` + `audit-verification-*.md` were tracked by default. | v0.51 `.gitignore` extended with new patterns. | `c206b54` |
| R8-F1 | fix-recommended (Claude — missed-by-both) | `losses.RecallAtLowFPR.__init__` did not validate `pos_weight > 0`; non-positive values produced degenerate-but-bounded loss values silently. | v0.51 eager validation matching sibling-kwarg pattern. | `4c5e140` |
| R8-F2 | minor (Claude — missed-by-both) | `metric_specs.ece(n_bins=)` validated `strategy` eagerly but deferred `n_bins` to compute time. | v0.51 eager `_validate_n_bins` call at factory level. | `4c5e140` |
| R8-F3 | minor (Claude — missed-by-both) | `analysis.CsvPredictionReader.read_predictions` silently filled missing CSV columns with empty strings → cryptic dtype error downstream. | v0.51 detects missing columns at read time; raises actionable `ValueError`. | `4c5e140` |

### Refuted (3) — recorded for audit trail; NO fix shipped

| ID | Severity (as auditor marked it) | Finding (rejected) | Rationale |
|----|---------|---------|-----------|
| R8-G2 | fix-recommended (Gemini) | "Cyclic import leakage→harness identified but left unresolved." | REFUTED: `harness.py:709-712` resolves the cycle via TYPE_CHECKING + lazy `noqa: PLC0415` import; comment explicitly states the resolution. Cycle is deliberately broken by an asymmetric-by-design pattern, not unresolved. |
| R8-G5 | minor (Gemini) | "Plotting tests use weak assertions (`assert fig is not None`)." | REFUTED: `tests/test_plotting_edge.py:125-140` does have that weak assertion, but the same file has structural assertions (line 92 `assert out is fig`; lines 220, 230 `.get_title()` equality). Cherry-picked; not representative. |
| R8-V1 | "Style + consistency: Exceptional" (Gemini) | Validation that the axis is exceptional. | REFUTED: Gemini's specific tooling claim (Ruff + Black + Mypy strict compliance) is true, but the axis is broader. Codex C1 + C5 + C6 + C7 + C8 + R8-G3 confirm substantive cross-API / docs / contract consistency drifts. Tooling-clean ≠ axis-exceptional. |
| R8-V2 | "Domain methodology: Masterclass" (Gemini) | Validation that methodology is masterclass. | REFUTED: Codex C1 + C3 + C4 + C6 confirm methodology bugs at fix-recommended-or-blocker severity. Methodology has real flaws; "Masterclass" overstates. Gemini's evidence (research-grounded tests exist) conflates research-citation discipline with methodology correctness. |
| R8-V3 | "scorer_error catches predictions safely" (Gemini) | Validation that the mechanism is safe. | CONFIRMED, recorded for completeness: `harness.py:468-474` correctly re-raises `MemoryError` + `AssertionError` BEFORE the broad `except Exception`, so critical exceptions escape the silent-capture path. |

### Deferred to v1.x (2)

| ID | Severity | Finding | Disposition |
|----|----------|---------|-------------|
| R8-G3 | fix-recommended (Gemini) | Codebase uses `ValueError` for every error condition (295 raises in src/; 0 custom exception classes). | Deferred to v1.x as Tier-2 additive. Custom exception hierarchy subclassing ValueError preserves catch-compatibility; adding it post-v1.0 is non-breaking. Reconsider when a downstream consumer requests it. |
| R8-G4 | minor (Gemini) | joblib OOM hazard documented at `_parallel.py:55-59` but not structurally mitigated (no memory-aware `n_jobs` capping in `harness.evaluate`). | Deferred to v1.x. Memory-aware capping needs RAM-measurement + dataframe-size accounting — non-trivial, no clear best-practice. Caller is responsible for `n_jobs` sizing under their RAM budget at v1.0. |

### Round 8 ship status

- **13 confirmed findings**: all RESOLVED in v0.51.0 via the
  per-finding commits above (Phase 1 + 2 + 3 + 4 on `release/v0.51.0`).
- **3 refuted findings + 2 over-confident Gemini validations**:
  recorded above; no shipped fix needed.
- **2 deferred findings** (G3 + G4): Tier-2 additive; v1.x or later.
- **Round 8 STOP-GATE status**: CLOSED via v0.51.0 ship. **Round 9 audit
  STOP-GATE** per Decision Y.2 opens against the v0.51.0 RC before
  `v1.0.0` tag can land.

---
