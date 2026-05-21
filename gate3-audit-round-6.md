# Round 6 independent methodology audit — eval-toolkit v0.46.0

## Why this audit

eval-toolkit is a small Python library (~20k LOC, single-author,
single-production-consumer) on the staggered path to v1.0 API stability.
v0.46.0 shipped on 2026-05-21 — the **first BREAKING release** of the v1.0
sprint (soft-deprecation; hard removal at v0.47). Your independent read at
this gate decides whether `release/v0.47.0` can open or whether v0.46
contracts need to be revisited before more code lands on top of them.

The release plan and methodology docs were developed in collaboration with
Claude (Anthropic). Your value comes from being a **different reasoning
trace** with different training corpora, catching things Anthropic-trained
reasoning may miss.

## What I'd like

A rigorous, independent read. Not a checklist — your own judgement applied
to what's in front of you. Where you disagree with a design decision, say
so. Where you spot a methodology mistake, dig into it. Where you suspect
"this was probably AI-co-written and may not have been deeply verified,"
flag it.

**The deliverable is a single review document — nothing else.** Do not
modify any files, do not open pull requests, do not commit changes, do not
propose patches as diffs. If you spot a problem, describe it in the review;
I'll decide what to do about it. The point is your independent assessment,
not your implementation.

## Context — what shipped between Round 5 and now

Round 5 (2026-05-21) audited the v0.44.0 state + the v1.0 plan. Twelve
findings were verified-real; all are tracked in the audit ledger
(see "Packet" below).

Since Round 5:

- **v0.45.0 shipped** — non-breaking `MetaLearner` Tier-2 Protocol +
  `LogisticStacker` reference impl (closes #52). Module:
  `src/eval_toolkit/stacking.py` (~370 LOC) + 24 tests in
  `tests/test_stacking.py`.
- **PR #62 merged** (pre-v0.46 precondition) — extended
  `SINGLE_CLASS_INCOMPATIBLE_METRICS` to recognize `pr_auc` / `roc_auc`
  aliases alongside `auroc` / `auprc`, so the v0.46 `scorecard()` skipped
  detection works with both naming conventions (Decision X.2).
- **v0.46.0 shipped** — primary metric surface (`scorecard()`,
  `metric_specs` namespace, `MetricSpec` Tier-2 Protocol, `MetricResult`,
  `Scorecard`); top-level scalar metric imports soft-deprecated via
  `__getattr__` shim (hard removal at v0.47). Closes #36.
- **ADRs 0001 + 0002 + 0003 published** — flat-module layout, scorecard
  as primary metric surface, tiered stability contract + Gate 3 framing
  (this audit's governance basis).

## Scope

Please look at:

1. **The v1.0 plan** at `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`
   — locked decisions A through Z, release sequence, semantic stop-gates,
   risks. (The plan now has Round 5's findings integrated; you don't need
   to re-derive them.)

2. **The audit findings ledger** at `docs/source/audit_findings.md` —
   Round 5's 12 findings with dispositions. This is what you should
   compare against: if you find an issue that matches a Round 5 entry, it
   either ISN'T resolved properly OR you've found a related-but-distinct
   issue (worth flagging separately).

3. **The methodology curriculum** at `docs/source/methodology/` (16
   chapters: leakage, splits, thresholds, calibration, comparison,
   fairness, reproducibility, testing, bootstrap, text dedup,
   versioning, length stratification, artifacts, claims, evidence,
   parallelism). Unchanged from Round 5; if you reviewed this before, you
   can re-skim rather than re-read.

4. **New since Round 5** (focus your time here):
   - `docs/source/adr/0001-flat-module-layout.md` — flat-module decision
     for v1.x with revised trigger criteria.
   - `docs/source/adr/0002-scorecard-as-primary-metric-surface.md` —
     scorecard surface design + alternatives considered + v2.0 trigger.
   - `docs/source/adr/0003-stability-contract-and-gate3-methodology.md` —
     tiered stability contract (Tier 1 strict / Tier 2 additive-only /
     Tier 3 free) + honest Gate 3 framing.
   - `docs/source/migration/v0.46.md` — consumer migration guide.
   - `src/eval_toolkit/stacking.py` (v0.45) — `MetaLearner` Protocol +
     `LogisticStacker` impl. Tier-2 Protocol freezes at v1.0.
   - `src/eval_toolkit/_scorecard.py` (v0.46, ~430 LOC) — `MetricSpec`
     Protocol + `MetricResult` + `Scorecard` + `scorecard()` function.
     Tier-2 Protocol freezes at v1.0.
   - `src/eval_toolkit/metric_specs.py` (v0.46) — first-party spec
     namespace: `pr_auc` / `roc_auc` / `brier` singletons + `ece` factory.
   - `src/eval_toolkit/__init__.py` — the `__getattr__` deprecation branch
     between BEGIN/END TRANSITIONAL markers + `_DEPRECATED_SCALARS` set.
     **Audit F4 invariant region** — the branch must coexist with the
     base lazy resolver without breaking non-deprecated symbol lookup.

5. **Source code** for cross-checking when the docs make specific claims.
   The library is small and the v1.0 plan + ADRs identify the most
   load-bearing modules.

You don't have to read it all linearly. Sample where your judgment is
sharpest.

## Focus areas for Round 6

In rough priority order — but if something else jumps out at you, surface
it:

### 1. Scorecard surface design lock-in (HIGH PRIORITY)

The `MetricSpec` Tier-2 Protocol freezes at v1.0. Method-signature changes
require a SemVer-major (v2.0) bump per ADR 0003. **Last cheap chance** to
catch contract gaps. Specifically:

- Does the `name: str` + `compute(y_true, y_score) -> float` Protocol
  cover what users will actually want from custom specs?
- The threshold-dependent metrics (F1, accuracy, precision, recall) are
  intentionally absent per Decision R. Is the rationale sound, or is this
  a v1.0 mistake the consumer will regret?
- Is the LRU-cached factory pattern (`ece(n_bins=15) is ece(n_bins=15)`)
  a stable v1.0 commitment, or does it lock in implementation details?

### 2. MetricResult cell-state contract

Does the `ok` / `skipped` / `error` vocabulary cover every relevant failure
mode? Are the reason strings useful for triage? Cases to consider:

- Single-class slice + ranking metric (current: `skipped`).
- `n < 10` samples + `bootstrap=True` (current: `ok` with `ci=None` +
  populated reason — is this right or should it be `error`?).
- Custom user spec raising deep inside compute (current: `error` with
  exception class + message).
- A spec name not in `is_metric_defined_for_slice`'s incompatibility
  registry on a single-class slice (current: falls through to
  compute-and-catch, becomes `error`).

### 3. `__getattr__` deprecation shim (Audit F4 invariant)

Round 5 caught the original plan's directive to "delete the whole
`__getattr__` block at v0.47" as a footgun — `__getattr__` is the
load-bearing lazy resolver for every `_EXPORTS` entry. The corrected
implementation extends with a discrete BEGIN/END branch. Verify:

- The branch correctly routes the 8 deprecated names (`pr_auc`,
  `roc_auc`, `brier_score`, 5 ECE variants) with appropriate warnings.
- The base resolver (`module_name = _EXPORTS.get(name); ...`) still
  fires for every non-deprecated name.
- Unknown names still raise `AttributeError` (the branch doesn't
  swallow).
- `tests/test_deprecated_scalars_shim.py` exercises the right surface —
  any test gap?

### 4. Spec name encoding for parameterized metrics

`ece(n_bins=15)` produces key `"ece_n_bins_15_strategy_uniform"`.
Alphabetized kwargs, snake-cased, joined by underscore. Concerns:

- Stable across Python versions? (kwargs ordering, str-of-int, etc.)
- What about user specs with multi-kwarg signatures — is the rule
  documented well enough that custom-spec authors can predict their own
  keys?
- Is this lockable as part of the v1.0 contract or does it leak
  implementation detail?

### 5. `Scorecard.to_pandas()` MultiIndex schema

First-time-public at v0.46. Schema: 1 row × MultiIndex columns
`(metric_name, field)` where `field ∈ {value, status, reason, ci_low,
ci_high, confidence}`. Concerns:

- Is the multi-index pattern the right shape for downstream pandas
  consumers, or would a long-format DataFrame (one row per metric) be
  more useful?
- NaN sentinel for skipped/error cells — does this cause aggregation
  surprises (e.g., `.mean()` skips NaN by default)?

### 6. Stability contract scope (ADR 0003)

Three tiers: Tier 1 strict (top-level `__all__` + 9 Tier-2 Protocols +
schemas) / Tier 2 additive-only (submodule public) / Tier 3 free
(docstring first lines + implementation internals + error messages). Is
this the right boundary? Particular concerns:

- Are there symbols in the current `_EXPORTS` that SHOULDN'T be Tier 1
  strict (e.g., constants that might want to evolve)?
- Are there Tier-2 Protocols that should be opt-in rather than strict
  (the way `Versioned` is)?
- Is the docstring-first-line skip the right Tier 3 boundary?

## Out of scope

- **Making any changes** — this is a review-only pass; output is a
  document, not edits.
- Stylistic preferences (variable names, formatting).
- Items explicitly marked "deferred to v1.x" or "out of scope" in the
  plan / roadmap.
- The single-consumer constraint itself.

## Known issues already in the v0.48 backlog (skip re-reporting)

Round 5 surfaced these and they're scheduled for v0.48 polish (§5E-prep
of the v1.0 plan). Skip them unless you find new dimensions:

1. **`cv_clt_ci` docstring** at `src/eval_toolkit/bootstrap.py:1156-1163`
   claims "Bayle et al. 2020 prove a CV-CLT with a correction factor"
   but the code does naive sample variance. Docstring oversells; code is
   correct per Bayle (2020) Theorem 3.1.
2. **`docs/source/methodology/parallelism.md:143-181`** says "harness not
   yet parallelized as of v0.34" + "once #29/#30 land". The harness has
   had `n_jobs` since v0.36. Parallelism table also needs to clarify
   `bootstrap_ci`'s `n_jobs` is studentized-only.
3. **`docs/source/methodology/testing.md:108-136`** says reference-
   equivalence gap "closing in PR 1.5" — but it closed long ago.
4. **`docs/source/methodology/calibration.md:15-18`** chapter intro lists
   only temperature / isotonic / Platt; Beta (v0.40) and the 4-binary-
   adapter family (`fit_*_binary`) should also appear.
5. **`docs/source/methodology/bootstrap.md`** two-level example passes
   the same array for `val_y` and `test_y` —
   `_paired_bootstrap_op_point_diff_step` resamples independently so
   this causes ~63.2% overlap. Both the docs example and a defensive
   code guard (`raise ValueError if val_y is test_y`) are scheduled for
   v0.48.
6. **DeLong contradictory public status**: `DeLongResult` +
   `delong_roc_variance` are exported (per `_EXPORTS` + API docs) but
   `methodology/comparison.md`, `methodology/reading_list.md`, and
   `roadmap.md` "Out of scope" describe DeLong as out-of-scope.
   Decision U (keep public, align docs) scheduled v0.48.
7. **`CostSensitiveSelector` framing**: implements the prior-corrected
   Elkan form, which double-counts the prior if `y_score` is already
   calibrated to the deployment prior. Existing docstring documents the
   intent but doesn't warn loudly enough. Sharpening scheduled v0.48.

If you find new findings adjacent to these (e.g., a different methodology
page that mismatches code, or a subtler edge case in the bootstrap
leakage example), surface those. Just don't re-flag the items above
without new information.

## How to respond

Whatever shape works for you. Things that help me make use of your read:

- **Cite specific file paths + line numbers / sections** when flagging
  issues. Example: `src/eval_toolkit/_scorecard.py:298` rather than
  "the scorecard function."
- **Mark severity in your own words** — "blocker for v1.0", "blocker
  before v0.47 (rework needed)", "worth a follow-up issue", "minor
  observation." Your call on the boundaries.
- **Be honest about confidence** — "potentially wrong; worth verifying"
  is a useful flag.
- **Don't anchor on the plan's framing.** If you think a design decision
  is fundamentally wrong, say so. Your independent read is the whole
  point.
- **Be brief when brief is honest.** If your read is "this is in good
  shape," that's a valid output. Don't fabricate concerns to fill space.

## Packet (attach alongside this briefing)

Send Codex / Gemini this briefing plus:

- `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` (v1.0 plan)
- `docs/source/audit_findings.md` (Round 5 ledger)
- `docs/source/methodology/*.md` (16 chapters)
- `docs/source/adr/0001-flat-module-layout.md` (**NEW**)
- `docs/source/adr/0002-scorecard-as-primary-metric-surface.md` (**NEW**)
- `docs/source/adr/0003-stability-contract-and-gate3-methodology.md` (**NEW**)
- `docs/source/migration/v0.46.md` (**NEW**)
- `docs/source/roadmap.md`
- `CHANGELOG.md`
- `src/eval_toolkit/*.py` (full source tree; focus on the new files
  + `__init__.py` deprecation branch + `metrics.py` aliases)

Take whatever time you need. The 7-day stop-gate timeout (per Decision
Y.2) is a fallback — if your read takes 2 days, that's fine; if 6 days,
also fine.
