# Round 7 independent methodology audit — eval-toolkit v0.47.0

## Why this audit

eval-toolkit is a small Python library (~20k LOC, single-author,
single-production-consumer) on the staggered path to v1.0 API stability.
v0.47.0 shipped on 2026-05-21 — the **second BREAKING release** of the
v1.0 sprint (hard removal of v0.46's deprecation shim + module-level
sweep consolidation + per-module-Protocol cleanup). Your independent
read at this gate decides whether `release/v0.48.0` can open or whether
v0.47 contracts need to be revisited before more code lands on top of
them.

The release plan + methodology docs + ADRs were developed in
collaboration with Claude (Anthropic). Your value comes from being a
**different reasoning trace** with different training corpora, catching
things Anthropic-trained reasoning may miss.

## What I'd like

A rigorous, independent read. Not a checklist — your own judgement
applied to what's in front of you. Where you disagree with a design
decision, say so. Where you spot a methodology mistake, dig into it.
Where you suspect "this was probably AI-co-written and may not have been
deeply verified," flag it.

**The deliverable is a single review document — nothing else.** Do not
modify any files, do not open pull requests, do not commit changes, do
not propose patches as diffs. If you spot a problem, describe it in the
review; I'll decide what to do about it. The point is your independent
assessment, not your implementation.

## Context — what shipped between Round 6 and now

Round 6 (2026-05-21) audited the v0.46.0 state. 11 findings were
verified-real; all are tracked in the audit ledger
(`docs/source/audit_findings.md` Round 6 section). Two findings shipped
as v0.46.1 hotfixes (R6-F1 ECE strategy validation, R6-F2 deprecation
warning content); the other 9 rolled forward to v0.47 per Decision R6-E.

Since Round 6:

- **v0.46.1 shipped** as a hotfix release per Decision Q + Decision R6-E.
  Codex R6-F1 (`metric_specs.ece(strategy="typo")` silently dispatched
  to quantile ECE) → factory + `_EceSpec.compute()` defence-in-depth
  validation. Codex+Gemini R6-F2 (broken migration snippets in all 5 ECE
  deprecation warnings) → restructured `_scorecard_spec_for()` returning
  `(factory_expr, scorecard_key, has_first_party)` tuples. ECE warnings
  use `n_bins=10` to preserve pre-v0.46 math per Decision R6-F + add a
  migration note about the v0.46+ factory default of `n_bins=15`. The 3
  ECE variants without first-party replacements (debiased / l2 /
  l2_debiased) route to submodule path per Decision R6-G.

- **v0.47.0 shipped** in 7 sub-PRs on `release/v0.47.0`:

  1. **Sub-PR 1** (R6-A docstring + R6-B duplicate-name guard +
     R6-F5 narrow except + R6-H `make_spec_name` helper) — small
     Round 6 follow-on items.
  2. **Sub-PR 2** (R6-C `Scorecard.to_pandas()` schema gains
     `n_resamples` + `method` columns; R6-D `tests/test_public_api.py`
     drift guard now captures `typing.Protocol` method signatures) —
     additive contract enforcement.
  3. **Sub-PR 3** (top-level `TextTransform` Protocol +
     `DelimitVariant` / `DatamarkVariant` / `EncodeVariant` preprocessing
     dataclasses) — closes Round 5 R5-F3 (Codex) plan-vs-code drift.
  4. **Sub-PR 4** (top-level `sweep()` with `TextTransform` Protocol
     keying + parity tests against the old module-level sweeps) —
     ADDITIVE only, prove-it-then-remove-it pattern.
  5. **Sub-PR 5** (6 advanced character-injection techniques:
     `BidiRTLInjection`, `TagStrippingInjection`, `SynonymSubstitution`,
     `TokenSplitting`, `UnicodeNormalization`, `InvisibleCharsInjection`)
     — closes the v0.43.0 forward-look per Decision Q11→11.3.
  6. **Sub-PR 6** — THE BIG BREAKING REMOVAL. Deleted v0.46
     `__getattr__` deprecation shim (`_DEPRECATED_SCALARS` +
     `_FIRST_PARTY_REPLACEMENTS` + `_deprecation_warning_for` helper +
     the BEGIN/END TRANSITIONAL block inside `__getattr__`). Removed
     module-level `adversarial.sweep` + `preprocessing.sweep`. Removed
     `character_injection` + `spotlighting` `SimpleNamespace` shortcuts.
     Removed `CharacterInjectionStrategy` per-module Protocol (replaced
     by top-level `TextTransform`).
  7. **Sub-PR 7** — release commit (version bump 0.46.1 → 0.47.0;
     `docs/source/migration/v0.47.md`; roadmap refresh per R6-F6;
     plus a follow-up fix renaming `sweep.py` → `_sweep.py` to resolve
     a module/function name collision under mypy + 40 doc-snippet
     migrations to the new API after CI surfaced them — see
     **Postmortem** section below).

  CI initially failed on Sub-PR 7 with 40 Sybil-collected snippet
  failures across 11 doc files. Root cause documented below for context
  — your review may want to verify the fix is durable.

- **Consumer pin bumped** v0.46.1 → v0.47.0 in
  `prompt-injection-detection-submission` with end-to-end smoke
  verification (all 7 removed top-level names → AttributeError; sweep
  composes mixed defence + attack; submodule scalar path still works;
  R6-B duplicate-name guard fires; R6-C to_pandas schema has the new
  columns; R6-H `make_spec_name` canonicalizes correctly).

## Scope

Please look at:

1. **The v1.0 plan** at
   `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` — locked
   decisions A through Z and R6-A through R6-H, release sequence,
   semantic stop-gates, risks. The plan now has Round 5 and Round 6
   findings integrated; you don't need to re-derive them.

2. **The audit findings ledger** at `docs/source/audit_findings.md` —
   Rounds 5–6 with dispositions. This is the comparison surface: if you
   find an issue matching an existing entry, it ISN'T resolved properly
   OR you've found a related-but-distinct issue (worth flagging
   separately).

3. **The methodology curriculum** at `docs/source/methodology/` (16
   chapters). Unchanged from Round 6 in shipped state; the 7 items in
   the v0.48 backlog list below are still pending. If you reviewed this
   in Round 6, you can re-skim.

4. **New since Round 6** (focus your time here):

   - **`src/eval_toolkit/protocols.py`** — `TextTransform` Protocol
     added; the 9th strict Tier-2 Protocol per ADR 0003 (locks at v1.0).
     Two attributes: `name: str` + `transform(text: str) -> str`.
   - **`src/eval_toolkit/preprocessing.py`** — 3 new frozen + slots
     dataclasses (`DelimitVariant` / `DatamarkVariant` /
     `EncodeVariant`) wrapping the existing functional API. Module-level
     `sweep()` and `spotlighting` `SimpleNamespace` were removed in
     Sub-PR 6.
   - **`src/eval_toolkit/adversarial.py`** — 6 new advanced
     character-injection dataclasses (`BidiRTLInjection`,
     `TagStrippingInjection`, `SynonymSubstitution`, `TokenSplitting`,
     `UnicodeNormalization`, `InvisibleCharsInjection`). New
     `ADVANCED_TECHNIQUES` + `ALL_TECHNIQUES` tuples. Module-level
     `sweep()`, `character_injection` `SimpleNamespace`, and
     `CharacterInjectionStrategy` Protocol all removed in Sub-PR 6.
   - **`src/eval_toolkit/_sweep.py`** — the new top-level `sweep()`
     function. Underscore-prefixed module to avoid module/function name
     collision (a v0.47 pre-tag fix; see Postmortem).
   - **`src/eval_toolkit/_scorecard.py`** — `scorecard()` updated:
     duplicate-name guard (R6-B), narrow `except` (R6-F5), docstring
     fix (R6-A), and `to_pandas()` schema expansion (R6-C).
   - **`src/eval_toolkit/metric_specs.py`** — `make_spec_name()` helper
     added (R6-H, exposed in `metric_specs.__all__` only per Decision
     R6-H).
   - **`tests/test_public_api.py`** — `_protocol_method_snapshot()`
     helper + `_TIER2_PROTOCOLS` coverage + `protocol_methods` snapshot
     entry per Decision R6-D. Golden snapshot regenerated.
   - **`tests/test_deprecated_scalars_shim.py`** — rewritten as v0.47
     hard-removal smoke tests verifying `AttributeError` on the 8
     removed names.
   - **`docs/source/migration/v0.47.md`** — consumer migration guide.
   - **`CHANGELOG.md`** v0.47.0 entry.
   - **`docs/source/roadmap.md`** — header refreshed to v0.47.0; v0.45 /
     v0.46 / v0.46.1 / v0.47 entries added; 9 strict Tier-2 + 1 opt-in
     Protocol count made explicit.

5. **Source code** for cross-checking when the docs make specific
   claims. The library is small and the v1.0 plan + ADRs identify the
   most load-bearing modules.

You don't have to read it all linearly. Sample where your judgment is
sharpest.

## Focus areas for Round 7

In rough priority order — but if something else jumps out at you,
surface it:

### 1. `TextTransform` Protocol lock-in (HIGHEST PRIORITY)

The 9th strict Tier-2 Protocol per ADR 0003. **Locks at v1.0** —
method-signature changes require SemVer-major bump. Last cheap chance
to catch contract gaps.

- The Protocol has `name: str` + `transform(text: str) -> str` only. Is
  this enough for the full attack/defence/preprocessing taxonomy long
  term? Concerns:
  - **No `seed` parameter** on `transform` — deterministic behaviour
    relies on the strategy carrying its own seed as a constructor
    kwarg. Is that the right contract or does the Protocol need
    `transform(text: str, *, seed: int | None = None) -> str`?
  - **No batch method** — strategies that have natural batched
    implementations (e.g., GPU-accelerated paraphrasing) have to
    re-implement batching in callers. Would a default-implemented
    `transform_batch(texts: Sequence[str]) -> list[str]` subprotocol
    be wiser? (ADR 0003 says additive subprotocols are permitted, so
    this isn't a v1.0 blocker — but the shape decision is now.)
  - **No `inverse()`** — for reversible variants (delimit, datamark,
    base64), having a `inverse(transformed: str) -> str` would
    formalize round-trip recovery. Currently there's no Protocol
    obligation; some implementations have it, others don't.

- Does the structural-subtyping story actually work? Concrete classes
  satisfy `TextTransform` without inheriting, but mypy under `--strict`
  needs runtime-checkable validation. Verify
  `tests/test_public_api.py::test_tier2_protocols_have_method_shape_snapshot`
  pins the shape correctly.

### 2. Removal of `__getattr__` deprecation shim (Audit F4 invariant)

Sub-PR 6 deleted the BEGIN/END TRANSITIONAL block from
`__getattr__` along with `_DEPRECATED_SCALARS` +
`_FIRST_PARTY_REPLACEMENTS` + `_deprecation_warning_for()`. Verify:

- The base resolver (`module_name = _EXPORTS.get(name); ...`) still
  fires for every non-deprecated name.
- The 8 deprecated names (`pr_auc`, `roc_auc`, `brier_score`, 5 ECE
  variants) now raise `AttributeError` from `__getattr__` cleanly — NOT
  partially resolved by a stale `globals()` cache from earlier
  introspection.
- `from eval_toolkit.metrics import pr_auc` continues to work (the
  internal-API escape hatch per Decision C / ADR 0002). Verify the
  submodule is intact + the scalar function signatures match v0.45
  exactly.
- `tests/test_deprecated_scalars_shim.py` (rewritten) covers the right
  surface — any test gap?

### 3. Module-level sweep removal + top-level `sweep()` design

`adversarial.sweep` + `preprocessing.sweep` removed. New top-level
`sweep(strategies, texts, *, scorer=None, attack_threshold=None)`.
Concerns:

- **Parity coverage**: `tests/test_sweep.py::test_parity_with_*` ran
  AGAINST the old module-level sweeps before Sub-PR 6 deleted them; the
  parity tests were also deleted in Sub-PR 6. Is the deletion order
  correct (`prove parity, THEN remove`), or should the parity assertions
  have been pinned as golden fixtures so they outlive the old code
  paths?
- **`attack_threshold` mandatory for `asr` column**: the old
  `adversarial.sweep` had `threshold=0.5` as a default. The new one
  refuses to emit `asr` without explicit `attack_threshold` (rationale:
  `methodology/thresholds.md` warns against magic 0.5 defaults). Is this
  the right call, or does it just push the 0.5 default into every
  caller without methodological gain?
- **Scorer ergonomics**: batched `predict_proba` per strategy (cheaper
  than per-row). What's the failure mode if a Scorer is stateful and
  the batch order matters?

### 4. Doc-snippet migration (40 fences across 11 files)

Sub-PR 7 had to migrate 40 Sybil-executed code fences in
`docs/source/**/*.md` + `README.md` because they used the now-removed
top-level scalar imports. The fix mostly mapped `from eval_toolkit
import pr_auc` → `from eval_toolkit.metrics import pr_auc`, with some
"before" snippets in migration guides converted to ```text fences. Concerns:

- **Pedagogical drift**: any of the 40 migrated snippets now teach the
  WRONG thing (e.g., a methodology page that used to demonstrate
  `scorecard()` shape now uses `from eval_toolkit.metrics import
  pr_auc` directly, undermining the v0.46-onward "scorecard is the
  primary surface" message)?
- **`docs/source/migration/v0.47.md`** itself — verify the migration
  recipes work end-to-end (the migration guide is itself
  Sybil-executed).

### 5. Round 6 follow-on integration

Five small changes in `_scorecard.py` + `metric_specs.py`:

- **R6-A** (`seed=None` deterministic-by-default) — docstring fix
  only; verify the actual behavior matches the documented contract
  via bit-for-bit equality of `seed=None` and `seed=0` runs.
- **R6-B** (duplicate `MetricSpec.name`) — `_validate_unique_spec_names`
  validates before any compute. Edge case: what if the dict that powers
  the duplicate check accepts insertion-order-dependent dups (e.g.,
  user passes `[spec, spec]` where both are the *same instance*)?
- **R6-C** (`to_pandas()` schema) — `n_resamples` (int / NaN) +
  `method` (str / "") added. Sentinel choice OK? Does pandas dtype
  coercion turn the mixed int/NaN column into float64 (potential
  surprise for downstream)?
- **R6-D** (Protocol method-shape drift guard) — verify
  `_protocol_method_snapshot()` correctly captures every public method
  + annotated attr on the 9 strict Tier-2 Protocols. Sanity check the
  snapshot file `tests/golden/public_api/snapshot.json` contains
  expected method shapes for `MetricSpec.compute`, `MetaLearner.fit`,
  `Scorer.predict_proba`, etc.
- **R6-F5** (narrow except) — `MemoryError` / `RecursionError` /
  `KeyboardInterrupt` / `SystemExit` re-raised in `_evaluate_spec()`.
  Other system-exit-class exceptions worth catching too? (e.g.,
  `GeneratorExit`?)
- **R6-H** (`make_spec_name`) — placement in `metric_specs.__all__`
  only (NOT top-level `_EXPORTS`) per Decision R6-H. Tier-2 additive
  contract — the helper can gain kwargs in v1.x. Is the kwarg-alphabetize
  rule discoverable enough for custom-spec authors?

### 6. Advanced-6 character-injection techniques

Six new dataclasses. Closes the v0.43.0 CHANGELOG forward-look. Concerns:

- **`SynonymSubstitution`**: whitelist (`_SYNONYMS`) has 6 entries —
  `ignore`, `instructions`, `system`, `secret`, `send`, `all`. Is this
  whitelist documented? Will users be surprised when the technique is
  a no-op on inputs that don't contain whitelist words?
- **`BidiRTLInjection`**: prepends `U+202E` + appends `U+202C`. No
  ratio parameter — always wraps the whole input. Different design
  choice from the core 6. Intentional?
- **`UnicodeNormalization`**: NFKC default. Folds fullwidth → ASCII.
  Is this an attack, a defence, or both? The docs treat it as an
  attack; an attacker could equally use it as input-canonicalization
  defence. Categorization rationale?
- **`TokenSplitting`**: splits at a random offset based on a seeded
  RNG. The random-offset choice — does this defeat the "human-readable
  preservation" invariant the core 6 maintain? (E.g., `"ignore"` →
  `"ig nore"` is still readable; `"ignore"` → `"i gnore"` is jarring.)
- **`InvisibleCharsInjection`** vs `ZeroWidthSpaceInjection`: the
  former samples from 5 invisible code points; the latter inserts only
  ZWSP. Are both worth keeping public, or is one strictly more general?

### 7. Stability contract scope (ADR 0003 — re-verification)

ADR 0003 was drafted at v0.46.0 and reviewed in Round 6. Round 7 has the
benefit of seeing how the contract plays out under an actual breaking
removal. Concerns:

- Did the v0.47 BREAKING release respect the documented tiers? Or did
  any Tier-2-additive-only commitment slip into a method-shape change?
- The `protocol_methods` snapshot entry is now actively enforcing the
  Tier-2 method-shape contract (R6-D). Verify the snapshot's coverage
  matches ADR 0003's promise (9 strict + 1 opt-in = 10 total Protocols).
- Did anything that ADR 0003 calls Tier-3 ("free", docstring first
  lines + implementation internals + error message wording) accidentally
  acquire a test that pins it as Tier-2 or Tier-1?

## Out of scope

- **Making any changes** — this is a review-only pass; output is a
  document, not edits.
- Stylistic preferences (variable names, formatting).
- Items explicitly marked "deferred to v1.x" or "out of scope" in the
  plan / roadmap.
- The single-consumer constraint itself.

## Known issues already in the v0.48 backlog (skip re-reporting)

These 7 items are scheduled for v0.48 polish (§5E-prep of the v1.0
plan). Surface new dimensions if you find them; don't re-flag the
items themselves.

1. **`cv_clt_ci` docstring** (`bootstrap.py:1156-1163`) — "correction
   factor" phrasing oversells the math. Code is correct.
2. **`docs/source/methodology/parallelism.md:143-181`** — v0.34 state
   ("not yet parallelized" + "once #29/#30 land"). v0.36 wired
   `n_jobs` into the harness.
3. **`docs/source/methodology/testing.md:108-136`** — "reference-
   equivalence gap closing in PR 1.5"; long since closed.
4. **`docs/source/methodology/calibration.md:15-18`** — lists only 3
   calibrators; Beta + 4-binary-adapter family also shipped.
5. **`docs/source/methodology/bootstrap.md`** two-level example —
   passes the same array for `val_y`/`test_y`; ~63.2% overlap. Both
   docs example AND defensive `raise ValueError if val_y is test_y`
   in `paired_bootstrap_op_point_diff` scheduled.
6. **DeLong contradictory public status** — `DeLongResult` +
   `delong_roc_variance` are exported, but methodology docs say
   "out of scope". Decision U → align docs.
7. **`CostSensitiveSelector` framing** — prior-corrected Elkan form
   double-counts the prior on already-calibrated scores; existing
   docstring documents the intent but doesn't warn loudly enough.

## Postmortem context (your review may want to verify the fix is durable)

Sub-PR 7's first CI run failed on 40 Sybil-executed snippets across 11
doc files (README.md + 10 under docs/source/) — every failure was an
`ImportError` on a top-level scalar (`from eval_toolkit import pr_auc`,
etc.). Root cause: the v0.46 `__getattr__` deprecation shim had been
resolving those imports with a `DeprecationWarning` since v0.46.0, so
Sybil executed them successfully throughout the v0.46 cycle.

Two compounding mistakes on the author (me) side:

1. **Wrong scope for pre-push verification**: I ran
   `pytest tests/ --no-cov -q --ignore=tests/benchmarks`. Passing
   `tests/` as a positional arg silently overrides the project's
   `[tool.pytest.ini_options] testpaths = ["tests", "README.md",
   "docs/source"]` config, dropping 159 Sybil items from collection
   (1498 collected with the path arg vs 1657 without).
2. **Graceful-degradation-layer hazard**: the deprecation shim hid 40
   latent failures across the entire v0.46 cycle (each fence still
   "executed successfully" by returning a value + DeprecationWarning).
   Removing the shim activated all 40 simultaneously.

Fix shipped: 13 fences migrated to the new API (`from
eval_toolkit.metrics import …` or `scorecard(...)`), 5 illustrative
"before" snippets in migration guides converted to ```text fences,
golden snapshot regenerated, version field bumped 0.46.1 → 0.47.0.

Your review may want to confirm:

- The remaining doc-snippet migrations are pedagogically sound (not
  just syntactically green).
- The plan's pre-tag gates in §Semantic-stop-gates are sufficient to
  prevent this class of incident on v0.48 / v1.0 releases.

## How to respond

Whatever shape works for you. Things that help me make use of your read:

- **Cite specific file paths + line numbers / sections** when flagging
  issues. Example: `src/eval_toolkit/_scorecard.py:298` rather than
  "the scorecard function".
- **Mark severity in your own words** — "blocker for v1.0", "blocker
  before v0.48 (rework needed)", "worth a follow-up issue", "minor
  observation". Your call on the boundaries.
- **Be honest about confidence** — "potentially wrong; worth verifying"
  is a useful flag.
- **Don't anchor on the plan's framing.** If you think a design
  decision is fundamentally wrong, say so. Your independent read is the
  whole point.
- **Be brief when brief is honest.** If your read is "this is in good
  shape," that's a valid output. Don't fabricate concerns to fill space.

## Packet (attach alongside this briefing)

Send Codex / Gemini this briefing plus:

- `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` (v1.0 plan,
  with Rounds 5–6 integrated)
- `docs/source/audit_findings.md` (Rounds 5–6 ledger)
- `docs/source/methodology/*.md` (16 chapters)
- `docs/source/adr/0001-flat-module-layout.md`
- `docs/source/adr/0002-scorecard-as-primary-metric-surface.md`
- `docs/source/adr/0003-stability-contract-and-gate3-methodology.md`
- `docs/source/migration/v0.46.md`
- `docs/source/migration/v0.47.md` (**NEW**)
- `docs/source/roadmap.md` (refreshed header — now v0.47.0)
- `CHANGELOG.md` (v0.47.0 entry + v0.46.1 hotfix entry)
- `src/eval_toolkit/*.py` (full source tree; focus on the NEW files
  + the removal sites in `__init__.py` / `adversarial.py` /
  `preprocessing.py` + the v0.47-specific edits in `_scorecard.py` /
  `metric_specs.py` / `protocols.py` / `_sweep.py`)
- `tests/test_public_api.py` (Round 6 R6-D extension; the
  `_protocol_method_snapshot` helper + `_TIER2_PROTOCOLS` coverage)
- `tests/golden/public_api/snapshot.json` (regenerated golden — the
  `protocol_methods` entries are the new R6-D enforcement surface)

Take whatever time you need. The 7-day stop-gate timeout (per Decision
Y.2) is a fallback — if your read takes 2 days, that's fine; if 6 days,
also fine.

## Cross-references

- **Prior round briefings** (untracked per `.gitignore`): same repo root,
  `gate3-audit-round-6.md` (immediately prior; v0.46.0 state).
- **Canonical prompt template**: `~/.claude/plans/gate3-audit-prompt.md`
  (local, never committed) — this Round 7 briefing was structured from
  it; you don't need to re-read the template.
- **Reports from your read** should land at the repo root as
  `gate3-audit-round-7-report.md` (one file per reviewer / model; the
  `.gitignore` keeps per-round reports untracked but per-round briefings
  tracked).
