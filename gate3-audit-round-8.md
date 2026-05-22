# Round 8 independent methodology audit — eval-toolkit v0.48.0

## Why this audit

eval-toolkit is a small Python library (~20k LOC, single-author,
single-production-consumer) on the staggered path to v1.0 API stability.
v0.48.0 shipped on 2026-05-22 — the **third + final BREAKING minor** of
the v1.0 sprint. **This is the final STOP-GATE before v1.0.0**, per
Decision Y.2. Your independent read at this gate decides whether
`v1.0.0` can tag or whether the stability contract needs revisiting
before the v1.0 SemVer-major commitment locks in.

The release plan + methodology docs + ADRs were developed in
collaboration with Claude (Anthropic). Your value comes from being a
**different reasoning trace** with different training corpora, catching
things Anthropic-trained reasoning may miss.

## What's different from Rounds 5–7

This is the **stability-commitment audit**. Rounds 5–7 had a forward
path — find issues, fold them into the next BREAKING minor. Round 8
is different: **the v0.48 state is what v1.0 will be**. Round 8
findings either fix-as-v0.48.1-hotfix (per Decision Q's
severity-tiered policy) OR get explicitly accepted as
"known-but-tolerated at v1.0" with rationale in `audit_findings.md`.
**There is no v0.49** — by design, per the staggered-releases plan +
the user's `feedback_staggered_breaking_releases` memory.

If you find a fundamental problem with the v1.0 contract, **say so
plainly**. v1.0 means SemVer-major to change; we'd rather take a
seventh round of audit than lock in a broken shape.

## What I'd like

A rigorous, independent read. Not a checklist — your own judgement
applied to what's in front of you. Where you disagree with a design
decision, say so. Where you spot a methodology mistake, dig into it.
Where you suspect "this was probably AI-co-written and may not have
been deeply verified," flag it.

**The deliverable is a single review document — nothing else.** Do
not modify any files, do not open pull requests, do not commit
changes, do not propose patches as diffs. If you spot a problem,
describe it in the review; I'll decide what to do about it.

## Context — what shipped between Round 7 and now

Round 7 (2026-05-21) audited the v0.47.0 state. Codex produced 3
substantive findings + Gemini 6 minor observations. The
audit-as-seed framing (user direction during plan refinement —
"use the audits as seeds for things to reconsider; make sure we
don't overlook things") expanded the v0.48 scope beyond the
explicit findings. See `docs/source/audit_findings.md` Round 7
section for full ledger.

Since Round 7, v0.48.0 shipped in **15 commits** on `release/v0.48.0`:

- **§5L** (`9878a54`) — `make pre-push` Makefile target running all
  3 doc-execution surfaces (Sybil + MyST-NB + `--doctest-modules`).
  Closes the v0.47 Sub-PR 7 incident class
  (`pytest tests/` silently overriding `testpaths`).
- **§5G** (`e07db16`) — full doc migration: 6 MyST-NB example
  notebooks + 4 module-level docstrings + 5 drifted `api/*.md`
  autosummary lists corrected + 8 new `api/*.md` pages created (the
  audit-as-seed expansion found 8 previously-undocumented public
  modules) + roadmap "Sybil-validated examples" wording fixed.
- **§5H** (`6349472`) — `nb_execution_raise_on_error = True` in
  `docs/source/conf.py` (Decision R7-A). Docs CI now fails on
  notebook execution errors instead of leaving them as advisory
  warnings.
- **§5M** (`58cf462`) — `.doctest-modules` expanded 11 → 21 modules
  (audit-as-seed; surface 3 audit yielded 82 PASS / 0 fail).
- **§5I** (`f454afe`) — **BREAKING**: `sweep()` schema grows by 1
  column (`strategy_id` between `text_id` + `variant`) AND rejects
  duplicate `strategy_id` at boundary. Mirrors R6-B's
  duplicate-`MetricSpec.name` rejection. Style invariants 1 + 2 + 4.
- **§5J** (`fcf99f0`) — **BREAKING**: `sweep()` validates scorer
  output shape with contextual `ValueError`. Replaces three
  failure modes (silent truncation, IndexError, TypeError) with
  one API-level error.
- **§5A** (`e4ab3b9`) — pin-exact-key-set regression-guards for
  every dict-returning metrics function. Audit revealed no drift;
  the tests pin existing keys so future drift fails CI loud.
- **§5B** (`58dd87d`) — **BREAKING**: `BootstrapCI.to_dict()` +
  `PairedBootstrapCI.to_dict()` schema rewrite. `"point_estimate"`
  → `"point"`; `"ci_95: [l, h]"` → `"low"` + `"high"` separate
  scalars. Self-describing — key names no longer contradict the
  `confidence` field.
- **§5C** (`632eabc`) — standardized `ImportError` messages across
  all lazy-extras surfaces.
- **§5E-prep** (`5c8e68d`) — Round 5 packet-drift fixes (7
  methodology pages) + `paired_bootstrap_op_point_diff()` defensive
  guard against `val_y is test_y` (Round 5 R5-F6e — the two-level
  bootstrap assumes disjoint partitions; passing the same array
  caused ~63.2% silent overlap).
- **§5K** (`6304cea`) — docstring polish: `SynonymSubstitution`
  whitelist `Notes`; `Scorecard.to_pandas()` dtype coercion
  `Notes`; `CostSensitiveSelector` calibrated-prior `Warning`.
- **§5N** (`76773dc`) — cross-API shape-validation consistency
  sweep beyond Codex R7-F3. Tightening commit for
  `metrics_at_threshold` silent threshold semantics.
- **§5E** (`56207ba`) — ADR 0001 (flat-module layout) finalized
  for v1.0.
- **§5F** (`0f91dfa`) — ADR 0003 (stability contract + Gate 3
  methodology) finalized for v1.0.
- **§5G follow-up** (`b946007`) — HFDatasetsLoader ImportError
  test alignment to §5C wording.
- **release commit** (`6d9a180`) — `_version.py` 0.47.0 → 0.48.0;
  golden snapshot regenerated; `migration/v0.48.md` added;
  `audit_findings.md` Round 7 entries marked CLOSED;
  CHANGELOG v0.48.0 entry; schema-drift cleanup in `schemas.md` +
  `methodology/claims.md` + `getting-started.md` for §5B's
  rewrite.

(Plus 2 small fix-ups during the PR review cycle:
`0e70fe3` migration-guide `python`→`text` fence conversion;
`6130a34` `_sweep.py` doctest example updated for the §5I
strategy_id column.)

The consumer pin bump v0.47.0 → v0.48.0 landed in the prompt-
injection-detection-submission consumer repo with end-to-end smoke
verifying all 6 BREAKING surfaces work as documented.

## Scope

Please look at:

1. **The v1.0 plan** at
   `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` — locked
   decisions A through Z + R6-A through R6-H + R7-A through R7-C,
   release sequence, semantic stop-gates, risks. The plan now has
   Rounds 5–7 findings + the audit-as-seed extensions integrated.

2. **The audit findings ledger** at `docs/source/audit_findings.md` —
   Rounds 5–7 with dispositions. **Round 7 entries are now marked
   RESOLVED with the v0.48 commit hashes that closed them.**

3. **The methodology curriculum** at `docs/source/methodology/` (16
   chapters). The 7 packet-drift items Round 5 surfaced are all
   fixed at v0.48 (§5E-prep). Re-skim if you reviewed in earlier
   rounds; focus on whether the fixes are sound, not whether they
   exist.

4. **New since Round 7** (focus your time here):

   - **`docs/source/migration/v0.48.md`** — the migration recipes
     for the §5B + §5I + §5J BREAKING changes. The page is itself
     Sybil-executable; the "before" snippets are intentionally in
     ```text fences per the `feedback_sybil_python_blocks` memory.
   - **`src/eval_toolkit/_sweep.py`** — the §5I + §5J changes.
     `_strategy_id_for()` helper; `_validate_unique_strategy_ids()`
     boundary check; `_validate_scorer_output()` boundary check.
     The `sweep()` Examples doctest covers the new shape.
   - **`src/eval_toolkit/bootstrap.py`** — `BootstrapCI.to_dict()`
     + `PairedBootstrapCI.to_dict()` rewrite; the docstrings
     include explicit "Before v0.48 / v0.48+" examples.
     `paired_bootstrap_op_point_diff()` gains the `val_y is test_y`
     defensive guard.
   - **`docs/source/adr/0001-flat-module-layout.md`** — finalized
     for v1.0. Trigger criteria for v2.0 subpackage restructure.
   - **`docs/source/adr/0003-stability-contract-and-gate3-methodology.md`** —
     finalized for v1.0. Tiered stability contract (Tier 1 strict /
     Tier 2 additive-only / Tier 3 free). 9 strict Tier-2 Protocols
     + 1 opt-in (Versioned).
   - **`docs/source/adr/0002-scorecard-as-primary-metric-surface.md`** —
     already finalized at v0.46 ship; verify it still reflects the
     post-v0.48 state.
   - **`Makefile`** — new `pre-push` target (§5L).
   - **`docs/source/conf.py`** — `nb_execution_raise_on_error =
     True` (§5H).
   - **`.doctest-modules`** — expanded 11 → 21 modules (§5M).
   - **`tests/test_lazy_extras_messages.py`** (new file) — pins
     the canonical ImportError message per §5C.
   - **`docs/source/api/*.md`** — 8 new module pages
     (`scorecard`, `metric_specs`, `sweep`, `adversarial`,
     `preprocessing`, `stacking`, `losses`, `probes`) + 5 fixed
     autosummary lists.
   - **`tests/golden/public_api/snapshot.json`** — regenerated at
     v0.48 release-prep; the protocol_methods entries (R6-D) now
     cover the strict Tier-2 set including TextTransform.

5. **Source code** for cross-checking when the docs make specific
   claims. The library is small and the v1.0 plan + ADRs identify
   the most load-bearing modules.

You don't have to read it all linearly. Sample where your judgment
is sharpest. The **v1.0 stability question** is the primary focus.

## Focus areas for Round 8

In rough priority order — but if something else jumps out at you,
surface it:

### 1. Stability contract coherence (HIGHEST PRIORITY)

ADR 0003 promises three tiers:
- **Tier 1 STRICT** (SemVer-major to change): top-level `__all__`
  + the 9 strict Tier-2 Protocols + 1 opt-in + current JSON schema
  versions.
- **Tier 2 ADDITIVE-ONLY** (SemVer-minor): submodule public symbols;
  Tier-2 Protocols can gain optional methods via subprotocols.
- **Tier 3 FREE** (SemVer-patch): docstring first lines (snapshot
  test updated to skip at v1.0), implementation internals, error
  message wording.

For v1.0, the question is: **does the actual shipped state respect
the contract?** Specifically:

- Does the `tests/test_public_api.py` golden snapshot reflect Tier 1
  STRICT — i.e., every name + signature + base list + Protocol method
  shape pinned?
- Did anything that ADR 0003 calls Tier-3 (docstring first lines, error
  message wording) accidentally acquire a test that pins it as
  Tier-1 or Tier-2? §5A's dict-key regression-guards are a candidate
  — they pin exact key sets, which is Tier-1 strict for those
  returned dicts. Is that intentional? Or should those dicts be
  Tier-2 additive-only (i.e., new keys OK, removed keys break)?
- The `protocol_methods` snapshot entry (R6-D) pins method
  signatures for the 9 strict Tier-2 Protocols. Is the snapshot
  semantics — capturing `inspect.signature()` of method bodies as a
  string + alphabetized annotated-attr strings — durable across Python
  versions? Python 3.14 might produce different signature reprs.
- Is the v1.0 snapshot-test plan documented? Per ADR 0003 +
  `tests/test_public_api.py:251-303`, at v1.0 the docstring-first-line
  capture is dropped (Tier-3 free). Is this orchestrated correctly —
  does the v1.0 release commit include the test edit + golden regen?

### 2. The §5I duplicate-`strategy_id` rejection — is the canonical-identifier format stable?

The `strategy_id` shape is `"{name}/{k1}={repr(v1)},{k2}={repr(v2)},..."`
(alphabetized kwargs). Concerns:

- The format uses `repr()` for value rendering. Python's repr is
  stable for primitives + standard strings, but objects with custom
  `__repr__` could produce surprising IDs. Should the format restrict
  values to `(str, int, float, bool, None)` and raise on other types?
- A user adding a new field to a dataclass strategy would silently
  change all existing `strategy_id`s for that class (the alphabetized
  iteration includes the new field). Tier-2 additive-only contract
  per ADR 0003 prohibits this at the Protocol level — but
  user-defined dataclass strategies aren't governed by the contract.
  Worth a docs note?
- Is the format documented anywhere downstream users would
  predictably find? It's in the `_strategy_id_for` docstring but
  that's a leading-underscore helper.

### 3. §5B schema migration — is the "before"/"after" guidance sufficient?

`BootstrapCI.to_dict()` rewrite is the only v0.48 schema change. The
migration guide at `migration/v0.48.md` documents it, but:

- Are there callers in `harness.py` / `evidence.py` / `claims.py` /
  the analysis layer that consume the dict and might still expect
  the old shape? §5G's audit cleaned `schemas.md` + `getting-started.md`
  + `methodology/claims.md` + `methodology/artifacts.md` — but those
  are doc-side. Verify the source-side consumers (e.g., harness
  result emission, claim gates) all read the new keys.
- Is there a v1.0 commitment about which `confidence` levels are
  supported? The schema says "self-describing, supports any
  confidence", but if downstream tooling assumes 0.95, the rewrite
  doesn't actually move that needle. Worth flagging at v1.0?

### 4. Cross-API consistency post-§5N

§5N landed one tightening commit (`metrics_at_threshold` silent
threshold semantics). The audit checked 5 public-API surfaces;
4 came out clean. Concerns:

- Is the §5N coverage actually comprehensive, or did the agent miss
  surfaces? Worth a spot-check: are there other dict-returning
  public functions in `harness.py`, `evidence.py`,
  `operating_points.py`, `claims.py` that have silent-failure
  surfaces or low-level-error leaks?
- The `Scorer.predict_proba` shape requirement is now enforced at
  the `sweep()` boundary (§5J) but NOT at other boundaries that
  also call Scorers (e.g., `evaluate()`, `evaluate_folded()` in
  `harness.py`). Is that intentional asymmetry, or should §5J's
  validation pattern be applied at every Scorer-calling boundary?

### 5. Methodology curriculum completeness for v1.0

The 7 Round-5 packet-drift items are all fixed at v0.48. Plus the
v0.48 §5G added 8 new `api/*.md` autosummary pages. Concerns:

- Re-read 1–2 methodology chapters at random and ask: is this what
  v1.0 should ship? Specifically, the `methodology/bootstrap.md`
  two-level disjoint-split example (§5E-prep item 5) — does the
  rewrite read naturally now?
- The DeLong disposition (Decision U): does the comparison.md +
  reading_list.md + roadmap.md framing read coherently? Or does
  the "bootstrap is preferred but DeLong is also shipped" message
  send mixed signals about what readers should reach for?
- Are there chapters that NEED to be in v1.0 that aren't there
  yet? E.g., is there a chapter on the v0.46 scorecard surface
  beyond the migration guide?

### 6. ADRs 0001 + 0003 finalization — are the trigger criteria + tier boundaries clear?

ADR 0001 documents v2.0 trigger criteria for subpackage restructure
((a) second consumer, (b) functional grouping, (c) discoverability
complaints). Are those criteria too vague to actually trigger
anything?

ADR 0003 documents the Tier 1/2/3 stability contract. Verify:

- Is the Tier-2 "additive-only" boundary durable? E.g., can
  `_scorecard.py`'s `_validate_unique_spec_names` helper signature
  change at v1.x? Documented as Tier-2 (submodule public) or Tier-3
  (internal)?
- Does the Tier-3 "docstring first lines free" decision actually
  hold? The snapshot test currently captures them; the plan says
  the v1.0 release commit drops that capture. Is that captured in
  the release-prep process so it doesn't get forgotten?

### 7. v0.46.1 hotfix legacy

v0.46.1 shipped the ECE strategy validation + deprecation warning
content per Decision Q hotfix policy. The shim it documented is now
gone (removed at v0.47). At v1.0, is anything still referencing
v0.46.1 that shouldn't be? E.g., `_deprecated.py` (the `@deprecated`
decorator infrastructure) — does it still describe the v0.46
deprecation cycle as if relevant? Should it be updated for v1.0+
deprecation cycles, OR removed if no v1.0 deprecation is planned?

## Out of scope

- **Making any changes** — this is a review-only pass; output is a
  document, not edits.
- Stylistic preferences (variable names, formatting).
- Items explicitly marked "deferred to v1.x" or "out of scope" in
  the plan / roadmap.
- The single-consumer constraint itself (Decision in plan — locked
  pre-v1.0).
- v1.0-and-beyond features (the v0.43 forward-look items all
  shipped via v0.47's advanced-6).

## Known issues — none currently in a backlog beyond this audit

Unlike Rounds 5–7, the v0.48 release closed all known drift. There
is no "v0.49 backlog" because there is no v0.49 — v1.0 follows
v0.48.0 directly. **Any finding you surface in Round 8 is either
a v0.48.1 hotfix candidate (per Decision Q) or an accepted-with-
rationale entry in `audit_findings.md` for the v1.0 ship.**

## Postmortem context from Round 7 → v0.48

The Round 7 audit + the user direction "use audits as seeds for
things to reconsider; make sure we don't overlook things" reshaped
v0.48 from "polish + Round 7 fixes" to "comprehensive audit-as-seed
sweep before v1.0 locks the contract." The expansion added:

- §5G grew from 4 Codex-flagged docstrings to a full module-
  docstring + autosummary sweep. Surfaced 8 entirely-undocumented
  public modules + 5 drifted autosummary lists.
- §5M (new) — third doc-execution surface audited (in-source
  docstrings). Result: 82 PASS / 0 fail. Coverage list expanded
  11 → 21 modules.
- §5N grew from "metrics_at_threshold key normalization" (which
  turned out to need no work) to a comprehensive cross-API shape-
  validation consistency sweep.

The Sub-PR 7 incident class (Sybil failures hidden by a
deprecation shim) recurred TWICE during v0.48 release:
- First: the `pytest tests/` path-override pattern resurfaced during
  the v0.48 release commit's verification run. Caught by CI; fixed
  via the `make pre-push` Makefile target (§5L) which encodes the
  lesson + 4 doc-side schema-drift cleanups for the §5B rewrite.
- Second: my own `migration/v0.48.md` had executable ```python
  fences demonstrating BREAKING failure modes; Sybil ran them and
  they failed. Caught by PR CI; fixed via `python`→`text` fence
  conversion (commit `0e70fe3`).

Your review may want to verify whether the pre-push gate (§5L) +
the doc-CI gate (§5H) are sufficient to prevent this class of
incident from recurring at v0.48.1 / v1.0.0 / v1.x.

## How to respond

Whatever shape works for you. Things that help me make use of your
read:

- **Cite specific file paths + line numbers / sections** when
  flagging issues. Example: `src/eval_toolkit/_sweep.py:265-280`
  rather than "the strategy_id helper."
- **Mark severity in your own words** — "blocker for v1.0", "v0.48.1
  hotfix candidate", "accept with rationale at v1.0", "v1.x
  follow-up." Your call on the boundaries.
- **Be honest about confidence** — "potentially wrong; worth
  verifying" is a useful flag.
- **Don't anchor on the plan's framing.** If you think a design
  decision is fundamentally wrong, say so. v1.0 means the contract
  locks; this is the last cheap moment to revisit.
- **Be brief when brief is honest.** If your read is "this is in
  good shape," that's a valid output and ideal for v1.0. Don't
  fabricate concerns to fill space.

## Packet (attach alongside this briefing)

Send Codex / Gemini this briefing plus:

- `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md` (v1.0 plan,
  with Rounds 5–7 integrated)
- `docs/source/audit_findings.md` (Rounds 5–7 ledger; Round 7
  RESOLVED)
- `docs/source/methodology/*.md` (16 chapters; the 7 §5E-prep
  packet-drift items are all fixed at v0.48 — skim for whether the
  fixes read well)
- `docs/source/adr/0001-flat-module-layout.md` (finalized)
- `docs/source/adr/0002-scorecard-as-primary-metric-surface.md`
  (finalized at v0.46)
- `docs/source/adr/0003-stability-contract-and-gate3-methodology.md`
  (finalized)
- `docs/source/migration/v0.46.md` + `v0.47.md` + `v0.48.md` (the
  v0.46→v0.48 migration trail)
- `docs/source/roadmap.md` (refreshed at v0.48 ship)
- `CHANGELOG.md` (v0.48.0 entry — full BREAKING + ADDED + CHANGED
  + FIXED sections)
- `src/eval_toolkit/*.py` (full source tree; focus on the v0.48
  changes — `_sweep.py`, `bootstrap.py`, `_scorecard.py`,
  `metric_specs.py`, `protocols.py`, plus the §5N tightening site
  in `metrics.py`)
- `tests/test_public_api.py` (R6-D Protocol method-shape snapshot
  semantics) + `tests/golden/public_api/snapshot.json` (the
  regenerated v0.48 golden — this is what v1.0 will inherit)
- `tests/test_lazy_extras_messages.py` (new — §5C ImportError
  message tests)
- `Makefile` (`pre-push` target — §5L)
- `docs/source/conf.py` (`nb_execution_raise_on_error` — §5H)
- `.doctest-modules` (expanded list — §5M)

Take whatever time you need. The 7-day stop-gate timeout (per
Decision Y.2) is a fallback — if your read takes 3 days, that's
fine; if 6 days, also fine.

## Cross-references

- **Prior round briefings** (tracked per `.gitignore`): same repo
  root, `gate3-audit-round-5.md`, `gate3-audit-round-6.md`,
  `gate3-audit-round-7.md`.
- **Canonical prompt template**:
  `~/.claude/plans/gate3-audit-prompt.md` (local, never committed) —
  this Round 8 briefing was structured from it.
- **Reports from your read** should land at the repo root as
  `gate3-audit-round-8-report.md` (one file per reviewer / model;
  the `.gitignore` keeps per-round reports untracked but per-round
  briefings tracked).
