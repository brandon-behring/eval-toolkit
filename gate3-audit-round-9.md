# Round 9 independent methodology audit — eval-toolkit v0.51.0

## Why this audit

eval-toolkit is a small Python library (~20k LOC, single-author,
single-production-consumer) on the staggered path to v1.0 API
stability. v0.51.0 shipped on 2026-05-24 — the **Round 8
rectification batch**, fixing 13 of 18 verified audit findings from
the v0.50.0 multi-LLM cross-review. **This is the final STOP-GATE
before v1.0.0**, per Decision Y.2 (Round 8 expected to be the last;
Round 9 is the verification round that the v0.51 fixes hold + no
regressions were introduced + nothing critical was missed in the
~10 modules neither Round-8 auditor cited).

Your independent read at this gate decides whether `v1.0.0` can tag
or whether the stability contract needs revisiting before the v1.0
SemVer-major commitment locks in.

The release plan + methodology docs + ADRs were developed in
collaboration with Claude (Anthropic). Your value comes from being
a **different reasoning trace** with different training corpora,
catching things Anthropic-trained reasoning may miss. The Round 8
verification report explicitly identified that Codex tends to pad
(C10) and Gemini tends to over-validate (V1, V2) — Round 9 is the
last chance to course-correct.

## What's different from Round 8

Round 8 audited the v0.50.0 state — find issues, fold into v0.51.
Round 9 is different: **the v0.51 state is what v1.0 will be**.
Round 9 findings either fix-as-v0.51.1-hotfix (per Decision Q's
severity-tiered policy) OR get explicitly accepted as
"known-but-tolerated at v1.0" with rationale in
`audit_findings.md`.

If you find a fundamental problem with the v1.0 contract, **say so
plainly**. v1.0 means SemVer-major to change; the cheapest moment
to revisit is now.

## What I'd like

A rigorous, independent read. Not a checklist — your own judgement
applied to what's in front of you. Where you disagree with a design
decision, say so. Where you spot a methodology mistake, dig into
it. Where you suspect "this was probably AI-co-written and may not
have been deeply verified," flag it.

**The deliverable is a single review document — nothing else.** Do
not modify any files, do not open pull requests, do not commit
changes, do not propose patches as diffs. If you spot a problem,
describe it in the review; I'll decide what to do.

## Context — what shipped between Round 8 and now

Round 8 (2026-05-24) verified Codex + Gemini reports against v0.50.0.
**Distribution: 13 confirmed / 3 refuted / 2 deferred.** v0.51.0
shipped 7 commits on `release/v0.51.0`:

- **`87453f6`** — R8-C4(b) `spawn_seed_sequences` respects Generator
  state (was: extracted seed-source entropy; Generator advancement
  ignored). Now draws fresh entropy via
  `rng.integers(0, 2**63-1, size=n)`; each call advances state.
- **`61964f6`** — R8-C4(a) `_score_all_slices` boundary spawn. The
  shared rng object was attached to every (slice, scorer) work_unit;
  joblib forked copies at the SAME generator state, so all workers
  used identical bootstrap samples in parallel mode. v0.51 spawns
  one independent SeedSequence per work unit at the dispatch
  boundary. SPEC-7 bit-for-bit parity contract restored.
- **`672d45f`** — R8-C3 `recall_at_fpr` sentinel for unsatisfiable
  `target_fpr`. Pre-v0.51 the fallback set `threshold=1.0` then
  computed `y_pred = (y_score >= 1.0)`, silently classifying score=1.0
  negatives as positives → `actual_fpr=1.0` returned in violation
  of the function's own ceiling. v0.51 returns
  `RecallAtFprResult(threshold=np.inf, actual_fpr=0.0, fp=0)` —
  caller filters via `np.isinf(...)`.
- **`f60d43c`** — R8-C1 `evaluate_folded` `reseed_splitter` callback
  (DeprecationWarning when multi-seed + None; warning persists past
  v1.0). R8-C2 `SourceDisjointKFoldSplitter.iter_folds` capped at
  `min(k, n_sources)` (matches `get_n_splits`); UserWarning when
  cap fires.
- **`4c5e140`** — Tier-2 additive validation rigor (R8-C6 calibration
  `_validate_calibrated_score` + `fit_temperature` bounds check;
  R8-F1 `RecallAtLowFPR.pos_weight` validation; R8-F2 `metric_specs.ece`
  eager `n_bins`; R8-F3 `CsvPredictionReader` missing-column
  diagnostic).
- **`c206b54`** — Docs + structure sweep (R8-C5 README link repointing
  + migration toctree extended through v0.51; R8-C8 SimilarityStrategy
  demoted; R8-G1 repo-strategy.md supersession note; R8-C9 GateResult
  to_dict docstring; R8-C10 gitignore audit patterns).
- **`b0044ae`** — Release commit: version bump 0.50.0 → 0.51.0;
  CHANGELOG header; Round 8 ledger populated; public-API snapshot
  regen; `_generate_population` test helper SPEC-7 follow-up
  (missed-by-both surfaced during pre-push).

Refuted in Round 8 (recorded in `audit_findings.md` for the audit
trail; no shipped fix): R8-G2 (cycle is deliberately broken by
TYPE_CHECKING + lazy import — Gemini's "unresolved" framing wrong);
R8-G5 (cherry-picked weak assertion in a file with stronger ones);
R8-V1 + R8-V2 (Gemini over-confident "Exceptional" / "Masterclass"
validations refuted by Codex's confirmed bugs).

Deferred to v1.x as Tier-2 additive: R8-G3 (no custom exception
hierarchy beyond ValueError); R8-G4 (joblib OOM hazard documented
but unmitigated).

## Scope

Round 9 focus areas — in rough priority order. If something else
jumps out at you, surface it.

### 1. Regression check on Round-8 fixes (HIGHEST PRIORITY)

The 13 v0.51 fixes claim to close specific Round-8 bugs. For each,
verify:

- The probe documented in the audit verification report still
  reproduces the FIXED behavior at v0.51 (the file
  `audit-verification-codex-gemini-v0.50.0.md` appendix has the
  exact commands; you should be able to re-run them and confirm).
- The regression test bundled with each fix actually pins the
  invariant — does it cover the original bug shape, or is it
  testing a weaker proxy?
- Adjacent code paths that the fix touched but didn't audit
  carefully — e.g., the C4(a) boundary spawn refactor in
  `harness.py:_score_all_slices` could have introduced a different
  bug; the C3 sentinel-return change to `recall_at_fpr` could have
  affected downstream callers that consumed the result dict.

### 2. The ~10 modules neither Round-8 auditor cited

The Round 8 verification report explicitly noted: "Both auditors
missed the same modules" — Codex and Gemini together cited none of
these:

- `src/eval_toolkit/metrics.py` (34 public defs — heaviest module
  by symbol count; BCa + percentile + bias-corrected bootstrap CI
  formulas; AUPRC + AUROC + ECE variants + Brier + reliability;
  threshold-derived metrics).
- `src/eval_toolkit/bootstrap.py` (24 defs — BCa jackknife
  acceleration; paired bootstrap diff; block bootstrap on folds;
  cv_clt_ci; mde_from_ci).
- `src/eval_toolkit/text_dedup.py` (5 strategies + orchestrators).
- `src/eval_toolkit/embeddings.py` (MiniLM convenience factory).
- `src/eval_toolkit/scorecards.py` (the v1.0 primary metric surface
  per ADR 0002).
- `src/eval_toolkit/_sweep.py` (strategy_id format stability — R7-B
  audit; scorer-output shape validation — R7-C).
- `src/eval_toolkit/probes.py` (ActivationDeltaProbe — TaskTracker-style
  linear probe; optional [probes] extra).
- `src/eval_toolkit/stacking.py` (MetaLearner Protocol +
  LogisticStacker reference impl).
- `src/eval_toolkit/preprocessing.py` (3 Spotlighting variants).
- `src/eval_toolkit/operating_points.py` (OperatingPointSpec +
  transferred operating points).

Each is potentially load-bearing for v1.0 claims and has had only
self-review (not multi-LLM peer review). Sample 2-4 modules where
your judgment is sharpest. **For math-heavy modules**
(`metrics.py`, `bootstrap.py`), verify the formulas against textbook
references where you have the background to do so. **For
domain-methodology modules** (`stacking.py`, `text_dedup.py`,
`scorecards.py`), audit whether the design choices defensibly
implement what they claim.

### 3. The v0.51 design decisions that could be wrong

Three specific decisions in v0.51 are judgment calls — each could
be wrong:

- **R8-C3 sentinel design**: returning `threshold=np.inf` instead of
  raising `RuntimeError`. Callers may not check `np.isinf` and
  silently use a meaningless result downstream. Is the sentinel
  actually safer than raising?
- **R8-C1 DeprecationWarning that persists past v1.0**: the
  `evaluate_folded` warning will fire on every multi-seed call
  indefinitely because the pre-v1.0 deprecation window doesn't
  close. Is this the right trade-off vs flipping the default to
  raise at v0.52 (which itself is a SemVer-major event post-v1.0)?
- **R8-C2 UserWarning instead of raising**: when k > n_sources,
  iter_folds caps + warns rather than rejecting. Permissive design
  vs strict — at v1.0, which is the right contract?

### 4. Stability contract coherence post-v0.51

ADR 0003 promises Tier 1 STRICT / Tier 2 ADDITIVE / Tier 3 FREE.
v0.51 changed the public API surface (R8-C1 added `reseed_splitter`
to `evaluate_folded`; R8-C3 changed `recall_at_fpr` return shape
semantics). Verify:

- `tests/test_public_api.py` golden snapshot was regenerated for
  the new signatures. Is the regenerated snapshot capturing the
  right shape? Does it pin anything that ADR 0003 calls Tier-3
  (docstring first lines, error message wording)?
- The R8-C9 GateResult.to_dict docstring change adds
  load-bearing-via-docs guarantees. Are the docs Tier-1-stable now,
  or still Tier-3? Inconsistent.

### 5. Methodology coherence post-Round-8

Round 8 made 5 doc changes (R8-C5 README + migration toctree;
R8-C8 SimilarityStrategy alignment; R8-G1 repo-strategy
supersession note; R8-C9 GateResult JSON-safety doc; R8-C10
gitignore). Re-read 1-2 of the touched docs as a whole and ask:
does the document still read coherently? Does the v0.51 patch
introduce internal contradictions?

### 6. The audit-process meta-question

Round 5-8 have been gate3-style methodology audits. The v0.51 ship
introduced the `~/Claude/audit-templates/audit-prompt.md` universal
template (a Claude-authored artifact). Look at the template and
ask: does it represent the audit-prompt-design lessons learned
from Rounds 5-8 accurately? Anything important missing?

This is meta but: if Round 9 surfaces process-level findings about
the audit machinery itself, those are first-class output.

## Out of scope

- **Making any changes** — review-only pass; output is a document.
- Stylistic preferences (variable names, formatting).
- Items explicitly marked "deferred to v1.x" (R8-G3, R8-G4).
- The single-consumer constraint (Decision in plan — locked
  pre-v1.0).
- Re-flagging refuted Round-8 items (R8-G2, R8-G5, R8-V1, R8-V2)
  unless your read genuinely disagrees with the verification
  rationale.

## Cross-cutting hunts (do all four)

Per the `~/Claude/audit-templates/audit-prompt.md` cross-cutting
hunt set:

1. **Design-doc claim verification.** Pick 3-5 claims from the
   updated v0.51 docs (CHANGELOG entries, audit_findings.md Round 8
   ledger entries, the 3 new migration guides v0.49/v0.50/v0.51).
   For each, find the code or test that backs the claim. Does the
   claim actually hold?

2. **Undocumented load-bearing decisions.** Pick the v0.51
   commits and identify decisions whose rationale you cannot
   reconstruct from the docs alone. R8-C1's "warning persists past
   v1.0" is one such decision — others?

3. **End-to-end feature trace.** Trace `recall_at_fpr` end-to-end
   at v0.51: public API → tests → docs → migration guide → audit
   ledger entry. Any seam where the story breaks?

4. **Weak-assertion test scan.** The Round-8 verification found C7
   weak-assertion patterns; the v0.51 fix bundled stronger tests
   per finding. Spot-check that NO new weak-assertion tests were
   introduced in the v0.51 commits.

## How to respond

Whatever shape works for you. Things that help me make use of your
read:

- **Cite specific file paths + line numbers / sections** when
  flagging issues. Example: `src/eval_toolkit/_rng.py:50-65` rather
  than "the spawn helper."
- **Mark severity in your own words** — "blocker for v1.0",
  "v0.51.1 hotfix candidate", "accept with rationale at v1.0",
  "v1.x follow-up." Your call on the boundaries.
- **Be honest about confidence** — "potentially wrong; worth
  verifying" is a useful flag.
- **Don't anchor on the plan's framing.** If you think a design
  decision is fundamentally wrong, say so. v1.0 means the contract
  locks; this is the last cheap moment to revisit.
- **Be brief when brief is honest.** If your read is "this is in
  good shape," that's a valid output and ideal for v1.0. Don't
  fabricate concerns to fill space.

## Packet

Send Codex / Gemini this briefing plus:

- `~/Claude/eval-toolkit/audit-verification-codex-gemini-v0.50.0.md`
  (the Round-8 verification ground-truth — what was confirmed /
  refuted / deferred; gitignored, so attach as a file)
- `~/Claude/eval-toolkit/comprehensive-audit-codex.md` +
  `audit-gemini.md` (the Round-8 source reports; gitignored, so
  attach as files)
- `~/Claude/audit-templates/audit-prompt.md` (the universal audit
  template authored alongside v0.51)
- `docs/source/audit_findings.md` (Round 8 section with the
  full 13-confirmed / 3-refuted / 2-deferred ledger)
- `docs/source/adr/0001-flat-module-layout.md` + `0002` + `0003` +
  `0004` (the 4 ADRs — the v1.0 contract documents)
- `docs/source/migration/v0.49.md`, `v0.50.md`, `v0.51.md` (the
  three new migration guides)
- `CHANGELOG.md` (v0.51.0 entry — full BREAKING + ADDED + FIXED
  sections)
- `src/eval_toolkit/*.py` (full source tree; focus on the v0.51
  changes — `_rng.py`, `harness.py`, `splits.py`, `thresholds.py`,
  `calibration.py`, `losses.py`, `metric_specs.py`, `analysis.py`,
  `claims.py` — PLUS the 10 modules neither Round-8 auditor cited)
- `tests/test_public_api.py` + `tests/golden/public_api/snapshot.json`
  (the regenerated v0.51 golden — this is what v1.0 will inherit)
- `tests/test_rng.py` (new at v0.51) +
  `tests/test_harness_parallelism.py` (R8-C4a regression tests) +
  `tests/test_harness_folded.py` (R8-C1 regression tests) +
  `tests/test_recall_at_fpr.py` (R8-C3 regression tests) +
  `tests/test_protocol_conformance.py` (R8-C2 regression tests)

Take whatever time you need. The 7-day stop-gate timeout (per
Decision Y.2) is a fallback — if your read takes 3 days, that's
fine; if 6 days, also fine.

## Cross-references

- **Prior round briefings** (tracked in repo root):
  `gate3-audit-round-5.md`, `-round-6.md`, `-round-7.md`,
  `-round-8.md`.
- **Canonical prompt template**:
  `~/.claude/plans/gate3-audit-prompt.md` (local, untracked) — this
  Round 9 briefing was structured from it plus the v0.51 audit-template.
- **Reports from your read** should land at the repo root as
  `gate3-audit-round-9-codex-report.md` /
  `gate3-audit-round-9-gemini-report.md` (gitignored per the
  `gate3-audit-round-*-report.md` rule).
- **Claude verification** of your reports will land at
  `~/Claude/eval-toolkit/audit-verification-round-9-v0.51.0.md`
  (gitignored per the `audit-verification-*.md` rule added at
  v0.51 R8-C10).

## Known issues — none currently in a backlog beyond this audit

Unlike Rounds 5-7, the v0.51 release closed all confirmed Round-8
findings (13/13 fixed). There is no v0.52 backlog — by design,
v1.0 follows v0.51 directly **if Round 9 lands clean**. Any
finding you surface in Round 9 is either a v0.51.1 hotfix candidate
(per Decision Q) or an accepted-with-rationale entry in
`audit_findings.md` for the v1.0 ship. The 2 deferred-to-v1.x
items (R8-G3 + R8-G4) are accepted-as-design at v1.0 and are NOT
Round-9 surfaces unless your read disagrees with the deferral
rationale.
