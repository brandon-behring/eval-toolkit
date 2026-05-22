# ADR 0003: v1.0 stability contract + Gate 3 methodology

**Status:** Accepted
**Date:** 2026-05-21 (drafted at v0.46-prep; finalized at v0.48)
**Deciders:** Brandon Behring (author), Round 5 / Round 6 / Round 7 audits
(Codex + Gemini)
**Supersedes:** N/A. **Superseded by:** N/A.

## Context

v1.0 is eval-toolkit's first **stability commitment** release. Two questions
must be answered precisely or the commitment is unspecified:

1. **Stability contract scope.** What exactly is frozen at v1.0? The
   maximally-strict reading — every public symbol, every signature,
   every docstring first line — taxes ongoing maintenance heavily.
   The maximally-loose reading — only top-level `__all__` — gives consumers
   a weak guarantee and forces submodule users to absorb refactoring as
   surprise breaking changes.

2. **Gate 3 methodology evidence.** The original v1.0 plan called for
   "methodology docs peer-reviewed by an external reader (statistics /
   methodology background, ideally not part of the `prompt_injection_*`
   core team)." For a single-author / single-consumer / one-month
   release window, identifying an external academic reviewer with the
   right expertise is high-variance calendar work. The plan-time
   alternative — multi-model LLM cross-review (author + Codex + Gemini)
   — is operationally cheaper but not the same evidence class.

Round 5 audit (Codex F7) flagged the second issue specifically: the plan
as originally drafted understated the difference between external human
peer review and model-assisted cross-review.

## Decision

Two interlinked decisions, finalized here:

### 1. Tiered stability contract (Decision M from the v1.0 plan)

v1.0 commits to **three tiers** of stability, each with explicit
SemVer-version cost-to-change:

#### Tier 1: STRICT (SemVer-major to change)

The following are part of the v1.0 strict contract and require a v2.0
bump to alter:

- **Symbols in `eval_toolkit.__all__`** (i.e., everything in `_EXPORTS`
  at v1.0 tag time). Examples: `scorecard`, `Scorecard`, `MetricSpec`,
  `MetricResult`, `bootstrap_ci`, `BootstrapCI`, `evaluate`,
  `RunManifest`, `LogisticStacker`, `MetaLearner`, etc.
- **Their signatures** — parameter names, defaults, kwarg-only markers,
  return types. A new optional kwarg with a default is technically
  additive, but it changes the signature snapshot — handle via the
  additive-Protocol path below or accept the major bump.
- **Tier-2 Protocols** (9 strict at v1.0, all shipped through v0.47.0):
  - `Scorer` (from `protocols.py`)
  - `LeakageCheck` (from `leakage.py`)
  - `Splitter` (from `splits.py`)
  - `ThresholdSelector` (from `thresholds.py`)
  - `DatasetLoader` (from `loaders.py`)
  - `MetricSpec` (from `_scorecard.py`, shipped v0.46)
  - `TextTransform` (from `protocols.py`, shipped v0.47)
  - `MetaLearner` (from `stacking.py`, shipped v0.45)
  - `Probe` (from `probes.py`, shipped v0.43)
  - PLUS 1 opt-in Protocol: `Versioned` (additive on top of Tier-2).

  Tier-2 Protocol method signatures are pinned by the Decision R6-D drift
  guard in `tests/test_public_api.py` (since v0.47.0): the golden
  snapshot captures each Protocol's method signatures + return types via
  `typing.get_type_hints` + `inspect.signature`, so a SemVer-major
  review fires on any shape change.
- **Current versioned JSON schemas** (`src/eval_toolkit/schemas/*.json`)
  per artifact type — `manifest.v3.json` (canonical), `manifest.v1.json`
  + `manifest.v2.json` (historical, kept for migration),
  `ood_manifest.v1.json`, `results.v1.json`, `results_full.v1.json`.
  Schema-version bumps within the v1.x line are additive only (e.g.,
  publishing `manifest.v4.json` is fine; renaming required fields in
  `manifest.v3.json` is not).

#### Tier 2: ADDITIVE-ONLY (SemVer-minor for changes)

The following may grow in v1.x minors but may not lose existing
functionality:

- **Submodule public symbols** (e.g., `eval_toolkit.metrics.pr_auc`,
  `eval_toolkit.bootstrap.paired_bootstrap_diff`,
  `eval_toolkit.calibration.fit_platt_calibrator`). These are
  documented as **internal API** per ADR 0002 — not part of Tier 1's
  strict freeze, but eval-toolkit commits to **not removing** them
  during v1.x. Refactor that adds new helpers + keeps old paths
  importable is acceptable.
- **Tier-2 Protocols can gain optional methods via subprotocols** —
  e.g., a future `MetricSpec` subprotocol could add
  `is_defined_on(y_true) -> tuple[bool, str]` for richer cell-state
  detection. Implementations that don't implement the subprotocol
  continue to satisfy the base Protocol; this is additive.

#### Tier 3: FREE (SemVer-patch)

The following may change in any release:

- **Docstring first lines** — `tests/test_public_api.py` currently
  captures them as part of the public-API snapshot. At v1.0, the
  snapshot test is modified to drop docstring-first-line capture from
  the golden (or gate it behind a `STRICT_DOCSTRINGS=1` env var for
  local power-user use). This removes the SemVer tax on docstring
  polish.
- **Implementation internals** (helpers without `__all__` entries,
  private `_-prefixed` symbols, module-level constants not in any
  `__all__`). These may move, rename, or disappear without a SemVer
  signal.
- **Error message wording**. The error TYPES (ValueError vs RuntimeError)
  are stable; the human-readable message text is not.

### 2. Gate 3 methodology cross-review

Gate 3 at v1.0 is **internal model-assisted cross-review**, **NOT**
external academic peer review. The process:

1. **Manual review by the author.** Author reads
   `docs/source/methodology/` (16 chapters) + new feature pages +
   ADRs 0001 / 0002 / 0003 / this file + the v1.0 plan with a critic's
   eye. The author is NOT an external reviewer; this is the
   highest-context but weakest-independence of the three reads.
2. **Codex independent report.** Provide the methodology curriculum +
   plan + relevant source files; ask for a methodology-focused review
   identifying gaps, contradictions, unstated assumptions, and
   statistical-correctness issues. Codex's training corpus differs from
   Anthropic's, so it surfaces issues Anthropic-trained reasoning may
   miss.
3. **Gemini independent report.** Same packet, different model. Google's
   training corpus differs from both Anthropic's and OpenAI's, so it
   provides a third independent read.

**Honest framing** (per Audit F7): this is **not** equivalent to external
academic peer review.

- **What multi-LLM cross-review catches well** (demonstrated in Round 5):
  plan/code contradictions, references to symbols that don't exist,
  doc-code drift, load-bearing instruction errors in implementation
  directives, mathematical claims that don't match implementation,
  public-API status contradictions. Round 5 surfaced 12 verified-real
  findings before any v0.46 code shipped.
- **What multi-LLM cross-review catches less reliably**: methodological
  judgments that depend on domain expertise outside the models' training
  corpora, deep statistical correctness on novel methods, whether the
  chosen methodology serves the practitioner's actual decision-making
  need.
- **What it does NOT substitute for**: external accountability. There is
  no third party who will be embarrassed if v1.0 turns out to have a
  methodology bug.

**Cycle structure** (per Decision Y.2):

- Round 5 (pre-v0.46 implementation) — **complete** 2026-05-21.
  Surfaced 12 verified-real findings, including F1 / F2 (scorecard
  shape blockers) and F7 (Gate 3 honesty — this ADR's framing).
- Round 6 (post-v0.46 ship) — **complete** 2026-05-21. STOP-GATE
  CLOSED before v0.47 opened. Codex R6-F1 (ECE strategy validation)
  + Codex R6-F2 (deprecation warning content) shipped as v0.46.1
  hotfix per Decision R6-E; the remaining findings (R6-A through R6-H)
  folded into v0.47.0.
- Round 7 (post-v0.47 ship) — **complete** 2026-05-21. STOP-GATE
  CLOSED before v0.48 opened. Codex R7-F1 (MyST-NB doc-execution gap)
  + R7-F2 (sweep strategy_id) + R7-F3 (sweep scorer output shape)
  folded into v0.48.0; Gemini observations folded into v0.48
  documentation polish (§5K) + the Makefile pre-push target (§5L).
- Round 8 (post-v0.48 ship) — final pre-v1.0 packet review.
- v1.0 tag requires Round 8 to close (or fall back per ADR amendment).

**Findings ledger**: [`docs/source/audit_findings.md`](../audit_findings.md)
records each round + the disposition of every blocker-severity finding.
Blocker findings also get a `p1-gate3`-labelled GitHub issue for
fix-tracking.

**Escalation path** (if a Round flags methodological judgment beyond
LLM capability): consult a human reviewer for the specific narrow
question. Don't require a full-curriculum external read; do require a
human signoff on any methodology claim that surfaces as "models disagree"
or "models flag uncertainty without confidence."

## Consequences

**Positive:**

- The tiered contract gives consumers a clear answer to "what can I rely
  on?" — Tier 1 is bedrock; Tier 2 is documented-as-additive; Tier 3 is
  understood-to-evolve.
- Submodule users (the `eval_toolkit.metrics.*` escape hatch) get an
  explicit additive-only commitment for v1.x — not the maximally-loose
  "no commitment" of a narrow contract.
- Gate 3 honesty: future maintainers can read this ADR and understand
  exactly what evidence the v1.0 release carries (and doesn't).
- Rounds 5 / 6 / 7 demonstrated that multi-LLM cross-review catches
  real issues at each stage (Round 5: 12 findings, Round 6: 11 findings,
  Round 7: 3 substantive Codex findings + 6 Gemini observations); the
  cycle structure ensures coverage of every breaking minor before the
  final stability tag.

**Negative:**

- The Tier 1 commitment to Tier-2 Protocol shapes makes future
  Protocol-evolution work harder. v1.x can ADD subprotocols but cannot
  MODIFY base method signatures without v2.0.
- The tiered contract is more documentation surface to maintain than a
  flat "everything is strict" or "only `__all__` is strict" contract.
- Multi-LLM cross-review has no formal external-accountability story.
  If a v1.0 methodology bug surfaces post-release, no third party shares
  responsibility for missing it.

## Alternatives considered

### Maximally-strict contract (every public symbol, signature, docstring)

Reject because the docstring-first-line capture would tax every
documentation-polish PR with a SemVer concern. Test snapshot would
trigger on every doc improvement.

### Maximally-loose contract (only top-level `__all__`)

Reject because the submodule escape hatch is real and used. Without a
commitment, consumer code at `from eval_toolkit.metrics import
expected_calibration_error` could break without SemVer warning, defeating
the consumer's purpose for pinning to a stable version.

### Strict + flat (no tiering, freeze submodule too)

Reject because the v1.x line needs room to grow. The 4-binary-calibrator
family added across v0.35 → v0.42 was exactly the kind of additive
submodule evolution the flat-strict contract would have penalized.

### External academic peer review as required Gate 3

Reject because the single-author, one-month release window made
reviewer-identification calendar risk a hard blocker. The Round 5 audit
demonstrated multi-LLM cross-review surfaces real findings (12 verified)
quickly; the trade-off (no external accountability) is documented here
rather than overclaimed.

### Defer Gate 3 entirely (self-review only)

Reject because Round 5 demonstrated that even at the author's level of
familiarity with the plan, Codex + Gemini each caught issues the author
missed. Removing the cross-review step would surrender that benefit.

## Trigger to revisit

This ADR is locked at v1.0. Revisiting requires **SemVer-major (v2.0)** or
explicit ADR amendment. Likely triggers:

- **Second production consumer** with materially different access
  patterns (e.g., heavy submodule use that exposes additive-only
  ambiguity).
- **A v1.x methodology bug** that multi-LLM cross-review missed but
  external review would have caught. In that case, re-tighten Gate 3
  for v2.0+ to require human review for specific claim types.
- **Schema bumps** beyond v3 (manifest schema is currently at v3). If
  v1.x publishes `manifest.v4.json` as canonical, this ADR's schema-list
  is amended (additive change to Tier 1 — does NOT require v2.0).

## References

- v1.0 plan: `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`
  (Decisions M + O).
- Round 5 audit ledger: [`audit_findings.md`](../audit_findings.md) —
  Audit F7 (Gate 3 honesty) is the key finding for this ADR.
- [ADR 0001 — flat module layout](0001-flat-module-layout.md) — the
  module-layout contract that operates inside the Tier-1 stability
  commitment defined here.
- [ADR 0002 — scorecard as primary metric surface](0002-scorecard-as-primary-metric-surface.md)
  — adds `MetricSpec` to Tier-2 Protocols list above; demotes
  `eval_toolkit.metrics.*` to the Tier-2 additive-only commitment.
- `tests/test_public_api.py` — currently captures docstring first lines;
  modified at v1.0 to skip them per Tier 3.
