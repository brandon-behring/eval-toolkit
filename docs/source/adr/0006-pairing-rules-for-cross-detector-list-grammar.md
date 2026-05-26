# ADR 0006: Pairing rules for cross-detector list-grammar in audit validators

**Status:** Accepted at v1.3.0 — applies to all future audit validators
in the `eval_toolkit.audit_*` flat-module family.

**Date:** 2026-05-26

**Deciders:** Brandon Behring (author), `/exploring-options` 2-round
review during #81 implementation, consumer-feedback audit Round 14.

**Supersedes:** N/A. **Superseded by:** N/A.

## Context

[ADR 0005](0005-structured-keys-for-audit-validators.md) introduced
a two-layer correctness model for audit validators:

1. **Layer 1 — Identity** (`BindingKey` frozen dataclass) — v1.1.0.
2. **Layer 2 — Scope** (content-type + context-keyword filters via
   `scope="narrative"`) — v1.1.0 + v1.2.0.

The v1.2.0 release shipped four context-aware filters (T1 delta,
T2 floor, T3 consume-on-match per-sentence, T4 sentence-boundary
detector-pair reject) under `scope="narrative"`, achieving 93%
total noise reduction (95 → 7) on the consumer's writeup. ADR 0005's
"Future work (deferred)" section explicitly named two remaining
failure modes — *sentence-boundary unawareness* (closed by v1.2.0's
T4) and *multi-detector list parsing in dense prose* (still
deferred). Consumer adoption at `prompt-injection-detection-submission@v1.3.12`
reduced their dogfood to **4 residual warnings**, all in this
deferred category.

Issue [#81](https://github.com/brandon-behring/eval-toolkit/issues/81)
documented the 4 residuals as three distinct prose patterns:

- **Pattern A — "for X" postfix**: `"versus 0.364 [...] for the
  frozen probe and 0.291 [...] for TF-IDF + LR"`. The validator's
  proximity-based detector pairing mis-attributes 0.291 to the
  nearer prior detector mention; the `"for TF-IDF + LR"` postfix
  is the authoritative binding signal.
- **Pattern B — possessive `'s`**: `"LoRA's pooled OOD AUROC is
  0.383 against frozen probe's 0.515"`. The `'s` construction
  isn't part of detector-alias regex matching; cross-detector
  confusion follows.
- **Pattern C — group subject**: `"... 0.38 AUROC, ~0.6 drop for
  the trained detectors; frozen probe's gap is 0.91 → 0.515"`.
  The value 0.38 belongs to "trained detectors" (a multi-detector
  group), not the next-mentioned single detector.

A fourth pattern emerged during v1.3.0 dogfood:

- **Pattern D — metric-axis confusion**: `"than the AUPRC delta
  suggests: LoRA's pooled OOD AUROC is 0.383"`. The proximity-based
  metric check finds "AUPRC" from a delta clause earlier in the
  prose, even though AUROC is the metric semantically owning
  0.383. This is symmetric to detector-axis pairing — the same
  positional heuristic fails for metric-axis in dense prose.

These four patterns are **pairing-rule problems**, not identity or
scope problems. They require a third correctness layer that
operates **on top of** identity + scope.

## Decision

Introduce **Layer 3 — pairing rules** as the third correctness
layer for audit validators:

| Layer | Correctness dimension | Mechanism | Release |
|---|---|---|---|
| 1 | Identity | Structured key with named fields | v1.1.0 |
| 2 | Scope | Content-type + context-keyword filters | v1.1.0 + v1.2.0 |
| **3** | **Pairing** | **Override or suppress proximity-based detector / metric pairing under explicit grammar cues** | **v1.3.0** |

Layer 3 ships under the existing `scope="narrative"` bundle (no new
public kwargs). Tier-1 ADDITIVE per
[ADR 0003](0003-stability-contract-and-gate3-methodology.md):
`scope="all"` callers see zero behavior change.

### Four pairing rules

**Pattern A — `"for {detector}"` postfix override.**
When a candidate value is followed (within +50 chars) by a `"for
{detector_alias}"` construct AND no other value pattern lies
between the value and the postfix (excluding values in CI brackets
per v1.1.0's scope='narrative' content-type filter), the postfix is
authoritative:
- If the postfix names THIS binding's detector → confirm pairing
  (bypass proximity check).
- If it names a DIFFERENT canonical detector → skip (the other
  detector's loop iteration will claim the value).
- If unresolved → fall through to proximity.

**Pattern B — `"{detector}'s"` possessive override.**
Same mechanics as Pattern A, but scanning −80 chars before the
value for `"{alias}'s"`. The LAST possessive in the pre-window is
authoritative IF its end position is within 30 chars of the value
start (covers both immediate `"frozen probe's 0.515"` and
short-clause `"LoRA's ... AUROC is 0.383"`). Last-match — not
first — is critical: an earlier possessive belonging to a different
preceding value must not bleed into a later value's check.

**Pattern C — group-subject suppression.**
When prose contains `"for the {trained|frozen|baseline|all|both|other}
detectors"` within ±60 chars of the value AND on the same side of
any sentence boundary, the value refers to a multi-detector group
statement that doesn't bind to a single canonical detector. The
candidate is SUPPRESSED (no override). Multi-detector inference is
deferred to v1.4.0+.

**Pattern D — metric-axis nearest-pairing.**
Symmetric to detector-axis pairing. Pre-collects ALL metric
positions per file (across consumer-supplied `metric_aliases`,
not just metrics tied to canonical bindings). Requires the NEAREST
metric mention to the value (by text-order last-before-first-after)
to be THIS binding's canonical metric. Catches prose with multiple
metrics in close proximity where the v1.2.0 window-based proximity
check picks up the wrong metric.

### Why suppression (Pattern C) rather than inference

ADR 0005's deferred-work section framed multi-detector list parsing
as a 200+ LOC parser-level problem. v1.3.0 takes a simpler path:
when prose explicitly names a multi-detector group (`"for the
trained detectors"`), the validator SUPPRESSES the candidate
rather than trying to infer which detectors own the value. This
matches the architectural pattern of v1.2.0's T1/T2 (recognize a
context cue, skip the candidate) and avoids the high-risk
multi-detector iteration path (~250 LOC, MODERATE-HIGH risk per
the Round 12 Explore analysis). Inference can be added as a
Layer 3 extension (multi-detector iteration) in v1.4.0+ if
consumer demand emerges.

## Scope of this ADR

**Applies to the `audit_*` flat-module family only.** Other parts
of the codebase (e.g., `MetricSpec`, `harness.evaluate`) are NOT
retroactively forced to adopt pairing-rule mechanics. Audit
validators are a coherent subfamily that share the closed-config
pattern + the consumer-prose-aware mission.

Future pairing-rule additions (e.g., enumeration parsing for the
`"X scored Y, Z, and W for A, B, C respectively"` pattern, or
multi-detector inference replacing Pattern C's suppression) join
Layer 3 as additional rule families under the same architectural
slot.

## Consequences

### Positive

- **Closes #81's 4 residuals.** Consumer-side dogfood reaches 0
  warnings (down from 4); HARD-gate promotion of
  `audit_value_bindings` becomes credible. Combined with v1.1.0 +
  v1.2.0, **100% reduction vs the pre-fix v1.0.5 baseline** on
  the consumer's writeup.
- **Architectural consistency.** Layer 3 is opt-in via the
  existing `scope="narrative"` bundle (no new kwargs);
  backward-compat preserved for `scope="all"` callers.
- **Symmetric metric-axis pairing.** Pattern D extends the
  positional pairing model from detector-axis to metric-axis,
  using the existing `_nearest_canonical_key` helper. Establishes
  "axis-by-axis nearest-pairing" as a reusable Layer 3 building
  block.
- **Bypass + confirm semantics.** Pattern A/B overrides are
  authoritative: they CONFIRM pairing (bypass proximity check)
  when they match THIS binding's detector. Avoids the bug where
  override + proximity disagree and the value is wrongly rejected.

### Negative

- **Layer 3 adds ~150 LOC.** Pattern helpers, per-call regex
  builds, inner-loop wiring. Within the flat-module
  maintainability bar (comparable to v1.2.0's T1-T4 = ~150 LOC).
- **Pattern A intervening-value check + Pattern C sentence-
  boundary check reuse the existing exclusion-ranges and
  sentence-positions infrastructure** from v1.1.0/v1.2.0. Adds
  cross-layer coupling: changes to bracket-exclusion logic or
  sentence-detection logic now also affect Layer 3 correctness.
  Mitigation: unit tests pin each rule's behavior under known
  prose patterns.
- **Pattern D requires `metric_aliases` for unbound metrics.**
  When prose mentions a metric (e.g., AUROC) that has no canonical
  binding, the consumer must still pass it in `metric_aliases`
  for Pattern D to recognize it. Without the alias, Pattern D
  falls through to the binding's own metric (legacy v1.2.0
  behavior). Documented in v1.3.0 CHANGELOG.

### Future work (post-v1.3.0)

- **Multi-detector inference for Pattern C.** Replace suppression
  with multi-detector ownership iteration: when "for the trained
  detectors" is found, iterate the value-comparison block once per
  detector in the implied group. ~250 LOC; MODERATE-HIGH risk.
  Track as v1.4.0+ if consumer demand emerges.
- **Enumeration parsing.** Prose like `"X scored Y, Z, and W for
  A, B, C respectively"` requires positional alignment between two
  lists. Not addressed by v1.3.0. Track as v1.4.0+ if needed.
- **Markdown AST parsing** (ADR 0005 §A4) — v2.0 territory.

## Alternatives considered

### A1 — Markdown AST parsing

Rejected for v1.x per ADR 0005 §A4. Too heavy, fragile to markdown
dialects, ~1000+ LOC dependency. Stays v2.0 territory.

### A2 — Pattern A + B only (defer C and D)

Closes 3 of 4 residuals. ~100 LOC. Rejected because the consumer's
HARD-gate promotion is blocked on ALL 4 warnings; a 75% close-rate
release would not unblock the consumer-side workflow that motivated
this work.

### A3 — Multi-detector inference for Pattern C (instead of suppression)

~250 LOC; replaces the simple "for the trained detectors" suppression
with explicit iteration over implied group detectors. Rejected for
v1.3.0 because suppression closes the same false positives at ~30
LOC; inference's marginal value isn't worth the complexity until
consumer prose surfaces a case where suppression hides a real bug.

### A4 — Public kwargs for pairing rules

Add `list_connectives: Sequence[str] | None = None`,
`possessive_patterns: ...`, etc. — let consumers extend the
hardcoded sets. Rejected per ADR 0005 §4 reasoning: YAGNI without
concrete consumer demand. The hardcoded frozensets cover the
consumer's actual prose patterns; runtime extension can be added
in a future v1.3.x patch if needed.

### A5 — Layer 3 as a separate `scope="strict"` tier

A new `scope` value that's `narrative` + pairing rules.
Rejected because it creates an ordering relationship consumers
must remember (`all` ⊂ `narrative` ⊂ `strict`) and grows the API
surface. The existing `scope="narrative"` bundle already
represents "opt-in narrative-prose-aware correctness mode"; Layer
3 fits within that mental model.

## Cross-references

- [ADR 0001](0001-flat-module-layout.md) — flat-module layout
  still applies; Layer 3 helpers live in `audit_value_bindings.py`
  alongside the v1.2.0 helpers, not in a subpackage.
- [ADR 0003](0003-stability-contract-and-gate3-methodology.md) —
  Tier-1 ADDITIVE classification. No new public kwargs; no
  signature drift; only `__version__` and the inner-loop logic
  change.
- [ADR 0005](0005-structured-keys-for-audit-validators.md) —
  introduces Layer 1 + 2 and explicitly defers Layer 3 to v1.3.0+.
  This ADR is the formal closure of that deferred work.
- [Round 14 audit findings](../audit_findings.md) — captures
  the v1.3.0 cycle dogfood + the four-pattern taxonomy.
- Issue [#81](https://github.com/brandon-behring/eval-toolkit/issues/81)
  — consumer-filed signal that triggered this ADR.
