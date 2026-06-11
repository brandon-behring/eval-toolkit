# ADR 0007: Three-layer architecture for audit validators (family-wide)

**Status:** Accepted at v1.4.0 — applies to all `eval_toolkit.audit_*`
flat-module validators going forward.

**Date:** 2026-05-26

**Deciders:** Brandon Behring (author), `/exploring-options` 3-round
review during #82 implementation, consumer-feedback audit Round 15.

**Supersedes:** N/A. **Superseded by:** N/A.

## Context

[ADR 0005](0005-structured-keys-for-audit-validators.md) (v1.1.0)
introduced a two-layer correctness model for `audit_value_bindings`:
identity + scope. [ADR 0006](0006-pairing-rules-for-cross-detector-list-grammar.md)
(v1.3.0) added Layer 3 (pairing rules) for the same validator. Both
ADRs were originally framed validator-specific. The R11→R14 cycle
demonstrated that the three-layer model is the canonical architecture
for the audit-validator family — not just for one validator.

Issue [#82](https://github.com/brandon-behring/eval-toolkit/issues/82)
surfaced the same architectural gap in `audit_citation_alignment`
(shipped v1.0.1; identity only). 188 residual warnings on the
consumer's HEAD; same root cause class as the v1.1.0 → v1.3.0
journey for `audit_value_bindings`. v1.4.0 ships Layer 2 + Layer 3
for `audit_citation_alignment` as a single coherent release.

The v1.4.0 release also extracts the shared narrative-prose helpers
into private flat module `eval_toolkit._narrative` so both validators
import the same implementation rather than maintaining parallel
copies (consistent with ADR 0001's flat-module precedent —
`_rng.py`, `_parallel.py`, `_sweep.py` are existing private modules).

This ADR codifies the three-layer model as the canonical architecture
for the entire `audit_*` validator family.

## Decision

All current and future `audit_*` validators in `eval_toolkit` adopt
the three-layer correctness model:

| Layer | Correctness dimension | Mechanism | Cross-reference |
|---|---|---|---|
| **1** | **Identity** | Canonical-identity types use frozen dataclasses with named fields, not positional tuples | [ADR 0005](0005-structured-keys-for-audit-validators.md) |
| **2** | **Scope** | Content-type filter via `scope: Literal["all", "narrative"] = "all"` opt-in kwarg; `"narrative"` excludes markdown tables, bracketed expressions, and fenced code blocks | ADR 0005 §"Layer 2"; v1.1.0 / v1.2.0 of `audit_value_bindings`; v1.4.0 of `audit_citation_alignment` |
| **3** | **Pairing** | Override or suppress proximity-based pairing under explicit grammar cues; activates under `scope='narrative'` | [ADR 0006](0006-pairing-rules-for-cross-detector-list-grammar.md); v1.3.0 of `audit_value_bindings`; v1.4.0 of `audit_citation_alignment` |

### Shared helpers (`eval_toolkit._narrative`)

Per the v1.4.0 refactor, narrative-prose primitives live in the
private flat module `eval_toolkit/_narrative.py`:

- Keyword frozensets: `_DELTA_KEYWORDS`, `_FLOOR_KEYWORDS`,
  `_GROUP_SUBJECT_KEYWORDS`, `_ABBREV_BEFORE_DOT`.
- Compiled patterns: `_DELTA_PATTERN`, `_FLOOR_PATTERN`,
  `_GROUP_SUBJECT_PATTERN`.
- Structural helpers: `_build_exclusion_ranges`, `_is_excluded`,
  `_compile_keyword_pattern`.
- Sentence-boundary helpers: `_is_sentence_terminator_dot`,
  `_sentence_boundary_positions`, `_sentence_id_of`,
  `_crosses_sentence_boundary`.
- Value-context helpers: `_is_signed_value` (imported by
  `audit_value_bindings` since v1.11.0; previously inventory-listed
  with zero importers), `_has_keyword_in_window`.
- Positional helpers: `_line_starts`, `_position_to_line`
  (consolidated v1.11.0, #99 — previously triplicated across the
  three validators; `audit_sister_doc_concept_drift` now imports
  from `_narrative` too).

All three validators import from this module (`audit_value_bindings`
v1.4.0+ refactor, signature-preserving; `audit_citation_alignment`
v1.4.0+ new adoption; `audit_sister_doc_concept_drift` v1.11.0+ for
the positional helpers). Future audit validators add their own
context-aware behavior on top.

The module is **private** (underscore-prefixed name, not in the
package's `_EXPORTS` resolver). Consumers don't import directly;
they use the public `audit_*` validators. Promotion to a public
module is YAGNI until concrete cross-consumer demand emerges.

## Validator family status (post-v1.4.0)

| Validator | Layer 1 (identity) | Layer 2 (scope) | Layer 3 (pairing) | Closes |
|---|---|---|---|---|
| `audit_value_bindings` | v1.1.0 | v1.1.0 + v1.2.0 | v1.3.0 | #71, #80, #81 |
| `audit_citation_alignment` | v1.0.1 (originally identity-only) | v1.4.0 | v1.4.0 | #73, #82 |
| `audit_sister_doc_concept_drift` | v1.0.4 (embedding-based; identity only) | (when consumer needs) | (when consumer needs) | #72 |

The three-layer model is the entry point for any new `audit_*`
validator. Implementations may ship Layer 1 only at their first
release (per `audit_citation_alignment` v1.0.1 precedent) and add
Layers 2 + 3 in follow-on minor releases as consumer feedback
surfaces context-correctness gaps. The library-first cycle (R11→R15
to date) is the canonical evolution mechanism.

## Consequences

### Positive

- **Architectural consistency across the family.** All audit
  validators share the same correctness vocabulary; consumer
  mental model transfers across validators.
- **Shared narrative helpers reduce drift.** Bugs in
  exclusion-ranges or sentence-boundary detection are fixed once,
  benefiting all validators.
- **Tier-1 ADDITIVE for layer additions.** Adopting Layers 2 + 3
  on an existing validator is a minor-version bump (default
  `scope="all"` preserves backward compat). Consumers opt in at
  their own pace.
- **Codifies the library-first cycle.** Future consumers and
  contributors have a clear template for filing issues and
  upstream design: "which layer is this gap in?"

### Negative

- **Some validators won't need all three layers.** For example,
  `audit_sister_doc_concept_drift` uses embedding similarity, not
  positional heuristics — Layer 3 pairing may not apply. The ADR
  doesn't force unused layers.
- **Layer 3 rule sets diverge across validators.** Each validator
  has its own rule names (Patterns A/B/C/D vs α/β/γ). Intentional —
  rules are prose-pattern-specific — but consumers reading both
  validators see different vocabularies. The unifying concept is
  "Layer 3 = override/suppress proximity pairing under grammar
  cues."
- **`_narrative` module grows over time.** As new validators add
  helpers, this private module accumulates. Future refactor may
  split into sub-modules (still private). Out of scope for v1.4.0;
  ADR 0001's flat-module commitment holds through v1.x.

## Alternatives considered

### A1 — Keep ADR 0005 / ADR 0006 validator-specific; no ADR 0007

Smaller diff. Rejected because the v1.4.0 cycle adopted the same
architecture for `audit_citation_alignment` — that's family-wide
behavior, not validator-specific. ADR 0007 documents what the
codebase ALREADY does.

### A2 — Public helpers (`eval_toolkit.audit_narrative`)

Promote `_narrative` to public API. Rejected: YAGNI. Consumers
don't need direct access; they use the public `audit_*` validators
which delegate to `_narrative`. Promoting is Tier-1 STRICT
addition with maintenance burden; not justified by current demand.

### A3 — Force Layer 2 + Layer 3 on `audit_sister_doc_concept_drift`

Apply the three-layer model uniformly to every validator regardless
of need. Rejected: `audit_sister_doc_concept_drift` uses embedding
similarity, not positional regex; the false-positive surface is
different. Add layers only when consumer dogfood surfaces gaps.

### A4 — Sub-package layout (`eval_toolkit.audit.{citation,value,...}`)

Rejected per ADR 0001: stay flat through v1.x. The flat-module
constraint is a v1.0 contract; restructuring to subpackages waits
for v2.0 (if ever).

## Cross-references

- [ADR 0001](0001-flat-module-layout.md) — flat-module commitment.
- [ADR 0003](0003-stability-contract-and-gate3-methodology.md) —
  Tier 1/2/3 stability contract.
- [ADR 0005](0005-structured-keys-for-audit-validators.md) —
  Layer 1 + Layer 2 origin (validator-specific framing).
- [ADR 0006](0006-pairing-rules-for-cross-detector-list-grammar.md)
  — Layer 3 origin (audit_value_bindings-specific framing).
- Issue [#82](https://github.com/brandon-behring/eval-toolkit/issues/82)
  — consumer-filed Round 15 trigger.
- Round 14 + 15 audit ledger entries.
