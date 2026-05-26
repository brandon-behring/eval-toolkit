# ADR 0005: Structured keys over positional tuples for canonical-identity types in audit validators

**Status:** Accepted at v1.1.0 — applies to all future audit validators
in the `eval_toolkit.audit_*` flat-module family.

**Date:** 2026-05-26

**Deciders:** Brandon Behring (author), `/exploring-options` 4-round
review during #80 implementation, consumer-feedback audit Round 12.

**Supersedes:** N/A. **Superseded by:** N/A.

## Context

The v1.0.3 release of `audit_value_bindings` (#71) shipped with a
canonical-binding schema keyed on a positional 2-tuple:

```text
BINDINGS: Mapping[tuple[str, str], float]
# key = (detector, metric); value = expected_value
```

Three days later (2026-05-26), consumer-side dogfooding produced
**96 warnings, ~95 false positives** because the same
`(detector, metric)` pair legitimately appears across multiple
measurement *slices* in research-writeup prose (`direct_validation`,
`pooled_ood`, paired-delta tables, random-floor mentions). The
2-tuple identity collapsed the slice axis — every cross-slice
mention produced a value mismatch against the single canonical
binding. Issue [#80](https://github.com/brandon-behring/eval-toolkit/issues/80)
proposed extending to a 3-tuple `(detector, metric, slice)` schema.

Two architectural questions arose during the design discussion:

1. **Should the third axis be a positional 3-tuple, or a structured
   identity type?** A positional tuple solves the immediate
   slice-axis problem but locks the validator into a recur-every-
   N-months schema-event pattern — the next axis addition (split,
   ci_kind, source_ref, ...) would again be a breaking schema
   change.
2. **Is identity correctness sufficient, or is there a second
   correctness layer?** First-pass dogfood of the slice-axis fix
   alone reduced noise only ~22% (95 → 74 warnings). ~80% of the
   residual noise came from **content-type confusion**: the
   validator was matching values inside CI brackets (`[0.286,
   0.301]`), markdown table cells, and fenced code blocks — none of
   which are narrative binding claims.

Both questions point at the same underlying gap: audit validators
that ship to v1.x stability must distinguish *what they're looking
at* (identity) from *where they look* (scope).

## Decision

Audit validators in `eval_toolkit.audit_*` use a two-layer
correctness model:

### Layer 1 — Identity correctness (structured keys)

Canonical-identity types for binding schemas are **frozen dataclasses
with named fields**, not positional tuples. Forward-extensible: new
identity axes are added as defaulted fields without breaking the
dict-key schema.

The v1.1.0 instantiation for `audit_value_bindings`:

```text
@dataclass(frozen=True)
class BindingKey:
    detector: str
    metric: str
    slice: str = "any"
    # Future: split: str = "any", ci_kind: str = "point", ...
```

`@dataclass(frozen=True)` provides:
- Immutability (safe as a dict key).
- Structural equality + hash (works as a dict key).
- Type-checker-friendly named fields (mypy/pyright see field types).
- Adding a defaulted field is **Tier-1 ADDITIVE** per ADR 0003
  (existing consumer code continues to work; the new field defaults
  to a value that preserves prior semantics).

#### Multi-shape input adapter (migration pattern)

To preserve backward compatibility for legacy positional-tuple inputs
during the v1.x line, validators accept multiple input shapes and
normalize internally via a per-key adapter:

```text
# All three shapes accepted by validate_reader_value_bindings:
BindingKey("tf-idf", "AUPRC", "direct_validation")  # canonical
("tf-idf", "AUPRC", "direct_validation")            # sugar 3-tuple
("tf-idf", "AUPRC")                                 # legacy (slice="any")
```

The adapter is a tactical migration aid, NOT a co-equal architectural
rule. New consumer code SHOULD use the canonical `BindingKey` form;
tuple shapes are syntactic sugar / backward-compat preservation.
Formal deprecation of tuple inputs is deferred to a future v2.0
cleanup pass when there is concrete payoff (see ADR 0003's
staggered-deprecation discipline).

### Layer 2 — Scope correctness (content-type filtering)

Audit validators scan markdown surfaces for binding claims. Not all
content in a markdown file is a candidate claim:

- **Narrative prose sentences** ARE candidate claims. The motivating
  bug class (V1.3.1 ADR-080) — and the bug `audit_value_bindings` is
  designed to catch — is misbindings in narrative prose.
- **Markdown table rows** (`| Detector | AUPRC | AUROC |`) are
  structured data, audited via different mechanisms (e.g., direct
  results-table verification). Table cells typically contain
  multiple metrics per row that positional heuristics cannot
  disambiguate.
- **Bracketed expressions** (`[CI 0.286, 0.301]`) contain bound
  values, not point estimates. The numeric content inside brackets
  is not a binding claim.
- **Fenced code blocks** contain code or literal data, not
  narrative claims.

Validators provide a `scope` parameter (Literal type) defaulting to
the v1.0.x-compatible `"all"` for backward compat, with
`"narrative"` opting into the content-type filter:

```text
def validate_reader_value_bindings(
    *,
    files: ...,
    bindings: ...,
    scope: Literal["all", "narrative"] = "all",  # NEW v1.1.0
    ...
) -> ValueBindingsReport:
```

When `scope="narrative"`: the validator pre-computes excluded
character ranges per file (table rows, bracketed expressions, code
blocks) and skips candidate values whose position falls inside an
excluded range. Compatible with the motivating misbinding bug class
(V1.3.1 ADR-080 was narrative prose; no recall loss).

#### Lint-design parallel

This convention mirrors production-quality linters:

| Linter | Scope predicate |
|---|---|
| `ruff` / `flake8` | `# noqa[: rule]` comments |
| `mypy` | `# type: ignore[code]` comments |
| `bandit` | `# nosec` comments |
| `eslint` | `// eslint-disable-line` comments |

Scope awareness is not optional in production-quality linters; an
audit validator that lacks scope predicates is incomplete.

## Scope of this ADR

**This rule applies to the `audit_*` flat-module family only.** Other
parts of the codebase (e.g., `MetricSpec` positional tuples,
`harness.evaluate` slice keys) are NOT retroactively forced to
migrate. Audit validators are a coherent subfamily that:

1. Share the closed-config pattern (consumer supplies the canonical
   table; validator owns parsing).
2. Ship to Tier-1 STRICT and live for the v1.x line.
3. Are designed for consumer prose, which has open-ended identity
   axes (any project's writeup style may introduce new slice / split
   / model-variant axes over time).

The non-audit modules don't share these properties to the same
degree; their tuple keys (where present) are stable and don't
exhibit the recur-every-N-months schema-event pattern. Forcing
them to migrate would be scope creep.

## Consequences

### Positive

- **Forward extensibility without schema events.** New identity axes
  added as `field: T = default` are Tier-1 ADDITIVE per ADR 0003.
  No more #80-redux when the next axis surfaces.
- **Validator usefulness in dense prose.** `scope="narrative"`
  brings noise reduction to ~80% on real consumer writeups,
  versus 22% for identity fix alone (v1.1.0 dogfood evidence).
- **Lint-design alignment.** Audit validators behave like
  production-quality linters (`ruff`/`mypy`/`bandit`), not like
  one-off pattern-matching scripts.
- **Multi-shape adapter as low-cost migration.** ~20 LOC of
  normalization preserves all pre-v1.1 consumer code; voluntary
  per-binding migration to `BindingKey` form happens at consumer's
  own pace.

### Negative

- **Slightly more code per validator.** A frozen dataclass + a
  normalization helper + a scope predicate adds ~50 LOC to a typical
  audit module. Validated by the v1.1.0 implementation
  (`audit_value_bindings.py`: 448 → ~620 LOC inclusive of the new
  helpers, dataclass, and docstring expansions).
- **Two valid input shapes during v1.x.** Consumers may see legacy
  2-tuple usage in old example code alongside `BindingKey` in new
  example code. Docstrings flag `BindingKey` as the canonical/
  recommended form; the multi-shape adapter ensures correctness
  regardless.
- **`scope="all"` default.** Backward compat requires the default to
  preserve v1.0.x behavior, which means consumers with new
  validator instances must explicitly opt into the better
  `"narrative"` scope. Documented in the validator's docstring +
  CHANGELOG.

### Future work (deferred)

The v1.1.0 dogfood revealed two non-identity, non-scope failure
modes that fall outside the original ADR 0005 scope:

1. **Sentence-boundary unawareness** — prose like "X scored 0.291.
   The pooled OOD random floor is 0.374" pairs 0.374 with detector
   X across a `.` boundary.
2. **Multi-detector list parsing in dense prose** — prose like
   "LoRA scored 0.293, versus 0.364 for the frozen probe and 0.291
   for TF-IDF + LR" over-credits the second detector in a list
   construction.

**v1.2.0 partial closure** (2026-05-26 follow-up release): the
first item is **resolved** via T4 (sentence-boundary
detector-pair reject); a related set of context-aware filters
T1–T3 was added under the same `scope="narrative"` opt-in:

| Filter | Failure mode addressed |
|---|---|
| T1 | Delta-magnitude values (signed or near `delta`/`drop`/`vs`/`below` keywords) |
| T2 | Random-floor / chance-baseline values (near `random`/`floor`/`chance`/`trivial`) |
| T3 | Same-binding duplicate flags within one sentence (catches "0.556 vs 0.519" enumerations from the same detector context) |
| T4 | Detector-value pairs spanning a sentence boundary |

Combined dogfood result: consumer's residual 36 (v1.1.0) → 7
(v1.2.0); 93% total reduction vs the pre-fix v1.0.5 baseline.

**Still deferred** (post-v1.2.0): the cross-detector
list-grammar problem proper — prose where a single
detector mention precedes multiple values that belong to
DIFFERENT detectors via list connectives ("and", "for X", "vs").
T3 only deduplicates the SAME binding within one sentence; it
doesn't infer that subsequent values belong to other detectors.
The 7 v1.2.0 residuals are all this shape. Track as v1.3.0+
with its own ADR design review; the path forward is either
shallow list-grammar parsing (~250 LOC, MODERATE-HIGH risk per
the Round 12 Explore agent's analysis) or markdown AST parsing
(ADR 0005 A4; v2.0 territory).

## Alternatives considered

### A1 — In-place 3-tuple (issue body's literal proposal)

Extend `BINDINGS: Mapping[tuple[str, str], float]` to
`Mapping[tuple[str, str, str], float]`. Preserve 2-tuple acceptance
for backward compat.

**Rejected** because: positional tuples are forward-fragile. The next
identity axis (split, ci_kind, ...) would require another schema
event with consumer migration, recurring every N months. The
`BindingKey` dataclass eliminates the recurrence at a one-time cost
of ~25 LOC (dataclass + normalization helper).

### A2 — Strict 3-tuple-only (no backward compat)

Drop 2-tuple support entirely; require all consumer code to migrate
atomically.

**Rejected** because: would force a v2.0 major bump for a Tier-1
STRICT signature change. Conflicts with the staggered-breaking-
releases discipline ([memory `feedback_staggered_breaking_releases`]).
The multi-shape adapter is ~20 LOC and consumes one if-elif-else
branch in the inner loop; the cost of preserving backward compat
is trivial relative to the cost of forcing a v2.0.

### A3 — Slice-axis fix only (no scope predicate)

Ship `BindingKey` + slice-aware matching as v1.1.0; file
narrative-scope as v1.2.0.

**Rejected** because: dogfood showed identity fix alone delivered
only 22% noise reduction; consumers would not adopt v1.1.0 in
practice (validator still produces ~74 warnings on typical
writeups, which can't be acted on). Releasing two minor versions
to deliver a coherent improvement violates the
"one coherent fix per minor" intent of staggered releases — they
should be unrelated breaks split across minors, not related
correctness improvements.

### A4 — Markdown AST parsing (proper structured input)

Replace the regex-based positional-heuristic matching with a
proper markdown parser (`markdown-it-py` or similar). Build a
semantic AST; match claims at the sentence/paragraph level.

**Rejected for v1.x** because: 1000+ LOC, months of work, fragile
to markdown dialects, adds a heavy dependency. Inconsistent with
eval-toolkit's flat-module simple-tool aesthetic. Worth considering
for v2.0 if the pairing-rule limitations (sentence-boundary,
list-parsing — see "Future work") become acute.

### A5 — Advisory-only validator (no gating)

Reframe `audit_value_bindings` as advisory rather than a HARD-gate
candidate; output ranked candidate violations; consumers manually
triage.

**Rejected** because: the bug class the validator is designed to
catch (V1.3.1 ADR-080-style misbindings) IS gateable in narrative
prose with the v1.1.0 fix. Reframing as advisory undersells what
the validator delivers. Consumers who want advisory-only behavior
can already inspect `ValueBindingsReport.violations` without
acting on them; making it the default would be a regression.

## Cross-references

- [ADR 0001](0001-flat-module-layout.md) — flat-module layout still
  applies; `BindingKey` lives in the same `audit_value_bindings.py`
  flat module, not a subpackage.
- [ADR 0003](0003-stability-contract-and-gate3-methodology.md) —
  Tier 1/2/3 stability contract. `BindingKey` ships as Tier-1
  STRICT; adding the new `scope` and `slice_aliases` kwargs is
  Tier-1 ADDITIVE (defaults preserve prior semantics).
- [ADR 0004](0004-naming-conventions.md) — naming conventions.
  `BindingKey` follows PascalCase + the `*Key` suffix convention
  (matches `Match`, `Violation`, `ValueBindingsReport` siblings in
  the same module).
- [Round 12 audit findings](../audit_findings.md#round-12-2026-05-26--schema-extensibility--scope-correctness-lesson-from-80)
  — the audit-trail capture of the v1.1.0 decision process.
- Issue [#80](https://github.com/brandon-behring/eval-toolkit/issues/80) — the consumer-feedback signal that triggered this ADR.
