# Architecture Decision Records

This directory captures architecturally-significant decisions that shape
`eval-toolkit`'s long-term design. ADRs are immutable historical records —
once accepted, a decision is not edited in place; if it changes, a new ADR
supersedes it.

## When to file an ADR

File a new ADR when a decision:

- **Locks in an interface or shape** that future code is expected to
  conform to (e.g., "metrics return type", "Protocol vs ABC").
- **Closes off alternatives** that were seriously considered, so the
  reasoning isn't lost.
- **Carries cost** to reverse (e.g., a public-API contract that promises
  stability across a release line).

Routine refactors, bug fixes, and internal-only patterns do not need ADRs —
the commit message + CHANGELOG entry are enough.

## Numbering

Sequential, zero-padded: `0001-flat-module-layout.md`,
`0002-scorecard-as-primary-metric-surface.md`, etc. Number is assigned
at the time of writing; if two ADRs are drafted in parallel, the second
to merge takes the next number.

## Format

Each ADR uses this skeleton (loosely based on MADR — Markdown ADR — without
the heavyweight template):

```markdown
# ADR NNNN: Title

**Status:** Proposed | Accepted | Superseded by ADR-MMMM
**Date:** YYYY-MM-DD
**Deciders:** (names or roles)

## Context

What's the situation that requires a decision? What constraints are at play?

## Decision

What did we decide?

## Consequences

What follows from this decision? (Both positive and negative.)

## Alternatives considered

What else was on the table, and why wasn't it chosen?

## Trigger to revisit

What would have to change for this decision to be reopened?
(Optional but useful — keeps the ADR self-documenting.)
```

## Cross-references

- [`docs/RELEASING.md`](../../RELEASING.md) — release-flow process; ADRs
  are typically drafted as part of release prep.
- [`docs/source/roadmap.md`](../roadmap.md) — long-term direction;
  ADRs explain how individual roadmap decisions were made.

## Index

(Updated as ADRs are added.)

| # | Title | Status | Date |
|---|---|---|---|
| _none yet_ | | | |
