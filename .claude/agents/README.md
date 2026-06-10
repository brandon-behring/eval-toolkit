# eval-toolkit review agents

Repo-local Claude Code subagents that enforce the **judgment** the deterministic
gates can't: SemVer impact, audit-validator architecture, silent failures,
docstring conformance, and dogfood noise. They are **advisory** — `ruff` / `black` / `mypy` / `pytest` /
coverage / the public-API snapshot remain the authoritative blocking gates. No
agent re-runs or replaces them.

## The agents

| Agent | Catches | Authoritative source |
|---|---|---|
| `etk-audit-validator-reviewer` | Three-layer conformance (identity/scope/pairing), `_narrative` reuse, UTF-8 | ADR 0007 / 0005 / 0006, STYLE.md §5 |
| `etk-api-stability-guardian` | Tier-1/2/3 SemVer class + public-API snapshot regen | ADR 0003, `tests/test_public_api.py`, STYLE.md §17 |
| `etk-silent-failure-auditor` | NaN/inf finiteness gaps, swallowed exceptions, encoding/IO, non-diagnostic raises | STYLE.md §1 / §6 / §7 |
| `etk-docstring-conformance-auditor` | NumPy sections, Raises↔code agreement, canonical param names, runnable Examples | STYLE.md §12 / §3a |
| `etk-dogfood-noise-analyst` | Classifies consumer residuals (real / FP / edge × layer) | runner: `scripts/dogfood_audit.py` |

## How to run

You never need to remember the names. Either:

- **Describe the task** — "review the changes I made to the citation validator" — and the main agent auto-routes by each agent's `description`; or
- **Run `/review-eval`** — the one handle that fans them out and synthesizes one verdict.

```
/review-eval                      # diff mode: git diff main...HEAD
/review-eval --pr 84              # review a GitHub PR diff
/review-eval --audit              # full baseline sweep (whole files, no diff)
/review-eval --audit api          # focused: just the public surface
/review-eval --audit validators   # focused: just audit_*.py + _narrative.py
/review-eval --audit docstrings   # focused: just public docstrings
/review-eval --refute             # adversarial second pass (quote-or-reject)
/review-eval --ledger             # persist a review entry under .claude/reviews/
```

Diff mode prints to the terminal; `--audit` also writes a machine-local entry
under `.claude/reviews/` (gitignored).

## Conventions baked in

- **Read-back discipline** — every finding quotes code with `path:line`; no quote, no finding (counters validation-without-reading).
- **High-confidence only** — plus a `suppressed N low-confidence` footer so nothing is silently dropped.
- **Hybrid rubric** — a tight inlined checklist that defers to `STYLE.md` / the ADRs as the single source of truth.
- **Structured verdict** — `PASS / CONCERNS / BLOCK`, per-agent and overall.

`tests/test_claude_agents.py` guards these files against pointer-rot (frontmatter
parses, `name` matches filename, every cited path exists).

## Escalation

For a full multi-round release audit (fan out N finders → dedup → adversarially
verify → synthesize a ledger), use the **Workflow** tool, not a single subagent.
`/review-eval --audit` is the lightweight precursor.
