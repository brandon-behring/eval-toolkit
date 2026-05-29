---
description: Fan out the eval-toolkit review subagents over a diff (default) or a full-scope audit, then synthesize one verdict.
argument-hint: "[--audit [target]] [--pr N] [--refute] [--ledger] [--consumer PATH]"
allowed-tools: Task, Bash(git diff:*), Bash(git show:*), Bash(gh pr diff:*), Bash(make dogfood:*), Read, Grep, Glob, Write
---

You are orchestrating eval-toolkit's repo-local review subagents. Parse `$ARGUMENTS` and run the appropriate mode. These agents are **advisory** — ruff/black/mypy/pytest/coverage/the public-API snapshot remain the authoritative blocking gates. Do not duplicate them.

## The agents

- `etk-audit-validator-reviewer` — three-layer model (identity/scope/pairing) for `audit_*.py` + `_narrative.py`.
- `etk-api-stability-guardian` — Tier-1/2/3 SemVer classification + snapshot-regen gate for the public surface.
- `etk-silent-failure-auditor` — NaN/inf finiteness gaps, swallowed exceptions, encoding/IO, non-diagnostic raises across `src/`.
- `etk-docstring-conformance-auditor` — NumPy docstring structure, Raises↔code agreement, canonical param names, doctest-runnable Examples.
- `etk-dogfood-noise-analyst` — classifies consumer residual findings (needs the dogfood runner output).

## Mode resolution

**Diff mode (default, no `--audit`).** Resolve the diff scope, then launch the relevant agents **in parallel** (one Task call each, in a single message), feeding each the **full changed files with the changed spans marked** (not just hunks — a NaN-bypass or scope gap often sits in unchanged lines of a touched function):
- Default scope: `git diff main...HEAD` (merge-base). If on `main` with no upstream divergence, fall back to `git diff --staged` then `git diff`.
- `--pr N`: use `gh pr diff N` for the diff.
- Route by what changed: `audit_*.py`/`_narrative.py` → audit-validator-reviewer; public-surface touches (`__all__`, public signatures, Protocol methods, exported constants, `__init__.py` `_EXPORTS`) → api-stability-guardian; error handling / numeric guards / IO / threshold comparisons → silent-failure-auditor; public docstring additions/edits → docstring-conformance-auditor. A single change often hits several of these — when more than one matches (or you are unsure), run all that apply in parallel. Skip dogfood-noise-analyst in diff mode unless `--consumer PATH` is given.

**Audit mode (`--audit [target]`).** Full files, no diff.
- No target → **full baseline sweep**: launch audit-validator-reviewer (over all `audit_*.py` + `_narrative.py`), api-stability-guardian (whole public surface), silent-failure-auditor (all of `src/`), and docstring-conformance-auditor (public docstrings, kernels in `.doctest-modules`), in parallel. Skip dogfood unless `--consumer PATH` is given.
- `--audit validators` → audit-validator-reviewer only. `--audit api` → api-stability-guardian only. `--audit docstrings` → docstring-conformance-auditor only. `--audit <path>` → run the agent(s) whose scope matches that path over that path.

**Dogfood (`--consumer PATH`, or `--audit dogfood`).** First run the deterministic runner: `make dogfood VALIDATOR=<v> CONSUMER=<PATH>` (default consumer `~/Claude/prompt-injection-detection-submission`; default validator `audit_citation_alignment` — **the only validator with a runner adapter today**; others raise `NotImplementedError`). Then pass its JSON output to `etk-dogfood-noise-analyst` for classification. The runner does the run; the agent does the judgment.

## Pass each agent

Give every launched agent a self-contained prompt (they start fresh — no conversation history): the resolved scope, the exact files/diff to review, and a pointer to its authoritative sources (`STYLE.md`, the relevant ADRs). Require **read-back discipline**: every finding must quote code with `path:line`, and only high-confidence findings are reported with a `suppressed N low-confidence` footer.

## `--refute` (opt-in adversarial pass)

After collecting findings, re-spawn each agent on **its own findings only** (no cross-agent context) with the instruction: "Try to refute each finding below. Quote the actual code; if the finding does not hold up under a close read — code misread, contradicted by a higher-authority source (cite it), or already caught by a deterministic gate — mark it refuted. Default to refuted when uncertain." Keep only survivors. Report how many findings were refuted.

## Synthesis (always)

Produce one report:
1. **Deduplicate first (the orchestrator owns dedup — agents always report what they see).** When several agents run, merge findings keyed on `(path, line, issue-class)`. For cross-cutting classes, keep the canonical owner's version and drop the duplicate: **encoding/IO → `etk-silent-failure-auditor`**. Report `deduplicated M → N` so nothing is silently hidden.
2. **Overall verdict:** `PASS` / `CONCERNS` / `BLOCK` (BLOCK if any agent returned BLOCK).
3. **Per-agent verdict line** with each agent's `PASS/CONCERNS/BLOCK`.
4. **Combined high-confidence findings table** (post-dedup) grouped by agent: `path:line · severity · confidence · finding · fix`.
5. **Per-agent `suppressed N low-confidence` footers.**
6. If `--refute` was used: a "refuted M / N findings" line.

## Output destination

- Diff mode → terminal only.
- `--audit` → also **write a structured ledger entry** to `.claude/reviews/review-<ISO8601>.md` (e.g. `review-2026-05-28T1430.md`), or the path given by `--out`. This is a *machine-local* artifact (`.claude/reviews/` is gitignored) — **never** reuse the `gate3-audit-round-*.md` prefix, which is the human release-audit ritual and would be clobbered. Confirm the target does not exist before writing (bump the timestamp if it does). Use Write only for this artifact.
- `--ledger` → force the same `.claude/reviews/` write even in diff mode.

Never `git add`/commit; never regenerate the public-API snapshot; never run ruff/black/mypy/pytest as a gate (only `git`/`gh`/`make dogfood` for scope resolution).
