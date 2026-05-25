# Repo Strategy

How `eval-toolkit` is organized today, the target shape, and the
rules for when to extract a sub-package into its own repo.

> **Status note (v0.51, 2026-05-24, R8-G1 audit fix):** The §4
> "target shape" 6-sub-package plan below was **deliberately
> superseded** by
> [ADR 0001 (flat-module layout)](adr/0001-flat-module-layout.md),
> which commits to the current flat layout through v1.x and defers any
> subpackage restructuring to v2.0 trigger criteria. Sections §4–§5
> below are preserved here for context on the v2.0-and-beyond
> conversation, not as a planned v1.x execution. Treat this doc as
> historical for any pre-v2.0 work; consult ADR 0001 first for
> the v1.0+ module-layout contract.

> **If you're an AI session entering this repo to propose
> reorganization, extraction, or a new module: read this entire doc
> before proposing changes.**
>
> **The current target shape is locked-in for v0.10.x:** in-place
> mono-repo with the 6 sub-package layout in [§4](#target-shape).
> Don't propose alternative groupings. Don't propose multi-repo
> splits unless an extraction passes the [§5 checklist](#extraction-checklist).
>
> **Before any extraction proposal, run the audit in [§6](#audit-cadence).**
> If the audit fails the 4-question checklist (≥ 2 yes), don't
> propose extraction; instead propose closing the gap (e.g., add a
> deprecation, ask for a downstream signal).
>
> **Next scheduled audit: v0.13.0** (every 3 minor releases; current
> is v0.10.0).

What's in this doc:

- §1 [Context](#context) — why this exists.
- §2 [Audit findings](#audit-findings) — the dependency-graph state
  as of v0.10.0.
- §3 [LOC + concern map](#loc-concern-map) — vocabulary the rest of
  the doc uses.
- §4 [Recommended target shape](#target-shape) — the 6-bucket layout.
- §5 [Extraction checklist](#extraction-checklist) — the 4 questions
  that decide whether to spin off a sub-package.
- §6 [Audit cadence](#audit-cadence) — how to re-run this audit.
- §7 [Anti-goals](#anti-goals) — paths explicitly ruled out.
- §8 [Open questions](#open-questions) — deferred to a future round.
- §9 [Cross-links](#cross-links) — related docs.

If you only read one section, read [§5](#extraction-checklist).

---

(context)=
## 1. Context
Two motivations drove this strategy doc:

**Architectural debt.** The README's three-tier architecture
(functional core / Protocol-based orchestration / reproducibility
scaffolding) is partly aspirational. Three Tier-2 helpers
(`leakage`, `splits`, `loaders`) import *upward* into `harness`,
which means you can't use a Splitter without taking the orchestrator
along. Two modules (`thresholds`, `operating_points`) straddle the
Tier 1 / Tier 2 boundary — they're labeled as Protocols but their
impls reach into the math kernels, so they behave like Tier 1
utilities. `metrics` has in-degree 4 (analysis,
operating_points, thresholds, harness all import it), which makes
"extract metrics" a refactor that ripples into half the package.

**Conceptual sprawl.** 24 modules, 12.3K LOC, six distinct concerns
under one name: math kernels, eval orchestration, evidence/claims,
reproducibility scaffolding, plotting, standalone utilities (text
dedup, markdown rendering, paths, config). The "eval-toolkit" name
fits the math + orchestration; it strains when applied to a 1300-LOC
text-deduplication library or a markdown renderer.

The constraint: stay mono-repo by default. Multi-repo splits incur
real coordination overhead (synchronized releases, version pin
management, cross-repo PRs). Split only when a candidate proves
independent users / release cadence / install-slimness payoff (the
4-question checklist in §5).

This is a **roadmap document**, not an implementation plan. No code
moves in v0.10.x.

(audit-findings)=
## 2. Audit findings
Snapshot from a dependency-graph audit run at v0.10.0
(`f6ef004`).

### Coupling map

**True leaves** (zero in-package imports — trivially extractable
*if* their consumers go too): `_version`, `protocols`, `seeds`,
`config`, `paths`, `evidence`, `artifacts`, `claims`, `docs`,
`plotting`, `provenance`, `text_dedup`, `calibration`, `bootstrap`,
`metrics`. That's 15 of 24 modules.

**Hub.** `src/eval_toolkit/harness.py` is the central orchestrator —
in-degree 3 (`splits`, `loaders`, `leakage` import it), out-degree 5
(it imports from `artifacts`, `bootstrap`, `metrics`,
`operating_points`, `protocols`, plus TYPE_CHECKING imports of
`leakage` and `splits`). This is the package's identity; extracting
it would leave a confusingly-named shell.

**High-leverage kernel.** `src/eval_toolkit/metrics.py` has
in-degree 4 (`analysis`, `operating_points`, `thresholds`,
`harness`). Anything that touches metrics ripples broadly. This
makes metrics foundational — and unextractable without taking its
consumers along.

### Tier 1/2 blurring

The README claims `thresholds` and `operating_points` are Tier 2
(Protocol-based orchestration). In practice:

- `src/eval_toolkit/thresholds.py` defines the `ThresholdSelector`
  Protocol and 6 reference impls. The impls import `metrics` and
  `calibration` (both Tier 1). A selector *uses* a metric — that's
  fine — but it means `thresholds` is half-Tier-1 in coupling
  terms.
- `src/eval_toolkit/operating_points.py` (178 LOC) imports `metrics`
  + `thresholds`. It's labeled Tier 2 but acts like a Tier 1
  utility for fit-then-apply threshold workflows.

This blurring is annotated in the doc, but the *fix* (relabeling +
moving `operating_points` under the math kernels) is deferred to the
v0.11.0 reorg in §4.

### Upward-import anti-pattern

Three Tier-2 helpers reach into the orchestrator for type shapes:

- `src/eval_toolkit/leakage.py` imports `harness` (for `EvalSlice`).
- `src/eval_toolkit/splits.py` imports `harness` (same).
- `src/eval_toolkit/loaders.py` imports `harness` + `provenance`.

These helpers should depend on a *type contract*, not the
orchestrator. The fix in v0.11.0 is to extract a small
`eval_toolkit/_types.py` module with the `EvalSlice` and `RunResult`
dataclasses, and have helpers + harness both depend on `_types`.
Pure internal refactor; no public API change. Eliminates the cycle
that blocks "I want a Splitter without the harness" use cases.

(loc-concern-map)=
## 3. LOC + concern map
| Concern bucket | Modules | LOC |
|---|---|---:|
| **Math kernels** (Tier 1) | metrics, bootstrap, calibration, thresholds, operating_points | 5389 |
| **Eval orchestration** (Tier 2) | harness, splits, loaders, leakage, analysis | 2698 |
| **Evidence layer** (v0.9) | claims, artifacts, evidence | 948 |
| **Reproducibility** (Tier 3) | manifest, provenance, seeds | 797 |
| **Visualization** | plotting | 956 |
| **Standalone utilities** | text_dedup, docs, paths, config, protocols, _version | 1973 |

Three modules — `metrics` (1550), `bootstrap` (1149), `text_dedup`
(1327) — are 32% of the LOC.

(target-shape)=
## 4. Recommended target shape
### Phase 1 — in-place mono-repo reorganization (target: v0.11.0)

Restructure `src/eval_toolkit/` into 6 sub-packages, one per bucket:

```
src/eval_toolkit/
├── __init__.py            # lazy re-exports preserved (back-compat)
├── _version.py
├── _types.py              # NEW: EvalSlice, RunResult (extracted from harness)
├── core/                  # math kernels — Tier 1
│   ├── metrics.py
│   ├── bootstrap.py
│   ├── calibration.py
│   ├── thresholds.py
│   └── operating_points.py
├── harness/               # eval orchestration — Tier 2
│   ├── __init__.py        # re-exports evaluate, evaluate_folded, EvalSlice
│   ├── orchestrator.py    # was harness.py
│   ├── splits.py
│   ├── loaders.py
│   ├── leakage.py
│   └── analysis.py
├── evidence/              # claims / artifacts (v0.9)
│   ├── claims.py
│   ├── artifacts.py
│   └── evidence.py
├── scaffolding/           # reproducibility — Tier 3
│   ├── manifest.py
│   ├── provenance.py
│   └── seeds.py
├── viz/                   # plotting (matplotlib-only)
│   └── plotting.py
└── utils/                 # standalone tools
    ├── text_dedup.py
    ├── docs.py            # markdown rendering
    ├── paths.py
    ├── config.py
    └── protocols.py       # the trait crate
```

Top-level `eval_toolkit/__init__.py` keeps every existing lazy
export working — consumers see no breaking change. Internally:

- Helpers in `harness/` depend on `_types` instead of importing
  `harness/orchestrator`. Cycle eliminated.
- A CI lint test (`tests/test_import_directions.py`) parses
  `src/eval_toolkit/**/*.py` ASTs and fails the build if a
  sub-package imports from a "higher" sub-package per a declared
  import order: `_types < core < (harness, evidence, scaffolding,
  viz, utils)`.
- The `protocols` module moves to `utils/` — it's a trait crate
  with zero runtime cost; placing it under utils makes the
  conceptual map cleaner (it's a *contract* used by orchestration
  but not part of it).

### Phase 2 — earn-then-extract (long-term, no committed schedule)

Three candidates, ranked by independence + value:

1. **`eval-toolkit-text-dedup`** — separate PyPI package + repo.
   1327 LOC, sklearn-heavy, zero in-package imports, conceptually
   orthogonal (it's a similarity/dedup library, not an eval
   kernel). The current `eval_toolkit.utils.text_dedup` becomes a
   thin re-export from the new package + a `DeprecationWarning`.
   **Trigger:** a downstream consumer asks for it standalone, or
   the sklearn-version pin starts costing the main package.

2. **`eval-toolkit-plotting`** — separate PyPI package + repo. 956
   LOC, matplotlib-only (already gated behind `[plotting]` extra),
   zero in-package imports, conceptually orthogonal. The split
   lets matplotlib churn (3.x bumps, deprecations) without
   touching the math core. **Trigger:** a matplotlib release
   breaks something and we want to ship a viz-only fix without
   bumping the main package.

3. **`eval-toolkit-evidence`** — separate package, only if v0.10+
   evidence-layer adoption proves it has independent consumers.
   948 LOC, recently added in v0.9, churning. The gate framework
   (`claims.py`) is generic enough to be useful outside
   eval-toolkit. But: deeply integrated with the `RunResult`
   shape, so extraction means freezing that contract first.
   **Trigger:** someone wants to use the gate framework against a
   non-eval-toolkit result payload.

### What we're NOT extracting

- `metrics`, `bootstrap`, `calibration` — too foundational;
  in-degree 4 on `metrics` means everything else falls in line
  behind it. These stay the math core of `eval-toolkit`.
- `harness` + the orchestration ring — this *is* the package's
  identity. Extracting it would leave a confusingly-named shell.
- `manifest` + `provenance` + `seeds` — small, tight Tier-3 pair,
  no consumer demand to use them outside an eval context.

(extraction-checklist)=
## 5. Extraction checklist
For any module `X` proposed for extraction into a separate PyPI
package, run all four questions. **Extract only if ≥ 2 of 4 are
yes.** Default = stay together.

Each question has a concrete command so the audit is reproducible
by a human or AI session.

### Q1 — Has X been touched in the last 2 minor releases for reasons unrelated to the rest of the package?

```bash
# Replace v0.<n-2> with the version 2 minors back from current
git log --oneline v0.<n-2>..HEAD -- src/eval_toolkit/<X>.py
```

If all listed commits' subject lines are X-only (no
co-modifications of `harness`, `metrics`, etc.), mark **yes**.

### Q2 — Has at least one downstream consumer asked to use X without pulling the rest of eval-toolkit?

```bash
# Search known consumer repos
gh search code "eval_toolkit.<X>" --owner <user>
# Or grep manually if consumer repos are private
```

Known consumer repos: `prompt-injection-detector`,
`prompt-injection-showcase`, `prompt-injection-sdd`. If any has
filed an issue, asked in chat, or imported just `<X>` without the
harness, mark **yes**.

### Q3 — Does X have heavy / churning external deps whose churn would otherwise force main-package bumps?

```bash
# Check release cadence of X's primary external dep
pip index versions <X-key-dep>
```

If the dep released ≥ 2 minor versions in the last 6 months *and*
we've seen breakage in the main package because of it, mark **yes**.

### Q4 — Is X's public API stable enough to commit to a separate release cadence?

```bash
git log --oneline -- src/eval_toolkit/<X>.py | head -20
grep -A 50 "^__all__" src/eval_toolkit/<X>.py
```

If `__all__` hasn't changed in the last 2 minor releases, mark
**yes**. (A churning API can't be split off because every change
forces a new release of the spun-off package.)

### Verdict

- **0–1 yes** — do not extract. Keep X in the main package.
- **2 yes** — extraction is justified. Open a planning round with
  this doc as the entry point.
- **3–4 yes** — extraction is overdue. Prioritize.

Document the verdict in this doc as an appendix, even if the
verdict is "do not extract." Silence is a valid audit result; an
explicit "ran the audit, nothing changed" entry prevents future
sessions from re-deriving the same conclusion.

(audit-cadence)=
## 6. Audit cadence
**Cadence:** every 3 minor releases. Next audit due at **v0.13.0**
(current is v0.10.0).

### How to run the full audit

These commands a human or AI session can copy-paste:

```bash
# 1. Refresh the dependency graph
grep -rnE "^from eval_toolkit\.|^import eval_toolkit\." src/eval_toolkit/ \
  | grep -v "TYPE_CHECKING" \
  | sort | uniq

# 2. Per-module LOC (re-confirms the LOC + concern map in §3)
wc -l src/eval_toolkit/**/*.py

# 3. Per-module __all__ size
for f in src/eval_toolkit/**/*.py; do
  printf "%-40s " "$f"
  grep -A 50 "^__all__" "$f" | grep -c '"[A-Za-z_]'
done

# 4. Coverage per module (re-confirms the v0.10.0 ≥90% per-module floor)
uv run pytest -q --cov=eval_toolkit --cov-report=term --no-header
```

Then for each candidate module, run the §5 checklist commands.

### Audit owner

Whoever opens a PR on or after the v0.13.0 milestone. The output
goes into:

- A new appendix to this doc (`§Audit log` — added in the same PR
  that triggers the audit), OR
- A comment on a tracking issue, with a link from this doc.

Either way, the audit produces a written artifact. **If no
extraction passes the §5 checklist, write "audit ran at v0.13.0,
nothing changed" and move on.** Silence is a valid result.

(anti-goals)=
## 7. Anti-goals
Things this strategy explicitly rules out:

- **Don't split into 6 separate repos at once.** Multi-repo
  coordination overhead (synchronized releases, version pin
  management, cross-repo PRs) is real; the value has to be earned
  per-extraction via the §5 checklist.
- **Don't propose alternative bucket groupings.** The 6-bucket cut
  in §4 is the agreed shape. Re-litigating it costs more than it
  buys. If a new module genuinely doesn't fit any bucket, that's a
  signal to write up a §8 open question, not to redraw the map.
- **Don't extract `metrics`, `bootstrap`, or `calibration` as
  separate packages.** Foundational math kernels with multiple
  in-package consumers; circular installs (`eval-toolkit-core`
  depending on `eval-toolkit-metrics` etc.) are worse than
  co-location.
- **Don't add a plugin / entry-point system.** The Protocol layer
  in `eval_toolkit/utils/protocols.py` already gives consumers
  Tier 2 extension; entry-points are a lot of machinery for the
  same outcome.
- **Don't rename the package.** `eval-toolkit` stays as the PyPI
  name. Sub-packages get sensible names within
  (`eval_toolkit.viz`, `eval_toolkit.core`, etc.).
- **Don't pre-emptively add deprecation warnings to today's API.**
  No `DeprecationWarning` lands until a Phase 2 extraction
  actually triggers and the back-compat shim is the only thing
  left in the main package.

### Parallelism (v0.34.0+)

The toolkit was historically single-threaded — not by policy, just because
no primitive needed parallelism. v0.34.0 codifies the explicit pattern:
**opt-in per-function `n_jobs` parameter** backed by the internal
[`_parallel.parallel_map`](methodology/parallelism.md) helper (joblib
loky backend; reproducibility-by-default via `np.random.SeedSequence`).

What this means for contributors:

- **Adding `n_jobs` to a new function is allowed and encouraged** when the
  function has a Python-level loop over independent work units with
  medium+ per-item cost. Follow the checklist in
  [methodology/parallelism.md §"When to add `n_jobs`"](methodology/parallelism.md).
- **Use the helper, don't roll your own.** All parallelism flows through
  `_parallel.parallel_map`; no inline `joblib.Parallel`, no
  `concurrent.futures`, no raw `multiprocessing`, no `asyncio`.
- **Default sequential.** `n_jobs: int = 1` keeps existing call sites
  unchanged and preserves reproducibility/traceback fidelity by default.

(open-questions)=
## 8. Open questions
Deferred to the v0.13.0 audit (or earlier if a trigger fires):

- Should `analysis.py` (CSV/JSONL prediction readers) move to
  `evidence/`? It's used by the evidence-core paired-diff workflow
  more than by the eval orchestrator. Currently lives in
  `harness/` per §4 because it imports `bootstrap` + `metrics`,
  but the *consumer* shape suggests it belongs with `claims` +
  `artifacts`.
- Should `protocols.py` become its own published package
  (`eval-toolkit-protocols`) so consumer libraries can declare a
  dependency on the contract without depending on any
  implementation? Useful only if there's ever a second harness
  implementation. Today there isn't.
- v1.0 timing — when does the API freeze trigger? The §5
  checklist's Q4 (`__all__` stability) is the relevant signal.

(cross-links)=
## 9. Cross-links
- [`roadmap.md`](roadmap.md) — broader feature roadmap. This doc
  is the *organizational* roadmap; that's the *feature* roadmap.
- [`methodology/versioning.md`](methodology/versioning.md) — the
  schema-evolution policy. Any extracted package's API contract
  should follow the same additive-fields,
  `additionalProperties: true` discipline documented there.
- [`MIGRATION.md`](MIGRATION.md) — index of per-version migration
  guides. The v0.11.0 reorg gets its own entry there; future
  Phase-2 extractions get their own entries too.
- [`extending.md`](extending.md) — the Protocol-by-Protocol guide
  for custom Scorers, Splitters, gates. Stays the recommended
  extension point regardless of how the package is internally
  organized.
