# ADR 0001: Flat single-file modules through v1.x

**Status:** Accepted
**Date:** 2026-05-21 (drafted at v0.46-prep; finalized at v0.48)
**Deciders:** Brandon Behring (author), Round 5 audit (Codex + Gemini)
**Supersedes:** N/A. **Superseded by:** N/A.

## Context

eval-toolkit organizes its public API as **flat single-file modules** under
`src/eval_toolkit/`. Every concern — metrics, bootstrap, calibration,
harness, leakage, splits, thresholds, stacking, scorecard, etc. — lives in
exactly one `.py` file, often several hundred to a couple thousand lines.

The v1.0 plan ([[evaluate-all-the-work-twinkly-kite]]) inherits this layout
from v0.x. Two pressure tests against the flat-module convention surfaced
during planning + Round 5 audit:

1. **Module size.** A ground-truth check at the planning round measured
   9 modules exceeding 800 LOC (`metrics.py` 1819, `bootstrap.py` 1796,
   `calibration.py` 1477, `harness.py` 1449, `text_dedup.py` 1403,
   `plotting.py` 1351, `leakage.py` 1215, `loaders.py` 1081,
   `thresholds.py` 953). The earlier "800 LOC trigger" criterion was
   already violated by current code.

2. **Functional grouping.** As v0.46 introduced `scorecard.py` and
   `metric_specs.py`, and v0.47 will add a top-level `sweep.py` +
   `TextTransform` Protocol, the question naturally arises: should
   "metrics" become a subpackage with `metrics/scorecard.py`,
   `metrics/specs.py`, `metrics/calibration.py`, etc.? Same for
   `attacks/` (adversarial + preprocessing) or `bootstrap/`.

Either move (size-driven or grouping-driven) would be **breaking** at v1.0
because consumer imports follow the flat module structure
(`from eval_toolkit.metrics import pr_auc`, not
`from eval_toolkit.metrics.calibration_metrics import pr_auc`).

## Decision

**Stay flat through v1.x.** v1.0 commits to the existing flat-module
layout; subpackage restructuring is deferred until at least one of three
concrete triggers (below) fires, at which point v2.0 is the appropriate
boundary.

Specifically:

1. The current `src/eval_toolkit/*.py` file structure is the v1.0 contract
   for *internal-API* module paths. `eval_toolkit.metrics`,
   `eval_toolkit.bootstrap`, `eval_toolkit.calibration`, etc., remain valid
   submodule paths through the v1.x release line.

2. The **strict** v1.0 contract is still scoped to top-level `__all__` +
   Tier-2 Protocols (per [ADR 0003](0003-stability-contract-and-gate3-methodology.md)).
   Submodule public symbols carry an *additive-only* commitment — they may
   gain functionality in v1.x minors but the existing path remains
   importable.

3. Internal helpers (anything not in any module's `__all__`) are free to
   move in v1.x minors.

## Trigger criteria for a v2.0 subpackage restructure

The flat-module convention holds until ANY of the following becomes true
(none of which currently apply):

### A. Second production consumer with materially different surface needs

The single-consumer / breaking-OK rationale that made the v0.46 → v1.0
breaking sequence cheap stops holding once a second downstream consumer
adopts eval-toolkit with materially different access patterns. If the
second consumer would benefit from a subpackage layout (e.g., importing
from `eval_toolkit.attacks` rather than `eval_toolkit.adversarial` +
`eval_toolkit.preprocessing` separately), the v2.0 prep cycle is the
appropriate time to evaluate.

### B. Clear functional grouping that the codebase asks for

Examples that could surface during v1.x development:

- **`attacks/`** — adversarial + preprocessing (defense) modules collapse
  into a single attack-surface namespace as `sweep()` + `TextTransform`
  become the dominant interfaces.
- **`calibration/`** — the 4-binary-adapter family (`fit_temperature_binary`,
  `fit_isotonic_binary`, `fit_platt_binary`, `fit_beta_binary`), the
  underlying calibrator fitters, and the calibration-error metrics (ECE
  family) split across `metrics.py` + `calibration.py` could naturally
  group.
- **`bootstrap/`** — `bootstrap_ci`, `paired_bootstrap_*`,
  `block_bootstrap_on_folds`, `DeLongResult`, `BootstrapCI`,
  `PairedBootstrapCI`, MDE estimation, and CV-CLT could group around the
  inference primitive theme.

None of these have been demanded by the consumer or by audit feedback as
of v0.46. The trigger is "asks for it," not "could conceivably be
grouped."

### C. Discoverability complaint from real users

If multiple users file issues like "I can't find where X is" or "the
module name doesn't match what it does," subpackage navigation may help.
The trigger is **two or more such issues** from independent reporters —
not internal tooling concerns or single-author preference.

### Not a trigger: per-module line count

The original v1.0 plan draft cited a "module crosses ~800 LOC" trigger.
Verification at planning time found 9 modules already exceed that — the
trigger criterion was already violated, but nobody (consumer or audit)
flagged module size as a problem.

The pattern that emerged: eval-toolkit's large modules are **cohesive**.
`metrics.py` is large because it contains every binary-classification
metric primitive; that's the right home for them. `bootstrap.py` is large
because it contains every bootstrap-CI primitive; same story.
`text_dedup.py` is large because it contains the full
similarity-strategy + leakage-check decomposition for one logical
problem.

Splitting a cohesive module on size alone produces arbitrary boundaries
that don't help discoverability. v1.0 explicitly rejects size as a
splitting trigger.

## Consequences

**Positive:**

- One module path per concern. `from eval_toolkit.metrics import pr_auc`
  is the canonical scalar-metric import; `from eval_toolkit.bootstrap
  import bootstrap_ci` is the canonical inference import. Consumers don't
  need to learn a deeper module taxonomy.
- v1.0 → v1.x can grow without coordinating subpackage moves.
- Test files mirror the module structure 1:1
  (`tests/test_metrics.py`, etc.), which is easy to reason about.

**Negative:**

- `metrics.py` at 1819 lines is genuinely cumbersome to navigate in an
  IDE. The mitigation is `__all__` (already in place) + the
  `metric_specs` namespace (v0.46+) which lets consumers reach individual
  metrics through a smaller surface without the IDE-navigation cost.
- A future second consumer may want a different organizational
  vocabulary; v2.0 is the appropriate boundary.

## Alternatives considered

### Subpackage restructure at v1.0

Split the largest modules into subpackages (`metrics/`, `bootstrap/`,
`calibration/`, `harness/`, `text_dedup/`). Rejected because:

- v1.0 commits to API stability; a flat→subpackage shift is **exactly**
  the kind of breaking change v1.0 should NOT introduce (a v0.46-style
  soft-deprecation shim would need to re-export from old paths for
  multiple releases).
- The single consumer reports no discoverability problems with the flat
  layout.
- Each subpackage would require its own `__init__.py` re-export choices,
  multiplying the public-API surface to maintain.

### Defer the layout decision past v1.0 without ADR

Make no commitment in either direction; revisit ad-hoc. Rejected because
v1.0 IS the API stability commitment — the subpackage question must be
explicitly resolved one way or the other. This ADR is the resolution.

### Hybrid: subpackage some, keep others flat

E.g., split `metrics/` but keep `bootstrap.py` flat. Rejected because the
hybrid pattern is harder to teach and creates inconsistent expectations:
new users would have to look up every module to know whether it's a
single file or a directory.

## Trigger to revisit

This ADR is locked at v1.0. Revisiting requires **SemVer-major (v2.0)**.
Specific triggers per the criteria above (any ONE is sufficient):

1. Second production consumer with materially different surface needs.
2. Functional grouping the codebase asks for (sweep + TextTransform may
   create grouping pressure once v0.47 ships — re-evaluate at v2.0 prep).
3. ≥2 independent discoverability complaints from real users.

## References

- v1.0 plan: `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`
  (Decision 2 + audit revision via Round 5 module-size ground-truth).
- Round 5 audit ledger:
  [`audit_findings.md`](../audit_findings.md) — Round 5 module-size
  verification.
- [ADR 0002 — scorecard as primary metric surface](0002-scorecard-as-primary-metric-surface.md)
  — companion ADR documenting the v0.46 surface design within the
  flat-module convention.
- [ADR 0003 — stability contract + Gate 3 methodology](0003-stability-contract-and-gate3-methodology.md)
  — defines the tiered v1.0 stability commitment that this ADR's
  flat-module choice operates inside.
