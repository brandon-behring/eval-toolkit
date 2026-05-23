# ADR 0002: `scorecard()` as the primary v1.0 metric surface

**Status:** Accepted
**Date:** 2026-05-21
**Deciders:** Brandon Behring (author), Round 5 audit (Codex + Gemini methodology cross-review)
**Supersedes:** N/A. **Superseded by:** N/A.

## Context

eval-toolkit v0.x exposed per-metric functions (`pr_auc`, `roc_auc`, `brier_score`,
the ECE family) as top-level imports. A typical eval pipeline imported each one
separately, computed a scalar, and called `bootstrap_ci` separately to get a
confidence interval. Issue [#36](https://github.com/brandon-behring/eval-toolkit/issues/36)
asked for an inline-CI surface on every metric (Inspect-AI / lm-eval scorecard
pattern). The single downstream consumer
(`prompt-injection-detection-submission`) filed two API sketches: a per-metric
`with_ci=True` kwarg vs. a top-level `scorecard()` factory.

Round 5 audit (Codex) surfaced a deeper concern about both sketches:
threshold-dependent metrics (F1, accuracy, precision, recall) cannot live in
a `(y_true, y_score) -> float` contract without baking in a default threshold
(against documented methodology) or hiding threshold provenance inside a spec.

The v1.0 release window is the cheapest moment to choose ONE primary surface;
post-v1.0 the contract is frozen (Decision M tiered stability — top-level
`__all__` strict, only additive subprotocols).

## Decision

At v0.46.0, ship a single primary metric surface:

```python
from eval_toolkit import scorecard, metric_specs as ms

r = scorecard(y_true, y_score,
              metrics=[ms.pr_auc, ms.brier, ms.ece(n_bins=15)],
              bootstrap=True, n_resamples=1000, confidence=0.95)
r["pr_auc"].value   # 0.873
r["pr_auc"].status  # 'ok' | 'skipped' | 'error'
r["pr_auc"].ci      # BootstrapCI(...)
```

Specifically:

1. **`scorecard()` is the public entry point.** Returns a `Scorecard`
   (`Mapping[str, MetricResult]`). Dict-subscript access by spec name
   (Decision I — type-safe under `mypy --strict`; typos raise `KeyError`
   instead of returning `Any`).

2. **`metric_specs` namespace ships threshold-free specs only** (Decision R).
   `pr_auc` / `roc_auc` / `brier` as singletons; `ece(n_bins, strategy)` as a
   LRU-cached factory. F1 / accuracy / precision / recall remain reachable
   via the existing `metrics_at_threshold` + `ThresholdSelector` machinery —
   they need a threshold, and threshold provenance is its own concern.

3. **`MetricResult` carries status-aware cells** (Decision S). Mirrors the
   existing `MetricState` vocabulary from `artifacts.py:30-61`: `ok` /
   `skipped` / `error`, plus a human-readable `reason` and optional
   `BootstrapCI`. Single-class slices produce `status="skipped"` rather than
   raising; per-cell exceptions become `status="error"` without aborting the
   whole scorecard.

4. **Skipped detection reuses the v0.39 primitive** (Decision X.2 +
   precondition): `is_metric_defined_for_slice(name, is_single_class=...)`.
   The primitive was extended at v0.46-prep to recognize both `pr_auc` /
   `roc_auc` (used by the new scorecard) and `auroc` / `auprc` (used by the
   v0.39 harness). No new public exception type; no new Protocol method.

5. **`MetricSpec` is a Tier-2 Protocol** per Decision M — frozen at v1.0
   modulo additive subprotocols. Method-signature changes (e.g., adding a
   threshold parameter to `compute()`) would require a SemVer-major bump.

6. **Top-level scalar metric imports are soft-deprecated at v0.46** and
   hard-removed at v0.47 (Decision L). The 8 deprecated names (`pr_auc`,
   `roc_auc`, `brier_score`, 5 ECE variants) remain reachable at v0.46 via a
   transitional `__getattr__` branch that emits `DeprecationWarning`; at
   v0.47 the branch is deleted (the names raise `AttributeError`).

7. **The `eval_toolkit.metrics` submodule path is the internal-API escape
   hatch.** `from eval_toolkit.metrics import pr_auc` works at v0.46, v0.47,
   v1.0+ without warning. **This is documented as internal API, subject to
   refactor in major versions** — not part of the v1.0 strict stability
   contract.

## Consequences

**Positive:**

- One canonical metric pipeline. Users learn `scorecard()` once; it covers
  every reasonable threshold-free metric + a uniform CI story.
- Status-aware cells eliminate a class of evaluation-pipeline bugs where
  degenerate single-class slices silently produced `1.0` PR-AUC values that
  entered downstream artifacts.
- Custom user specs (any object satisfying `MetricSpec` structurally) compose
  alongside first-party specs without registration. The Protocol contract is
  small (`name`, `compute()`) and stable.
- The submodule escape hatch (`from eval_toolkit.metrics import pr_auc`)
  keeps power-user / Monte-Carlo inner-loop patterns viable without paying
  the `scorecard()` orchestration cost.

**Negative:**

- Consumer pays a migration cost: every top-level scalar import either
  rewrites to `scorecard(...)` or moves to the submodule path. v0.46's
  DeprecationWarning shim lets this happen gradually across v0.46-v0.47;
  hard removal at v0.47 forces completion.
- Threshold-dependent metrics (F1, accuracy, precision, recall) are
  intentionally NOT in `metric_specs` at v0.46. Users wanting "give me F1
  with a bootstrap CI" must combine `MaxF1Selector` + `metrics_at_threshold`
  + a bootstrap loop themselves. Future work (deferred to v1.x) may add an
  operating-point spec family with explicit threshold provenance.
- `Scorecard.to_pandas()` requires the `[dataframe]` extra; the core
  `to_dict()` is stdlib-only.

## Alternatives considered

### `with_ci=True` kwarg on every per-metric function (consumer sketch #1)

Each metric function takes `with_ci: bool = False`; when `True`, returns
`(value, BootstrapCI)`. Rejected because:

- Doesn't address the threshold-dependent metrics gap (F1 etc. still need a
  threshold argument).
- Keeps the per-metric return-shape sprawl as v1.0 contract (sometimes
  `float`, sometimes `(float, BootstrapCI)`, sometimes `dict[str, float]` —
  the audit's #2 finding from the v0.43 design).
- Forces every scalar metric function to grow a `bootstrap` kwarg path —
  signature noise across ~10 functions vs. one `scorecard()` orchestrator.

### `scorecard()` + per-metric `with_ci=True` (Decision G — Round 1 Q1 option C)

Both surfaces public + maintained. Rejected because Daisy's "no legacy
design at v1.0" stance argued against two-surfaces-for-one-job.

### Operating-point spec family alongside threshold-free specs

A separate `op_metric_specs` namespace with explicit `threshold` /
`selector` / `fitted_op` provenance + explicit CI-policy contract
(fix-vs-refit per resample). Rejected for v0.46 as scope-creep; deferred to
v1.x if user demand surfaces. The methodology infrastructure is already in
place (`metrics_at_threshold`, `ThresholdSelector`, `OperatingPointSpec`,
`FittedOperatingPoint`) so adding the spec family is non-breaking when the
need is clear.

## Trigger to revisit

This decision is locked at v1.0 modulo Protocol-additive evolutions.
Reopening would require **SemVer-major (v2.0)**. Concrete triggers:

- **Second production consumer with materially different metric-surface
  needs.** The single-consumer / breaking-OK rationale that made this v1.0
  contract cheap stops holding. Re-litigate the contract during v2.0 prep.
- **Threshold-dependent metric demand from the consumer** surfaces as a
  release-blocker. Add an operating-point spec family (additive at v1.x);
  consider lifting `MetricSpec.compute()` to a richer signature at v2.0.
- **Performance pressure on the orchestration cost.** `scorecard()` adds a
  small per-call constant overhead vs. raw scalar functions. Acceptable for
  offline eval pipelines; a Monte-Carlo inner loop calling `scorecard()`
  millions of times would suffer. The submodule escape hatch documents the
  workaround; if the gap becomes user-visible, consider a fast-path.

## References

- v1.0 plan: `~/.claude/plans/evaluate-all-the-work-twinkly-kite.md`
  (Decisions A / B / C / I / J / L / M / R / S / X / Y / Z).
- Round 5 audit findings: `docs/source/audit_findings.md`
  (F1 / F2 / F4 drove this ADR).
- Issue [#36](https://github.com/brandon-behring/eval-toolkit/issues/36) —
  original consumer ask (inline bootstrap CI).
- `src/eval_toolkit/scorecards.py` — implementation (renamed from `_scorecard.py` at v0.49.0; see ADR 0004 + STYLE.md §3d for the asymmetric-promotion principle).
- `src/eval_toolkit/metric_specs.py` — first-party spec namespace.
- `src/eval_toolkit/artifacts.py:30-61` — `MetricState` vocabulary reused
  by `MetricResult`.
