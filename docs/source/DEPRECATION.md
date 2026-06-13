# Deprecation policy

This document defines eval-toolkit's deprecation policy and how
contributors mark public-API symbols for removal.

## SemVer + the v1.x tier contract

eval-toolkit follows [Semantic Versioning](https://semver.org/). Since
v1.0, what a version bump may change is defined by the
[ADR 0003](adr/0003-stability-contract-and-gate3-methodology.md)
stability tiers:

- **Tier 1 (STRICT)** — top-level `eval_toolkit.__all__` symbols and
  their signatures: removing or renaming requires **MAJOR** (v2.0).
- **Tier 2 (ADDITIVE-ONLY)** — submodule public symbols (e.g.,
  `eval_toolkit.eda.*`, `eval_toolkit.metrics.*`): may evolve in
  **MINOR** releases under the deprecation policy below.
- **Tier 3 (FREE)** — internals, error-message wording, docstring
  bodies: may change in any **PATCH**.

This means: a Tier-2 symbol could *technically* change in a MINOR
release without prior warning. But that's a bad citizen move for
users who have built on the library. **The deprecation policy below
extends what the tier contract permits with what we actually commit
to.**

(Historical note: through v0.x this section documented SemVer's
pre-1.0 rules — breaking changes allowed in MINOR bumps during the
0.x series. The 0.x examples below are retained for the record; the
mechanics are unchanged.)

## Promise: minimum 2 minor versions of warning before removal

For every public symbol (anything in `eval_toolkit.__all__` or
documented in `docs/api/`), we commit to:

1. **Announce in version N**: mark the symbol with the
   `@deprecated(deadline="X.Y.Z")` decorator. The deadline is the
   future version at which it will be removed. Every call emits a
   `DeprecationWarning`. The CHANGELOG `### Deprecated` section
   notes it.
2. **Maintain in versions N+1, N+2, …**: the symbol continues to
   work, continues to warn.
3. **Remove in version M** where `M >= N + 2 minor versions`.
   For Tier-2/Tier-3 surfaces, the completed warning window is what
   licenses removal in a MINOR bump (ADR 0003's Tier-2 contract is
   additive-only by default; its 2026-06-12 amendment routes
   non-additive changes through this document's process). Tier-1
   removals additionally wait for the next MAJOR.

Concrete example: a function deprecated in `0.29.0` with
`deadline="0.31.0"` works (with warnings) in `0.29.x`, `0.30.x`,
and is removed in `0.31.0`.

The two-minor-version window gives external consumers time to migrate
without surprise. **Exceptions** require a clear CHANGELOG note and
should be rare.

## How to deprecate a public symbol

1. Decide the removal deadline. Pick at least two minor versions
   ahead of the current release (e.g., if HEAD is `0.29-dev`, the
   earliest sane deadline is `0.31.0`).
2. Apply the decorator:

   ```python
   from eval_toolkit._deprecated import deprecated

   @deprecated(
       "0.31.0",
       reason="see eval_toolkit.metrics.replacement_metric — same signature, fixes the off-by-one in the legacy version",
       use_instead="eval_toolkit.metrics.replacement_metric",
   )
   def legacy_metric(y, score):
       ...
   ```

3. Add a `### Deprecated` subsection to the current `[Unreleased]`
   CHANGELOG entry:

   ```markdown
   ### Deprecated

   - `legacy_metric()` (will be removed in `0.31.0`). Use
     `replacement_metric()` instead. The new function has the same
     signature; the only behavior change is the off-by-one fix.
     See issue #N for context.
   ```

4. If the replacement exists, link both in docstrings. If it doesn't
   yet exist, file a tracking issue for adding it before the deadline.

## How removal works at deadline

When the release that contains the deadline ships (e.g., `0.31.0`):

1. **Delete the deprecated function** + its decorator + any associated
   tests that exercised it for deprecation-warning correctness.
2. Add a `### Removed` subsection to the release's CHANGELOG entry.
3. The drift-guard test (`tests/test_public_api.py`) will catch the
   removal — regenerate the snapshot in the same release commit.
4. The `tests/test_deprecations.py::test_no_expired_deprecation_deadlines`
   test asserts no `@deprecated` decorator references a deadline ≤
   the current `eval_toolkit.__version__`. If a removal is forgotten,
   this test fails loudly.

## What the @deprecated decorator does

- Validates the deadline string at decoration time (i.e., at import
  time). Typos like `"0..31.0"` fail loudly the moment the module
  loads.
- Wraps the function so every call emits
  `DeprecationWarning(message)` with a structured message including
  the deadline, reason, and recommended replacement.
- Preserves `__name__`, `__doc__`, `__wrapped__` via
  `functools.wraps`. Tools that introspect (Sphinx, mkdocstrings,
  IDEs) see the original function.
- Stashes metadata as `__deprecated_deadline__` /
  `__deprecated_reason__` / `__deprecated_use_instead__` attributes
  so tests can introspect without parsing the warning message
  string.

## What NOT to deprecate

- **Private symbols** (underscore-prefixed names, modules in
  `eval_toolkit._*`) — these are not subject to the deprecation
  policy. Internal refactors are free to rename / remove them
  without notice.
- **Internal classes that just happen to be importable**: if they're
  not in `__all__` and not documented in `docs/api/`, they're not
  public. Treat as private.

## Active deprecations

_None as of v1.12.0 — every prior deprecation has either been hard-removed
or formally re-classified as a no-op kept for backward compatibility
(see below)._

## No-op extras kept for backward compatibility

These extras still resolve via `pip install` but install nothing —
their underlying dependencies moved to base deps. They are retained
indefinitely so that downstream `pip` pins do not break.

| Extra | Originally introduced | No-op since | Notes |
|---|---|---|---|
| `[validation]` | pre-v0.16.0 | v0.16.0 | `jsonschema>=4.21` moved to base deps. The error message in `eval_toolkit/__main__.py` still recommends `pip install 'eval-toolkit[validation]'` for users who hit the missing-jsonschema path; the resolve-as-no-op behavior is deliberate. Originally announced as a deprecation in v0.30.1 with target removal v0.33.0; reclassified at v0.49.0 (R3) as permanent-no-op rather than removed because hard removal would break consumer pip pins of the form `eval-toolkit[validation]` for no functional benefit. |

## One-time exceptions to the 2-minor-version warning policy

The 2-minor-version warning is a **policy**, not a hard SemVer rule
(ADR 0003's Tier-2 contract is additive-only by default; its 2026-06-12
amendment routes non-additive Tier-2 changes through this document —
normally the warning window above). Rarely, an exception is justified when the cost of
the warning window exceeds its benefit — known consumer set is small + the
deprecation alias would carry forever-debt + every known consumer can be
notified directly via cross-repo issue.

Every exception below is documented with: announced version, justification,
and notification mechanism.

| Symbol | Renamed/removed | Version | Justification | Notification |
|---|---|---|---|---|
| `eval_toolkit.bootstrap.mde_from_ci(paired=...)` parameter rename | Renamed to `ci=...` and type widened to `BootstrapCI | PairedBootstrapCI` | v0.34.0 | Pre-1.0 SemVer; only 2 known consumers (`prompt-injection-detection-submission`, `post-transformers`), both use positional form per audit; deprecation alias would add forever-debt for a clean-API win. Cleaning the name now (before widespread adoption) beats living with the awkward `paired=` parameter name forever. | Cross-repo issues filed on both known consumers with explicit migration step: `mde_from_ci(paired=x)` → `mde_from_ci(ci=x)`. Positional `mde_from_ci(x)` unaffected. |
| `eval_toolkit.eda.*` parameter renames: `random_state=` (5 functions: `median_bandwidth`, `proxy_a_distance`, `maximum_mean_discrepancy`, `distribution_shift`, `competency_baselines`) and `n_bootstrap=` (3 functions) | `random_state=` → `rng=` (SPEC-7 typed: `RNGLike \| SeedLike \| None`, default `0` unchanged); `n_bootstrap=` → `n_resamples=` | v1.12.0 | Tier-2 evolvable surface (`eda/__init__.py` declares it; zero entries in the public-API snapshot — verified); closes the ADR 0004 §D4 canonical-vocabulary deviation flagged by the 2026-06-09 audit (#100); cross-repo grep found **zero** consumers of any eda symbol, so a 2-minor alias window would carry pure forever-debt with no beneficiary. | Zero consumers found by grep (sole production consumer `prompt-injection-detection-submission` imports no `eval_toolkit.eda` symbol); CHANGELOG migration snippet + GitHub Release notes callout stand in for consumer issues. |

**Future exception criteria** (must satisfy all):

1. **Small known consumer set** (≤ 3 repos) — verifiable via grep across
   sibling repos
2. **Cross-repo notification feasible** — issues filed on every consumer
   before / with the release
3. **API-debt cost > warning-window cost** — the alias would carry
   non-trivial future maintenance (e.g., long-lived `paired=` accepted
   forever, special-cased in docstrings, etc.)
4. **Documented here at announce time** — not retroactively

If any of these don't hold, follow the standard 2-minor-version
deprecation process above.

## See also

- [`src/eval_toolkit/_deprecated.py`](https://github.com/brandon-behring/eval-toolkit/blob/main/src/eval_toolkit/_deprecated.py) — implementation
- [`tests/test_deprecations.py`](https://github.com/brandon-behring/eval-toolkit/blob/main/tests/test_deprecations.py) — deprecation tests
- [`docs/RELEASING.md`](RELEASING.md) — release runbook
- [`CONTRIBUTING.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CONTRIBUTING.md) — general contribution flow
