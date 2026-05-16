# Deprecation policy

This document defines eval-toolkit's deprecation policy and how
contributors mark public-API symbols for removal.

## SemVer + pre-1.0 rules

eval-toolkit follows [Semantic Versioning](https://semver.org/). Per
SemVer pre-1.0 expectations:

- **Breaking changes** are allowed in **MINOR** bumps (`0.X.0`) during
  the 0.x series.
- **PATCH** bumps (`0.X.Y`) remain backward-compatible (security
  fixes + bug fixes that don't change documented behavior).

This means: removing a public symbol in `0.29.0` is *technically*
allowed without prior warning. But that's a bad citizen move for
users who have built on the library. **The deprecation policy below
extends what SemVer permits with what we actually commit to.**

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
   Removal is a breaking change appropriate for a MINOR bump per
   the SemVer pre-1.0 contract.

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

| Symbol / artifact | Announced | Removal | Reason |
|---|---|---|---|
| `[validation]` optional-dependency extra | v0.30.1 | v0.33.0 | No-op since v0.16.0 (jsonschema moved to base deps). Extras cannot emit `DeprecationWarning` at import time, so the deprecation is documentation-only. `pip install eval-toolkit[validation]` will continue to resolve cleanly through v0.32.x and will be removed in v0.33.0. |

## See also

- [`src/eval_toolkit/_deprecated.py`](https://github.com/brandon-behring/eval-toolkit/blob/main/src/eval_toolkit/_deprecated.py) — implementation
- [`tests/test_deprecations.py`](https://github.com/brandon-behring/eval-toolkit/blob/main/tests/test_deprecations.py) — deprecation tests
- [`docs/RELEASING.md`](RELEASING.md) — release runbook
- [`CONTRIBUTING.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CONTRIBUTING.md) — general contribution flow
