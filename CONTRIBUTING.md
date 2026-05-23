# Contributing to eval-toolkit

Thanks for your interest in contributing. This doc is the operational guide —
how to set up a dev environment, run tests, submit changes, and ship a release.
The coding-standards reference lives separately in [`STYLE.md`](STYLE.md); read
that for naming conventions, type-hint rules, docstring format, and the
anti-overengineering principles that shape PR review.

For the formal naming-convention decision record (including industry
citations and the asymmetric-promotion principle), see
[ADR 0004 — Naming conventions](docs/source/adr/0004-naming-conventions.md).
STYLE.md is the day-to-day quick reference; ADR 0004 is the authoritative
source if the two ever diverge.

## Dev setup

Requirements: **Python ≥ 3.13** (RunPod-parity floor; see `pyproject.toml`'s
`requires-python` field) and [uv](https://github.com/astral-sh/uv) for env +
package management.

```bash
git clone https://github.com/brandon-behring/eval-toolkit
cd eval-toolkit
make install    # uv venv + uv pip install -e ".[dev]"
source .venv/bin/activate
```

The `[dev]` extra pulls in all sub-extras (`dataframe`, `plotting`, `property`,
`yaml`, `parquet`) plus the test/lint/type toolchain (`pytest`, `pytest-cov`,
`pytest-mpl`, `sybil`, `ruff`, `black`, `mypy`, `pre-commit`).

## Hooks

Pre-commit hooks gate ruff + black at commit time and mypy at push time:

```bash
make hooks   # `pre-commit install` + `pre-commit install --hook-type pre-push`
```

If you have a global `core.hooksPath` set (common with personal hook chains
like gitleaks), `make hooks` will detect the conflict and print remediation
instead of silently failing. Alternatives:

```bash
uv run pre-commit run --all-files            # one-off manual run; ignores core.hooksPath
uv run pre-commit run --hook-stage pre-push  # also runs mypy
```

## Test loop

```bash
make test-fast    # fast iteration: pytest -m "not slow" -q (~10s)
make test         # full suite incl. doctests + slow markers (~70s)
make type         # mypy strict on src/
make ci           # everything CI runs (lint + type + test + coverage gate)
```

The coverage gate enforces **92% line coverage** (`--cov-fail-under=92`).
`make ci` mirrors the GitHub Actions matrix locally; if it's green, CI will
almost certainly be too.

For the doctest sub-suite specifically (math kernels + utility modules):

```bash
make test-doctest    # reads .doctest-modules for the canonical module list
```

## Coding standards

See [`STYLE.md`](STYLE.md). The short version:

- Black (100-char line length); ruff with `E/F/W/I/N/UP/B/SIM/C4` enabled
- mypy strict on `src/` — every public function has type hints
- Docstrings: NumPy style; `Raises:` sections required where documented
  exceptions exist; doctests welcome on math kernels
- pytest markers: `unit`, `property`, `smoke`, `golden`, `slow`,
  `monte_carlo` (opt-out for fast loops via `pytest -m "not slow and not monte_carlo"`)
- Anti-overengineering (STYLE.md §5): don't add abstractions without a second
  concrete use site

## Logging conventions

eval-toolkit follows the [PEP-recommended library-logging pattern](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library):

- `src/eval_toolkit/__init__.py` attaches a `NullHandler` to the
  `eval_toolkit` root logger. Library is **silent by default**; the
  consuming application configures handlers.
- Each module instantiates its own logger:
  `_logger = logging.getLogger(__name__)`. This produces names like
  `eval_toolkit.harness`, `eval_toolkit.bootstrap`, etc. — matching
  the import path so consumers can filter granularly:
  `logging.getLogger("eval_toolkit.harness").setLevel(logging.DEBUG)`.

**Log-level conventions**:

| Level | When to use |
|---|---|
| `DEBUG` | Internal events: slice transitions, bootstrap-resample iteration milestones, leakage-check completions, loader split construction. Anything a contributor would set the level for when debugging. |
| `INFO` | User-relevant progress signal — sparingly. The harness's per-slice "n=200, positives=100" line is INFO because it's the user-facing summary of a long-running operation. Library code should rarely use INFO; consumers add their own. |
| `WARNING` | **Reserved for `warnings.warn(...)`, NOT `logger.warning(...)`**. The library uses `DeprecationWarning` / `UserWarning` via the `warnings` module, not the logging module. Double-emitting via both pollutes the warning channel. |
| `ERROR` | **Should not exist in library code** — raise an exception instead. The caller decides whether the exception is logged or handled. |

For consumers who want all logging on stderr at DEBUG level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# eval_toolkit logs now flow through the basicConfig StreamHandler
```

For consumers who want only `harness` logging at DEBUG:

```python
logging.basicConfig(level=logging.WARNING)  # silence everything else
logging.getLogger("eval_toolkit.harness").setLevel(logging.DEBUG)
```

## Parallelism

eval-toolkit codified its parallelism story in **v0.34.0**: a single
internal helper, opt-in per-function `n_jobs` parameters, joblib loky
backend, and a reproducibility-by-default contract. Full design rationale
lives in
[`docs/source/methodology/parallelism.md`](docs/source/methodology/parallelism.md).

**Short version** for contributors adding parallelism to a new function:

1. **Use the helper, not inline `joblib.Parallel`.** Every parallel-
   capable function in the toolkit calls into
   `eval_toolkit._parallel.parallel_map`. No `concurrent.futures`, no raw
   `multiprocessing`, no `threading`, no `asyncio` for CPU-bound work.
2. **Default `n_jobs: int = 1`** (sequential). Preserves traceback fidelity
   + reproducibility for existing callers.
3. **Reproducibility contract**: when the loop body uses random state,
   the function MUST use `np.random.SeedSequence(seed).spawn(n)` to
   derive per-item seeds. `n_jobs > 1` MUST give bit-for-bit-identical
   output to `n_jobs == 1` for the same caller-supplied seed. Add a
   `test_*_n_jobs_reproducibility` test asserting this.
4. **Picklability**: when `n_jobs != 1`, the loop body must be a module-
   level function (not a lambda or local closure). The helper does a
   `pickle.dumps(fn)` sniff-test up front and raises a helpful `TypeError`
   if violated.
5. **Smart defaults** are handled by the helper: `n_jobs=0` raises;
   `n_jobs > os.cpu_count()` caps with WARNING; `n_jobs=1` with
   `>= 1000` items emits a one-shot INFO guidance log per process (the
   toolkit's *only* INFO log — keep it that way unless you have an
   equally strong "user-relevant progress" case).

Currently parallel-capable: the 5 public bootstrap functions
(`bootstrap_ci`, `paired_bootstrap_diff`, `paired_bootstrap_ece_diff`,
`paired_bootstrap_op_point_diff`, `paired_mde`). See
[methodology/parallelism.md "When to add `n_jobs`"](docs/source/methodology/parallelism.md)
for the checklist if you're considering wiring a new site.

## Submitting changes

1. Branch from `main` (the repo uses direct-to-main for solo work and PRs
   for collaborative work)
2. Run `make ci` locally before pushing
3. Open a PR — the [PR template](.github/PULL_REQUEST_TEMPLATE.md) will
   pre-fill the body; fill in Summary / Testing / CHANGELOG / Risk
4. CI must be green:
   - `CI` workflow (lint + type + tests across ubuntu/macos/windows on Python 3.13)
5. Reference the issue with `Closes #N` if applicable

## Release flow

eval-toolkit follows [Semantic Versioning](https://semver.org/). Per SemVer
pre-1.0 expectations, breaking changes are allowed in MINOR bumps (`0.X.0`)
during the 0.x series; PATCH bumps (`0.X.Y`) remain backward-compatible.

> **Full runbook (incl. known gotchas + recovery recipes)**:
> [`docs/RELEASING.md`](docs/RELEASING.md). Read that for anything
> beyond the smooth happy path.

The release checklist:

1. Update `__version__` in `src/eval_toolkit/_version.py` (single source of
   truth — `pyproject.toml` reads it dynamically via `[tool.hatch.version]`,
   so editing it in two places is no longer possible)
2. Add a `## [X.Y.Z] — YYYY-MM-DD — <short description>` section to
   `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/)
   format. Use `### Added` / `### Changed` / `### Fixed` / `### Internal`
   sub-headings as relevant.
3. Commit: `git commit -m "release: vX.Y.Z — <short description>"` and push
   to `main`; wait for CI green
4. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z — <short description>"` and
   `git push origin vX.Y.Z`
5. The `Publish to PyPI` workflow (`.github/workflows/publish.yml`) runs
   automatically:
   - Prerelease tags (`v*rc*`, `v*a*`, `v*b*`, `v*dev*`) → **TestPyPI**
   - Stable tags (any other `v*`) → **PyPI**
6. Verify: `pip install eval-toolkit==X.Y.Z` in a clean venv prints the
   expected `__version__`
7. The `v4 sibling smoke` workflow also runs on `v*` tags as one extra
   contract check on the release commit

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no API tokens are stored in the repo. The `pypi` and `testpypi`
GitHub Environments scope which workflow file may publish. No manual approval
gate is configured on the `pypi` environment; the TestPyPI rehearsal step
(via a prerelease tag) is the safety net before a real release.

**Rehearsing a release on TestPyPI**: tag a prerelease first, e.g.
`git tag -a vX.Y.Zrc1 -m "rc1"` then `git push origin vX.Y.Zrc1`. The
workflow publishes `X.Y.Zrc1` to TestPyPI; install it in a clean venv with
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ eval-toolkit==X.Y.Zrc1`
(the `--extra-index-url` lets pip resolve numpy/scipy from real PyPI). Once
verified, tag the stable `vX.Y.Z` for the real publish.

Prerelease tags do **not** require editing `_version.py` — the workflow
validates that the tag's base release (`X.Y.Z`) matches `_version.py`'s
current version, then overrides `_version.py` with the tag-derived
prerelease version for the build only. `_version.py` on `main` always
carries the next planned *stable* release.

**Rollback**: PyPI does not allow re-uploading the same filename. If a
release is broken, yank it on PyPI (this hides it from new `pip install`
resolution while preserving pinned installs) and ship `0.X.Y+1` with the
fix. There is no fix-and-re-tag path; the TestPyPI rehearsal catches the
common configuration errors before they touch real PyPI.

## Filing issues

Use the YAML-form templates at
[github.com/brandon-behring/eval-toolkit/issues/new/choose](https://github.com/brandon-behring/eval-toolkit/issues/new/choose):

- **Bug report** — required affected-module dropdown, env block, reproduction,
  expected-vs-actual. Use this for unexpected behavior, crashes, or divergence
  from documented behavior.
- **Feature request** — required motivation, proposed API, alternatives
  considered. Use for concrete API proposals. Open-ended ideas are better
  suited to GitHub Discussions (if enabled on the repo).

Blank issues are disabled; the templates ensure enough context to act on the
report.

## Plan archive

Significant architectural decisions and multi-step refactors are planned in a
local plan file at `~/.claude/plans/what-can-be-done-parsed-wadler.md` (not
committed to the repo). Tier A through Tier E in that doc track the project's
quality-and-distribution cycle as of v0.27.0; future contributors won't see
the prior plans but the CHANGELOG entries are the durable record.
