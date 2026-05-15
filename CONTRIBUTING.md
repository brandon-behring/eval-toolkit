# Contributing to eval-toolkit

Thanks for your interest in contributing. This doc is the operational guide —
how to set up a dev environment, run tests, submit changes, and ship a release.
The coding-standards reference lives separately in [`STYLE.md`](STYLE.md); read
that for naming conventions, type-hint rules, docstring format, and the
anti-overengineering principles that shape PR review.

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
- pytest markers: `unit`, `property`, `smoke`, `golden`, `slow` (opt-out
  for fast loops)
- Anti-overengineering (STYLE.md §5): don't add abstractions without a second
  concrete use site

## Submitting changes

1. Branch from `main` (the repo uses direct-to-main for solo work and PRs
   for collaborative work)
2. Run `make ci` locally before pushing
3. Open a PR — the [PR template](.github/PULL_REQUEST_TEMPLATE.md) will
   pre-fill the body; fill in Summary / Testing / CHANGELOG / Risk
4. CI must be green:
   - `CI` workflow (lint + type + tests across ubuntu/macos/windows on Python 3.13)
   - `v4 sibling smoke` (advisory during the trial; see "Downstream contract"
     below) — a red v4-smoke check needs diagnosis but doesn't block merge
     during the trial period
5. Reference the issue with `Closes #N` if applicable

## Downstream contract: v4 sibling-smoke

eval-toolkit has a live downstream consumer: `brandon-behring/prompt-injection-v4`.
The `.github/workflows/v4-smoke.yml` workflow checks out v4 at `main` and runs
its fast `-m smoke` suite against this repo's current PR head — catching contract
regressions at PR time rather than after-merge.

**HF_TOKEN secret:** v4's smoke fixtures load gated HuggingFace datasets. The
workflow needs an `HF_TOKEN` repo secret to be set at:
`https://github.com/brandon-behring/eval-toolkit/settings/secrets/actions`.

The workflow is **advisory** (`continue-on-error: true`) during a 2–3 week
trial to characterize the false-positive rate (independent v4 main breakage,
HF rate-limits). It will be promoted to a required gate after the trial. If
your PR sees a red v4-smoke check, comment with the diagnosis — workflow
hiccup, real contract break, or v4 main pre-existing breakage.

See `README.md`'s "Downstream contract testing (v4 sibling-smoke)" section
for the full design rationale.

## Release flow

eval-toolkit follows [Semantic Versioning](https://semver.org/). Per SemVer
pre-1.0 expectations, breaking changes are allowed in MINOR bumps (`0.X.0`)
during the 0.x series; PATCH bumps (`0.X.Y`) remain backward-compatible.

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
