# Releasing eval-toolkit

This is the operational runbook for cutting a new eval-toolkit release.
It documents the workflow, the **specific gotchas encountered in past
releases**, and the recovery steps for each known failure mode.

For the high-level release philosophy (SemVer, deprecation policy, etc.),
see [DEPRECATION.md](DEPRECATION.md). For ongoing contributor flow
(branch model, hooks, test loop), see [CONTRIBUTING.md](../CONTRIBUTING.md).

## TL;DR — the happy-path checklist

```
1. Edit src/eval_toolkit/_version.py → bump X.Y.Z
2. Regen tests/golden/public_api/snapshot.json:
   REGEN_PUBLIC_API_GOLDEN=1 uv run python -m pytest tests/test_public_api.py
3. Add [X.Y.Z] entry to CHANGELOG.md (convert from [Unreleased])
4. Commit: "release: vX.Y.Z — <short description>"
5. Push to main; wait for CI green (CI + CodeQL + Deploy docs)
6. Tag: git tag -a vX.Y.Z -m "vX.Y.Z — <short description>"
7. Push tag: git push origin vX.Y.Z
8. Watch publish.yml + docs.yml fire
9. Smoke-test: pip install eval-toolkit==X.Y.Z in a clean Py3.13 venv
10. Update memory: project_etk_on_pypi reflects the new version
```

## Detailed runbook

### Pre-release

#### 1. Version bump

Edit `src/eval_toolkit/_version.py` — the **single source of truth**.
`pyproject.toml` reads it dynamically via `[tool.hatch.version]`; do
NOT edit pyproject's version (it's `dynamic = ["version"]`).

#### 2. Public-API snapshot regen (load-bearing!)

```bash
REGEN_PUBLIC_API_GOLDEN=1 uv run python -m pytest tests/test_public_api.py
```

The public-API drift-guard test (`tests/test_public_api.py`) pins
`__version__` as one of the snapshot's value entries. **If you skip
this step, CI will fail on the release commit with**:

```
AssertionError: Public API entry drift (signatures/bases/docs/values):
    __version__.value: actual="'X.Y.Z'" expected="'A.B.C'"
```

**Recovery:** regen the snapshot, amend the release commit (or push a
follow-up `fix(release): regen public_api snapshot` commit).
This bit us in v0.28.0 → fixed in `58e120a`.

#### 3. CHANGELOG

Convert the `[Unreleased]` section to `## [X.Y.Z] — YYYY-MM-DD — <short>`.
Add a brief summary paragraph and the section list. Use today's UTC
date for the `YYYY-MM-DD`.

Keep `## [Unreleased]` as an empty placeholder above the new entry.

#### 4. Commit

Stage explicitly (never `git add .` — `.env.local` and personal scratch
files must stay unstaged):

```bash
git add src/eval_toolkit/_version.py CHANGELOG.md tests/golden/public_api/snapshot.json
git commit -m "release: vX.Y.Z — <short description>"
```

Push to main:

```bash
git push origin main
```

#### 5. Wait for CI green

Three workflows fire on a push to main:

- **CI** — full test matrix (3 OS × Py3.13) + lint + type + coverage gate + base-install + pip-audit
- **CodeQL** — static security analysis
- **Deploy docs** — builds + deploys mkdocs site to GitHub Pages

All three must be green before tagging. Verify via:

```bash
gh run list --branch main --limit 5
```

Or via the web UI: `https://github.com/brandon-behring/eval-toolkit/actions`.

### Release

#### 6. Tag

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <short description>

<longer release notes — paste relevant CHANGELOG section here>"
```

Use an **annotated tag** (`-a`), not a lightweight one. The publish
workflow keys off `refs/tags/v*`; annotated tags carry the release
notes that GitHub's Releases UI surfaces.

#### 7. Push tag

```bash
git push origin vX.Y.Z
```

This triggers `publish.yml` (→ PyPI via Trusted Publishing OIDC) and
the `Deploy docs` workflow re-fires with the new tag.

#### 8. Watch the publish

```bash
gh run watch --workflow=publish.yml
```

Expected: ~3 min wall time. The `publish-testpypi` job will be
SKIPPED (only fires on `*rcN` / `*aN` / `*bN` / `*devN` tags); the
`publish-pypi` job uploads sdist + wheel to real PyPI.

### Post-release

#### 9. Verify install

PyPI's simple-index has eventual-consistency caching — a fresh release
can take **30-60 seconds** to appear in the index even after
publish.yml completes successfully. **The first `pip install
eval-toolkit==X.Y.Z` may fail with "no version found"; retry in a
minute.** This bit us in v0.28.0 verification.

```bash
TS=$(date +%s)
uv venv --python 3.13 "/tmp/etk-verify-$TS"
source "/tmp/etk-verify-$TS/bin/activate"
uv pip install --no-cache --refresh "eval-toolkit==X.Y.Z"
python -c "
import eval_toolkit
print(f'__version__: {eval_toolkit.__version__}')
import importlib.metadata as M
assert eval_toolkit.__version__ == M.version('eval-toolkit') == 'X.Y.Z'
print('version consistency: OK')
"
```

#### 10. Update memory

Update the `project_etk_on_pypi` memory file to reflect the new
current PyPI version + note any new public API in the release.

## Known gotchas + recovery recipes

### "Public-API snapshot drift" on the release commit

**Symptom:** CI fails on the release commit with
`AssertionError: Public API entry drift (signatures/bases/docs/values): __version__.value: actual="'X.Y.Z'" expected="'A.B.C'"`

**Cause:** Forgot to regen the snapshot after bumping `_version.py`.

**Recovery:** Run `REGEN_PUBLIC_API_GOLDEN=1 uv run python -m pytest
tests/test_public_api.py`. Commit the regenerated
`tests/golden/public_api/snapshot.json` as
`fix(release): regen public_api snapshot for X.Y.Z`. Push.

This is the most common gotcha (hit it on every other release).
**Belt-and-suspenders fix**: incorporate the regen into a `make
release-prep VERSION=X.Y.Z` Makefile target. Listed as a future
chore in `docs/whats-new.md` roadmap.

### "Tag already exists" when retagging

**Symptom:** `git tag -a vX.Y.Z` fails with "tag 'vX.Y.Z' already
exists".

**Cause:** The version was used as an internal milestone tag before
PyPI publishing infrastructure existed (bit us at v0.27.0).

**Recovery:** Bump to the next patch (X.Y.Z+1). The pre-existing tag
stays as a historical reference; the new patch is the first PyPI
release on this minor line. Don't force-move the old tag — destroys
the audit trail.

### "PyPI install can't find new release"

**Symptom:** `pip install eval-toolkit==X.Y.Z` fails immediately after
publish.yml goes green, with "no version found".

**Cause:** PyPI's simple-index CDN propagation lag (~30-60 seconds).
The release exists at `https://pypi.org/pypi/eval-toolkit/X.Y.Z/json`
but the `https://pypi.org/simple/eval-toolkit/` index hasn't updated
yet.

**Recovery:** Wait 60 seconds; retry with `--no-cache --refresh`.

### "GitHub Pages deploy fails"

**Symptom:** `Deploy docs` workflow's deploy job fails on "Deploy to
GitHub Pages" step.

**Cause (first time):** GitHub Pages isn't enabled in repo settings.
Site source must be set to "GitHub Actions" via
`https://github.com/<org>/<repo>/settings/pages`. **This is a
one-time manual setup** that the docs.yml workflow assumes is done.

**Cause (subsequent):** Pages quota or transient GitHub-side issue.

**Recovery:** First-time → enable Pages → re-run the workflow.
Subsequent → re-run the failed job; if persistent, check
`https://www.githubstatus.com/`.

### "publish-testpypi failure pre-runner"

**Symptom:** publish-testpypi job fails instantly with `steps: []`
and no runner assigned (happened on v0.27.0rc2).

**Cause:** The `testpypi` GitHub Environment has a deployment-branch
restriction set as type **"Branch"** when it should be **"Tag"** (or
have no restriction at all). GitHub's UI defaults the type selector
to Branch, rejecting tag-triggered deployments pre-run.

**Recovery:** Edit the environment via
`https://github.com/<org>/<repo>/settings/environments`. Set
"Deployment branches and tags" to allow tag refs matching `v*` (or
remove the restriction entirely).

### "Wrong version pin: prerelease tag fails the guard"

**Symptom:** Tagging `vX.Y.Zrc1` (a prerelease) makes the publish
workflow fail at the "Verify tag matches package __version__" step.

**Cause:** Source `_version.py` carries the stable next version
(e.g., `X.Y.Z`); the tag's stripped form is `X.Y.Zrc1`. The workflow's
guard validates that the **base release** matches (X.Y.Z == X.Y.Z) and
sed-rewrites `_version.py` to the tag-derived version for the build
only. If you hit a strict-equality variant of this error, the
workflow patch is at `b7946d4`.

**Recovery:** Make sure `_version.py` carries the next stable release
(not a prerelease string). The workflow handles the prerelease
suffix at build time.

## Rollback policy

**PyPI does NOT allow re-uploading the same filename.** If a release
ships broken:

1. **Yank** the release on PyPI's web UI
   (`https://pypi.org/manage/project/eval-toolkit/releases/X.Y.Z/`).
   Yank hides the release from new `pip install` resolution but
   preserves pinned installs (`==X.Y.Z` still works for users who
   already pinned).
2. **Fix on main.**
3. **Ship X.Y.Z+1** with the fix. CHANGELOG should reference the
   yank and document what was wrong with the broken release.

**No fix-and-re-tag path exists.** TestPyPI rehearsals catch the
common config errors (workflow misconfig, schema validation) before
they touch real PyPI; the in-repo `test-base-install` job catches
import-path bugs.

## Two-stage release pattern

For bundles that include both **security-only** and **feature**
changes (like the v0.28.0 → v0.28.1 → v0.29.0 plan from May 2026),
ship the security patch first:

1. Land security commits to main → tag patch release (e.g., `v0.28.1`)
   → publish.
2. Continue with feature work on main → tag minor release (e.g.,
   `v0.29.0`) → publish.

Pattern: security-signal arrives in days, not weeks. Users on the
older minor line can `pip install -U eval-toolkit~=X.Y.0` and get
the security patch without picking up feature changes.

## Cross-references

- [CONTRIBUTING.md](../CONTRIBUTING.md) — ongoing contributor flow,
  hooks, test loop. References this doc for release specifics.
- [DEPRECATION.md](DEPRECATION.md) — when and how to deprecate
  public API (forthcoming).
- `.github/workflows/publish.yml` — the publish pipeline
- `.github/workflows/docs.yml` — the docs-deploy pipeline
- `.github/workflows/ci.yml` — the PR/push CI gate
- `tests/test_public_api.py` — the snapshot drift-guard test
