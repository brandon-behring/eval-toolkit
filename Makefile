.PHONY: help install hooks lint format test test-fast test-unit test-property test-smoke test-doctest type ci coverage clean release-prep pre-push dogfood

PYTHON := .venv/bin/python
VENV := .venv

# Canonical doctest module list. Shared with noxfile.py, tox.ini, and
# .github/workflows/ci.yml — all four read .doctest-modules to stay in sync.
DOCTEST_MODULES := $(shell tr '\n' ' ' < .doctest-modules)

help:
	@echo "Targets:"
	@echo "  install       Create .venv via uv sync (lock-faithful) with dev + docs extras"
	@echo "  hooks         Install pre-commit hooks (ruff+black at commit, mypy at push)"
	@echo "  lint          ruff check + black --check + mypy"
	@echo "  format        black + ruff --fix"
	@echo "  test          pytest: tests/ + README/docs Sybil fences + doctests"
	@echo "  test-fast     pytest -m 'not slow' (fast iteration loop)"
	@echo "  test-unit     pytest -m unit"
	@echo "  test-property pytest -m property"
	@echo "  test-smoke    pytest -m smoke"
	@echo "  test-doctest  pytest --doctest-modules over the curated .doctest-modules list"
	@echo "  type          mypy strict on src/"
	@echo "  coverage      pytest with coverage report"
	@echo "  ci            lint + test + coverage gate"
	@echo "  clean         remove .venv, caches, build artifacts"
	@echo "  release-prep  bump _version.py + regen public-api snapshot (VERSION=X.Y.Z)"
	@echo "  pre-push      mirror CI doc-execution gate: pytest (NO path arg) + sphinx-build + --doctest-modules"
	@echo "  dogfood       run an audit_* validator against its consumer, emit residual findings as JSON"

# Lock-faithful install (issue #102): `uv pip install -e` resolves fresh and
# drifts from uv.lock (observed: black 26.5.1 resolved vs 26.3.1 locked/CI).
# `uv sync` creates .venv, installs the project editable, and resolves from
# uv.lock — matching what CI runs.
install:
	uv sync --extra dev --extra docs
	@echo "Activate: source $(VENV)/bin/activate"

hooks:
	@if git config --get core.hooksPath >/dev/null 2>&1; then \
		echo "core.hooksPath is set to '$$(git config --get core.hooksPath)'."; \
		echo "pre-commit refuses to install over a custom hooks path."; \
		echo ""; \
		echo "Choose one:"; \
		echo "  1) Disable for this repo only:"; \
		echo "       git config --local --unset-all core.hooksPath"; \
		echo "       make hooks"; \
		echo "  2) Run hooks manually without installing:"; \
		echo "       uv run pre-commit run --all-files"; \
		echo "  3) Chain pre-commit into your global hooks dir manually."; \
		exit 1; \
	fi
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	@echo "Hooks installed: ruff+black at commit, mypy at push."

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts
	$(PYTHON) -m mypy src scripts

format:
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

# Collection roots are listed explicitly (= pyproject testpaths) because
# positional args override testpaths: a bare `tests` here silently dropped
# the README/docs Sybil fences (v0.47 §5L incident class; 2026-06-09 audit).
test:
	$(PYTHON) -m pytest tests README.md docs/source --doctest-modules $(DOCTEST_MODULES)

test-fast:
	$(PYTHON) -m pytest -m "not slow" -q

test-unit:
	$(PYTHON) -m pytest -m unit

test-property:
	$(PYTHON) -m pytest -m property

test-smoke:
	$(PYTHON) -m pytest -m smoke

test-doctest:
	$(PYTHON) -m pytest --doctest-modules $(DOCTEST_MODULES)

type:
	$(PYTHON) -m mypy src scripts

coverage:
	$(PYTHON) -m pytest --cov=eval_toolkit --cov-report=term-missing --cov-report=json --cov-fail-under=92 -m "not monte_carlo and not benchmark and not integration"
	$(PYTHON) scripts/check_module_floors.py

ci: lint test coverage

# dogfood — run an audit_* validator against its real consumer and emit the
# residual findings as JSON. The deterministic *run*; classification of the
# residuals is the etk-dogfood-noise-analyst agent's job (see .claude/agents/).
# Override the defaults: make dogfood VALIDATOR=audit_citation_alignment \
#     CONSUMER=~/Claude/prompt-injection-detection-submission SCOPE=narrative
VALIDATOR ?= audit_citation_alignment
CONSUMER ?= $(HOME)/Claude/prompt-injection-detection-submission
SCOPE ?= narrative
dogfood:
	$(PYTHON) scripts/dogfood_audit.py --validator $(VALIDATOR) --consumer $(CONSUMER) --scope $(SCOPE)

clean:
	rm -rf $(VENV) build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Cleaned"

# release-prep VERSION=X.Y.Z — canonical "step 1" of the release flow.
# Closes the public_api snapshot-drift gotcha that hit v0.28.0 / v0.28.1 /
# v0.29.0 / v0.30.0 releases (forgetting to regen the snapshot after a
# version bump). See docs/RELEASING.md for the full runbook.
#
# Accepts PEP 440 final + prerelease versions: 0.30.1, 0.31.0rc1,
# 0.32.0a2, 0.33.0b1, 0.34.0.dev3.
release-prep:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make release-prep VERSION=0.30.1"; \
		exit 1; \
	fi
	@echo "$(VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+|a[0-9]+|b[0-9]+|\.dev[0-9]+)?$$' || { \
		echo "Error: VERSION '$(VERSION)' does not match PEP 440 (X.Y.Z[rcN|aN|bN|.devN])"; \
		exit 2; \
	}
	@echo '"""Single lightweight version source."""' >  src/eval_toolkit/_version.py
	@echo ''                                          >> src/eval_toolkit/_version.py
	@echo '__all__ = ["__version__"]'                 >> src/eval_toolkit/_version.py
	@echo ''                                          >> src/eval_toolkit/_version.py
	@echo '__version__ = "$(VERSION)"'                >> src/eval_toolkit/_version.py
	@echo "[release-prep] wrote src/eval_toolkit/_version.py with __version__ = '$(VERSION)'"
	REGEN_PUBLIC_API_GOLDEN=1 $(PYTHON) -m pytest tests/test_public_api.py -q
	@echo ""
	@echo "[release-prep] DONE. Next steps:"
	@echo "  1. Edit CHANGELOG.md: convert [Unreleased] header to '## [$(VERSION)] — $$(date +%Y-%m-%d) — <theme>'"
	@echo "  2. Review diff: git diff src/eval_toolkit/_version.py tests/golden/public_api/snapshot.json CHANGELOG.md"
	@echo "  2b. Classify the snapshot diff against ADR 0003 tiers (Tier-1/2/3) and label"
	@echo "      the CHANGELOG entry to match — see the v1.5.0 erratum (#101)."
	@echo "  3. Commit:      git add src/eval_toolkit/_version.py tests/golden/public_api/snapshot.json CHANGELOG.md"
	@echo "                  git commit -m 'chore(release): v$(VERSION) — <theme>'"
	@echo "  4. Push:        git push origin main"
	@echo "  5. After CI green: git tag -a v$(VERSION) -m 'v$(VERSION) — <theme>' && git push origin v$(VERSION)"

# pre-push — local mirror of CI's full doc-execution + test surface.
#
# Sub-PR-7 (v0.47.0) incident postmortem: my pre-push command was
# `pytest tests/ --no-cov -q --ignore=tests/benchmarks`. Passing `tests/`
# as a positional argument SILENTLY OVERRIDES the pyproject testpaths
# config `["tests", "README.md", "docs/source"]`, dropping 159 sybil items
# from collection. Removing the v0.46 __getattr__ deprecation shim then
# activated 40 latent failures in those 159 items — and my pre-push gate
# missed all of them. CI caught them; fix was the doc-migration commit.
# See `feedback_sybil_python_blocks` + `feedback_degradation_layer_removal_hazard`.
#
# The fix is to run all three doc-execution surfaces with the correct
# collection scopes:
#
#   Surface 1 (Sybil .md fences + tests/): bare pytest (no positional)
#     so testpaths applies — covers tests/ + README.md + docs/source/.
#   Surface 2 (MyST-NB example notebooks): sphinx-build runs the
#     {code-cell} blocks per nb_execution_mode="cache". A hard gate:
#     conf.py sets nb_execution_raise_on_error=True (§5H, landed v0.48),
#     so this exits non-zero on notebook execution errors.
#   Surface 3 (in-source >>> docstring examples): --doctest-modules
#     over the curated DOCTEST_MODULES list.
#
# Each surface has a different collection scope; ensuring all three
# are run is the v0.48 §5L lesson from the v0.47 release sequence.
pre-push:
	@echo "[pre-push] Surface 1: tests/ + Sybil .md fences (no positional path arg)"
	$(PYTHON) -m pytest --no-cov -q --ignore=tests/benchmarks
	@echo ""
	@echo "[pre-push] Surface 2: MyST-NB example notebooks via sphinx-build"
	@echo "           (hard gate: conf.py sets nb_execution_raise_on_error=True)"
	$(PYTHON) -m sphinx -b html -n docs/source/ docs/build/html/
	@echo ""
	@echo "[pre-push] Surface 3: in-source >>> docstring examples (curated DOCTEST_MODULES list)"
	$(PYTHON) -m pytest --doctest-modules $(DOCTEST_MODULES) --no-cov -q
	@echo ""
	@echo "[pre-push] ALL THREE SURFACES GREEN. Safe to push."
