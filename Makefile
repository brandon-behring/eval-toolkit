.PHONY: help install hooks lint format test test-fast test-unit test-property test-smoke test-doctest type ci coverage clean release-prep

PYTHON := .venv/bin/python
VENV := .venv

# Canonical doctest module list. Shared with noxfile.py, tox.ini, and
# .github/workflows/ci.yml — all four read .doctest-modules to stay in sync.
DOCTEST_MODULES := $(shell tr '\n' ' ' < .doctest-modules)

help:
	@echo "Targets:"
	@echo "  install       Create .venv via uv and install dev dependencies"
	@echo "  hooks         Install pre-commit hooks (ruff+black at commit, mypy at push)"
	@echo "  lint          ruff check + black --check + mypy"
	@echo "  format        black + ruff --fix"
	@echo "  test          pytest (all markers + doctests)"
	@echo "  test-fast     pytest -m 'not slow' (fast iteration loop)"
	@echo "  test-unit     pytest -m unit"
	@echo "  test-property pytest -m property"
	@echo "  test-smoke    pytest -m smoke"
	@echo "  test-doctest  pytest --doctest-modules src/eval_toolkit/{metrics,bootstrap,calibration,text_dedup,thresholds,leakage,manifest,paths,provenance}.py"
	@echo "  type          mypy strict on src/"
	@echo "  coverage      pytest with coverage report"
	@echo "  ci            lint + test + coverage gate"
	@echo "  clean         remove .venv, caches, build artifacts"
	@echo "  release-prep  bump _version.py + regen public-api snapshot (VERSION=X.Y.Z)"

install:
	uv venv
	uv pip install -e ".[dev]"
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
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m black src tests
	$(PYTHON) -m ruff check --fix src tests

test:
	$(PYTHON) -m pytest tests --doctest-modules $(DOCTEST_MODULES)

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
	$(PYTHON) -m mypy src

coverage:
	$(PYTHON) -m pytest --cov=eval_toolkit --cov-report=term-missing --cov-report=json --cov-fail-under=92 -m "not monte_carlo and not benchmark and not integration"
	$(PYTHON) scripts/check_module_floors.py

ci: lint test coverage

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
	@echo "  3. Commit:      git add src/eval_toolkit/_version.py tests/golden/public_api/snapshot.json CHANGELOG.md"
	@echo "                  git commit -m 'chore(release): v$(VERSION) — <theme>'"
	@echo "  4. Push:        git push origin main"
	@echo "  5. After CI green: git tag -a v$(VERSION) -m 'v$(VERSION) — <theme>' && git push origin v$(VERSION)"
