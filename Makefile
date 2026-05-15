.PHONY: help install hooks lint format test test-fast test-unit test-property test-smoke test-doctest type ci coverage clean

PYTHON := .venv/bin/python
VENV := .venv

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
	$(PYTHON) -m pytest tests \
		--doctest-modules src/eval_toolkit/metrics.py src/eval_toolkit/bootstrap.py \
		src/eval_toolkit/calibration.py src/eval_toolkit/text_dedup.py \
		src/eval_toolkit/thresholds.py src/eval_toolkit/leakage.py \
		src/eval_toolkit/manifest.py src/eval_toolkit/paths.py \
		src/eval_toolkit/provenance.py src/eval_toolkit/seeds.py \
		src/eval_toolkit/docs.py

test-fast:
	$(PYTHON) -m pytest -m "not slow" -q

test-unit:
	$(PYTHON) -m pytest -m unit

test-property:
	$(PYTHON) -m pytest -m property

test-smoke:
	$(PYTHON) -m pytest -m smoke

test-doctest:
	$(PYTHON) -m pytest --doctest-modules \
		src/eval_toolkit/metrics.py src/eval_toolkit/bootstrap.py \
		src/eval_toolkit/calibration.py src/eval_toolkit/text_dedup.py \
		src/eval_toolkit/thresholds.py src/eval_toolkit/leakage.py \
		src/eval_toolkit/manifest.py src/eval_toolkit/paths.py \
		src/eval_toolkit/provenance.py src/eval_toolkit/seeds.py \
		src/eval_toolkit/docs.py

type:
	$(PYTHON) -m mypy src

coverage:
	$(PYTHON) -m pytest --cov=eval_toolkit --cov-report=term-missing --cov-fail-under=92

ci: lint test coverage

clean:
	rm -rf $(VENV) build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Cleaned"
