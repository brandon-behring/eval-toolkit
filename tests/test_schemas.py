"""Validate runtime JSON outputs against the v1 schemas in src/eval_toolkit/schemas/.

A breaking change to the JSON shape without bumping ``schema_version`` fails
this test loudly. Per-file schemas (``results.v1.json``,
``results_full.v1.json``, ``manifest.v1.json``) live alongside the package
so downstream consumers can pin against them.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

import eval_toolkit
from eval_toolkit.claims import ClaimReport, GateResult
from eval_toolkit.harness import EvalSlice, evaluate, with_claim_report, write_run_result
from eval_toolkit.manifest import build_manifest, write_manifest

SCHEMAS_DIR = Path(eval_toolkit.__file__).parent / "schemas"


def _load_schema(filename: str) -> dict:
    return json.loads((SCHEMAS_DIR / filename).read_text())


@pytest.mark.unit
def test_schemas_exist() -> None:
    """All three v1 schemas ship with the package."""
    for name in ("results.v1.json", "results_full.v1.json", "manifest.v1.json"):
        assert (SCHEMAS_DIR / name).exists(), f"missing schema: {name}"


@pytest.mark.unit
def test_schemas_are_valid_json_schemas() -> None:
    """Each schema itself validates against draft 2020-12."""
    for name in ("results.v1.json", "results_full.v1.json", "manifest.v1.json"):
        schema = _load_schema(name)
        # Constructor validates the schema against the meta-schema.
        Draft202012Validator(schema)


@pytest.fixture
def fixture_run_result() -> dict:
    """Build a real RunResult by running evaluate() and convert to dict."""

    class FixedScorer:
        def predict_proba(self, X):
            rng = np.random.default_rng(42)
            return rng.uniform(0, 1, size=len(X))

    df = pd.DataFrame({"text": [f"t{i}" for i in range(60)], "label": [i % 2 for i in range(60)]})
    slice_ = EvalSlice(name="test", df=df)
    result = evaluate({"rng": FixedScorer()}, [slice_], run_id="r")
    return result.to_dict()


@pytest.mark.unit
def test_results_full_validates_against_v1_schema(fixture_run_result: dict) -> None:
    schema = _load_schema("results_full.v1.json")
    Draft202012Validator(schema).validate(fixture_run_result)


@pytest.mark.unit
def test_results_compact_validates_against_v1_schema(fixture_run_result: dict) -> None:
    """Compact form (scores stripped) — written by write_run_result."""

    class _Scorer:
        def predict_proba(self, X):
            return np.full(len(X), 0.5)

    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    slice_ = EvalSlice(name="test", df=df)
    result = evaluate({"s": _Scorer()}, [slice_], run_id="r-compact")
    with tempfile.TemporaryDirectory() as d:
        compact_path, _ = write_run_result(result, Path(d))
        compact = json.loads(compact_path.read_text())
    schema = _load_schema("results.v1.json")
    Draft202012Validator(schema).validate(compact)


@pytest.mark.unit
def test_results_with_claim_report_validate_against_v1_schemas() -> None:
    class _Scorer:
        def predict_proba(self, X):
            return np.full(len(X), 0.5)

    df = pd.DataFrame({"text": [f"r{i}" for i in range(40)], "label": [i % 2 for i in range(40)]})
    result = evaluate({"s": _Scorer()}, [EvalSlice(name="test", df=df)], run_id="claims")
    report = ClaimReport(claims={"claim": [GateResult(name="gate", passed=True, message="ok")]})
    result = with_claim_report(result, report)

    with tempfile.TemporaryDirectory() as d:
        compact_path, full_path = write_run_result(result, Path(d))
        compact = json.loads(compact_path.read_text())
        full = json.loads(full_path.read_text())

    Draft202012Validator(_load_schema("results.v1.json")).validate(compact)
    Draft202012Validator(_load_schema("results_full.v1.json")).validate(full)
    assert compact["claim_report"]["has_failures"] is False
    assert full["claim_report"]["claims"]["claim"][0]["passed"] is True


@pytest.mark.unit
def test_manifest_validates_against_v1_schema() -> None:
    m = build_manifest(run_id="r", config={"k": 5})
    with tempfile.TemporaryDirectory() as d:
        path = write_manifest(m, d)
        loaded = json.loads(path.read_text())
    schema = _load_schema("manifest.v1.json")
    Draft202012Validator(schema).validate(loaded)


@pytest.mark.unit
def test_manifest_with_source_roles_and_guardrails_validates() -> None:
    m = build_manifest(
        run_id="r",
        config={},
        source_roles=[
            {"source": "train_pool", "role": "train", "n_rows": 10},
            {"source": "final", "role": "locked_final_holdout", "metadata": {"locked": True}},
        ],
        guardrails=["do not tune on final holdout"],
    )
    with tempfile.TemporaryDirectory() as d:
        loaded = json.loads(write_manifest(m, d).read_text())
    schema = _load_schema("manifest.v1.json")
    Draft202012Validator(schema).validate(loaded)


@pytest.mark.unit
def test_manifest_with_leakage_report_validates() -> None:
    """Schema accepts a populated leakage_report (the most complex optional field)."""
    from eval_toolkit.leakage import LeakageFinding, LeakageReport

    finding = LeakageFinding(
        check_name="ExactDuplicateCheck",
        severity="warning",
        drop_indices={"test": [0, 1]},
        evidence={"foo": "bar"},
        message="found 2",
        n_affected=2,
    )
    report = LeakageReport(findings=[finding])
    m = build_manifest(run_id="r", config={}, leakage_report=report)
    with tempfile.TemporaryDirectory() as d:
        loaded = json.loads(write_manifest(m, d).read_text())
    schema = _load_schema("manifest.v1.json")
    Draft202012Validator(schema).validate(loaded)
