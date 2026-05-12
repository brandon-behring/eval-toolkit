"""Coverage-focused tests for eval_toolkit.claims.

Pairs with the happy-path coverage in ``test_claims.py`` and the
invariant pinning in ``test_claims_props.py``. This file targets
the validation-error and structural-branch paths that those suites
don't exercise: ``ClaimSpec`` post-init guards, every gate's
``not isinstance(..., Mapping)`` defensive branch, the
``external_diagnostic_gate`` two-mode logic, the private
``_as_mapping`` / ``_as_int`` / ``_as_float`` / ``_compare``
helpers, and the gate-implementer-exception → typed-failure
contract for every exception class that ``EvidenceGate.evaluate``
catches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from eval_toolkit.claims import (
    ClaimReport,
    ClaimSpec,
    EvidenceGate,
    GateResult,
    evaluate_claims,
    external_diagnostic_gate,
    low_fpr_feasibility_gate,
    minimum_slice_size_gate,
    no_leakage_errors_gate,
    no_scorer_errors_gate,
    source_role_gate,
    strict_artifact_gate,
)

# ---------------------------------------------------------------------------
# ClaimSpec __post_init__ guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_claim_spec_rejects_empty_name() -> None:
    gate = EvidenceGate(name="g", check=lambda r, m: GateResult(name="g", passed=True))
    with pytest.raises(ValueError, match="name must be non-empty"):
        ClaimSpec(name="", gates=(gate,))


@pytest.mark.unit
def test_claim_spec_rejects_empty_gates() -> None:
    with pytest.raises(ValueError, match="gates must be non-empty"):
        ClaimSpec(name="c", gates=())


@pytest.mark.unit
def test_claim_spec_rejects_empty_mode() -> None:
    gate = EvidenceGate(name="g", check=lambda r, m: GateResult(name="g", passed=True))
    with pytest.raises(ValueError, match="mode must be non-empty"):
        ClaimSpec(name="c", gates=(gate,), mode="")


# ---------------------------------------------------------------------------
# low_fpr_feasibility_gate parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_low_fpr_feasibility_rejects_bad_max_fpr() -> None:
    with pytest.raises(ValueError, match="max_fpr must be in"):
        low_fpr_feasibility_gate("s", max_fpr=0.0)
    with pytest.raises(ValueError, match="max_fpr must be in"):
        low_fpr_feasibility_gate("s", max_fpr=1.5)


@pytest.mark.unit
def test_low_fpr_feasibility_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        low_fpr_feasibility_gate("s", max_fpr=0.01, confidence=0.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        low_fpr_feasibility_gate("s", max_fpr=0.01, confidence=1.0)


# ---------------------------------------------------------------------------
# minimum_slice_size_gate: missing-slice branch (line 237)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minimum_slice_size_missing_slice_fails_cleanly() -> None:
    gate = minimum_slice_size_gate("absent", min_n=10)
    result = gate.check({"by_slice": {}}, None)
    assert result.passed is False
    assert "missing slice" in result.message


# ---------------------------------------------------------------------------
# no_scorer_errors_gate: all three defensive branches (417, 420, 423)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_scorer_errors_when_by_slice_not_mapping() -> None:
    gate = no_scorer_errors_gate()
    out = gate.check({"by_slice": "not-a-mapping"}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_scorer_errors_when_slice_block_not_mapping() -> None:
    gate = no_scorer_errors_gate()
    out = gate.check({"by_slice": {"s": "not-a-mapping"}}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_scorer_errors_when_by_scorer_not_mapping() -> None:
    gate = no_scorer_errors_gate()
    out = gate.check({"by_slice": {"s": {"by_scorer": "not-a-mapping"}}}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_scorer_errors_collects_errors() -> None:
    gate = no_scorer_errors_gate()
    out = gate.check(
        {"by_slice": {"s": {"by_scorer": {"m": {"error": "boom"}}}}},
        None,
    )
    assert out.passed is False
    assert "1 scorer error" in out.message
    assert out.evidence == {"errors": ["s.m: boom"]}


# ---------------------------------------------------------------------------
# no_leakage_errors_gate: config / manifest / report shape branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_leakage_errors_when_config_not_mapping() -> None:
    gate = no_leakage_errors_gate()
    out = gate.check({"config": "not-a-mapping"}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_leakage_errors_pulls_from_manifest_when_config_empty() -> None:
    gate = no_leakage_errors_gate()
    out = gate.check(
        {},
        {"leakage_report": {"findings": [{"severity": "error", "kind": "x"}]}},
    )
    assert out.passed is False
    assert len(out.evidence["errors"]) == 1


@pytest.mark.unit
def test_no_leakage_errors_skips_non_mapping_report() -> None:
    gate = no_leakage_errors_gate()
    out = gate.check({"config": {"leakage_report": "not-a-mapping"}}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_leakage_errors_skips_non_sequence_findings() -> None:
    gate = no_leakage_errors_gate()
    out = gate.check({"config": {"leakage_report": {"findings": "not-a-sequence"}}}, None)
    assert out.passed is True


@pytest.mark.unit
def test_no_leakage_errors_skips_non_mapping_finding() -> None:
    gate = no_leakage_errors_gate()
    out = gate.check(
        {"config": {"leakage_report": {"findings": ["not-a-mapping", {"severity": "warning"}]}}},
        None,
    )
    assert out.passed is True


# ---------------------------------------------------------------------------
# source_role_gate: defensive branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_role_with_no_manifest_reports_all_missing() -> None:
    gate = source_role_gate(["train", "eval"])
    out = gate.check({}, None)
    assert out.passed is False
    assert out.evidence["missing"] == ["eval", "train"]


@pytest.mark.unit
def test_source_role_when_source_roles_not_sequence() -> None:
    gate = source_role_gate(["train"])
    out = gate.check({}, {"source_roles": "not-a-sequence"})
    # Strings ARE Sequence — but iterating won't yield Mapping records,
    # so missing roles is still reported. The defensive branch is the
    # "skip non-Mapping record" path.
    assert out.passed is False


@pytest.mark.unit
def test_source_role_skips_record_without_role_string() -> None:
    gate = source_role_gate(["train"])
    out = gate.check(
        {},
        {"source_roles": ["not-a-mapping", {"source": "x"}, {"role": 42}, {"role": "train"}]},
    )
    assert out.passed is True
    assert out.evidence["present_roles"] == ["train"]


# ---------------------------------------------------------------------------
# external_diagnostic_gate: validation + two-mode logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_external_diagnostic_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="path must be non-empty"):
        external_diagnostic_gate("")


@pytest.mark.unit
def test_external_diagnostic_rejects_mismatched_op_and_threshold() -> None:
    with pytest.raises(ValueError, match="op and threshold must be supplied together"):
        external_diagnostic_gate("p", op=">", threshold=None)
    with pytest.raises(ValueError, match="op and threshold must be supplied together"):
        external_diagnostic_gate("p", op=None, threshold=0.5)


@pytest.mark.unit
def test_external_diagnostic_falls_back_to_manifest() -> None:
    gate = external_diagnostic_gate("diag.score")
    out = gate.check({}, {"diag": {"score": 0.7}})
    assert out.passed is True
    assert out.evidence["payload"] == "manifest"
    assert out.evidence["value"] == 0.7


@pytest.mark.unit
def test_external_diagnostic_threshold_satisfied() -> None:
    gate = external_diagnostic_gate("d", op=">=", threshold=0.5)
    out = gate.check({"d": 0.75}, None)
    assert out.passed is True
    assert out.evidence["value"] == 0.75


@pytest.mark.unit
def test_external_diagnostic_threshold_failed() -> None:
    gate = external_diagnostic_gate("d", op=">=", threshold=0.5)
    out = gate.check({"d": 0.25}, None)
    assert out.passed is False
    assert "failed" in out.message


@pytest.mark.unit
def test_external_diagnostic_threshold_with_missing_value() -> None:
    gate = external_diagnostic_gate("d", op=">=", threshold=0.5)
    out = gate.check({}, None)
    assert out.passed is False  # numeric is None ⇒ passed False


# ---------------------------------------------------------------------------
# strict_artifact_gate: manifest path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strict_artifact_inspects_manifest_too() -> None:
    gate = strict_artifact_gate()
    out = gate.check({"ok": 1.0}, {"bad": float("nan")})
    assert out.passed is False
    assert any("manifest" in p for p in out.evidence["non_finite_paths"])


# ---------------------------------------------------------------------------
# evaluate_claims via TypeError input (exercises _as_mapping rejection)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_claims_rejects_non_mapping_input() -> None:
    gate = EvidenceGate(name="g", check=lambda r, m: GateResult(name="g", passed=True))
    spec = ClaimSpec(name="c", gates=(gate,))
    with pytest.raises(TypeError, match="expected mapping or object with to_dict"):
        evaluate_claims(42, [spec])


@pytest.mark.unit
def test_evaluate_claims_accepts_object_with_to_dict() -> None:
    class _R:
        def to_dict(self) -> dict[str, Any]:
            return {"by_slice": {"s": {"n": 10, "n_positive": 5}}}

    gate = minimum_slice_size_gate("s", min_n=5)
    spec = ClaimSpec(name="c", gates=(gate,))
    report = evaluate_claims(_R(), [spec])
    assert report.has_failures() is False


@pytest.mark.unit
def test_evaluate_claims_rejects_to_dict_returning_non_mapping() -> None:
    class _R:
        def to_dict(self) -> str:
            return "not-a-mapping"

    gate = EvidenceGate(name="g", check=lambda r, m: GateResult(name="g", passed=True))
    spec = ClaimSpec(name="c", gates=(gate,))
    with pytest.raises(TypeError, match="expected mapping or object with to_dict"):
        evaluate_claims(_R(), [spec])


@pytest.mark.unit
def test_evaluate_claims_with_manifest_object_with_to_dict() -> None:
    class _M:
        def to_dict(self) -> dict[str, Any]:
            return {"source_roles": [{"role": "train"}]}

    gate = source_role_gate(["train"])
    spec = ClaimSpec(name="c", gates=(gate,))
    report = evaluate_claims({}, [spec], manifest=_M())
    assert report.has_failures() is False


# ---------------------------------------------------------------------------
# _compare: every operator branch via metric_threshold_gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "op,value,threshold,expected",
    [
        ("<", 0.3, 0.5, True),
        ("<", 0.7, 0.5, False),
        ("<=", 0.5, 0.5, True),
        (">", 0.7, 0.5, True),
        (">", 0.3, 0.5, False),
        (">=", 0.5, 0.5, True),
        ("==", 0.5, 0.5, True),
        ("==", 0.4, 0.5, False),
    ],
)
def test_metric_threshold_every_operator(
    op: str, value: float, threshold: float, expected: bool
) -> None:
    from eval_toolkit.claims import metric_threshold_gate

    gate = metric_threshold_gate("s", "m", "k", op=op, threshold=threshold)  # type: ignore[arg-type]
    out = gate.check({"by_slice": {"s": {"by_scorer": {"m": {"k": value}}}}}, None)
    assert out.passed is expected


@pytest.mark.unit
def test_external_diagnostic_with_invalid_op_propagates_via_evaluate() -> None:
    """Constructed via the gate factory, an unsupported op surfaces as a
    typed-message failure thanks to EvidenceGate.evaluate's broad-exception
    contract. (The _compare helper raises ValueError on unknown ops.)"""
    # external_diagnostic_gate validates op via its `Literal[..]` typing but
    # not at runtime — feed a bogus op to exercise the raise in _compare.
    gate = external_diagnostic_gate("d", op="!=", threshold=0.5)  # type: ignore[arg-type]
    out = gate.evaluate({"d": 0.5}, None)
    assert out.passed is False
    assert out.message.startswith("ValueError:")


# ---------------------------------------------------------------------------
# _as_int and _as_float: bool rejection (lines 598, 606) + non-numeric
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_minimum_slice_size_rejects_bool_n() -> None:
    """bool is a subclass of int — _as_int must return None for it."""
    gate = minimum_slice_size_gate("s", min_n=0)
    out = gate.check({"by_slice": {"s": {"n": True, "n_positive": False}}}, None)
    assert out.passed is False
    # Evidence reflects None coercion
    assert out.evidence["n"] is None
    assert out.evidence["n_positive"] is None


@pytest.mark.unit
def test_metric_threshold_rejects_bool_value() -> None:
    """bool value coerces to None via _as_float, so the gate fails."""
    from eval_toolkit.claims import metric_threshold_gate

    gate = metric_threshold_gate("s", "m", "k", op=">=", threshold=0.5)
    out = gate.check({"by_slice": {"s": {"by_scorer": {"m": {"k": True}}}}}, None)
    assert out.passed is False
    assert out.evidence["value"] is None


@pytest.mark.unit
def test_metric_threshold_rejects_non_numeric_value() -> None:
    from eval_toolkit.claims import metric_threshold_gate

    gate = metric_threshold_gate("s", "m", "k", op=">=", threshold=0.5)
    out = gate.check({"by_slice": {"s": {"by_scorer": {"m": {"k": "not-a-number"}}}}}, None)
    assert out.passed is False


# ---------------------------------------------------------------------------
# EvidenceGate.evaluate: exception normalization for every caught class
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc_class",
    [KeyError, ValueError, TypeError, RuntimeError, AttributeError, LookupError],
)
def test_evidence_gate_normalizes_each_handled_exception(exc_class: type[Exception]) -> None:
    def _bad(_r: Mapping[str, Any], _m: Mapping[str, Any] | None) -> GateResult:
        raise exc_class("boom")

    gate = EvidenceGate(name="exploding", check=_bad)
    out = gate.evaluate({}, None)
    assert out.passed is False
    assert out.name == "exploding"
    assert out.message.startswith(f"{exc_class.__name__}:")


@pytest.mark.unit
def test_evidence_gate_does_not_swallow_unhandled_exception() -> None:
    """Implementer bugs (NameError, AssertionError, ImportError) propagate."""

    def _bad(_r: Mapping[str, Any], _m: Mapping[str, Any] | None) -> GateResult:
        raise NameError("undefined")

    gate = EvidenceGate(name="exploding", check=_bad)
    with pytest.raises(NameError):
        gate.evaluate({}, None)


@pytest.mark.unit
def test_evidence_gate_rewrites_name_severity_drift() -> None:
    """A check returning a GateResult with a different name/severity is rewritten."""

    def _drift(_r: Mapping[str, Any], _m: Mapping[str, Any] | None) -> GateResult:
        return GateResult(
            name="some-other-name",
            passed=True,
            severity="info",  # different from gate's "warning"
            message="drift",
            evidence={"x": 1},
        )

    gate = EvidenceGate(name="canonical", check=_drift, severity="warning")
    out = gate.evaluate({}, None)
    assert out.name == "canonical"
    assert out.severity == "warning"
    # Other fields preserved from the inner result
    assert out.message == "drift"
    assert out.evidence == {"x": 1}


# ---------------------------------------------------------------------------
# ClaimReport.has_failures with include_warnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_claim_report_has_failures_includes_warnings_when_asked() -> None:
    report = ClaimReport(
        claims={
            "c": [
                GateResult(name="g1", passed=True, severity="error"),
                GateResult(name="g2", passed=False, severity="warning"),
            ]
        }
    )
    assert report.has_failures(include_warnings=False) is False
    assert report.has_failures(include_warnings=True) is True
