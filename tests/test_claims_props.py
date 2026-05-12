"""Hypothesis property tests for eval_toolkit.claims invariants.

Pins down the contract that claim evidence is order-independent,
that warning-severity failures don't trip ``has_failures()``, that
gate-implementation exceptions are normalized to typed-message
failures, and that the per-gate ``name`` round-trips through
``EvidenceGate.evaluate``.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eval_toolkit.claims import (
    ClaimReport,
    ClaimSpec,
    EvidenceGate,
    GateResult,
    evaluate_claims,
)


def _always_pass_check(severity: str) -> EvidenceGate:
    def _check(result: Mapping[str, object], manifest: Mapping[str, object] | None) -> GateResult:
        return GateResult(name="always_pass", passed=True, severity=severity, message="ok")

    return EvidenceGate(name="always_pass", check=_check, severity=severity)


def _always_fail_check(severity: str, *, suffix: str = "") -> EvidenceGate:
    name = f"always_fail{suffix}"

    def _check(result: Mapping[str, object], manifest: Mapping[str, object] | None) -> GateResult:
        return GateResult(name=name, passed=False, severity=severity, message="nope")

    return EvidenceGate(name=name, check=_check, severity=severity)


def _exception_check(exc_class: type[Exception], message: str) -> EvidenceGate:
    name = f"raises_{exc_class.__name__}"

    def _check(result: Mapping[str, object], manifest: Mapping[str, object] | None) -> GateResult:
        raise exc_class(message)

    return EvidenceGate(name=name, check=_check)


severities = st.sampled_from(["error", "warning", "info"])
catchable_exceptions = st.sampled_from(
    [KeyError, ValueError, TypeError, RuntimeError, AttributeError, LookupError]
)


@pytest.mark.property
@given(n_gates=st.integers(min_value=1, max_value=8), sev=severities)
def test_all_pass_means_no_error_failures(n_gates: int, sev: str) -> None:
    """A ClaimSpec where every gate passes never reports failures."""
    gates = tuple(_always_pass_check(sev) for _ in range(n_gates))
    spec = ClaimSpec(name="all-pass claim", gates=gates)
    report = evaluate_claims({}, [spec])
    assert report.has_failures(include_warnings=False) is False
    assert report.has_failures(include_warnings=True) is False
    assert len(report.claims["all-pass claim"]) == n_gates


@pytest.mark.property
@given(
    n_pass=st.integers(min_value=0, max_value=4),
    n_fail=st.integers(min_value=1, max_value=4),
)
def test_any_error_failure_trips_has_failures(n_pass: int, n_fail: int) -> None:
    """At least one failing error-severity gate ⇒ has_failures() is True."""
    pass_gates = tuple(_always_pass_check("error") for _ in range(n_pass))
    fail_gates = tuple(_always_fail_check("error", suffix=f"_{i}") for i in range(n_fail))
    spec = ClaimSpec(name="mixed claim", gates=pass_gates + fail_gates)
    report = evaluate_claims({}, [spec])
    assert report.has_failures() is True


@pytest.mark.property
@given(n_warn_fail=st.integers(min_value=1, max_value=5))
def test_warning_failures_do_not_trip_default_has_failures(n_warn_fail: int) -> None:
    """Warning-severity failures stay quiet unless include_warnings=True."""
    gates = tuple(_always_fail_check("warning", suffix=f"_{i}") for i in range(n_warn_fail))
    spec = ClaimSpec(name="warning-only claim", gates=gates)
    report = evaluate_claims({}, [spec])
    assert report.has_failures(include_warnings=False) is False
    assert report.has_failures(include_warnings=True) is True


@pytest.mark.property
@given(
    exc_class=catchable_exceptions,
    message=st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            blacklist_characters="'\"\\",
        ),
        min_size=0,
        max_size=50,
    ),
)
def test_gate_exception_normalized_to_typed_failure(
    exc_class: type[Exception], message: str
) -> None:
    """A gate raising any whitelisted exception yields a failing GateResult.

    The resulting message is prefixed with the exception type name —
    documented contract from claims.py:77.
    """
    gate = _exception_check(exc_class, message)
    spec = ClaimSpec(name="exception claim", gates=(gate,))
    report = evaluate_claims({}, [spec])
    result = report.claims["exception claim"][0]
    assert result.passed is False
    assert result.message.startswith(f"{exc_class.__name__}: ")
    # KeyError applies repr() to its arg; other exceptions don't. Skip the
    # substring check for KeyError but still verify the type prefix above.
    if exc_class is not KeyError:
        assert message in result.message


@pytest.mark.property
@given(
    pass_count=st.integers(min_value=1, max_value=3),
    fail_count=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=999),
)
def test_gate_order_does_not_affect_claim_verdict(
    pass_count: int, fail_count: int, seed: int
) -> None:
    """Permuting the gates within a ClaimSpec doesn't change has_failures()."""
    import random

    rng = random.Random(seed)
    pass_gates = [_always_pass_check("error") for _ in range(pass_count)]
    fail_gates = [_always_fail_check("error", suffix=f"_{i}") for i in range(fail_count)]
    base = pass_gates + fail_gates
    shuffled = base.copy()
    rng.shuffle(shuffled)

    spec_a = ClaimSpec(name="ordered", gates=tuple(base))
    spec_b = ClaimSpec(name="ordered", gates=tuple(shuffled))

    report_a = evaluate_claims({}, [spec_a])
    report_b = evaluate_claims({}, [spec_b])
    assert report_a.has_failures() == report_b.has_failures()
    # Each verdict (pass/fail) shows up the same number of times.
    a_passed = sorted(r.passed for r in report_a.claims["ordered"])
    b_passed = sorted(r.passed for r in report_b.claims["ordered"])
    assert a_passed == b_passed


@pytest.mark.property
@given(
    gate_name=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-:."),
        min_size=1,
        max_size=40,
    )
)
def test_evaluate_normalizes_gate_result_name_to_spec_name(gate_name: str) -> None:
    """``EvidenceGate.evaluate`` returns a GateResult whose name matches the gate.

    Even if the inner check returns a GateResult with a different name,
    ``evaluate`` re-emits it under the canonical gate name. Pinning this
    invariant prevents drift between the spec's claimed gate name and
    the report's recorded one.
    """

    def _check(result: Mapping[str, object], manifest: Mapping[str, object] | None) -> GateResult:
        # Intentionally use a different name in the inner result.
        return GateResult(name="inner-name", passed=True, severity="error", message="ok")

    gate = EvidenceGate(name=gate_name, check=_check)
    out = gate.evaluate({}, None)
    assert out.name == gate_name


@pytest.mark.property
@given(n_specs=st.integers(min_value=1, max_value=5))
def test_claim_report_serialization_round_trip(n_specs: int) -> None:
    """ClaimReport.to_dict() preserves per-claim verdicts and has_failures flag."""
    specs = []
    for i in range(n_specs):
        # Half pass, half fail — guarantees variation in has_failures.
        if i % 2 == 0:
            gates = (_always_pass_check("error"),)
        else:
            gates = (_always_fail_check("error", suffix=f"_{i}"),)
        specs.append(ClaimSpec(name=f"claim-{i}", gates=gates))

    report = evaluate_claims({}, specs)
    payload = report.to_dict()
    assert isinstance(payload, dict)
    assert payload["has_failures"] == report.has_failures()
    assert set(payload["claims"]) == {spec.name for spec in specs}
    for spec in specs:
        assert len(payload["claims"][spec.name]) == len(spec.gates)


@pytest.mark.unit
def test_claim_report_dataclass_is_constructible_directly() -> None:
    """Sanity check that ClaimReport works without the evaluate_claims wrapper."""
    report = ClaimReport(
        claims={
            "x": [GateResult(name="g", passed=True, severity="error", message="")],
        }
    )
    assert report.has_failures() is False
