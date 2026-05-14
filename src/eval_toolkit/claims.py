"""Generic evidence gates for classification claims.

This module does not render reports and does not encode domain claims. It
evaluates caller-supplied claim specs against result/manifest payloads and
returns machine-readable pass/fail evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ClaimReport",
    "ClaimSpec",
    "EvidenceGate",
    "GateResult",
    "evaluate_claims",
    "external_diagnostic_gate",
    "headline_present_gate",
    "low_fpr_feasibility_gate",
    "metric_threshold_gate",
    "minimum_slice_size_gate",
    "no_leakage_errors_gate",
    "no_scorer_errors_gate",
    "paired_diff_present_gate",
    "required_metric_gate",
    "required_scorer_gate",
    "required_slice_gate",
    "source_role_gate",
    "strict_artifact_gate",
]

GateSeverity = Literal["error", "warning", "info"]
GateCheck = Callable[[Mapping[str, Any], Mapping[str, Any] | None], "GateResult"]


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of one evidence gate."""

    name: str
    passed: bool
    severity: GateSeverity = "error"
    message: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class EvidenceGate:
    """Named callable gate used inside a :class:`ClaimSpec`."""

    name: str
    check: GateCheck
    description: str = ""
    severity: GateSeverity = "error"

    def evaluate(
        self,
        result: Mapping[str, Any],
        manifest: Mapping[str, Any] | None = None,
    ) -> GateResult:
        """Run the gate and normalize unexpected exceptions to failures."""
        try:
            gate_result = self.check(result, manifest)
        except (KeyError, ValueError, TypeError, RuntimeError, AttributeError, LookupError) as exc:
            return GateResult(
                name=self.name,
                passed=False,
                severity=self.severity,
                message=f"{type(exc).__name__}: {exc}",
            )
        if gate_result.name == self.name and gate_result.severity == self.severity:
            return gate_result
        return GateResult(
            name=self.name,
            passed=gate_result.passed,
            severity=self.severity,
            message=gate_result.message,
            evidence=gate_result.evidence,
        )


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    """A claim plus the gates required before it can be treated as supported."""

    name: str
    gates: tuple[EvidenceGate, ...]
    mode: str = "claim"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate minimum claim shape."""
        if not self.name:
            raise ValueError("ClaimSpec.name must be non-empty")
        if not self.gates:
            raise ValueError("ClaimSpec.gates must be non-empty")
        if not self.mode:
            raise ValueError("ClaimSpec.mode must be non-empty")


@dataclass(frozen=True, slots=True)
class ClaimReport:
    """Machine-readable result of evaluating claim specs."""

    claims: dict[str, list[GateResult]]

    def has_failures(self, *, include_warnings: bool = False) -> bool:
        """Return True if any claim has a failing error gate."""
        failing_severities = {"error", "warning"} if include_warnings else {"error"}
        return any(
            (not result.passed) and result.severity in failing_severities
            for results in self.claims.values()
            for result in results
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "claims": {
                claim: [result.to_dict() for result in results]
                for claim, results in self.claims.items()
            },
            "has_failures": self.has_failures(),
        }


def evaluate_claims(
    result: object,
    claim_specs: Sequence[ClaimSpec],
    *,
    manifest: object | None = None,
) -> ClaimReport:
    """Evaluate claim specs against a result payload and optional manifest."""
    result_dict = _as_mapping(result)
    manifest_dict = _as_mapping(manifest) if manifest is not None else None
    claims: dict[str, list[GateResult]] = {}
    for spec in claim_specs:
        claims[spec.name] = [gate.evaluate(result_dict, manifest_dict) for gate in spec.gates]
    return ClaimReport(claims=claims)


def required_slice_gate(slice_name: str, *, severity: GateSeverity = "error") -> EvidenceGate:
    """Require ``by_slice.<slice_name>`` to exist."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        by_slice = result.get("by_slice", {})
        passed = isinstance(by_slice, Mapping) and slice_name in by_slice
        return GateResult(
            name=f"required_slice:{slice_name}",
            passed=passed,
            severity=severity,
            message="slice present" if passed else f"missing slice {slice_name!r}",
        )

    return EvidenceGate(name=f"required_slice:{slice_name}", check=_check, severity=severity)


def required_scorer_gate(
    slice_name: str,
    scorer_name: str,
    *,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require a scorer result under a slice."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        block = _get_path(result, f"by_slice.{slice_name}.by_scorer.{scorer_name}")
        passed = isinstance(block, Mapping)
        return GateResult(
            name=f"required_scorer:{slice_name}:{scorer_name}",
            passed=passed,
            severity=severity,
            message="scorer present" if passed else "missing scorer result",
        )

    return EvidenceGate(
        name=f"required_scorer:{slice_name}:{scorer_name}",
        check=_check,
        severity=severity,
    )


def required_metric_gate(
    slice_name: str,
    scorer_name: str,
    metric_path: str,
    *,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require a metric path under one scorer result."""
    path = f"by_slice.{slice_name}.by_scorer.{scorer_name}.{metric_path}"

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        value = _get_path(result, path)
        passed = value is not None
        return GateResult(
            name=f"required_metric:{slice_name}:{scorer_name}:{metric_path}",
            passed=passed,
            severity=severity,
            message="metric present" if passed else f"missing metric path {path!r}",
            evidence={"path": path, "value": value} if passed else {"path": path},
        )

    return EvidenceGate(
        name=f"required_metric:{slice_name}:{scorer_name}:{metric_path}",
        check=_check,
        severity=severity,
    )


def minimum_slice_size_gate(
    slice_name: str,
    *,
    min_n: int = 0,
    min_positive: int = 0,
    min_negative: int = 0,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require minimum total/positive/negative counts for a slice."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        block = _get_path(result, f"by_slice.{slice_name}")
        if not isinstance(block, Mapping):
            return GateResult(
                name=f"minimum_slice_size:{slice_name}",
                passed=False,
                severity=severity,
                message=f"missing slice {slice_name!r}",
            )
        n = _as_int(block.get("n"))
        n_positive = _as_int(block.get("n_positive"))
        n_negative = None if n is None or n_positive is None else n - n_positive
        passed = (
            n is not None
            and n_positive is not None
            and n_negative is not None
            and n >= min_n
            and n_positive >= min_positive
            and n_negative >= min_negative
        )
        return GateResult(
            name=f"minimum_slice_size:{slice_name}",
            passed=passed,
            severity=severity,
            message="slice size sufficient" if passed else "slice size below requirement",
            evidence={
                "n": n,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "min_n": min_n,
                "min_positive": min_positive,
                "min_negative": min_negative,
            },
        )

    return EvidenceGate(name=f"minimum_slice_size:{slice_name}", check=_check, severity=severity)


def low_fpr_feasibility_gate(
    slice_name: str,
    *,
    max_fpr: float,
    confidence: float = 0.95,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require enough negatives for a low-FPR claim to be statistically feasible.

    This is not an observed-performance gate. It asks whether a slice could
    support a claim of ``FPR <= max_fpr`` even in the best empirical case of
    zero false positives. The best-case upper bound is the Wilson score upper
    confidence bound for ``0 / n_negative``.

    Raises
    ------
    ValueError
        If ``max_fpr`` is not in (0, 1] or ``confidence`` is not in (0, 1).
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        block = _get_path(result, f"by_slice.{slice_name}")
        if not isinstance(block, Mapping):
            return GateResult(
                name=f"low_fpr_feasibility:{slice_name}",
                passed=False,
                severity=severity,
                message=f"missing slice {slice_name!r}",
            )

        n = _as_int(block.get("n"))
        n_positive = _as_int(block.get("n_positive"))
        n_negative = None if n is None or n_positive is None else n - n_positive
        empirical_step = None if n_negative is None or n_negative <= 0 else 1.0 / n_negative
        best_case_high = (
            None
            if n_negative is None or n_negative <= 0
            else _wilson_zero_success_upper(n_negative, confidence=confidence)
        )
        passed = best_case_high is not None and best_case_high <= max_fpr
        return GateResult(
            name=f"low_fpr_feasibility:{slice_name}",
            passed=passed,
            severity=severity,
            message=(
                "negative count can support requested FPR"
                if passed
                else "negative count cannot support requested FPR"
            ),
            evidence={
                "n": n,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "max_fpr": max_fpr,
                "confidence": confidence,
                "empirical_fpr_step": empirical_step,
                "best_case_fpr_ci_high": best_case_high,
            },
        )

    return EvidenceGate(name=f"low_fpr_feasibility:{slice_name}", check=_check, severity=severity)


def paired_diff_present_gate(
    slice_name: str,
    diff_key: str,
    *,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require a paired-difference comparison under a slice."""
    path = f"by_slice.{slice_name}.paired_diffs.{diff_key}"

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        value = _get_path(result, path)
        passed = isinstance(value, Mapping) and "skipped" not in value and "error" not in value
        return GateResult(
            name=f"paired_diff_present:{slice_name}:{diff_key}",
            passed=passed,
            severity=severity,
            message="paired diff present" if passed else "missing, skipped, or errored paired diff",
            evidence={"path": path},
        )

    return EvidenceGate(
        name=f"paired_diff_present:{slice_name}:{diff_key}",
        check=_check,
        severity=severity,
    )


def headline_present_gate(
    path: str = "headline", *, severity: GateSeverity = "error"
) -> EvidenceGate:
    """Require a non-null headline/comparison block at an arbitrary path."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        value = _get_path(result, path)
        passed = value is not None
        return GateResult(
            name=f"headline_present:{path}",
            passed=passed,
            severity=severity,
            message="headline present" if passed else f"missing headline at {path!r}",
            evidence={"path": path},
        )

    return EvidenceGate(name=f"headline_present:{path}", check=_check, severity=severity)


def metric_threshold_gate(
    slice_name: str,
    scorer_name: str,
    metric_path: str,
    *,
    op: Literal["<", "<=", ">", ">=", "=="],
    threshold: float,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require a numeric metric to satisfy a threshold comparison."""
    path = f"by_slice.{slice_name}.by_scorer.{scorer_name}.{metric_path}"

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        raw = _get_path(result, path)
        value = _as_float(raw)
        passed = value is not None and _compare(value, op, threshold)
        return GateResult(
            name=f"metric_threshold:{slice_name}:{scorer_name}:{metric_path}",
            passed=passed,
            severity=severity,
            message="metric threshold satisfied" if passed else "metric threshold failed",
            evidence={"path": path, "value": value, "op": op, "threshold": threshold},
        )

    return EvidenceGate(
        name=f"metric_threshold:{slice_name}:{scorer_name}:{metric_path}",
        check=_check,
        severity=severity,
    )


def no_scorer_errors_gate(*, severity: GateSeverity = "error") -> EvidenceGate:
    """Fail if any scorer block contains an ``error`` field."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        errors: list[str] = []
        by_slice = result.get("by_slice", {})
        if isinstance(by_slice, Mapping):
            for slice_name, slice_block in by_slice.items():
                if not isinstance(slice_block, Mapping):
                    continue
                by_scorer = slice_block.get("by_scorer", {})
                if not isinstance(by_scorer, Mapping):
                    continue
                for scorer_name, scorer_block in by_scorer.items():
                    if isinstance(scorer_block, Mapping) and "error" in scorer_block:
                        errors.append(f"{slice_name}.{scorer_name}: {scorer_block['error']}")
        return GateResult(
            name="no_scorer_errors",
            passed=not errors,
            severity=severity,
            message="no scorer errors" if not errors else f"{len(errors)} scorer error(s)",
            evidence={"errors": errors},
        )

    return EvidenceGate(name="no_scorer_errors", check=_check, severity=severity)


def no_leakage_errors_gate(*, severity: GateSeverity = "error") -> EvidenceGate:
    """Fail if result config or manifest leakage report has error-severity findings."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        reports = []
        config = result.get("config", {})
        if isinstance(config, Mapping):
            reports.append(config.get("leakage_report"))
        if manifest is not None:
            reports.append(manifest.get("leakage_report"))
        errors: list[object] = []
        for report in reports:
            if not isinstance(report, Mapping):
                continue
            findings = report.get("findings", [])
            if not isinstance(findings, Sequence):
                continue
            for finding in findings:
                if isinstance(finding, Mapping) and finding.get("severity") == "error":
                    errors.append(dict(finding))
        return GateResult(
            name="no_leakage_errors",
            passed=not errors,
            severity=severity,
            message="no leakage errors" if not errors else f"{len(errors)} leakage error(s)",
            evidence={"errors": errors},
        )

    return EvidenceGate(name="no_leakage_errors", check=_check, severity=severity)


def source_role_gate(
    required_roles: Sequence[str],
    *,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require source-role metadata with the requested roles in the manifest."""
    required = tuple(required_roles)

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        roles = set()
        if manifest is not None:
            source_roles = manifest.get("source_roles", [])
            if isinstance(source_roles, Sequence):
                for record in source_roles:
                    if isinstance(record, Mapping) and isinstance(record.get("role"), str):
                        roles.add(str(record["role"]))
        missing = sorted(set(required) - roles)
        return GateResult(
            name="source_role_presence",
            passed=not missing,
            severity=severity,
            message="required source roles present" if not missing else "missing source roles",
            evidence={
                "required_roles": list(required),
                "present_roles": sorted(roles),
                "missing": missing,
            },
        )

    return EvidenceGate(name="source_role_presence", check=_check, severity=severity)


def external_diagnostic_gate(
    path: str,
    *,
    op: Literal["<", "<=", ">", ">=", "=="] | None = None,
    threshold: float | None = None,
    severity: GateSeverity = "error",
) -> EvidenceGate:
    """Require an external diagnostic payload, optionally thresholded.

    The gate first checks the result payload and then the manifest payload.
    That keeps diagnostics generic: consumers can store them in results when
    computed during analysis, or in manifests when they are precomputed source
    evidence.

    Raises
    ------
    ValueError
        If ``path`` is empty, or if exactly one of ``op``/``threshold`` is
        supplied (both required together, or both ``None`` for
        presence-only check).
    """
    if not path:
        raise ValueError("path must be non-empty")
    if (op is None) != (threshold is None):
        raise ValueError("op and threshold must be supplied together")

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        value = _get_path(result, path)
        payload = "result"
        if value is None and manifest is not None:
            value = _get_path(manifest, path)
            payload = "manifest"
        if op is None or threshold is None:
            passed = value is not None
            message = "external diagnostic present" if passed else "missing external diagnostic"
            evidence: dict[str, object] = {"path": path, "payload": payload}
            if passed:
                evidence["value"] = value
        else:
            numeric = _as_float(value)
            passed = numeric is not None and _compare(numeric, op, threshold)
            message = (
                "external diagnostic threshold satisfied"
                if passed
                else "external diagnostic threshold failed"
            )
            evidence = {
                "path": path,
                "payload": payload,
                "value": numeric,
                "op": op,
                "threshold": threshold,
            }
        return GateResult(
            name=f"external_diagnostic:{path}",
            passed=passed,
            severity=severity,
            message=message,
            evidence=evidence,
        )

    return EvidenceGate(name=f"external_diagnostic:{path}", check=_check, severity=severity)


def strict_artifact_gate(*, severity: GateSeverity = "error") -> EvidenceGate:
    """Fail if result or manifest contains non-finite numeric values."""

    def _check(result: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> GateResult:
        findings = _non_finite_paths(result, prefix="result")
        if manifest is not None:
            findings.extend(_non_finite_paths(manifest, prefix="manifest"))
        return GateResult(
            name="strict_artifact",
            passed=not findings,
            severity=severity,
            message="artifacts are strict JSON safe" if not findings else "non-finite values found",
            evidence={"non_finite_paths": findings},
        )

    return EvidenceGate(name="strict_artifact", check=_check, severity=severity)


def _as_mapping(obj: object) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, Mapping):
            return out
    raise TypeError(f"expected mapping or object with to_dict(), got {type(obj).__name__}")


def _get_path(payload: Mapping[str, Any], path: str) -> object | None:
    cur: object = payload
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if np_isfinite(out):
            return out
    return None


def _non_finite_paths(value: object, *, prefix: str) -> list[str]:
    """Return dotted/indexed paths to non-finite floats in a JSON-like object."""
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [] if np_isfinite(value) else [prefix]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_non_finite_paths(child, prefix=f"{prefix}.{key}"))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for idx, child in enumerate(value):
            paths.extend(_non_finite_paths(child, prefix=f"{prefix}[{idx}]"))
        return paths
    return []


def np_isfinite(value: float) -> bool:
    """Tiny local finite check to avoid importing numpy for gate traversal."""
    return value == value and value not in (float("inf"), float("-inf"))


def _wilson_zero_success_upper(n: int, *, confidence: float) -> float:
    """Wilson upper confidence bound for zero successes in ``n`` trials."""
    from scipy.stats import norm  # noqa: PLC0415

    z = float(norm.ppf(0.5 + confidence / 2.0))
    z2 = z * z
    return z2 / (n + z2)


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unsupported operator {op!r}")
