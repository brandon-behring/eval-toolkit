"""Hypothesis property tests for v0.7.0 LeakageCheck reference impls.

Restores coverage on `src/eval_toolkit/leakage.py` toward the 90 % gate
(v0.7.1 / PR 1.5).
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eval_toolkit import EvalSlice
from eval_toolkit.leakage import (
    ExactDuplicateCheck,
    GroupLeakageCheck,
    LabelConflictCheck,
    LeakageFinding,
    NormalizedFormLeakageCheck,
    _aggressive_normalize,
    run_leakage_checks,
)


def _build_slice(name: str, texts: list[str], labels: list[int]) -> EvalSlice:
    """Helper: assemble an EvalSlice from parallel lists."""
    return EvalSlice(
        name=name,
        df=pd.DataFrame({"text": texts, "label": labels}),
    )


# ---------------------------------------------------------------------------
# run_leakage_checks invariants
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_empty_check_list_returns_clean_report() -> None:
    """run_leakage_checks([], splits) -> empty findings; has_errors False."""
    splits = {"test": _build_slice("test", ["a", "b"], [0, 1])}
    report = run_leakage_checks([], splits)
    assert report.findings == []
    assert not report.has_errors()
    assert report.errors() == []
    assert report.warnings() == []
    assert report.merged_drop_indices() == {}


@pytest.mark.property
@given(
    n=st.integers(2, 50),
    seed=st.integers(0, 9999),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much])
def test_unique_texts_produce_no_errors(n: int, seed: int) -> None:
    """A split with all-unique texts has no error-severity findings."""
    texts = [f"unique_text_{seed}_{i}" for i in range(n)]
    labels = [i % 2 for i in range(n)]
    splits = {"test": _build_slice("test", texts, labels)}
    report = run_leakage_checks(
        [
            ExactDuplicateCheck(),
            NormalizedFormLeakageCheck(),
            LabelConflictCheck(),
        ],
        splits,
    )
    assert not report.has_errors()


# ---------------------------------------------------------------------------
# NormalizedFormLeakageCheck — encoding obfuscation always caught
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    base=st.text(
        min_size=5, max_size=30, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    n_zw=st.integers(1, 5),
)
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_zero_width_injection_collides_with_base(base: str, n_zw: int) -> None:
    """A zero-width-injected variant of `base` collides with `base` under normalization."""
    # Inject n_zw zero-width characters at random positions.
    zwj = "‍"  # ZERO WIDTH JOINER
    obfuscated = zwj.join(list(base[: n_zw + 1])) + base[n_zw + 1 :]
    if obfuscated == base:
        return  # degenerate
    splits = {
        "test": _build_slice("test", [base, obfuscated, "totally_unrelated"], [0, 1, 0]),
    }
    finding = NormalizedFormLeakageCheck().validate(splits)
    assert isinstance(finding, LeakageFinding)
    # Either the zero-width strip catches the collision, or the normalized
    # forms genuinely don't match (depends on the unicode category of the
    # base characters).
    assert finding.severity == "error"


@pytest.mark.property
@given(text=st.text(min_size=0, max_size=100))
@settings(deadline=None, max_examples=20)
def test_aggressive_normalize_idempotent(text: str) -> None:
    """Applying _aggressive_normalize twice equals applying it once."""
    once = _aggressive_normalize(text)
    twice = _aggressive_normalize(once)
    assert once == twice


# ---------------------------------------------------------------------------
# LeakageFinding round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    name=st.text(
        min_size=1, max_size=30, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    severity=st.sampled_from(["error", "warning", "info"]),
    n_affected=st.integers(0, 100),
)
@settings(deadline=None, max_examples=20)
def test_leakage_finding_to_dict_preserves_fields(
    name: str, severity: str, n_affected: int
) -> None:
    """LeakageFinding.to_dict() keys match the dataclass fields."""
    f = LeakageFinding(
        check_name=name,
        severity=severity,  # type: ignore[arg-type]
        drop_indices={"test": [0, 1]},
        evidence={"foo": "bar"},
        message="msg",
        n_affected=n_affected,
    )
    d = f.to_dict()
    for key in ("check_name", "severity", "drop_indices", "evidence", "message", "n_affected"):
        assert key in d
    assert d["check_name"] == name
    assert d["severity"] == severity
    assert d["n_affected"] == n_affected


# ---------------------------------------------------------------------------
# merged_drop_indices invariants
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_merged_drop_indices_unions_findings() -> None:
    """LeakageReport.merged_drop_indices unions per-split indices across findings."""
    df = pd.DataFrame({"text": ["dup", "dup", "x"], "label": [0, 1, 0]})
    splits = {"test": EvalSlice(name="test", df=df)}
    # Both checks may flag overlapping indices; merged should be the union.
    report = run_leakage_checks(
        [ExactDuplicateCheck(severity="error"), NormalizedFormLeakageCheck()],
        splits,
    )
    merged = report.merged_drop_indices()
    if "test" in merged:
        # Verify union: every index in any finding's drop_indices for "test"
        # is in the merged result.
        all_indices: set[int] = set()
        for f in report.findings:
            all_indices.update(f.drop_indices.get("test", []))
        assert set(merged["test"]) == all_indices
        # Sorted invariant
        assert merged["test"] == sorted(merged["test"])


# ---------------------------------------------------------------------------
# GroupLeakageCheck — cross-split detection
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n_train=st.integers(3, 20),
    n_test=st.integers(2, 20),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_group_leakage_caught_when_group_spans(n_train: int, n_test: int) -> None:
    """If a group ID appears in both train and test, GroupLeakageCheck catches it."""
    # Group 0 appears in BOTH splits → must be caught.
    train_df = pd.DataFrame(
        {
            "text": [f"t_{i}" for i in range(n_train)],
            "label": [i % 2 for i in range(n_train)],
            "group_id": [0] + list(range(1, n_train)),
        }
    )
    test_df = pd.DataFrame(
        {
            "text": [f"e_{i}" for i in range(n_test)],
            "label": [i % 2 for i in range(n_test)],
            "group_id": [0] + list(range(100, 100 + n_test - 1)),
        }
    )
    splits = {
        "train": EvalSlice(name="train", df=train_df),
        "test": EvalSlice(name="test", df=test_df),
    }
    finding = GroupLeakageCheck(group_col="group_id").validate(splits)
    assert finding.severity == "error"
    assert finding.n_affected >= 2  # at least the train row + test row of group 0
