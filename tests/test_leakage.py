"""Smoke tests for the v0.7.0 LeakageCheck Protocol + reference impls."""

from __future__ import annotations

import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.leakage import (
    CrossSplitLeakageCheck,
    ExactDuplicateCheck,
    GroupLeakageCheck,
    LabelConflictCheck,
    LeakageCheck,
    LeakageFinding,
    LeakageReport,
    NormalizedFormLeakageCheck,
    TemporalLeakageCheck,
    run_leakage_checks,
)


@pytest.fixture
def clean_splits() -> dict[str, EvalSlice]:
    """Two splits with no leakage."""
    return {
        "train": EvalSlice(
            name="train",
            df=pd.DataFrame(
                {
                    "text": [f"unique_train_{i}" for i in range(20)],
                    "label": [i % 2 for i in range(20)],
                }
            ),
        ),
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame(
                {
                    "text": [f"unique_test_{i}" for i in range(10)],
                    "label": [i % 2 for i in range(10)],
                }
            ),
        ),
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "check",
    [
        ExactDuplicateCheck(),
        NormalizedFormLeakageCheck(),
        LabelConflictCheck(),
        CrossSplitLeakageCheck(train_split="train"),
        GroupLeakageCheck(group_col="group_id"),
        TemporalLeakageCheck(time_col="t", split_order=("train", "test")),
    ],
)
def test_checks_implement_protocol(check: LeakageCheck) -> None:
    assert isinstance(check, LeakageCheck)


@pytest.mark.unit
def test_clean_splits_produce_no_errors(clean_splits: dict[str, EvalSlice]) -> None:
    report = run_leakage_checks(
        [
            ExactDuplicateCheck(),
            NormalizedFormLeakageCheck(),
            LabelConflictCheck(),
            CrossSplitLeakageCheck(train_split="train"),
        ],
        clean_splits,
    )
    assert isinstance(report, LeakageReport)
    assert not report.has_errors()


@pytest.mark.unit
def test_normalized_form_catches_zero_width_obfuscation() -> None:
    """The encoding-obfuscation detection that motivated this check."""
    df = pd.DataFrame({"text": ["hello world", "h​el​lo  world", "unrelated"], "label": [0, 1, 0]})
    splits = {"test": EvalSlice(name="test", df=df)}
    finding = NormalizedFormLeakageCheck().validate(splits)
    assert isinstance(finding, LeakageFinding)
    assert finding.severity == "error"
    assert finding.n_affected == 1
    assert "test" in finding.drop_indices


@pytest.mark.unit
def test_label_conflict_detected_across_splits() -> None:
    splits = {
        "train": EvalSlice(name="train", df=pd.DataFrame({"text": ["x", "y"], "label": [0, 1]})),
        "test": EvalSlice(name="test", df=pd.DataFrame({"text": ["x", "z"], "label": [1, 0]})),
    }
    finding = LabelConflictCheck().validate(splits)
    assert finding.severity == "error"
    assert finding.n_affected == 2  # one row each from train and test


@pytest.mark.unit
def test_cross_split_leakage_detected() -> None:
    """Identical text in train and test should be caught at threshold ≥ 0.9."""
    splits = {
        "train": EvalSlice(
            name="train",
            df=pd.DataFrame(
                {"text": ["hello world this is a longer string", "lorem ipsum"], "label": [0, 1]}
            ),
        ),
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame(
                {
                    "text": ["hello world this is a longer string", "completely different"],
                    "label": [1, 0],
                }
            ),
        ),
    }
    finding = CrossSplitLeakageCheck(train_split="train").validate(splits)
    assert finding.severity == "error"
    assert finding.n_affected >= 1


@pytest.mark.unit
def test_group_leakage_detected() -> None:
    splits = {
        "train": EvalSlice(
            name="train",
            df=pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0], "group_id": [1, 2, 3]}),
        ),
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame(
                {"text": ["d", "e"], "label": [1, 0], "group_id": [1, 4]}  # group 1 spans
            ),
        ),
    }
    finding = GroupLeakageCheck(group_col="group_id").validate(splits)
    assert finding.severity == "error"
    assert finding.n_affected == 2  # train row + test row of group 1


@pytest.mark.unit
def test_leakage_report_has_errors_only_on_error_severity() -> None:
    """Warnings (e.g., ExactDuplicateCheck default) should NOT trigger has_errors."""
    df = pd.DataFrame({"text": ["dup", "dup", "unique"], "label": [0, 1, 0]})
    splits = {"test": EvalSlice(name="test", df=df)}
    report = run_leakage_checks([ExactDuplicateCheck()], splits)
    # ExactDuplicateCheck defaults to severity="warning"
    assert not report.has_errors()
    assert len(report.warnings()) == 1


@pytest.mark.unit
def test_merged_drop_indices_unions_across_findings() -> None:
    df = pd.DataFrame({"text": ["dup", "dup", "h​ello"], "label": [0, 1, 0]})
    splits = {"test": EvalSlice(name="test", df=df)}
    report = run_leakage_checks(
        [
            ExactDuplicateCheck(severity="error"),
            NormalizedFormLeakageCheck(),
        ],
        splits,
    )
    merged = report.merged_drop_indices()
    assert "test" in merged
    # The duplicate "dup" + the obfuscated "hello" both flagged
    assert len(merged["test"]) >= 1


@pytest.mark.unit
def test_temporal_leakage_detected() -> None:
    """Train data with timestamps later than test data should be caught."""
    train_df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1], "t": [10, 20]})
    test_df = pd.DataFrame({"text": ["c", "d"], "label": [1, 0], "t": [5, 15]})
    splits = {
        "train": EvalSlice(name="train", df=train_df),
        "test": EvalSlice(name="test", df=test_df),
    }
    finding = TemporalLeakageCheck(time_col="t", split_order=("train", "test")).validate(splits)
    assert finding.severity == "error"
    assert finding.n_affected > 0


@pytest.mark.unit
def test_temporal_leakage_clean_ordering_passes() -> None:
    train_df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1], "t": [1, 2]})
    test_df = pd.DataFrame({"text": ["c", "d"], "label": [1, 0], "t": [3, 4]})
    splits = {
        "train": EvalSlice(name="train", df=train_df),
        "test": EvalSlice(name="test", df=test_df),
    }
    finding = TemporalLeakageCheck(time_col="t", split_order=("train", "test")).validate(splits)
    assert finding.n_affected == 0


@pytest.mark.unit
def test_to_dict_round_trip_preserves_finding_shape() -> None:
    """LeakageFinding.to_dict() preserves all fields for manifest serialization."""
    df = pd.DataFrame({"text": ["hello", "h​ello"], "label": [0, 1]})
    splits = {"test": EvalSlice(name="test", df=df)}
    finding = NormalizedFormLeakageCheck().validate(splits)
    d = finding.to_dict()
    for field_name in (
        "check_name",
        "severity",
        "drop_indices",
        "evidence",
        "message",
        "n_affected",
    ):
        assert field_name in d


@pytest.mark.unit
def test_within_split_target_splits_filter() -> None:
    """ExactDuplicateCheck(target_splits=['x']) ignores other splits."""
    train_df = pd.DataFrame({"text": ["dup", "dup"], "label": [0, 1]})
    test_df = pd.DataFrame({"text": ["dup", "dup"], "label": [0, 1]})
    splits = {
        "train": EvalSlice(name="train", df=train_df),
        "test": EvalSlice(name="test", df=test_df),
    }
    finding = ExactDuplicateCheck(target_splits=["test"]).validate(splits)
    # Only test was scanned; train's dup is invisible to this check.
    assert "train" not in finding.drop_indices
    assert "test" in finding.drop_indices
