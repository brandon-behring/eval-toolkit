"""Error-path coverage for ``eval_toolkit.leakage`` (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``leakage.py`` has 8 ``raise KeyError`` sites and zero tests
asserting they fire as expected. Each test in this module:

1. Builds a minimal ``EvalSlice``-keyed splits dict that violates
   one specific check's pre-conditions.
2. Invokes the check.
3. Asserts ``pytest.raises(KeyError, match="<expected_phrase>")``.

The ``match=`` regex assertions also pin the error message phrasing
so a regression that silently changes the exception text (and
breaks downstream consumers parsing error messages) would fail
this suite.

Coverage matrix (raise-site → test):

| File:line in leakage.py | Test |
|---|---|
| L257 (_select_targets) | test_target_splits_validation_raises_keyerror |
| L687 (CrossSplitLeakageCheck.validate, train) | test_cross_split_missing_train_split_raises_keyerror |
| L699 (CrossSplitLeakageCheck.validate, eval) | test_cross_split_missing_eval_split_raises_keyerror |
| L751 (validate_label_split, train) | test_validate_label_split_missing_train_raises_keyerror |
| L769 (validate_label_split, eval) | test_validate_label_split_missing_eval_raises_keyerror |
| L882 (GroupLeakageCheck.validate) | test_group_leakage_missing_group_column_raises_keyerror |
| L949 (TemporalLeakageCheck.validate, split) | test_temporal_leakage_missing_split_raises_keyerror |
| L951 (TemporalLeakageCheck.validate, time_col) | test_temporal_leakage_missing_time_column_raises_keyerror |
"""

from __future__ import annotations

import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.leakage import (
    CrossSplitLeakageCheck,
    ExactDuplicateCheck,
    GroupLeakageCheck,
    TemporalLeakageCheck,
)


def _make_slice(name: str, *, n: int = 4) -> EvalSlice:
    """Build a minimal EvalSlice with text + binary label columns."""
    return EvalSlice(
        name=name,
        df=pd.DataFrame(
            {
                "text": [f"{name}_text_{i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
            }
        ),
    )


def test_target_splits_validation_raises_keyerror() -> None:
    """``ExactDuplicateCheck(target_splits=...)`` raises on a missing target.

    Covers ``leakage.py:257`` — the ``_select_targets`` helper, used
    by every within-split check that accepts an explicit
    ``target_splits`` filter.
    """
    splits = {"train": _make_slice("train"), "test": _make_slice("test")}
    check = ExactDuplicateCheck(target_splits=("train", "test", "missing_split"))
    with pytest.raises(KeyError, match="target_splits not in splits"):
        check.validate(splits)


def test_cross_split_missing_train_split_raises_keyerror() -> None:
    """``CrossSplitLeakageCheck.validate`` raises when ``train_split`` is absent.

    Covers ``leakage.py:687``.
    """
    splits = {"test": _make_slice("test")}  # no "train" key
    check = CrossSplitLeakageCheck(train_split="train")
    with pytest.raises(KeyError, match="train_split.*not in splits"):
        check.validate(splits)


def test_cross_split_missing_eval_split_raises_keyerror() -> None:
    """``CrossSplitLeakageCheck.validate`` raises when an explicit eval split is absent.

    Covers ``leakage.py:699``.
    """
    splits = {"train": _make_slice("train")}  # no "test" key
    check = CrossSplitLeakageCheck(train_split="train", eval_splits=("test",))
    with pytest.raises(KeyError, match="eval split.*not in splits"):
        check.validate(splits)


def test_validate_label_split_missing_train_raises_keyerror() -> None:
    """``CrossSplitLeakageCheck.validate_label_split`` raises on missing train.

    Covers ``leakage.py:751``.
    """
    splits = {"test": _make_slice("test")}
    check = CrossSplitLeakageCheck(train_split="train")
    with pytest.raises(KeyError, match="train_split.*not in splits"):
        check.validate_label_split(splits)


def test_validate_label_split_missing_eval_raises_keyerror() -> None:
    """``CrossSplitLeakageCheck.validate_label_split`` raises on missing eval.

    Covers ``leakage.py:769``.
    """
    splits = {"train": _make_slice("train")}
    check = CrossSplitLeakageCheck(train_split="train", eval_splits=("test",))
    with pytest.raises(KeyError, match="eval split.*not in splits"):
        check.validate_label_split(splits)


def test_group_leakage_missing_group_column_raises_keyerror() -> None:
    """``GroupLeakageCheck.validate`` raises when ``group_col`` is absent in a target split.

    Covers ``leakage.py:882``.
    """
    # Build a slice whose dataframe has text + label but no "group" column.
    splits = {"train": _make_slice("train"), "test": _make_slice("test")}
    check = GroupLeakageCheck(group_col="group")
    with pytest.raises(KeyError, match="missing group column"):
        check.validate(splits)


def test_temporal_leakage_missing_split_raises_keyerror() -> None:
    """``TemporalLeakageCheck.validate`` raises when a ``split_order`` entry is absent.

    Covers ``leakage.py:949``. The fixture must ensure earlier splits in
    ``split_order`` *do* have the configured ``time_col`` (otherwise the
    ``missing time column`` raise on L951 fires first), so the missing-
    split raise on L949 can be reached by a later entry.
    """
    df_with_ts = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "ts": [1, 2, 3, 4],
        }
    )
    splits = {"train": EvalSlice(name="train", df=df_with_ts)}  # no "val" key
    check = TemporalLeakageCheck(time_col="ts", split_order=("train", "val"))
    with pytest.raises(KeyError, match=r"in split_order.*not in splits"):
        check.validate(splits)


def test_temporal_leakage_missing_time_column_raises_keyerror() -> None:
    """``TemporalLeakageCheck.validate`` raises when ``time_col`` is absent in a split.

    Covers ``leakage.py:951``.
    """
    # Splits exist, but neither has the configured "ts" column.
    splits = {"train": _make_slice("train"), "val": _make_slice("val")}
    check = TemporalLeakageCheck(time_col="ts", split_order=("train", "val"))
    with pytest.raises(KeyError, match="missing time column"):
        check.validate(splits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
