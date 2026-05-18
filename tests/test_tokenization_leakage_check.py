"""Tests for ``TokenizationLeakageCheck`` (v0.37.0, closes #35)."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from eval_toolkit.harness import EvalSlice
from eval_toolkit.leakage import (
    LeakageCheck,
    LeakageFinding,
    TokenizationLeakageCheck,
)


def _byte_tokenizer(text: str) -> Mapping[str, object]:
    """Deterministic byte-level "tokenizer" for tests.

    Returns the lowercased bytes as ``input_ids``. Two texts that lower-
    case to the same string collide post-tokenization — mirrors the
    encoding-obfuscation use case without needing real ``transformers``.
    """
    return {"input_ids": list(text.lower().encode("utf-8"))}


def _batched_byte_tokenizer(text: str) -> Mapping[str, object]:
    """Same as ``_byte_tokenizer`` but emits a one-level-nested batched form.

    Some tokenizer wrappers always emit ``[[ids...]]`` even for a single
    string. The check should unwrap that.
    """
    return {"input_ids": [list(text.lower().encode("utf-8"))]}


@pytest.mark.unit
def test_implements_leakage_check_protocol() -> None:
    check = TokenizationLeakageCheck(tokenizer=_byte_tokenizer)
    assert isinstance(check, LeakageCheck)
    assert check.name == "TokenizationLeakageCheck"


@pytest.mark.unit
def test_clean_split_produces_no_findings() -> None:
    """Distinct texts → no collisions → empty drop_indices."""
    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["alpha", "beta", "gamma"], "label": [0, 1, 0]}),
        )
    }
    finding = TokenizationLeakageCheck(tokenizer=_byte_tokenizer).validate(splits)
    assert isinstance(finding, LeakageFinding)
    assert finding.n_affected == 0
    assert finding.drop_indices == {}
    assert "no tokenization-level duplicates" in finding.message


@pytest.mark.unit
def test_collisions_detected_via_lowercase_collision() -> None:
    """Texts differing only by case tokenize to identical input_ids."""
    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame(
                {
                    "text": ["Hello", "hello", "world", "WORLD"],
                    "label": [0, 1, 0, 1],
                }
            ),
        )
    }
    finding = TokenizationLeakageCheck(tokenizer=_byte_tokenizer).validate(splits)
    assert finding.n_affected == 2  # "hello" (dup of "Hello") + "WORLD" (dup of "world")
    assert finding.drop_indices == {"test": [1, 3]}
    assert finding.severity == "error"
    collisions = finding.evidence["collisions_by_split"]
    assert collisions["test"] == [(1, 0), (3, 2)]


@pytest.mark.unit
def test_unwraps_batched_output() -> None:
    """Tokenizer that emits ``[[ids...]]`` should produce same dedup as flat form."""
    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["Hello", "hello"], "label": [0, 1]}),
        )
    }
    flat = TokenizationLeakageCheck(tokenizer=_byte_tokenizer).validate(splits)
    batched = TokenizationLeakageCheck(tokenizer=_batched_byte_tokenizer).validate(splits)
    assert flat.n_affected == batched.n_affected == 1
    assert flat.drop_indices == batched.drop_indices == {"test": [1]}


@pytest.mark.unit
def test_default_severity_is_error() -> None:
    """Tokenization-level overlap defaults to error (mirrors NormalizedFormLeakageCheck)."""
    check = TokenizationLeakageCheck(tokenizer=_byte_tokenizer)
    assert check.severity == "error"


@pytest.mark.unit
def test_severity_override_to_warning() -> None:
    check = TokenizationLeakageCheck(tokenizer=_byte_tokenizer, severity="warning")
    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["X", "x"], "label": [0, 1]}),
        )
    }
    finding = check.validate(splits)
    assert finding.severity == "warning"
    assert finding.n_affected == 1


@pytest.mark.unit
def test_target_splits_restricts_scope() -> None:
    """target_splits=['test'] should ignore 'train'."""
    splits = {
        "train": EvalSlice(
            name="train",
            df=pd.DataFrame({"text": ["Foo", "foo"], "label": [0, 1]}),
        ),
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["Bar", "bar"], "label": [0, 1]}),
        ),
    }
    finding = TokenizationLeakageCheck(tokenizer=_byte_tokenizer, target_splits=["test"]).validate(
        splits
    )
    assert set(finding.drop_indices.keys()) == {"test"}
    assert finding.n_affected == 1  # only "bar" from the test split


@pytest.mark.unit
def test_target_splits_missing_raises_keyerror() -> None:
    splits = {"test": EvalSlice(name="test", df=pd.DataFrame({"text": ["a"], "label": [0]}))}
    check = TokenizationLeakageCheck(tokenizer=_byte_tokenizer, target_splits=["nonexistent"])
    with pytest.raises(KeyError, match="nonexistent"):
        check.validate(splits)


@pytest.mark.unit
def test_single_row_split_does_not_crash() -> None:
    """Splits with <= 1 row are skipped (no possible collision)."""
    splits = {"test": EvalSlice(name="test", df=pd.DataFrame({"text": ["only"], "label": [0]}))}
    finding = TokenizationLeakageCheck(tokenizer=_byte_tokenizer).validate(splits)
    assert finding.n_affected == 0
    assert finding.drop_indices == {}


@pytest.mark.unit
def test_empty_tokenizer_output_yields_empty_tuple_key() -> None:
    """A tokenizer returning {'input_ids': []} should treat empty as a valid key.

    Two rows that both tokenize to empty should collide.
    """

    def empty_tokenizer(text: str) -> Mapping[str, object]:
        return {"input_ids": []}

    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]}),
        )
    }
    finding = TokenizationLeakageCheck(tokenizer=empty_tokenizer).validate(splits)
    # All three tokenize to the empty key — rows 1 and 2 are dropped.
    assert finding.n_affected == 2
    assert finding.drop_indices == {"test": [1, 2]}


@pytest.mark.unit
def test_dataclass_is_frozen() -> None:
    """Following the LeakageCheck convention (frozen=True, slots=True)."""
    check = TokenizationLeakageCheck(tokenizer=_byte_tokenizer)
    with pytest.raises((AttributeError, TypeError)):
        check.severity = "info"  # type: ignore[misc]


@pytest.mark.unit
def test_message_format_when_dupes_found() -> None:
    splits = {
        "test": EvalSlice(
            name="test",
            df=pd.DataFrame({"text": ["A", "a"], "label": [0, 1]}),
        )
    }
    finding = TokenizationLeakageCheck(tokenizer=_byte_tokenizer).validate(splits)
    assert "tokenization-level duplicates: 1 rows" in finding.message
