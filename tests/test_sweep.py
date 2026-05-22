"""Tests for top-level ``eval_toolkit.sweep`` (v0.47 — Decisions K + D).

Coverage:

- Pure neutral text-transform enumeration (defence-only / attack-only / mixed)
- Scorer integration (``original_score`` + ``transformed_score`` columns)
- Explicit ``attack_threshold`` → ``asr`` column materialization
- Error paths (empty strategies, threshold-without-scorer, malformed strategy)

The Sub-PR 6 parity-against-module-sweeps tests were removed when the
module-level ``preprocessing.sweep`` / ``adversarial.sweep`` themselves were
removed at v0.47 (Decision N + plan §4E).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_toolkit import (
    DatamarkVariant,
    DelimitVariant,
    EncodeVariant,
    TextTransform,
    sweep,
)
from eval_toolkit.adversarial import (
    ZeroWidthSpaceInjection,
)

# ─────────────────────────────────────────────────────────────────────────────
# Basic shape: text_id × variant × transformed_text
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_defence_only_returns_text_columns() -> None:
    """Default call (no scorer) returns 4 columns: id, strategy_id, variant, text.

    v0.48 Decision R7-B (§5I): `strategy_id` is the canonical per-row
    identifier carrying the strategy's configured kwargs; `variant` keeps
    the dataclass `.name` attribute for backward-compat grouping.
    """
    df = sweep([DelimitVariant(), DatamarkVariant(), EncodeVariant()], ["hello"])
    assert df.shape == (3, 4)
    assert list(df.columns) == ["text_id", "strategy_id", "variant", "transformed_text"]
    assert set(df["variant"]) == {"delimit", "datamark", "encode"}


def test_sweep_attack_only() -> None:
    """Attack-side strategies work the same way (TextTransform structural sharing)."""
    df = sweep([ZeroWidthSpaceInjection()], ["payload"])
    assert df.shape == (1, 4)
    assert df.iloc[0]["variant"] == "zero_width_space"


def test_sweep_mixed_defence_attack() -> None:
    """Defence + attack strategies compose in one call (Decision K key win)."""
    df = sweep(
        [DelimitVariant(), ZeroWidthSpaceInjection()],
        ["hello", "world"],
    )
    assert df.shape == (4, 4)
    assert set(df["variant"]) == {"delimit", "zero_width_space"}


def test_sweep_preserves_text_id_order() -> None:
    """text_id is the 0-based input position; consistent across strategies."""
    df = sweep([DelimitVariant()], ["a", "b", "c"])
    assert df["text_id"].tolist() == [0, 1, 2]


def test_sweep_variant_rows_grouped_by_strategy() -> None:
    """All rows for strategy[0] come before strategy[1] (variant-major order)."""
    df = sweep([DelimitVariant(), DatamarkVariant()], ["a", "b"])
    # Order: delimit-a, delimit-b, datamark-a, datamark-b
    assert df.iloc[0]["variant"] == "delimit"
    assert df.iloc[1]["variant"] == "delimit"
    assert df.iloc[2]["variant"] == "datamark"
    assert df.iloc[3]["variant"] == "datamark"


# ─────────────────────────────────────────────────────────────────────────────
# Scorer integration (opt-in)
# ─────────────────────────────────────────────────────────────────────────────


class _FixedScorer:
    """Deterministic toy scorer: 0.9 if 'ignore' in text else 0.1."""

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return np.array([0.9 if "ignore" in t else 0.1 for t in X])


def test_sweep_with_scorer_adds_score_columns() -> None:
    df = sweep([DelimitVariant()], ["ignore me", "hello"], scorer=_FixedScorer())
    assert "original_score" in df.columns
    assert "transformed_score" in df.columns
    assert "asr" not in df.columns  # no threshold → no ASR
    # Original scores match what _FixedScorer would emit
    assert df[df["text_id"] == 0]["original_score"].iloc[0] == 0.9
    assert df[df["text_id"] == 1]["original_score"].iloc[0] == 0.1


def test_sweep_attack_threshold_adds_asr_column() -> None:
    df = sweep(
        [DelimitVariant()],
        ["ignore me", "hello"],
        scorer=_FixedScorer(),
        attack_threshold=0.5,
    )
    assert "asr" in df.columns
    # DelimitVariant("ignore me") → "<<ignore me>>"; _FixedScorer still scores 0.9
    # So ASR is False (no degradation past threshold).
    assert not bool(df[df["text_id"] == 0]["asr"].iloc[0])


def test_sweep_attack_threshold_actually_flags_success() -> None:
    """Construct a strategy that DOES strip the trigger word so ASR fires."""

    class _StripIgnore:
        name = "strip_ignore"

        def transform(self, text: str) -> str:
            return text.replace("ignore", "")

    df = sweep(
        [_StripIgnore()],
        ["ignore me", "hello"],
        scorer=_FixedScorer(),
        attack_threshold=0.5,
    )
    # text_id=0: original=0.9 (>=0.5), transformed=" me" (no "ignore") → 0.1 (<0.5) → ASR=True
    # text_id=1: original=0.1 (<0.5), gate fails → ASR=False
    assert bool(df[df["text_id"] == 0]["asr"].iloc[0])
    assert not bool(df[df["text_id"] == 1]["asr"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_empty_strategies_raises() -> None:
    with pytest.raises(ValueError, match="strategies must be non-empty"):
        sweep([], ["x"])


def test_sweep_threshold_without_scorer_raises() -> None:
    with pytest.raises(ValueError, match="attack_threshold requires scorer"):
        sweep([DelimitVariant()], ["x"], attack_threshold=0.5)


def test_sweep_malformed_strategy_raises() -> None:
    class _BadStrategy:
        # Missing the .name attribute
        def transform(self, text: str) -> str:
            return text

    with pytest.raises(ValueError, match="does not satisfy TextTransform"):
        sweep([_BadStrategy()], ["x"])


# ─────────────────────────────────────────────────────────────────────────────
# Custom user strategy via Protocol
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_accepts_custom_text_transform() -> None:
    """Any object satisfying TextTransform structurally can be passed in."""

    class _ReverseStrategy:
        name = "reverse"

        def transform(self, text: str) -> str:
            return text[::-1]

    df = sweep([_ReverseStrategy()], ["hello"])
    assert df.iloc[0]["transformed_text"] == "olleh"
    assert isinstance(_ReverseStrategy(), TextTransform)


def test_sweep_returns_pandas_dataframe() -> None:
    df = sweep([DelimitVariant()], ["x"])
    assert isinstance(df, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Decision R7-B / §5I (v0.48): strategy_id column + duplicate rejection.
#
# Style-coherent with R6-B (scorecard duplicate MetricSpec.name rejection):
# canonical identifier + reject duplicates IN THE CANONICAL DIMENSION.
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_strategy_id_distinguishes_configurations() -> None:
    """Two DelimitVariant instances with different delimiters share `variant`
    but have distinct `strategy_id`. The whole point of R7-B.
    """
    df = sweep(
        [DelimitVariant(delimiter="<<"), DelimitVariant(delimiter="[[")],
        ["hello"],
    )
    assert df.shape == (2, 4)
    # Both rows share variant
    assert set(df["variant"]) == {"delimit"}
    # But strategy_id distinguishes them
    sids = df["strategy_id"].tolist()
    assert sids[0] != sids[1]
    assert "delimiter='<<'" in sids[0]
    assert "delimiter='[['" in sids[1]


def test_sweep_strategy_id_distinguishes_ratios() -> None:
    """Same mechanism for adversarial-side dataclasses with numeric kwargs."""
    df = sweep(
        [ZeroWidthSpaceInjection(ratio=0.1), ZeroWidthSpaceInjection(ratio=0.9)],
        ["hello"],
    )
    assert df.shape == (2, 4)
    sids = df["strategy_id"].tolist()
    assert "ratio=0.1" in sids[0]
    assert "ratio=0.9" in sids[1]


def test_sweep_rejects_duplicate_strategy_id() -> None:
    """Two strategy instances with identical configured state must raise."""
    with pytest.raises(ValueError, match="duplicate strategy_id"):
        sweep([DelimitVariant(), DelimitVariant()], ["hello"])


def test_sweep_duplicate_rejection_reports_both_indices() -> None:
    """The error message names both the duplicate and the original index."""
    with pytest.raises(ValueError) as excinfo:
        sweep(
            [DelimitVariant(delimiter="<<"), DatamarkVariant(), DelimitVariant(delimiter="<<")],
            ["hello"],
        )
    msg = str(excinfo.value)
    assert "index 2" in msg
    assert "index 0" in msg


def test_sweep_duplicate_rejection_catches_identical_instances() -> None:
    """Passing the SAME instance twice is the simplest duplicate case."""
    s = DelimitVariant()
    with pytest.raises(ValueError, match="duplicate strategy_id"):
        sweep([s, s], ["hello"])


def test_sweep_groupby_strategy_id_disambiguates() -> None:
    """The canonical use case: groupby('strategy_id') for per-config rollup.

    Regression-guard: this is the analysis pattern R7-B exists to enable.
    """
    df = sweep(
        [
            DelimitVariant(delimiter="<<"),
            DelimitVariant(delimiter="[["),
            DelimitVariant(delimiter="(("),
        ],
        ["a", "b"],
    )
    # 3 configs × 2 texts = 6 rows
    assert len(df) == 6
    # groupby('variant') merges everything (the silent-merge footgun
    # R7-B closes)
    by_variant = df.groupby("variant").size()
    assert by_variant.shape == (1,)  # all 6 rows under variant='delimit'
    # groupby('strategy_id') distinguishes the 3 configs cleanly
    by_id = df.groupby("strategy_id").size()
    assert by_id.shape == (3,)
    assert (by_id == 2).all()  # 2 texts per strategy_id


def test_sweep_strategy_id_for_plain_protocol() -> None:
    """Non-dataclass TextTransform-Protocol implementations fall back to `name`.

    The strategy_id helper has no __dataclass_fields__ to alphabetize, so
    it just returns strategy.name. Two plain implementations with the same
    name → rejection. Two with distinct names → distinct strategy_ids.
    """

    class _Reverse:
        name = "reverse"

        def transform(self, text: str) -> str:
            return text[::-1]

    class _Upper:
        name = "upper"

        def transform(self, text: str) -> str:
            return text.upper()

    df = sweep([_Reverse(), _Upper()], ["hello"])
    assert df["strategy_id"].tolist() == ["reverse", "upper"]


def test_sweep_strategy_id_argument_order_invariant() -> None:
    """A strategy's strategy_id doesn't depend on kwarg construction order.

    Mirrors the metric_specs.make_spec_name argument-order-invariance contract.
    """
    a = DelimitVariant(delimiter="<<", end=">>")
    b = DelimitVariant(end=">>", delimiter="<<")
    from eval_toolkit._sweep import _strategy_id_for

    assert _strategy_id_for(a) == _strategy_id_for(b)
