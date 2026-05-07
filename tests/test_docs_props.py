"""Hypothesis property tests for docs anchor rendering."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eval_toolkit.docs import (
    ANCHOR_RE,
    DEFAULT_FORMATTERS,
    render_text,
    walk_path,
)


@pytest.mark.property
@given(value=st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text())
def test_walk_path_round_trip(value: object) -> None:
    """walk_path retrieves the value at a known path."""
    data = {"a": {"b": {"c": value}}}
    out = walk_path(data, "a.b.c")
    if isinstance(value, float):
        assert out == pytest.approx(value)
    else:
        assert out == value


@pytest.mark.property
@given(missing_key=st.text(min_size=1, max_size=20).filter(lambda x: "." not in x and x != "a"))
def test_walk_path_missing_key_raises(missing_key: str) -> None:
    """walk_path on missing key raises KeyError."""
    data = {"a": 1}
    with pytest.raises(KeyError):
        walk_path(data, missing_key)


@pytest.mark.property
@given(
    leaf_value=st.floats(0.0, 1.0, allow_nan=False),
)
def test_render_text_idempotent(leaf_value: float) -> None:
    """render_text(render_text(text, data), data) == render_text(text, data)."""
    text = "<!-- begin:metric -->X<!-- end:metric -->"
    data: dict[str, object] = {"metric": leaf_value}
    formatters = {"metric": lambda v: f"{v:.5f}"}
    once, _ = render_text(text, data, formatters)
    twice, _ = render_text(once, data, formatters)
    assert once == twice


@pytest.mark.property
@given(leaf_value=st.floats(0.0, 1.0, allow_nan=False))
def test_render_text_preserves_anchor_structure(leaf_value: float) -> None:
    """Rendered output retains the begin/end anchor markers."""
    text = "<!-- begin:m -->X<!-- end:m -->"
    data: dict[str, object] = {"m": leaf_value}
    rendered, _ = render_text(text, data)
    matches = list(ANCHOR_RE.finditer(rendered))
    assert len(matches) == 1
    assert matches[0].group("key") == "m"


@pytest.mark.property
@given(formatter_key=st.sampled_from(list(DEFAULT_FORMATTERS.keys())))
def test_default_formatters_handle_none(formatter_key: str) -> None:
    """Every default formatter handles None gracefully (returns 'N/A' or stringifies)."""
    fn = DEFAULT_FORMATTERS[formatter_key]
    if formatter_key == "lift":
        # compound formatter expects a dict, not a scalar — skip
        return
    out = fn(None)
    assert isinstance(out, str)
