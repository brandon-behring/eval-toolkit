"""Hypothesis property tests for docs anchor rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eval_toolkit.docs import (
    ANCHOR_RE,
    DEFAULT_FORMATTERS,
    render_files,
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


# v0.8.2: cover the walk_path branches not exercised by happy-path tests
# (list-index path + non-int index error + non-descendable type).
@pytest.mark.unit
def test_walk_path_list_index_branch() -> None:
    """walk_path traverses through a list when the path component is an int."""
    data = {"items": [{"v": 10}, {"v": 20}, {"v": 30}]}
    assert walk_path(data, "items.0.v") == 10
    assert walk_path(data, "items.2.v") == 30


@pytest.mark.unit
def test_walk_path_non_int_list_index_raises_keyerror() -> None:
    """A non-integer path component on a list yields KeyError, not ValueError."""
    data = {"items": [1, 2, 3]}
    with pytest.raises(KeyError, match="non-integer index"):
        walk_path(data, "items.foo")


@pytest.mark.unit
def test_walk_path_descend_into_scalar_raises_keyerror() -> None:
    """Trying to descend into a scalar (str/int) raises KeyError with a useful message."""
    data = {"a": 42}
    with pytest.raises(KeyError, match="cannot descend into"):
        walk_path(data, "a.b")


# v0.8.2: cover render_text/render_files type-validation guards
# (docs.py:209-212) and the multi-file errors-aggregation path
# (docs.py:270, render_files mode='check').
@pytest.mark.unit
def test_render_text_rejects_non_str_text() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        render_text(123, {"a": 1})  # type: ignore[arg-type]


@pytest.mark.unit
def test_render_text_rejects_non_dict_metrics() -> None:
    with pytest.raises(TypeError, match="metrics must be a dict"):
        render_text("<!-- begin:a -->X<!-- end:a -->", "not a dict")  # type: ignore[arg-type]


@pytest.mark.unit
def test_render_files_check_mode_collects_errors_and_drift(tmp_path: Path) -> None:
    """render_files(mode='check') reports per-file errors AND diff drift without writing."""
    # File 1: anchor refers to a key absent from `metrics` → error in errors[file1].
    file1 = tmp_path / "report1.md"
    file1.write_text("score: <!-- begin:metrics.missing_key -->X<!-- end:metrics.missing_key -->")
    # File 2: anchor would render to a different value than the placeholder → drift.
    file2 = tmp_path / "report2.md"
    file2.write_text("score: <!-- begin:metric.pr_auc -->placeholder<!-- end:metric.pr_auc -->")

    metrics = {"metric": {"pr_auc": 0.881}}
    result = render_files([file1, file2], metrics, mode="check")
    # File 1 has the error; file 2 has drift.
    assert str(file1) in result["errors"]
    assert str(file2) in result["drift"]
    # Original files are untouched in check mode.
    assert "placeholder" in file2.read_text()


@pytest.mark.unit
def test_render_files_missing_path_recorded_in_errors(tmp_path: Path) -> None:
    """A non-existent target path is recorded in the errors dict, not raised."""
    missing = tmp_path / "nope.md"
    result = render_files([missing], {"x": 1}, mode="check")
    assert str(missing) in result["errors"]
    assert "file not found" in result["errors"][str(missing)][0]


@pytest.mark.unit
def test_render_files_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        render_files([], {}, mode="apply-then-check")  # type: ignore[arg-type]


# v0.8.2: explicit per-helper None-handling tests so coverage doesn't depend
# on which keys hypothesis happens to sample (covers docs.py:91, 97, 109).
@pytest.mark.unit
def test_signed_and_4digit_formatters_handle_none() -> None:
    """_fmt_signed_3, _fmt_signed_4, and _fmt_4 should return 'N/A' for None."""
    from eval_toolkit.docs import _fmt_4, _fmt_signed_3, _fmt_signed_4

    assert _fmt_signed_3(None) == "N/A"
    assert _fmt_signed_4(None) == "N/A"
    assert _fmt_4(None) == "N/A"
    # Sanity: real values still format correctly.
    assert _fmt_signed_3(0.123) == "+0.123"
    assert _fmt_signed_4(-0.0001) == "-0.0001"
    assert _fmt_4(3.14159) == "3.1416"
