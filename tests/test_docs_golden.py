"""Golden snapshot tests for the docs anchor renderer.

Snapshot fixtures in tests/golden/docs/. Each value is anchored to analytical
truth: e.g., pr_auc=0.951234 → 0.951 via the default _fmt_3 formatter
(round-half-to-even at 3 decimals). lift formatter compounds delta + CI
bounds: +0.123 [+0.050, +0.200].

If the formatter behavior intentionally changes, regenerate the expected.md
with `render_files(targets, metrics, mode='apply')` and commit the diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_toolkit.docs import render_files, render_text

GOLDEN_DIR = Path(__file__).parent / "golden" / "docs"


@pytest.fixture
def golden_input() -> str:
    return (GOLDEN_DIR / "input.md").read_text()


@pytest.fixture
def golden_metrics() -> dict[str, object]:
    return json.loads((GOLDEN_DIR / "metrics.json").read_text())


@pytest.fixture
def golden_expected() -> str:
    return (GOLDEN_DIR / "expected.md").read_text()


@pytest.mark.golden
def test_render_text_matches_expected_snapshot(
    golden_input: str, golden_metrics: dict[str, object], golden_expected: str
) -> None:
    """render_text(input, metrics) matches the canonical expected.md exactly."""
    rendered, errors = render_text(golden_input, golden_metrics)
    assert errors == [], f"unexpected anchor errors: {errors}"
    assert rendered == golden_expected, (
        f"Rendered output diverged from golden expected.\n"
        f"--- expected ---\n{golden_expected}\n"
        f"--- actual ---\n{rendered}\n"
    )


@pytest.mark.golden
def test_render_files_apply_writes_expected(
    tmp_path: Path,
    golden_input: str,
    golden_metrics: dict[str, object],
    golden_expected: str,
) -> None:
    """render_files in apply mode writes the rendered content to disk."""
    target = tmp_path / "report.md"
    target.write_text(golden_input)
    result = render_files([target], golden_metrics, mode="apply")
    assert str(target) in result["updated"]
    assert target.read_text() == golden_expected
    assert result["errors"] == {}


@pytest.mark.golden
def test_render_files_check_reports_drift(
    tmp_path: Path,
    golden_input: str,
    golden_metrics: dict[str, object],
) -> None:
    """In check mode, render_files reports drift but does NOT write."""
    target = tmp_path / "report.md"
    target.write_text(golden_input)
    result = render_files([target], golden_metrics, mode="check")
    # File was NOT modified
    assert target.read_text() == golden_input
    # Drift reported
    assert str(target) in result["drift"]
    drift_text = result["drift"][str(target)]
    assert "TBD" in drift_text  # original had TBD; rendered replaced it


@pytest.mark.golden
def test_render_files_check_no_drift_when_already_rendered(
    tmp_path: Path,
    golden_metrics: dict[str, object],
    golden_expected: str,
) -> None:
    """When the file already matches the rendered output, check reports unchanged."""
    target = tmp_path / "report.md"
    target.write_text(golden_expected)
    result = render_files([target], golden_metrics, mode="check")
    assert str(target) in result["unchanged"]
    assert result["drift"] == {}


@pytest.mark.golden
def test_render_files_validates_mode(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    target.write_text("hi")
    with pytest.raises(ValueError, match="mode"):
        render_files([target], {}, mode="invalid")  # type: ignore[arg-type]


@pytest.mark.golden
def test_render_files_missing_file_reports_error(
    tmp_path: Path, golden_metrics: dict[str, object]
) -> None:
    """Missing file path produces an error entry, not a crash."""
    missing = tmp_path / "no_such.md"
    result = render_files([missing], golden_metrics, mode="apply")
    assert str(missing) in result["errors"]


@pytest.mark.golden
def test_lift_compound_formatter_format(
    golden_metrics: dict[str, object],
) -> None:
    """The lift compound formatter walks to the parent dict, not the leaf."""
    text = "<!-- begin:slices.test.diffs.lift -->X<!-- end:slices.test.diffs.lift -->"
    rendered, errors = render_text(text, golden_metrics)
    assert errors == []
    # +0.123 [+0.050, +0.200]
    assert "+0.123" in rendered
    assert "+0.050" in rendered
    assert "+0.200" in rendered


@pytest.mark.golden
def test_unknown_key_reports_error_leaves_unchanged(
    golden_metrics: dict[str, object],
) -> None:
    """Unknown anchor key leaves the body unchanged and appends a diagnostic."""
    text = "<!-- begin:slices.test.no_such_key -->ORIGINAL<!-- end:slices.test.no_such_key -->"
    rendered, errors = render_text(text, golden_metrics)
    assert "ORIGINAL" in rendered  # unchanged
    assert len(errors) == 1
    assert "no_such_key" in errors[0]


def test_render_files_apply_mode_utf8_round_trip(tmp_path: Path) -> None:
    """#97: apply mode reads + writes UTF-8 explicitly, so non-ASCII consumer
    content survives byte-for-byte regardless of the platform locale codec
    (Windows cp1252 previously corrupted it cumulatively on every apply)."""
    target = tmp_path / "résumé.md"
    content = "# Résumé — Δ§é ≥ 0.95\n\n" "pr_auc: <!-- begin:pr_auc -->stale<!-- end:pr_auc -->\n"
    target.write_text(content, encoding="utf-8")
    result = render_files([target], {"pr_auc": 0.951234}, mode="apply")
    assert str(target) in result["updated"]
    out = target.read_text(encoding="utf-8")
    assert "Résumé — Δ§é ≥ 0.95" in out  # non-ASCII prose untouched
    assert "0.951" in out  # anchor rendered
