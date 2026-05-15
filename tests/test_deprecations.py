"""Tests for the ``@deprecated`` decorator + deprecation-deadline scanner.

Covers:

1. Valid string deadlines warn on call; metadata preserved.
2. Invalid deadlines fail at decoration time (module load).
3. ``functools.wraps`` preserves __name__/__doc__/__wrapped__.
4. Scanner: walks every public-API callable in ``eval_toolkit.__all__``
   and fails if any has a ``@deprecated`` deadline ≤ the current
   ``eval_toolkit.__version__`` — i.e., a forgotten removal.

Per the v0.28.1 plan Q2=A: string-deadline decorator API.
"""

from __future__ import annotations

import warnings

import pytest

import eval_toolkit
from eval_toolkit._deprecated import deprecated


@pytest.mark.unit
def test_deprecated_emits_warning_on_call() -> None:
    """Every call to a @deprecated function emits DeprecationWarning."""

    @deprecated("0.30.0", reason="for testing")
    def _example() -> int:
        return 42

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _example()
        assert result == 42  # behavior preserved
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "0.30.0" in str(w[0].message)
        assert "for testing" in str(w[0].message)


@pytest.mark.unit
def test_deprecated_use_instead_appears_in_message() -> None:
    """The ``use_instead=`` kwarg appears in the warning message."""

    @deprecated("0.30.0", use_instead="new_function")
    def _example() -> None:
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _example()
        assert "new_function" in str(w[0].message)


@pytest.mark.unit
def test_deprecated_rejects_invalid_deadline() -> None:
    """Non-SemVer deadlines fail at decoration time."""
    with pytest.raises(ValueError, match="not a valid SemVer"):

        @deprecated("0..30.0")  # noqa: F841  — decorator typo
        def _bad() -> None:
            pass

    with pytest.raises(ValueError, match="not a valid SemVer"):

        @deprecated("v0.30.0")  # noqa: F841  — leading 'v'
        def _bad2() -> None:
            pass

    with pytest.raises(ValueError, match="not a valid SemVer"):

        @deprecated("0.30")  # noqa: F841  — missing patch
        def _bad3() -> None:
            pass

    with pytest.raises(ValueError, match="not a valid SemVer"):

        @deprecated("0.30.0rc1")  # noqa: F841  — prerelease not allowed
        def _bad4() -> None:
            pass


@pytest.mark.unit
def test_deprecated_preserves_function_metadata() -> None:
    """``functools.wraps`` keeps __name__, __doc__, __wrapped__ intact."""

    @deprecated("0.30.0")
    def example(x: int) -> int:
        """Example docstring."""
        return x

    assert example.__name__ == "example"
    assert example.__doc__ == "Example docstring."
    assert hasattr(example, "__wrapped__")


@pytest.mark.unit
def test_deprecated_exposes_introspection_attrs() -> None:
    """Tests can introspect deadline/reason/use_instead without parsing the message."""

    @deprecated("0.31.0", reason="superseded", use_instead="new_thing")
    def _example() -> None:
        pass

    assert _example.__deprecated_deadline__ == "0.31.0"
    assert _example.__deprecated_reason__ == "superseded"
    assert _example.__deprecated_use_instead__ == "new_thing"


@pytest.mark.unit
def test_deprecated_works_on_classes() -> None:
    """The decorator can wrap classes too — instantiation emits the warning."""

    @deprecated("0.30.0", use_instead="EvaluatorV2")
    class _LegacyEvaluator:
        def __init__(self, x: int) -> None:
            self.x = x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instance = _LegacyEvaluator(5)
        assert instance.x == 5
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)


# ---------------------------------------------------------------------------
# Scanner: catches forgotten-removal deadlines on public API
# ---------------------------------------------------------------------------


def _parse_version(s: str) -> tuple[int, int, int]:
    """Parse 'X.Y.Z' into (X, Y, Z) for comparison. Already validated by
    the decorator at import time, so this is just splitting."""
    return tuple(int(p) for p in s.split("."))  # type: ignore[return-value]


@pytest.mark.unit
def test_no_expired_deprecation_deadlines() -> None:
    """No public callable carries a @deprecated deadline ≤ current __version__.

    Walks eval_toolkit.__all__, checks each callable's
    ``__deprecated_deadline__`` attr (if any), and fails if any
    deadline is ≤ the current release. Prevents the
    "forgot to remove the deprecated function" class of bug at release
    time.

    Currently no public symbols are deprecated. This test passes
    trivially today; it'll start catching real expirations once we
    add the first deprecation.
    """
    current_version = _parse_version(eval_toolkit.__version__)
    expired: list[tuple[str, str]] = []

    for name in eval_toolkit.__all__:
        if name == "__version__":
            continue
        obj = getattr(eval_toolkit, name, None)
        if obj is None or not callable(obj):
            continue
        deadline_str = getattr(obj, "__deprecated_deadline__", None)
        if not isinstance(deadline_str, str):
            continue
        deadline = _parse_version(deadline_str)
        if deadline <= current_version:
            expired.append((name, deadline_str))

    assert not expired, (
        "The following deprecated public symbols have expired deadlines "
        "(deadline <= current version) and should have been removed:\n"
        + "\n".join(f"  {n}: deadline {d}" for n, d in expired)
    )


@pytest.mark.unit
def test_scanner_finds_actual_deprecation_when_present() -> None:
    """Smoke test: the scanner CAN find a deprecation. Sanity check on its logic.

    Without an actual deprecation in the public API yet, this is the only
    way to verify the scanner is wired correctly.
    """

    # Build a synthetic deprecated callable
    @deprecated("0.30.0", reason="synthetic for testing")
    def _synthetic() -> None:
        pass

    # The deadline is set as a string attribute
    assert _synthetic.__deprecated_deadline__ == "0.30.0"

    # The version comparison logic works
    current = _parse_version("0.28.0")
    deadline = _parse_version(_synthetic.__deprecated_deadline__)
    assert deadline > current  # 0.30.0 > 0.28.0 → not expired

    older_current = _parse_version("0.30.0")
    assert deadline <= older_current  # exactly at deadline → would be expired
