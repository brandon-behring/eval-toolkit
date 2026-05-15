"""Deprecation infrastructure for public-API surface.

Public surface: the :func:`deprecated` decorator. Use it on any public
function/method/class to telegraph its planned removal:

    >>> from eval_toolkit._deprecated import deprecated
    >>> @deprecated("0.30.0", reason="see new_fn()", use_instead="new_fn")
    ... def old_fn(x: int) -> int:
    ...     return x + 1
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     _ = old_fn(0)
    ...     assert issubclass(w[0].category, DeprecationWarning)
    ...     assert "0.30.0" in str(w[0].message)

The decorator:

1. Validates the deadline string is a valid SemVer release at
   decoration time (typos fail at import, not at runtime).
2. Wraps the function so each call emits a ``DeprecationWarning``
   with a structured message.
3. Preserves the original function via ``functools.wraps`` —
   ``__name__`` / ``__doc__`` / ``__wrapped__`` survive.

See :doc:`/DEPRECATION` for the project's deprecation-window policy.

Tier β #3 of the v0.29.0 best-practice gap audit. Per the
v0.28.1 plan Q2=A: string-deadline API, no ``packaging.Version``
dep, validated at import time.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable
from typing import TypeVar

__all__ = ["deprecated"]

# SemVer release pattern: X.Y.Z (no prereleases — deprecation deadlines
# point at stable releases).
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

F = TypeVar("F", bound=Callable[..., object])


def deprecated(
    deadline: str,
    *,
    reason: str = "",
    use_instead: str | None = None,
) -> Callable[[F], F]:
    """Mark a public callable as deprecated.

    Emits a :class:`DeprecationWarning` on every call. The wrapped
    function's behavior is otherwise unchanged.

    Parameters
    ----------
    deadline : str
        Version at which this callable will be removed. Must be a
        valid SemVer ``X.Y.Z`` release (no prereleases). Validated
        at decoration time — typos fail at import, not at call time.
    reason : str, optional
        Free-text rationale included in the warning message.
        Default ``""``.
    use_instead : str or None, optional
        Recommended replacement symbol name (e.g., ``"new_fn"``).
        Default ``None``. Included in the warning message when
        provided.

    Returns
    -------
    callable
        Decorator. Apply to the target callable.

    Raises
    ------
    ValueError
        At decoration time (i.e., module import time), if
        ``deadline`` is not a valid SemVer release.

    Examples
    --------
    >>> @deprecated("0.30.0", reason="superseded by new_metric")
    ... def old_metric(y, s):
    ...     return 0.0

    >>> @deprecated("1.0.0", use_instead="EvaluatorV2")
    ... class EvaluatorV1:
    ...     pass

    Notes
    -----
    The decorator validates the deadline string at decoration time
    via regex match against ``^\\d+\\.\\d+\\.\\d+$``. Invalid
    deadlines fail with ``ValueError`` at module load — the project
    treats this as a typo-protection rather than runtime validation.

    For the "remove-on-deadline" reverse check (a test that asserts
    no callable has an expired deadline), see
    ``tests/test_deprecations.py::test_no_expired_deprecation_deadlines``.

    See Also
    --------
    eval_toolkit._deprecated.deprecated : This function.

    References
    ----------
    .. [1] Project deprecation policy: ``docs/DEPRECATION.md``.
    """
    if not _SEMVER_RE.fullmatch(deadline):
        raise ValueError(
            f"deprecated() deadline {deadline!r} is not a valid SemVer "
            f"release (expected 'X.Y.Z'). Typo? See docs/DEPRECATION.md."
        )

    def _decorator(fn: F) -> F:
        message_parts = [
            f"{fn.__qualname__} is deprecated and will be removed in eval-toolkit {deadline}."
        ]
        if reason:
            message_parts.append(reason)
        if use_instead:
            message_parts.append(f"Use {use_instead} instead.")
        message = " ".join(message_parts)

        @functools.wraps(fn)
        def _wrapper(*args: object, **kwargs: object) -> object:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        # Stash the metadata so tests can introspect without parsing the
        # warning message string.
        _wrapper.__deprecated_deadline__ = deadline  # type: ignore[attr-defined]
        _wrapper.__deprecated_reason__ = reason  # type: ignore[attr-defined]
        _wrapper.__deprecated_use_instead__ = use_instead  # type: ignore[attr-defined]

        return _wrapper  # type: ignore[return-value]

    return _decorator
