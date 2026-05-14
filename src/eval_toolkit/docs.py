"""Anchor-based markdown rendering with formatter registry.

Anchor format: ``<!-- begin:KEY -->old_value<!-- end:KEY -->``.

``KEY`` is a dot-path into a metrics dict, e.g.
``slices.test.scorers.deberta.pr_auc``. Compound formatters (``...lift``,
``...lift_with_mde``) address the parent dict, not the leaf.

Pure functions: :func:`walk_path`, :func:`render_text`.
IO wrapper: :func:`render_files` (mode='check' returns drift report; mode='apply' writes).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "ANCHOR_RE",
    "DEFAULT_FORMATTERS",
    "render_files",
    "render_text",
    "walk_path",
]

ANCHOR_RE = re.compile(
    r"(<!--\s*begin:(?P<key>[^\s>]+)\s*-->)(?P<body>.*?)(<!--\s*end:(?P=key)\s*-->)",
    re.DOTALL,
)


def walk_path(data: Any, dotted_path: str) -> Any:
    """Walk a dot-path into nested dicts/lists.

    Parameters
    ----------
    data : Any
        Root mapping or list to walk.
    dotted_path : str
        Path like ``"a.b.c"`` or ``"a.0.x"``.

    Returns
    -------
    Any
        The value at the path.

    Raises
    ------
    KeyError
        If a key is missing or a non-dict/non-list is encountered mid-walk.

    Examples
    --------
    >>> walk_path({"a": {"b": 42}}, "a.b")
    42
    >>> walk_path({"a": [10, 20, 30]}, "a.1")
    20
    >>> try:
    ...     walk_path({"a": {}}, "a.b")
    ... except KeyError as e:
    ...     print("KeyError")
    KeyError
    """
    cur: Any = data
    for part in dotted_path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"missing key {part!r} at {dotted_path!r}")
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError as exc:
                raise KeyError(f"non-integer index {part!r} at {dotted_path!r}") from exc
            cur = cur[idx]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {dotted_path!r}")
    return cur


# Default leaf-name formatters. Callers extend via the ``formatters`` arg of
# :func:`render_text` / :func:`render_files`.


def _fmt_signed_3(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):+.3f}"


def _fmt_signed_4(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):+.4f}"


def _fmt_3(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.3f}"


def _fmt_4(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.4f}"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "N/A"
    return str(int(v))


def _fmt_lift(d: dict[str, Any]) -> str:
    """Render a paired-bootstrap CI as ``+0.097 [+0.020, +0.199]``."""
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


DEFAULT_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "pr_auc": _fmt_3,
    "roc_auc": _fmt_3,
    "f1": _fmt_3,
    "precision": _fmt_3,
    "recall": _fmt_3,
    "delta": _fmt_signed_3,
    "ci_low": _fmt_signed_3,
    "ci_high": _fmt_signed_3,
    "ece": _fmt_3,
    "ece_equal_width": _fmt_3,
    "ece_equal_mass": _fmt_3,
    "temperature": _fmt_4,
    "nll_pre": _fmt_3,
    "nll_post": _fmt_3,
    "improvement": _fmt_signed_4,
    "threshold": _fmt_4,
    "n": _fmt_int,
    "n_positive": _fmt_int,
    "n_negative": _fmt_int,
    "mean": _fmt_3,
    "std": _fmt_3,
    "min": _fmt_3,
    "max": _fmt_3,
    # compound formatters (called with the parent dict, not the leaf)
    "lift": _fmt_lift,
}


def _render_value(
    metrics: dict[str, Any],
    key: str,
    formatters: dict[str, Callable[[Any], str]],
    compound_keys: frozenset[str],
) -> str:
    """Apply the matching formatter to ``key`` looked up in ``metrics``."""
    leaf = key.rsplit(".", 1)[-1]
    if leaf in compound_keys:
        parent = walk_path(metrics, key.rsplit(".", 1)[0])
        return formatters[leaf](parent)
    value = walk_path(metrics, key)
    fmt = formatters.get(leaf)
    if fmt is None:
        return str(value)
    return fmt(value)


def render_text(
    text: str,
    metrics: dict[str, Any],
    formatters: dict[str, Callable[[Any], str]] | None = None,
    *,
    compound_keys: frozenset[str] = frozenset({"lift"}),
) -> tuple[str, list[str]]:
    """Replace anchored bodies in ``text`` with values looked up in ``metrics``.

    Parameters
    ----------
    text : str
        Markdown source containing ``<!-- begin:KEY -->...<!-- end:KEY -->`` anchors.
    metrics : dict[str, Any]
        Nested dict (or list) to look up KEYs in via :func:`walk_path`.
    formatters : dict[str, Callable] or None, optional
        Leaf-name → formatter callable. If ``None``, uses
        :data:`DEFAULT_FORMATTERS`. Caller extends via dict-merge.
    compound_keys : frozenset[str], optional
        Leaf names that should be passed the *parent dict* rather than the
        leaf value (e.g., ``lift`` is a compound formatter that needs
        ``delta``, ``ci_low``, ``ci_high`` together).

    Returns
    -------
    tuple[str, list[str]]
        ``(rendered_text, errors)``. Unknown keys leave the body unchanged
        and append a diagnostic; errors list does not raise.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str`` or ``metrics`` is not a ``dict``.

    Examples
    --------
    >>> text = "<!-- begin:metric.pr_auc -->X<!-- end:metric.pr_auc -->"
    >>> data = {"metric": {"pr_auc": 0.951}}
    >>> rendered, errs = render_text(text, data)
    >>> "0.951" in rendered
    True
    >>> errs
    []
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")
    fmts = formatters if formatters is not None else DEFAULT_FORMATTERS
    errors: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        key = m.group("key")
        try:
            new_body = _render_value(metrics, key, fmts, compound_keys)
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
            return m.group(0)
        return f"{m.group(1)}{new_body}{m.group(4)}"

    return ANCHOR_RE.sub(_sub, text), errors


def render_files(
    targets: list[Path],
    metrics: dict[str, Any],
    *,
    mode: str = "apply",
    formatters: dict[str, Callable[[Any], str]] | None = None,
    compound_keys: frozenset[str] = frozenset({"lift"}),
) -> dict[str, Any]:
    """IO wrapper: read each target file, render anchors, optionally write.

    Parameters
    ----------
    targets : list of pathlib.Path
        Markdown files to render.
    metrics : dict[str, Any]
    mode : {"apply", "check"}, optional
        ``"apply"`` writes rendered output back to each file (default).
        ``"check"`` does NOT write; instead returns the diff per file in the
        result dict (useful for CI to detect drift).
    formatters, compound_keys : see :func:`render_text`.

    Returns
    -------
    dict[str, Any]
        ``{"updated": [...], "unchanged": [...], "drift": {...}, "errors": {...}}``
        where ``drift`` is populated only in check mode.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{"apply", "check"}``.
    """
    if mode not in {"apply", "check"}:
        raise ValueError(f"mode must be 'apply' or 'check', got {mode!r}")

    updated: list[str] = []
    unchanged: list[str] = []
    drift: dict[str, str] = {}
    errors: dict[str, list[str]] = {}

    for path in targets:
        if not path.exists():
            errors[str(path)] = [f"file not found: {path}"]
            continue
        original = path.read_text()
        rendered, errs = render_text(original, metrics, formatters, compound_keys=compound_keys)
        if errs:
            errors[str(path)] = errs
        if rendered == original:
            unchanged.append(str(path))
            continue
        if mode == "apply":
            path.write_text(rendered)
            updated.append(str(path))
        else:
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    rendered.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=f"{path} (rendered)",
                    n=1,
                )
            )
            drift[str(path)] = diff
            updated.append(str(path))

    return {
        "updated": updated,
        "unchanged": unchanged,
        "drift": drift,
        "errors": errors,
    }
