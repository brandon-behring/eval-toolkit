"""Public-API drift guard — locks `eval_toolkit.__all__` against accidental changes.

Snapshots every name in `eval_toolkit.__all__` along with its kind
(function/class/value), signature (for callables), bases (for classes),
and the first line of its docstring. Stored at
``tests/golden/public_api/snapshot.json``.

The test fails on:

- Adding or removing an exported name (set-equality drift)
- Changing a function signature — param names, defaults, kwarg-only markers
- Changing a class's base list
- Changing the first-line docstring (catches accidental doc rewording)
- For constants: type changes, primitive-value changes, dict-key drift

Pairs with the CHANGELOG release flow: if you change a signature, update
the golden in the same commit. The golden update IS the API change
acknowledgment.

Regenerating
------------
``REGEN_PUBLIC_API_GOLDEN=1 uv run python -m pytest tests/test_public_api.py``
then review the diff and commit alongside the API-change CHANGELOG entry.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import eval_toolkit

# frozenset / set repr ordering is hash-dependent (can vary across processes).
# Canonicalize to sorted-member form so the snapshot is reproducible.
_SET_LITERAL_RE = re.compile(r"(frozenset)\(\{([^}]*)\}\)")


def _canonicalize_signature(sig: str) -> str:
    """Sort tokens inside frozenset({...}) literals so the signature str is stable."""

    def _sort(match: re.Match[str]) -> str:
        kind = match.group(1)
        members = [m.strip() for m in match.group(2).split(",") if m.strip()]
        members.sort()
        return f"{kind}({{{', '.join(members)}}})"

    return _SET_LITERAL_RE.sub(_sort, sig)


GOLDEN_DIR = Path(__file__).parent / "golden" / "public_api"
GOLDEN_PATH = GOLDEN_DIR / "snapshot.json"
REGEN = os.environ.get("REGEN_PUBLIC_API_GOLDEN") == "1"


def _value_summary(obj: Any) -> dict[str, Any]:
    """Address-free, repr-stable summary of a non-callable public value."""
    t = type(obj)
    tname = f"{t.__module__}.{t.__name__}" if t.__module__ != "builtins" else t.__name__
    # Atomic primitives: include exact value (stable repr)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return {"type": tname, "value": repr(obj)}
    # Tuples/lists of primitives: include exact value
    if isinstance(obj, (tuple, list)) and all(isinstance(x, (int, float, str, bool)) for x in obj):
        return {"type": tname, "value": repr(obj)}
    # Dicts: capture sorted key set (values may be function refs with addresses)
    if isinstance(obj, dict):
        return {"type": tname, "keys": sorted(str(k) for k in obj)}
    # Sets / frozensets: include sorted member list of primitive members
    if isinstance(obj, (set, frozenset)) and all(
        isinstance(x, (int, float, str, bool)) for x in obj
    ):
        return {"type": tname, "value": repr(sorted(obj))}
    # Everything else (e.g., compiled regex): just the type
    return {"type": tname}


def _entry_for(name: str, obj: Any) -> dict[str, Any]:
    """Build a stable snapshot entry for one exported symbol."""
    docstring = inspect.getdoc(obj)
    doc_first = docstring.splitlines()[0] if docstring else ""

    if inspect.isclass(obj):
        bases = [b.__name__ for b in obj.__bases__]
        try:
            sig = _canonicalize_signature(str(inspect.signature(obj)))
        except (TypeError, ValueError):
            sig = "<no signature>"
        return {
            "kind": "class",
            "signature": sig,
            "bases": bases,
            "doc_first_line": doc_first,
        }

    if callable(obj):
        try:
            sig = _canonicalize_signature(str(inspect.signature(obj)))
        except (TypeError, ValueError):
            sig = "<no signature>"
        return {
            "kind": "function",
            "signature": sig,
            "doc_first_line": doc_first,
        }

    return {
        "kind": "value",
        **_value_summary(obj),
        "doc_first_line": doc_first,
    }


def _build_snapshot() -> dict[str, Any]:
    """Build the full snapshot dict of every name in eval_toolkit.__all__."""
    names = sorted(eval_toolkit.__all__)
    return {
        "__all__": names,
        "entries": {name: _entry_for(name, getattr(eval_toolkit, name)) for name in names},
    }


@pytest.mark.golden
def test_public_api_drift_guard() -> None:
    """Public-API snapshot matches the pinned golden.

    Failure modes that this catches:

    - Removed export: name in golden but missing from current __all__
    - Added export: new name in __all__ without a CHANGELOG entry
    - Renamed kwarg: signature diff under an unchanged name
    - Changed default value: signature diff
    - Reordered class bases: bases diff
    - Reworded first docstring line: doc_first_line diff

    Regenerate with ``REGEN_PUBLIC_API_GOLDEN=1`` AFTER deciding the
    change is intentional and writing a corresponding CHANGELOG line.
    """
    actual = _build_snapshot()

    if REGEN or not GOLDEN_PATH.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        if REGEN:
            pytest.skip(
                f"REGEN: wrote {GOLDEN_PATH}; re-run without "
                "REGEN_PUBLIC_API_GOLDEN to validate."
            )
        pytest.skip(f"Initial golden written to {GOLDEN_PATH}; re-run to validate.")

    expected = json.loads(GOLDEN_PATH.read_text())

    # 1) Symbol-set equality (the highest-signal check)
    actual_names = set(actual["__all__"])
    expected_names = set(expected["__all__"])
    added = sorted(actual_names - expected_names)
    removed = sorted(expected_names - actual_names)
    assert not added and not removed, (
        "Public API drift detected:\n"
        f"  ADDED   ({len(added)}): {added}\n"
        f"  REMOVED ({len(removed)}): {removed}\n"
        "If intentional, regenerate the golden + add a CHANGELOG entry."
    )

    # 2) Per-entry equality (signatures, bases, docstring first lines, value summaries)
    drift: list[str] = []
    for name in sorted(actual_names):
        a = actual["entries"][name]
        e = expected["entries"][name]
        if a != e:
            # Build a focused per-key diff so failure message is actionable
            keys = sorted(set(a.keys()) | set(e.keys()))
            for k in keys:
                if a.get(k) != e.get(k):
                    drift.append(f"  {name}.{k}: actual={a.get(k)!r} expected={e.get(k)!r}")
    assert not drift, (
        "Public API entry drift (signatures/bases/docs/values):\n"
        + "\n".join(drift[:30])  # cap to first 30 for readability
        + (f"\n... ({len(drift) - 30} more)" if len(drift) > 30 else "")
        + "\nIf intentional, regenerate the golden + add a CHANGELOG entry."
    )
