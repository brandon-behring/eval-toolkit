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


def _protocol_method_snapshot(cls: type) -> dict[str, str]:
    """Capture method + annotated-attr signatures for a ``typing.Protocol`` class.

    Decision R6-D (Round 6 audit, Codex R6-F5): ADR 0003 promises strict
    Tier-2 Protocol method-shape stability, but the bare ``inspect.signature``
    of a Protocol class only yields ``(*args, **kwargs)`` — the public-API
    drift guard cannot see changes to ``MetricSpec.compute``,
    ``MetaLearner.fit``, etc. This helper introspects each public method on
    the Protocol and returns a ``name → signature-string`` mapping so the
    snapshot covers method shapes.

    Only methods declared on the class itself (``cls.__dict__``) are
    captured — Protocol / object machinery inherited from the base is
    skipped. Underscore-prefixed names are skipped. ``@property`` accessors
    are captured with signature ``"<property>"``. Annotated class
    attributes (the ``name: str`` declarations Protocols use for
    structural typing) are captured as ``"<attr: ...>"`` since their type
    is part of the contract.
    """
    methods: dict[str, str] = {}
    for attr_name, attr in cls.__dict__.items():
        if attr_name.startswith("_"):
            continue
        if isinstance(attr, property):
            methods[attr_name] = "<property>"
            continue
        if callable(attr):
            try:
                methods[attr_name] = _canonicalize_signature(str(inspect.signature(attr)))
            except (TypeError, ValueError):
                methods[attr_name] = "<no signature>"
    annotations = getattr(cls, "__annotations__", {})
    for ann_name, ann_type in annotations.items():
        if ann_name.startswith("_"):
            continue
        methods.setdefault(ann_name, f"<attr: {ann_type!r}>")
    return dict(sorted(methods.items()))


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
        entry: dict[str, Any] = {
            "kind": "class",
            "signature": sig,
            "bases": bases,
            "doc_first_line": doc_first,
        }
        # Decision R6-D (Round 6 audit, Codex R6-F5): for ``typing.Protocol``
        # classes, capture method signatures so the public-API drift guard
        # actually enforces the Tier-2 method-shape stability contract
        # promised by ADR 0003. The bare class signature
        # (``(*args, **kwargs)``) does not change when a Protocol method
        # is added / renamed / has its kwargs altered.
        if getattr(obj, "_is_protocol", False):
            entry["protocol_methods"] = _protocol_method_snapshot(obj)
        return entry

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


# Tier-2 Protocols (ADR 0003 strict-stability tier). Snapshotting their
# method shapes is the v0.47 enforcement gap closed by Decision R6-D.
# Update this list when a new Protocol is added to ``__all__`` so the
# coverage test fails fast if someone introduces a Tier-2 Protocol without
# also wiring it into the drift guard.
#
_TIER2_PROTOCOLS = frozenset(
    {
        "DatasetLoader",
        "LeakageCheck",
        "MetaLearner",
        "MetricSpec",
        "Probe",
        "Scorer",
        "Splitter",
        "TextTransform",  # added in v0.47 release/v0.47.0 Sub-PR 3 (Decision K)
        "ThresholdSelector",
    }
)


def test_tier2_protocols_have_method_shape_snapshot() -> None:
    """Decision R6-D: each Tier-2 Protocol must have its methods snapshotted.

    Fails fast when a new Protocol is exported without method-shape
    coverage, or when one of the existing Tier-2 Protocols loses its
    ``protocol_methods`` entry in the snapshot.
    """
    if not GOLDEN_PATH.exists():
        pytest.skip("Golden snapshot missing; run test_public_api_drift_guard first.")
    expected = json.loads(GOLDEN_PATH.read_text())
    entries = expected["entries"]

    # Every Tier-2 Protocol must (a) be in __all__ and (b) have method-shape coverage
    missing_from_all = sorted(p for p in _TIER2_PROTOCOLS if p not in entries)
    assert not missing_from_all, (
        f"Tier-2 Protocols missing from __all__ / snapshot: {missing_from_all}. "
        "Either remove from _TIER2_PROTOCOLS or restore the export."
    )
    missing_methods = sorted(p for p in _TIER2_PROTOCOLS if "protocol_methods" not in entries[p])
    assert not missing_methods, (
        f"Tier-2 Protocols missing protocol_methods snapshot: {missing_methods}. "
        "Regenerate the golden after extending _entry_for() Protocol-aware branch."
    )


def test_protocol_methods_only_on_protocol_classes() -> None:
    """``protocol_methods`` only appears on entries whose live class is a Protocol.

    Sanity-check that the Decision R6-D conditional in ``_entry_for()``
    only fires on actual ``typing.Protocol`` subclasses — otherwise the
    snapshot would grow noisy keys on every concrete class.

    Per-module Protocols (e.g. ``CharacterInjectionStrategy``) ARE Protocols
    and so are allowed to carry ``protocol_methods`` here; this test does
    NOT enforce the "Tier-2 only" subset (that is the
    ``_TIER2_PROTOCOLS`` test's job).
    """
    if not GOLDEN_PATH.exists():
        pytest.skip("Golden snapshot missing; run test_public_api_drift_guard first.")
    expected = json.loads(GOLDEN_PATH.read_text())
    leaks: list[str] = []
    for name, entry in expected["entries"].items():
        if entry.get("kind") != "class" or "protocol_methods" not in entry:
            continue
        live_obj = getattr(eval_toolkit, name, None)
        if live_obj is None:
            continue
        if not getattr(live_obj, "_is_protocol", False):
            leaks.append(name)
    assert not leaks, (
        f"Non-Protocol classes have protocol_methods in snapshot: {leaks}. "
        "The Decision R6-D conditional in _entry_for() is misfiring."
    )
