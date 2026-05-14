#!/usr/bin/env python
"""Audit docstring `Raises:` sections vs actual raise sites.

For each public function/method in ``src/eval_toolkit/``, compare the set of
exceptions raised in the function body against the exceptions documented in
the NumPy-style ``Raises`` section of its docstring. Report mismatches.

Behavior:
- Only walks public top-level functions and public methods on public classes.
- Ignores raise sites inside nested functions (those have their own docstrings).
- Treats ``NotImplementedError`` as an abstract marker; not required to document.
- Treats bare ``raise`` (re-raise) as not an introduced contract; ignores.
- ``--strict`` also reports documented-but-not-raised exceptions.

Exit code: 0 on no mismatches, 1 otherwise.

Usage:
    python scripts/audit_raises_sections.py [--strict]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "eval_toolkit"

# Skip files that have no public functions worth auditing.
SKIP_FILES = {"_version.py", "__init__.py", "__main__.py"}

# Exceptions raised by abstract Protocol methods or NotImplementedError stubs
# do not need documenting; the abstract contract carries the meaning.
ABSTRACT_MARKERS = {"NotImplementedError"}


def _exception_name_from_call(node: ast.AST) -> str | None:
    """Return the exception class name from a ``raise X(...)`` or ``raise X`` node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _exception_name_from_call(node.func)
    return None


def _find_raise_types(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect exception class names raised in ``func`` body, excluding nested functions."""
    raises: set[str] = set()
    for node in ast.walk(func):
        if node is not func and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.Raise) and node.exc is not None:
            name = _exception_name_from_call(node.exc)
            if name and name not in ABSTRACT_MARKERS:
                raises.add(name)
    return raises


# NumPy-style Raises section header: "Raises\n------\n"
RAISES_HEADER_RE = re.compile(r"^[ \t]*Raises[ \t]*\n[ \t]*-{3,}[ \t]*\n", re.MULTILINE)
# Class names matching standard exception suffixes.
EXCEPTION_LINE_RE = re.compile(
    r"^[ \t]{0,12}([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b",
    re.MULTILINE,
)
# Section-header rule: a capitalized word followed by a dashes line.
SECTION_HEADER_RE = re.compile(r"\n[ \t]*[A-Z][A-Za-z]+[ \t]*\n[ \t]*-{3,}")


def _find_documented_raises(docstring: str | None) -> set[str]:
    """Parse exception class names listed in the docstring's ``Raises`` section."""
    if not docstring:
        return set()
    m = RAISES_HEADER_RE.search(docstring)
    if not m:
        return set()
    after = docstring[m.end() :]
    next_header = SECTION_HEADER_RE.search(after)
    if next_header:
        after = after[: next_header.start()]
    return set(EXCEPTION_LINE_RE.findall(after))


def _public_functions(
    tree: ast.Module,
) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield ``(qualified_name, func_node)`` for public top-level fns and public methods on public classes."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                yield node.name, node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef) and (
                    not sub.name.startswith("_") or sub.name == "__init__"
                ):
                    yield f"{node.name}.{sub.name}", sub


Finding = tuple[str, str, set[str], set[str]]


def audit_file(path: Path) -> list[Finding]:
    """Audit one file. Return list of (rel_path, qualname, undocumented, extra)."""
    tree = ast.parse(path.read_text())
    findings: list[Finding] = []
    for qualname, fn in _public_functions(tree):
        raised = _find_raise_types(fn)
        documented = _find_documented_raises(ast.get_docstring(fn))
        undocumented = raised - documented
        extra = documented - raised
        if undocumented or extra:
            findings.append((str(path), qualname, undocumented, extra))
    return findings


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    files = sorted(p for p in SRC.glob("*.py") if p.name not in SKIP_FILES)
    total_undocumented = 0
    total_extra = 0
    for path in files:
        findings = audit_file(path)
        rel = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
        for _, qualname, undocumented, extra in findings:
            if undocumented:
                print(f"{rel}::{qualname}: undocumented raises: {sorted(undocumented)}")
                total_undocumented += 1
            if extra and strict:
                print(f"{rel}::{qualname}: documented but not raised: {sorted(extra)}")
                total_extra += 1
    if total_undocumented == 0 and (not strict or total_extra == 0):
        print("OK: all public functions document their raised exceptions.")
        return 0
    print(f"\n{total_undocumented} undocumented + {total_extra} extra mismatch(es) found.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
