"""Deterministic dogfood runner for eval-toolkit audit validators.

Runs an ``audit_*`` validator against its real consumer repo and emits the
raw residual findings as JSON on stdout. The *run* is deterministic and
lives here; the *classification* of residuals (real / false-positive /
edge-case x layer) is the job of the ``etk-dogfood-noise-analyst`` agent,
which consumes this JSON.

This generalizes the ad-hoc ``.scratch/dogfood_v1_4_0_citation.py`` into a
committed, parameterized tool so validator releases have a repeatable
acceptance-evidence step (see ADR 0007 on the library-first cycle).

Usage
-----
    python scripts/dogfood_audit.py --validator audit_citation_alignment \
        --consumer ~/Claude/prompt-injection-detection-submission \
        --scope narrative

Only ``audit_citation_alignment`` is wired today; other validators raise
``NotImplementedError`` loudly (never a silent empty result) so a missing
adapter is impossible to mistake for "zero findings".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_CONSUMER = Path.home() / "Claude" / "prompt-injection-detection-submission"


def _load_consumer_module(script_path: Path) -> dict[str, Any]:
    """Exec a consumer script for its module-level fixtures, without running main().

    The consumer's audit script defines constants (CATEGORY_KEYWORDS,
    SURFACE_GLOBS, ...) and helper functions at module scope and guards its
    entry point with ``if __name__ == "__main__"``. Setting ``__name__`` to a
    sentinel suppresses that entry point while exposing the fixtures.

    Parameters
    ----------
    script_path : Path
        Path to the consumer's audit script.

    Returns
    -------
    dict
        The executed module globals.

    Raises
    ------
    FileNotFoundError
        If the consumer script does not exist.
    """
    if not script_path.is_file():
        raise FileNotFoundError(
            f"consumer audit script not found: {script_path}. "
            f"Pass --consumer pointing at the consumer repo root."
        )
    namespace: dict[str, Any] = {
        "__file__": str(script_path),
        "__name__": "__dogfood__",  # not "__main__" -> suppresses the script's main()
    }
    source = script_path.read_text(encoding="utf-8")
    # exec() imports the consumer script's module-level fixtures. This trusts the
    # local consumer checkout (a developer's own repo), not untrusted input.
    exec(compile(source, str(script_path), "exec"), namespace)
    return namespace


def _run_citation_alignment(
    consumer: Path, scope: Literal["all", "narrative"]
) -> list[dict[str, Any]]:
    """Run audit_citation_alignment against the consumer and return raw findings.

    Parameters
    ----------
    consumer : Path
        Consumer repo root.
    scope : str
        ``"all"`` or ``"narrative"`` (the v1.4.0 Layer-2 opt-in).

    Returns
    -------
    list of dict
        One record per finding: file, line, cited_adr_id, claim_category,
        adr_actual_category, surrounding_text.

    Raises
    ------
    FileNotFoundError
        If the consumer script or its fixtures are missing.
    KeyError
        If the consumer script lacks a required fixture (CATEGORY_KEYWORDS,
        SURFACE_GLOBS) — surfaced loudly rather than silently skipped.
    """
    sys.path.insert(0, str(consumer))
    sys.path.insert(0, str(consumer / "scripts"))

    # Lazy import: requires sys.path (above) to include the consumer + its scripts.
    from eval_toolkit.audit_citation_alignment import validate_citations

    script_path = consumer / "scripts" / "audit_citation_alignment.py"
    g = _load_consumer_module(script_path)

    for required in ("CATEGORY_KEYWORDS", "SURFACE_GLOBS"):
        if required not in g:
            raise KeyError(
                f"consumer script {script_path} does not define {required!r}; "
                f"cannot run dogfood without it"
            )
    category_keywords = g["CATEGORY_KEYWORDS"]
    surface_globs = g["SURFACE_GLOBS"]
    skip_patterns = g.get("SKIP_PATTERNS", set())

    # The consumer script owns ADR-frontmatter parsing; we require its
    # build_adr_subjects() rather than reimplementing it (a parallel parser here
    # was dead code — the consumer always defines it — and brittle).
    if "build_adr_subjects" not in g:
        raise KeyError(
            f"consumer script {script_path} does not define build_adr_subjects(); "
            f"cannot map ADRs without it"
        )
    adr_subjects = g["build_adr_subjects"]()

    surfaces: list[Path] = []
    for pattern in surface_globs:
        surfaces.extend(consumer.glob(pattern))
    surfaces = sorted(
        {
            p
            for p in surfaces
            if not any(skip in str(p.relative_to(consumer)) for skip in skip_patterns)
        }
    )

    findings: list[dict[str, Any]] = []
    for path in surfaces:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Consumer content may contain non-UTF-8 bytes; skip the file but
            # never let it abort the whole run — and never skip silently:
            # a dropped surface file is missing acceptance evidence.
            print(f"WARNING: skipping non-UTF-8 surface file {path}: {exc}", file=sys.stderr)
            continue
        for f in validate_citations(
            markdown_text=text,
            markdown_path=path,
            adr_subjects=adr_subjects,
            category_keywords=category_keywords,
            scope=scope,
        ):
            try:
                rel = str(f.file.relative_to(consumer))
            except ValueError:
                rel = str(f.file)
            findings.append(
                {
                    "file": rel,
                    "line": f.line,
                    "cited_adr_id": f.cited_adr_id,
                    "claim_category": f.claim_category,
                    "adr_actual_category": f.adr_actual_category,
                    "surrounding_text": f.surrounding_text[:280],
                }
            )
    return findings


# Validator name -> adapter. Add new validators here as their consumer
# fixtures stabilize; each adapter returns a list of JSON-able finding dicts.
_ADAPTERS = {
    "audit_citation_alignment": _run_citation_alignment,
}


def run(validator: str, consumer: Path, scope: Literal["all", "narrative"]) -> dict[str, Any]:
    """Run a validator's dogfood and return a JSON-able report.

    Parameters
    ----------
    validator : str
        Validator module name, e.g. ``"audit_citation_alignment"``.
    consumer : Path
        Consumer repo root.
    scope : str
        ``"all"`` or ``"narrative"``.

    Returns
    -------
    dict
        ``{"validator", "consumer", "scope", "count", "per_file", "findings"}``.

    Raises
    ------
    NotImplementedError
        If no adapter exists for ``validator`` (loud, never a silent empty run).
    FileNotFoundError
        If the consumer repo root does not exist.
    """
    if not consumer.is_dir():
        raise FileNotFoundError(f"consumer repo not found: {consumer}")
    adapter = _ADAPTERS.get(validator)
    if adapter is None:
        raise NotImplementedError(
            f"no dogfood adapter for validator {validator!r}; "
            f"available: {sorted(_ADAPTERS)}. Add one in scripts/dogfood_audit.py."
        )
    findings = adapter(consumer, scope)
    per_file = Counter(f["file"] for f in findings)
    return {
        "validator": validator,
        "consumer": str(consumer),
        "scope": scope,
        "count": len(findings),
        "per_file": dict(per_file.most_common()),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run a dogfood and print the JSON report to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validator",
        default="audit_citation_alignment",
        help="audit_* validator module name (default: audit_citation_alignment)",
    )
    parser.add_argument(
        "--consumer",
        type=Path,
        default=DEFAULT_CONSUMER,
        help=f"consumer repo root (default: {DEFAULT_CONSUMER})",
    )
    parser.add_argument(
        "--scope",
        default="narrative",
        choices=["all", "narrative"],
        help="validator scope (default: narrative)",
    )
    args = parser.parse_args(argv)

    scope = cast(Literal["all", "narrative"], args.scope)
    report = run(args.validator, args.consumer.expanduser(), scope)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
