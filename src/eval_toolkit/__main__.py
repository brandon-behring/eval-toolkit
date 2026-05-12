"""eval_toolkit CLI entry: ``python -m eval_toolkit ...``.

Subcommands:
  schemas list                       List bundled JSON schema names.
  schemas show <name>                Pretty-print a single schema.
  validate <file> <schema-name>      Validate a JSON file against a bundled schema
                                     (requires [validation] extra).

Exit codes:
  0 — ok
  1 — validation failed
  2 — bad arg (file or schema not found)
  3 — missing optional dependency (jsonschema for validate)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import eval_toolkit


def _schemas_dir() -> Path:
    return Path(eval_toolkit.__file__).parent / "schemas"


def _cmd_schemas_list(_args: argparse.Namespace) -> int:
    for f in sorted(_schemas_dir().glob("*.json")):
        print(f.stem)
    return 0


def _cmd_schemas_show(args: argparse.Namespace) -> int:
    name: str = args.name
    candidate = _schemas_dir() / f"{name}.json"
    if not candidate.exists():
        # Tolerate users typing the full filename (e.g., results.v1.json).
        candidate_with_ext = _schemas_dir() / name
        if candidate_with_ext.exists() and candidate_with_ext.suffix == ".json":
            candidate = candidate_with_ext
        else:
            print(f"unknown schema: {name}", file=sys.stderr)
            return 2
    print(json.dumps(json.loads(candidate.read_text()), indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        import jsonschema  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        print(
            "validate requires the [validation] extra: " "pip install 'eval-toolkit[validation]'",
            file=sys.stderr,
        )
        return 3
    schema_name: str = args.schema
    schema_path = _schemas_dir() / f"{schema_name}.json"
    if not schema_path.exists():
        # Tolerate users typing the full filename.
        alt = _schemas_dir() / schema_name
        if alt.exists() and alt.suffix == ".json":
            schema_path = alt
        else:
            print(f"unknown schema: {schema_name}", file=sys.stderr)
            return 2
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"file not found: {args.file}", file=sys.stderr)
        return 2
    schema = json.loads(schema_path.read_text())
    payload = json.loads(file_path.read_text())
    import jsonschema as _js  # noqa: PLC0415

    try:
        _js.validate(payload, schema)
    except _js.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        print(f"VALIDATION ERROR at {loc}: {exc.message}", file=sys.stderr)
        return 1
    print(f"{args.file}: OK against {schema_name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m eval_toolkit`` and the ``eval-toolkit`` script."""
    parser = argparse.ArgumentParser(
        prog="eval-toolkit",
        description="eval-toolkit CLI: schema discovery + payload validation",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    schemas = sub.add_parser("schemas", help="bundled JSON schema discovery")
    schemas_sub = schemas.add_subparsers(dest="schemas_cmd", required=True)
    schemas_list = schemas_sub.add_parser("list", help="list bundled schema names")
    schemas_list.set_defaults(func=_cmd_schemas_list)
    schemas_show = schemas_sub.add_parser("show", help="pretty-print a single schema")
    schemas_show.add_argument("name", help="schema name (e.g., 'results.v1')")
    schemas_show.set_defaults(func=_cmd_schemas_show)

    validate = sub.add_parser("validate", help="validate a JSON file against a bundled schema")
    validate.add_argument("file", help="path to JSON file to validate")
    validate.add_argument("schema", help="schema name (e.g., 'results.v1')")
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
