"""Refresh the dedup-holdout golden fixture from the consumer repo.

Per v0.34.0 plan §C.18 — one-command path for re-syncing when the
consumer's holdout dataset evolves (re-labels, expanded pair count, etc.).

Usage::

    python scripts/refresh_dedup_holdout.py
    python scripts/refresh_dedup_holdout.py --consumer-path /path/to/consumer

Steps:

1. Validate ``<consumer-path>/data/dedup_holdout.jsonl`` exists.
2. Copy it to ``tests/golden/data/dedup_holdout.jsonl`` (bytes-identical
   guard: skips the copy if already up-to-date).
3. Update ``tests/golden/data/dedup_holdout_provenance.md`` with the
   consumer's current git SHA + commit date.
4. Regenerate ``tests/golden/data/dedup_holdout_expected.json`` by running
   the golden test under ``REGEN_DEDUP_HOLDOUT_GOLDEN=1``.

Re-verify the source-dataset licenses at refresh time (see provenance.md
for the table).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONSUMER_PATH = _REPO_ROOT.parent / "prompt-injection-detection-submission"
_FIXTURE_DEST = _REPO_ROOT / "tests" / "golden" / "data" / "dedup_holdout.jsonl"
_PROVENANCE_PATH = _REPO_ROOT / "tests" / "golden" / "data" / "dedup_holdout_provenance.md"


def _git_sha(consumer_path: Path) -> str:
    """Return the consumer repo's current HEAD SHA."""
    return subprocess.check_output(
        ["git", "-C", str(consumer_path), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_commit_date(consumer_path: Path) -> str:
    """Return the consumer's current HEAD commit date (ISO 8601 + timezone)."""
    return subprocess.check_output(
        ["git", "-C", str(consumer_path), "log", "-1", "--format=%ci"], text=True
    ).strip()


def _bytes_match(path_a: Path, path_b: Path) -> bool:
    """SHA-256 equality check on file bytes (avoids reading large files into memory)."""
    if not path_a.exists() or not path_b.exists():
        return False

    def _hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return _hash(path_a) == _hash(path_b)


def _update_provenance_sha(sha: str, date: str) -> None:
    """Replace the SHA + commit-date lines in provenance.md."""
    text = _PROVENANCE_PATH.read_text()
    text = re.sub(
        r"git SHA\s*\*\*`[0-9a-f]+`\*\*\s*\(commit dated[^)]+\)",
        f"git SHA **`{sha}`** (commit dated {date})",
        text,
        count=1,
    )
    _PROVENANCE_PATH.write_text(text)


def _regen_snapshot() -> None:
    """Re-run the golden test under REGEN_DEDUP_HOLDOUT_GOLDEN=1."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/golden/test_dedup_holdout_calibration.py::test_dedup_holdout_calibration_deterministic_strategies",
            "--no-cov",
            "-q",
        ],
        cwd=_REPO_ROOT,
        env={"REGEN_DEDUP_HOLDOUT_GOLDEN": "1", **_env_passthrough()},
        check=True,
    )


def _env_passthrough() -> dict[str, str]:
    """Copy the current env (so the test's venv/PATH are preserved)."""
    import os

    return dict(os.environ)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consumer-path",
        type=Path,
        default=_DEFAULT_CONSUMER_PATH,
        help=f"Path to the consumer repo (default: {_DEFAULT_CONSUMER_PATH})",
    )
    args = parser.parse_args()

    consumer_path: Path = args.consumer_path.resolve()
    fixture_src = consumer_path / "data" / "dedup_holdout.jsonl"

    if not fixture_src.exists():
        print(
            f"error: consumer fixture not found at {fixture_src}",
            file=sys.stderr,
        )
        return 1
    if not (consumer_path / ".git").exists():
        print(
            f"error: {consumer_path} is not a git repo (need git SHA for provenance)",
            file=sys.stderr,
        )
        return 1

    # Step 1: copy fixture (skip if bytes-identical).
    if _bytes_match(fixture_src, _FIXTURE_DEST):
        print(f"fixture up-to-date: {_FIXTURE_DEST}")
    else:
        print(f"copying fixture: {fixture_src} -> {_FIXTURE_DEST}")
        _FIXTURE_DEST.write_bytes(fixture_src.read_bytes())

    # Step 2: update provenance with current consumer SHA + date.
    sha = _git_sha(consumer_path)
    date = _git_commit_date(consumer_path)
    print(f"updating provenance: SHA={sha[:8]}, date={date}")
    _update_provenance_sha(sha, date)

    # Step 3: regenerate snapshot.
    print("regenerating dedup_holdout_expected.json snapshot...")
    _regen_snapshot()

    print("\nDONE. Review the diff:")
    print(
        "  git diff tests/golden/data/dedup_holdout.jsonl "
        "tests/golden/data/dedup_holdout_provenance.md "
        "tests/golden/data/dedup_holdout_expected.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
