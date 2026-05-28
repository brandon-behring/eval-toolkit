"""Drift guard for the repo-local Claude Code review agents and command.

The ``.claude/agents/*.md`` review subagents and the ``/review-eval`` command
embed pointers into the codebase (``STYLE.md`` sections, ADR paths,
``tests/test_public_api.py``, ``src/eval_toolkit/_narrative.py``, ...). Those
pointers rot silently when docs are renamed or modules move — exactly the
class of failure the public-API snapshot guards against for the API surface.
This test applies the same instinct to the agent files: their frontmatter
must parse, their ``name`` must match the filename, and every concrete
repo-relative path they cite must still exist.

No PyYAML dependency: the frontmatter we need (``name``, ``model``) is parsed
with line regexes so the test runs in any environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
COMMANDS_DIR = REPO_ROOT / ".claude" / "commands"

# Concrete repo-relative paths (src/tests/scripts/docs prefixes) with a file
# extension. Globs (``*``) and placeholders (``<...>``) are filtered out below.
_PATH_RE = re.compile(r"(?:src|tests|scripts|docs)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+")
# Top-level docs the agents reference by bare name.
_TOPLEVEL_RE = re.compile(r"\b(STYLE|CONTRIBUTING|CHANGELOG)\.md\b")
# Accepted `model:` values (tier aliases or a full claude-* id).
_VALID_MODELS = {"inherit", "opus", "sonnet", "haiku"}
# Prose ADR references like "ADR 0007" must resolve to a real ADR file.
_ADR_REF_RE = re.compile(r"\bADR[\s-]+0*(\d{1,4})\b")


def _agent_files() -> list[Path]:
    return sorted(p for p in AGENTS_DIR.glob("*.md") if p.name != "README.md")


def _frontmatter(text: str, path: Path) -> dict[str, str]:
    """Extract the leading ``---`` YAML frontmatter as a flat key map.

    Only top-level ``key: value`` lines are captured (sufficient for the
    fields this test checks). Raises if the frontmatter block is missing.
    """
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing leading '---' frontmatter fence")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError(f"{path.name}: unterminated frontmatter (no closing '---')")
    block = text[3:end]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def _cited_paths(text: str) -> set[str]:
    paths = set(_PATH_RE.findall(text))
    paths |= {f"{stem}.md" for stem in _TOPLEVEL_RE.findall(text)}
    # Drop globs and placeholder tokens — those are illustrative, not concrete.
    return {p for p in paths if "*" not in p and "<" not in p}


def test_agents_directory_is_populated() -> None:
    assert AGENTS_DIR.is_dir(), f"missing {AGENTS_DIR}"
    assert _agent_files(), "no .claude/agents/*.md review agents found"


@pytest.mark.parametrize("agent", _agent_files(), ids=lambda p: p.stem)
def test_agent_frontmatter_parses_and_name_matches(agent: Path) -> None:
    fields = _frontmatter(agent.read_text(encoding="utf-8"), agent)
    assert fields.get("name") == agent.stem, (
        f"{agent.name}: frontmatter name {fields.get('name')!r} "
        f"does not match filename stem {agent.stem!r}"
    )
    assert fields.get("description"), f"{agent.name}: missing description"
    model = fields.get("model", "")
    assert model in _VALID_MODELS or model.startswith("claude-"), (
        f"{agent.name}: invalid model {model!r} (expected one of {sorted(_VALID_MODELS)} "
        f"or a claude-* id)"
    )
    assert fields.get("tools"), f"{agent.name}: missing tools (least-privilege grant)"
    assert fields.get("color"), f"{agent.name}: missing color"


@pytest.mark.parametrize("agent", _agent_files(), ids=lambda p: p.stem)
def test_agent_cited_paths_exist(agent: Path) -> None:
    text = agent.read_text(encoding="utf-8")
    missing = [p for p in sorted(_cited_paths(text)) if not (REPO_ROOT / p).exists()]
    assert not missing, f"{agent.name} cites non-existent path(s): {missing}"


def test_review_eval_command_references_real_agents() -> None:
    cmd = COMMANDS_DIR / "review-eval.md"
    assert cmd.is_file(), f"missing {cmd}"
    text = cmd.read_text(encoding="utf-8")
    referenced = set(re.findall(r"\betk-[a-z-]+\b", text))
    existing = {p.stem for p in _agent_files()}
    unknown = referenced - existing
    assert not unknown, f"/review-eval references unknown agents: {sorted(unknown)}"
    # Every shipped agent should be wired into the command.
    assert (
        existing <= referenced
    ), f"agents not referenced by /review-eval: {sorted(existing - referenced)}"


def _adr_refs_resolve(text: str) -> list[str]:
    """Return prose `ADR NNNN` references that have no matching ADR file."""
    adr_dir = REPO_ROOT / "docs" / "source" / "adr"
    missing = []
    for num in _ADR_REF_RE.findall(text):
        if not list(adr_dir.glob(f"{int(num):04d}-*.md")):
            missing.append(f"ADR {num}")
    return missing


@pytest.mark.parametrize("agent", _agent_files(), ids=lambda p: p.stem)
def test_agent_adr_references_resolve(agent: Path) -> None:
    missing = _adr_refs_resolve(agent.read_text(encoding="utf-8"))
    assert not missing, f"{agent.name} references non-existent ADR(s): {missing}"


def test_review_eval_command_cited_paths_exist() -> None:
    cmd = COMMANDS_DIR / "review-eval.md"
    text = cmd.read_text(encoding="utf-8")
    missing = [p for p in sorted(_cited_paths(text)) if not (REPO_ROOT / p).exists()]
    assert not missing, f"{cmd.name} cites non-existent path(s): {missing}"
    adr_missing = _adr_refs_resolve(text)
    assert not adr_missing, f"{cmd.name} references non-existent ADR(s): {adr_missing}"
