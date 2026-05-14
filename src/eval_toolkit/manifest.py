"""Run manifest: NeurIPS-aligned reproducibility sidecar.

A :class:`RunManifest` is the metadata sidecar for a single run, written to
``<run_dir>/manifest.json`` next to the existing ``results.json`` /
``results_full.json`` produced by :func:`eval_toolkit.harness.write_run_result`.

Aligned with the [NeurIPS Reproducibility Checklist](https://neurips.cc/public/guides/PaperChecklist):
captures git provenance (with dirty-flag), code versions, seeds, data hashes,
config hash, environment fingerprint, GPU info, wall-clock time, and any
:class:`~eval_toolkit.leakage.LeakageReport` produced inline by the harness.

Per-object versions of any :class:`~eval_toolkit.leakage.Versioned`
implementation (Scorer, LeakageCheck, Splitter, ThresholdSelector,
DatasetLoader) are auto-captured into ``versioned_objects`` so cross-version
metric comparisons can be invalidated explicitly. Mirrors the
``lm-evaluation-harness`` task ``VERSION`` field pattern.

Pure / IO split mirrors :func:`eval_toolkit.harness.evaluate` /
:func:`eval_toolkit.harness.write_run_result` — :func:`build_manifest` is
deterministic and side-effect-free; :func:`write_manifest` is the sole IO sink.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval_toolkit._version import __version__ as eval_toolkit_version
from eval_toolkit.artifacts import PredictionArtifactRef, sanitize_for_json, write_json_strict
from eval_toolkit.provenance import FileHash, capture_git_sha, compute_file_hash

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "SourceRoleRecord",
    "build_manifest",
    "gpu_info",
    "validate_source_roles",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class SourceRoleRecord:
    """Generic source-role metadata for data-quality and evidence audits.

    Roles are intentionally user-defined. Recommended values include
    ``train``, ``calibration``, ``locked_eval``, ``external_diagnostic``,
    and ``excluded``.
    """

    source: str
    role: str
    n_rows: int | None = None
    notes: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation with ``None`` fields omitted."""
        out: dict[str, object] = {
            "source": self.source,
            "role": self.role,
            "notes": self.notes,
            "metadata": self.metadata,
        }
        if self.n_rows is not None:
            out["n_rows"] = self.n_rows
        return out


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Reproducibility metadata sidecar for a single run.

    Pure value type. JSON-serializable via :meth:`to_dict`. Written to
    ``manifest.json`` next to ``results.json`` by :func:`write_manifest`.

    Parameters
    ----------
    run_id : str
        Caller-supplied run identifier (timestamp / UUID).
    captured_at : str
        ISO-8601 UTC timestamp captured at :func:`build_manifest` call time
        (``"YYYY-MM-DDTHH:MM:SSZ"``). v2 field — distinguishes wall-clock
        capture from caller-meaningful ``run_id``.
    git_sha : str or None
        Repo HEAD commit SHA. ``None`` if not in a git repo or git unavailable.
    dirty_flag : bool
        True iff the working tree had uncommitted changes when the run was
        captured. NeurIPS-checklist concern: a "clean" replay needs no dirty
        bits.
    code_versions : dict[str, str]
        ``{package_name: version_string}`` for the toolkit and any caller-
        declared key dependencies (e.g. transformers, torch, sklearn).
    data_revisions : dict[str, str]
        ``{logical_name: revision_string}`` for input dataset and model
        revisions. v2 field — extracts dataset/model provenance out of
        ``code_versions`` (where it sat awkwardly with prefixes like
        ``hf_dataset:`` and ``hf_model:`` in v1 consumers).
    metadata : dict[str, str]
        ``{label_name: label_value}`` for caller-supplied run-time labels
        (CLI arguments, environment selectors, etc.). v2 field — replaces
        the v1 pattern of stuffing labels into ``code_versions`` under a
        ``meta:`` prefix.
    seeds : dict[str, int]
        Per-source seeds (``"global"``, ``"bootstrap"``, ``"torch"``, etc.)
        as captured by the caller.
    data_hashes : dict[str, str]
        ``{filename: "sha256:..."}`` for every artifact the run consumed.
        Populated from :func:`eval_toolkit.provenance.file_sha256`.
    config_hash : str
        SHA-256 of the canonical-JSON eval config (sorted keys, no whitespace).
    env : dict[str, str]
        Environment fingerprint: Python version, platform, key dep versions.
    gpu_info : dict[str, object]
        GPU model / count / memory captured via ``nvidia-smi`` (empty dict if
        unavailable). NeurIPS compute-resources field. v2 tightens
        ``count`` to ``int`` and ``memory_gb`` to ``float``; ``name``
        remains ``str``.
    cuda_version : str or None
        CUDA toolkit / driver version (empty if no CUDA).
    wall_clock_seconds : float or None
        Caller-supplied total run duration in seconds.
    versioned_objects : dict[str, str]
        ``{object_name: version_string}`` auto-captured for every
        :class:`~eval_toolkit.leakage.Versioned` Tier-2 implementation in
        the run (scorers, checks, splitters, etc.). Used to invalidate
        cross-version metric comparisons.
    leakage_report : dict[str, object] or None
        Serialized :class:`~eval_toolkit.leakage.LeakageReport` produced
        inline by the harness, or ``None`` if no checks ran.
    source_roles : list[dict[str, object]]
        Optional generic source-role records for data-quality and evidence
        audits. Roles are caller-defined; the toolkit only validates shape.
    guardrails : list[str]
        Optional predeclared guardrails for claim/evidence discipline.
    prediction_artifacts : list[dict[str, object]]
        Optional references to retained prediction artifacts.
    schema_version : str
        Manifest schema version. ``"v2"`` starting v0.14.0; ``"v1"`` for
        manifests written by v0.7.0–v0.13.x.
    """

    run_id: str
    captured_at: str = ""
    git_sha: str | None = None
    dirty_flag: bool = False
    code_versions: dict[str, str] = field(default_factory=dict)
    data_revisions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    data_hashes: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    env: dict[str, str] = field(default_factory=dict)
    gpu_info: dict[str, object] = field(default_factory=dict)
    cuda_version: str | None = None
    wall_clock_seconds: float | None = None
    versioned_objects: dict[str, str] = field(default_factory=dict)
    leakage_report: dict[str, object] | None = None
    source_roles: list[dict[str, object]] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    prediction_artifacts: list[dict[str, object]] = field(default_factory=list)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        out = sanitize_for_json(asdict(self))
        if not isinstance(out, dict):
            raise TypeError("RunManifest.to_dict expected a mapping payload")
        return out


def _hash_canonical_json(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the JSON-canonical encoding of ``payload``."""
    blob = json.dumps(
        sanitize_for_json(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(blob.encode()).hexdigest()}"


def _is_git_dirty(repo_root: Path | str | None = None) -> bool:
    """True iff ``git status --porcelain`` shows uncommitted changes.

    Returns ``False`` if git is unavailable or not in a repo.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def gpu_info() -> tuple[dict[str, object], str | None]:
    """Capture ``(gpu_info_dict, cuda_version)`` via ``nvidia-smi``.

    Returns ``({}, None)`` if ``nvidia-smi`` is unavailable, fails, or
    times out. Never raises — meant to be called unconditionally from
    :func:`build_manifest` with graceful fallback for CPU-only environments.

    Returns
    -------
    tuple[dict[str, object], str | None]
        First element: ``{"name": "...", "count": 1, "memory_gb": 40.0}``
        or ``{}`` if no GPU. ``count`` is ``int`` and ``memory_gb`` is
        ``float`` since v0.14.0 (v0.13.x and earlier returned strings).
        Second element: CUDA driver version string, or ``None``.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}, None
    if result.returncode != 0 or not result.stdout.strip():
        return {}, None
    rows = [r.strip() for r in result.stdout.strip().splitlines() if r.strip()]
    if not rows:
        return {}, None
    # Each row: "Tesla A100, 40960, 535.86.10"
    first_parts = [p.strip() for p in rows[0].split(",")]
    if len(first_parts) < 3:
        return {}, None
    name = first_parts[0]
    memory_mb = first_parts[1]
    driver_version = first_parts[2]
    info: dict[str, object] = {"name": name, "count": len(rows)}
    with contextlib.suppress(ValueError):
        info["memory_gb"] = round(int(memory_mb) / 1024, 1)
    return info, driver_version


def _capture_env() -> dict[str, str]:
    """Snapshot Python + platform + key library versions."""
    env: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "eval_toolkit": eval_toolkit_version,
    }
    # Optional libraries: only record if importable.
    for lib in ("numpy", "scipy", "sklearn", "pandas", "torch", "transformers"):
        try:
            mod = __import__(lib)
            env[lib] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            continue
    return env


def _collect_versioned(objects: Sequence[object] | Mapping[str, object]) -> dict[str, str]:
    """Auto-capture ``version`` from any object that exposes it.

    Accepts a sequence or a mapping. For sequences the key is the object's
    class name; for mappings the user-supplied key is used.
    """
    out: dict[str, str] = {}
    items = (
        objects.items()
        if isinstance(objects, Mapping)
        else ((type(o).__name__, o) for o in objects)
    )
    for key, obj in items:
        version = getattr(obj, "version", None)
        if isinstance(version, str):
            out[key] = version
    return out


def _source_role_to_dict(record: SourceRoleRecord | Mapping[str, object]) -> dict[str, object]:
    """Normalize a source-role record to a plain dict."""
    if isinstance(record, SourceRoleRecord):
        return record.to_dict()
    return dict(record)


def validate_source_roles(
    source_roles: Sequence[SourceRoleRecord | Mapping[str, object]],
    *,
    required_roles: Sequence[str] = (),
) -> list[str]:
    """Return validation errors for generic source-role records.

    Validation is deliberately about shape, not taxonomy. Consumers choose
    role names that fit their domain.

    Uniqueness contract (relaxed in v0.14.2, closing F4.5): each
    ``(source, role)`` pair must be unique. The same upstream ``source``
    may appear multiple times as long as each occurrence has a distinct
    ``role`` (e.g. the same dataset feeding both a training pool and an
    OOD diagnostic slice). Prior versions required ``source`` to be
    globally unique, which forced callers to synthesize per-slice source
    names like ``notinject_train_neg_pool`` and ``notinject_ood_fpr_only``
    just to satisfy the validator.
    """
    errors: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    roles_seen: set[str] = set()
    for idx, record in enumerate(source_roles):
        row = _source_role_to_dict(record)
        prefix = f"source_roles[{idx}]"
        source = row.get("source")
        role = row.get("role")
        source_ok = isinstance(source, str) and source.strip()
        role_ok = isinstance(role, str) and role.strip()
        if not source_ok:
            errors.append(f"{prefix}: source must be a non-empty string")
        if not role_ok:
            errors.append(f"{prefix}: role must be a non-empty string")
        if source_ok and role_ok:
            assert isinstance(source, str)
            assert isinstance(role, str)
            pair = (source, role)
            if pair in seen_pairs:
                errors.append(f"{prefix}: duplicate (source, role) pair " f"({source!r}, {role!r})")
            else:
                seen_pairs.add(pair)
            roles_seen.add(role)
        n_rows = row.get("n_rows")
        if n_rows is not None and (
            not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows < 0
        ):
            errors.append(f"{prefix}: n_rows must be a non-negative integer when present")
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            errors.append(f"{prefix}: notes must be a string when present")
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            errors.append(f"{prefix}: metadata must be an object when present")

    missing = sorted(set(required_roles) - roles_seen)
    if missing:
        errors.append(f"missing required source role(s): {missing}")
    return errors


def _utc_now_iso8601() -> str:
    """Return current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` (1-second resolution)."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    run_id: str,
    config: Mapping[str, Any],
    data_files: Mapping[str, Path | str] | None = None,
    seeds: Mapping[str, int] | None = None,
    extra_code_versions: Mapping[str, str] | None = None,
    data_revisions: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
    versioned: Sequence[object] | Mapping[str, object] | None = None,
    leakage_report: object | None = None,
    source_roles: Sequence[SourceRoleRecord | Mapping[str, object]] | None = None,
    required_source_roles: Sequence[str] = (),
    guardrails: Sequence[str] | None = None,
    prediction_artifacts: Sequence[PredictionArtifactRef | Mapping[str, object]] | None = None,
    wall_clock_seconds: float | None = None,
    repo_root: Path | str | None = None,
    git_sha: str | None = None,
) -> RunManifest:
    """Pure builder: assemble a :class:`RunManifest` from components.

    Parameters
    ----------
    run_id : str
        Caller-supplied run identifier.
    config : Mapping
        Eval config; SHA-256 of its canonical-JSON encoding becomes
        :attr:`RunManifest.config_hash`.
    data_files : Mapping[str, Path | str] or None
        ``{logical_name: filesystem_path}`` for every input artifact.
        Each path is hashed via :func:`provenance.file_sha256` and the
        results land in :attr:`RunManifest.data_hashes`.
    seeds : Mapping[str, int] or None
        ``{source: seed}`` (``"global"``, ``"bootstrap"``, ``"torch"``).
    extra_code_versions : Mapping[str, str] or None
        Caller-declared package versions to record alongside the auto-
        captured env (e.g. for in-repo subpackages with no PyPI version).
    data_revisions : Mapping[str, str] or None
        v2 — dataset / model revisions, keyed by logical name. Recommended
        key prefixes: ``"hf_dataset:<name>"`` and ``"hf_model:<name>"``.
    metadata : Mapping[str, str] or None
        v2 — caller-supplied run-time labels (CLI arguments, environment
        selectors, etc.). Use this for non-semantic provenance that does
        not belong in ``config`` (which feeds ``config_hash``).
    versioned : sequence of objects or Mapping[str, object] or None
        Tier-2 implementations whose ``version`` attribute should be
        captured. Mapping keys override the default class-name keying.
    leakage_report : object or None
        Mapping or object with ``to_dict()``, serialized into
        :attr:`RunManifest.leakage_report`.
    source_roles : sequence of SourceRoleRecord or mapping, optional
        Generic source-role records. Recommended role names are documented in
        :class:`SourceRoleRecord`, but not enforced.
    required_source_roles : sequence of str, optional
        Roles that must appear in ``source_roles`` if source-role metadata is
        supplied.
    guardrails : sequence of str, optional
        Predeclared guardrails for claim/evidence discipline.
    prediction_artifacts : sequence, optional
        Manifest references to retained prediction artifacts.
    wall_clock_seconds : float or None
    repo_root : Path or str or None
        Where to run ``git status`` for the dirty-flag check.
    git_sha : str or None, optional
        Explicit git SHA override. When provided, used as
        :attr:`RunManifest.git_sha` directly (callers that have already
        captured the SHA — e.g. from a CI environment variable on a pod
        with no ``.git/`` directory — skip the
        :func:`~eval_toolkit.provenance.capture_git_sha` fallback). When
        ``None`` (default), :func:`capture_git_sha` is used to auto-detect.

    Returns
    -------
    RunManifest
        With ``captured_at`` auto-populated to the current UTC time (ISO-8601,
        1-second resolution).

    Examples
    --------
    >>> from eval_toolkit.manifest import build_manifest
    >>> m = build_manifest(run_id="demo", config={"k": 5})
    >>> m.run_id, m.schema_version
    ('demo', 'v2')
    """
    data_hashes: dict[str, str] = {}
    if data_files:
        for logical_name, path in data_files.items():
            # Manifest format prefixes hashes with their algorithm for
            # self-describing JSON (matches Croissant's distribution.sha256).
            # v0.15.0: sentinel-typed compute_file_hash replaces the
            # str | None file_sha256; pattern-match clarifies the miss path.
            digest = compute_file_hash(path)
            if isinstance(digest, FileHash):
                data_hashes[logical_name] = f"sha256:{digest.sha256}"
            else:
                data_hashes[logical_name] = ""

    code_versions: dict[str, str] = {"eval_toolkit": eval_toolkit_version}
    if extra_code_versions:
        code_versions.update(dict(extra_code_versions))

    source_role_dicts = (
        [_source_role_to_dict(record) for record in source_roles] if source_roles else []
    )
    source_role_errors = validate_source_roles(
        source_role_dicts,
        required_roles=required_source_roles,
    )
    if source_role_errors:
        joined = "; ".join(source_role_errors)
        raise ValueError(f"invalid source_roles: {joined}")

    guardrail_list = list(guardrails) if guardrails else []
    bad_guardrails = [g for g in guardrail_list if not isinstance(g, str) or not g.strip()]
    if bad_guardrails:
        raise ValueError("guardrails must be non-empty strings")

    prediction_artifact_dicts = (
        [_prediction_artifact_to_dict(record) for record in prediction_artifacts]
        if prediction_artifacts
        else []
    )

    g_info, cuda = gpu_info()

    return RunManifest(
        run_id=run_id,
        captured_at=_utc_now_iso8601(),
        git_sha=git_sha if git_sha is not None else capture_git_sha(repo_root),
        dirty_flag=_is_git_dirty(repo_root),
        code_versions=code_versions,
        data_revisions=dict(data_revisions) if data_revisions else {},
        metadata=dict(metadata) if metadata else {},
        seeds=dict(seeds) if seeds else {},
        data_hashes=data_hashes,
        config_hash=_hash_canonical_json(dict(config)),
        env=_capture_env(),
        gpu_info=g_info,
        cuda_version=cuda,
        wall_clock_seconds=wall_clock_seconds,
        versioned_objects=_collect_versioned(versioned) if versioned else {},
        leakage_report=_object_to_dict(leakage_report) if leakage_report is not None else None,
        source_roles=source_role_dicts,
        guardrails=guardrail_list,
        prediction_artifacts=prediction_artifact_dicts,
        schema_version=MANIFEST_SCHEMA_VERSION,
    )


def write_manifest(manifest: RunManifest, run_dir: Path | str) -> Path:
    """Write ``manifest.json`` to ``run_dir``. Sole IO sink.

    Parameters
    ----------
    manifest : RunManifest
    run_dir : Path or str
        Destination directory; created if it doesn't exist.

    Returns
    -------
    Path
        Path to the written ``manifest.json``.
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.json"
    return write_json_strict(manifest.to_dict(), out_path)


def _object_to_dict(obj: object) -> dict[str, object]:
    """Normalize mapping or ``to_dict()`` object to a strict-JSON-safe dict."""
    if isinstance(obj, Mapping):
        out = sanitize_for_json(dict(obj))
    else:
        to_dict = getattr(obj, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"expected mapping or object with to_dict(), got {type(obj).__name__}")
        out = sanitize_for_json(to_dict())
    if not isinstance(out, dict):
        raise TypeError(f"expected mapping payload, got {type(out).__name__}")
    return out


def _prediction_artifact_to_dict(
    artifact: PredictionArtifactRef | Mapping[str, object],
) -> dict[str, object]:
    """Normalize a prediction artifact reference to a plain dict."""
    if isinstance(artifact, PredictionArtifactRef):
        return artifact.to_dict()
    out = sanitize_for_json(dict(artifact))
    if not isinstance(out, dict):
        raise TypeError("prediction artifact must normalize to a mapping")
    if not isinstance(out.get("uri"), str) or not out.get("uri"):
        raise ValueError("prediction artifact must include a non-empty uri")
    if not isinstance(out.get("media_type"), str) or not out.get("media_type"):
        raise ValueError("prediction artifact must include a non-empty media_type")
    columns = out.get("columns")
    if not isinstance(columns, Mapping):
        raise ValueError("prediction artifact must include a columns object")
    if not isinstance(columns.get("label"), str) or not columns.get("label"):
        raise ValueError("prediction artifact columns must include label")
    if not isinstance(columns.get("score"), str) or not columns.get("score"):
        raise ValueError("prediction artifact columns must include score")
    return out
