"""Dataset loading: pluggable :class:`DatasetLoader` Protocol + reference impls.

A :class:`DatasetLoader` produces a ``dict[str, EvalSlice]`` keyed by split
name (HuggingFace ``DatasetDict`` shape). For un-split data, return
``{"all": slice}`` and pipe through a :class:`~eval_toolkit.splits.Splitter`
for cross-validation. For pre-split data (``{"train", "validation", "test"}``),
the harness consumes the dict directly.

Reference impls cover the four prompt-injection-* projects' shapes:
:class:`DataFrameLoader` (in-memory dataframe + a split column),
:class:`SingleSliceLoader` (already-built EvalSlice as ``{"all": ...}``),
:class:`ParquetGlobLoader` (load + concat parquet files; hashes for the
manifest), and :class:`HFDatasetsLoader` (optional, soft-imports
``datasets`` only if installed).

The :meth:`describe` method on every loader returns a Croissant-compatible
metadata subset (``name``, ``description``, ``cite_as``, ``license``, ``url``,
``distribution: list[{name, contentUrl, sha256, contentSize}]``) so manifests
interoperate with the MLCommons standard without forcing full Croissant
production.

References
----------
.. [1] Croissant: A Metadata Format for ML-Ready Datasets. arXiv:2403.19546.
.. [2] HuggingFace ``datasets.DatasetDict`` — the convergent named-splits
       shape adopted by ``lm-evaluation-harness``, ``HELM``, and Inspect AI.
"""

from __future__ import annotations

import glob as _glob
import json as _json
import logging
import urllib.request as _urlrequest
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from eval_toolkit.harness import EvalSlice
from eval_toolkit.provenance import file_sha256

_HF_HUB_BASE = "https://huggingface.co"
_HF_FETCH_TIMEOUT_SEC = 15


def _hf_get_json(path: str) -> Any:
    """GET ``https://huggingface.co{path}`` and return parsed JSON.

    Stdlib-only (no ``requests`` / ``huggingface_hub`` dep). Raises
    ``OSError`` (urllib error) or ``ValueError`` (JSON decode) on
    failure — callers catch both. The 15-second timeout caps any one
    fetch so CI doesn't hang on a slow HF Hub.
    """
    url = f"{_HF_HUB_BASE}{path}"
    req = _urlrequest.Request(url, headers={"User-Agent": "eval-toolkit"})
    with _urlrequest.urlopen(req, timeout=_HF_FETCH_TIMEOUT_SEC) as resp:
        return _json.loads(resp.read().decode("utf-8"))


_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DataFrameLoader",
    "DatasetLoader",
    "HFDatasetsLoader",
    "OodManifestLoader",
    "ParquetGlobLoader",
    "SingleSliceLoader",
    "ood_dataset_from_manifest",
]


def _check_required_columns(
    df: pd.DataFrame,
    required_cols: tuple[str, ...] | list[str],
    *,
    context: str,
) -> None:
    """Raise ``KeyError`` if any required column is missing from ``df``.

    Centralizes the column-validation pattern used by every loader's
    ``__post_init__`` / ``load_splits`` (v0.30.0 refactor #2 — DRY).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to validate.
    required_cols : tuple or list of str
        Column names that must be present.
    context : str
        Caller-supplied prefix for the error message (e.g.,
        ``"DataFrameLoader"`` or ``"ParquetGlobLoader: split 'train'"``).
        Identifies *which* loader / split is missing the column.

    Raises
    ------
    KeyError
        On the first missing column. Error message includes the
        missing column name and the available columns list.
    """
    available = list(df.columns)
    for col in required_cols:
        if col not in available:
            raise KeyError(f"{context}: missing column {col!r}; available: {available}")


@runtime_checkable
class DatasetLoader(Protocol):
    """Yields one or more :class:`EvalSlice` keyed by split name.

    Mirrors HuggingFace ``DatasetDict``. For un-split data, return
    ``{"all": slice}`` and let a :class:`~eval_toolkit.splits.Splitter`
    produce folds. Tensor-agnostic: the
    :class:`~eval_toolkit.harness.Scorer` handles tokenization / tensor
    conversion / device placement.
    """

    def load_splits(self) -> dict[str, EvalSlice]:  # pragma: no cover
        """Return a ``{split_name: EvalSlice}`` dict."""
        ...

    def describe(self) -> dict[str, object]:  # pragma: no cover
        """Croissant-subset metadata for the manifest.

        Recommended keys:

        - ``name``: dataset name
        - ``description``: short description
        - ``cite_as``: BibTeX or arXiv id
        - ``license``: SPDX identifier
        - ``url``: canonical URL
        - ``distribution``: list of ``{name, contentUrl, sha256, contentSize}``
        """
        ...


@dataclass(frozen=True, slots=True)
class DataFrameLoader:
    """Wraps a single in-memory dataframe + a split column.

    Splits the dataframe by the unique values in ``split_col``. Covers the
    four prompt-injection-* projects' shape (a single parquet with a
    ``split`` column). For pre-split data without a single column,
    concatenate first and synthesize a ``split`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    split_col : str
        Column whose unique values name the splits (e.g. ``"split"`` with
        values ``"train" / "val" / "test"``).
    feature_col : str, optional
        Column carrying the feature passed to ``Scorer.predict_proba``.
        Default ``"text"``.
    label_col : str, optional
        Column with binary labels in ``{0, 1}``. Default ``"label"``.
    strata_col : str or None, optional
        Optional stratifier column for per-stratum recall reporting.
    name : str, optional
        Dataset name used in :meth:`describe`. Default ``""``.
    description : str, optional
        Free-text description used in :meth:`describe`. Default ``""``.
    cite_as : str, optional
        Citation string (BibTeX, arXiv id) used in :meth:`describe`.
    license : str, optional
        SPDX license identifier used in :meth:`describe`.
    url : str, optional
        Canonical URL used in :meth:`describe`.
    """

    df: pd.DataFrame
    split_col: str
    feature_col: str = "text"
    label_col: str = "label"
    strata_col: str | None = None
    name: str = ""
    description: str = ""
    cite_as: str = ""
    license: str = ""
    url: str = ""

    def __post_init__(self) -> None:
        """Validate the dataframe has the columns we'll read."""
        _check_required_columns(
            self.df,
            (self.split_col, self.feature_col, self.label_col),
            context="DataFrameLoader",
        )
        if self.strata_col is not None and self.strata_col not in self.df.columns:
            raise KeyError(
                f"DataFrameLoader: missing strata column {self.strata_col!r}; "
                f"available: {list(self.df.columns)}"
            )

    def load_splits(self) -> dict[str, EvalSlice]:
        """Group rows by ``split_col`` and build one EvalSlice per group."""
        out: dict[str, EvalSlice] = {}
        for split_name, sub_df in self.df.groupby(self.split_col, sort=False):
            sub_df_reset = sub_df.reset_index(drop=True)
            out[str(split_name)] = EvalSlice(
                name=str(split_name),
                df=sub_df_reset,
                description=self.description,
                feature_col=self.feature_col,
                label_col=self.label_col,
                strata_col=self.strata_col,
            )
        _logger.debug(
            "DataFrameLoader.load_splits: %d splits constructed (%s)",
            len(out),
            ", ".join(f"{k}={len(v.df)}" for k, v in out.items()),
        )
        return out

    def describe(self) -> dict[str, object]:
        """Croissant-subset metadata. ``distribution`` empty (no file artifacts)."""
        return {
            "name": self.name or "DataFrameLoader",
            "description": self.description,
            "citeAs": self.cite_as,
            "license": self.license,
            "url": self.url,
            "distribution": [],
            "n_total_rows": int(len(self.df)),
            "split_col": self.split_col,
        }


@dataclass(frozen=True, slots=True)
class SingleSliceLoader:
    """Wraps a single pre-built :class:`EvalSlice` as ``{"all": slice}``.

    The trivial entry point into the :class:`~eval_toolkit.splits.Splitter`
    pipeline when you already have an EvalSlice and want to run K-fold or
    holdout on it.

    Parameters
    ----------
    slice_ : EvalSlice
        The parent slice. Will be re-keyed as ``"all"``.
    name : str, optional
        Dataset name for :meth:`describe`. Default ``""``.
    description : str, optional
        Free-text description. Default ``""``.
    """

    slice_: EvalSlice
    name: str = ""
    description: str = ""

    def load_splits(self) -> dict[str, EvalSlice]:
        """Return ``{"all": <renamed-slice>}``."""
        renamed = EvalSlice(
            name="all",
            df=self.slice_.df,
            description=self.description or self.slice_.description,
            feature_col=self.slice_.feature_col,
            label_col=self.slice_.label_col,
            strata_col=self.slice_.strata_col,
        )
        return {"all": renamed}

    def describe(self) -> dict[str, object]:
        """Croissant-subset metadata."""
        return {
            "name": self.name or "SingleSliceLoader",
            "description": self.description,
            "citeAs": "",
            "license": "",
            "url": "",
            "distribution": [],
            "n_total_rows": int(len(self.slice_.df)),
        }


@dataclass(frozen=True, slots=True)
class ParquetGlobLoader:
    """Load + concat parquet files matched by a glob; hash each for the manifest.

    For each glob, all matching files are loaded with
    :func:`pandas.read_parquet` and concatenated. Each file's SHA-256 is
    captured in :meth:`describe` so manifests are reproducible.

    Parameters
    ----------
    splits : dict[str, str]
        Mapping ``{split_name: glob_pattern}``. Each pattern's matching
        files become one split. Patterns are evaluated relative to the
        process CWD; pass absolute paths for determinism.
    feature_col : str, optional
        Column name. Default ``"text"``.
    label_col : str, optional
        Column name. Default ``"label"``.
    strata_col : str or None, optional
    name, description, cite_as, license, url : str, optional
        Croissant metadata fields.
    """

    splits: dict[str, str]
    feature_col: str = "text"
    label_col: str = "label"
    strata_col: str | None = None
    name: str = ""
    description: str = ""
    cite_as: str = ""
    license: str = ""
    url: str = ""

    def _resolve_files(self) -> dict[str, list[Path]]:
        """Expand each glob pattern to a sorted list of Path objects."""
        out: dict[str, list[Path]] = {}
        for split_name, pattern in self.splits.items():
            files = sorted(Path(p) for p in _glob.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"ParquetGlobLoader: glob {pattern!r} matched no files for split "
                    f"{split_name!r}"
                )
            out[split_name] = files
        return out

    def load_splits(self) -> dict[str, EvalSlice]:
        """Read + concat each split's parquet files into an EvalSlice.

        Raises
        ------
        KeyError
            If any split's loaded DataFrame is missing ``feature_col`` or
            ``label_col``.
        FileNotFoundError
            If any split's glob pattern matches no files (the error names
            the pattern and the split).
        """
        import pandas as pd  # function-local: pandas is in the [parquet]/[dataframe] extra

        files_by_split = self._resolve_files()
        out: dict[str, EvalSlice] = {}
        for split_name, files in files_by_split.items():
            parts = [pd.read_parquet(p) for p in files]
            df = pd.concat(parts, axis=0, ignore_index=True)
            _check_required_columns(
                df,
                (self.feature_col, self.label_col),
                context=f"ParquetGlobLoader: split {split_name!r}",
            )
            out[split_name] = EvalSlice(
                name=split_name,
                df=df,
                description=self.description,
                feature_col=self.feature_col,
                label_col=self.label_col,
                strata_col=self.strata_col,
            )
        return out

    def describe(self) -> dict[str, object]:
        """Croissant-subset metadata with per-file SHA-256 in ``distribution``.

        Raises
        ------
        FileNotFoundError
            If any split's glob pattern matches no files (same resolution
            as :meth:`load_splits`).
        """
        files_by_split = self._resolve_files()
        distribution: list[dict[str, object]] = []
        for split_name, files in files_by_split.items():
            for f in files:
                size = f.stat().st_size if f.exists() else 0
                sha = file_sha256(f, strict=False)
                distribution.append(
                    {
                        "name": f"{split_name}/{f.name}",
                        "contentUrl": str(f),
                        "sha256": sha,
                        "contentSize": int(size),
                    }
                )
        return {
            "name": self.name or "ParquetGlobLoader",
            "description": self.description,
            "citeAs": self.cite_as,
            "license": self.license,
            "url": self.url,
            "distribution": distribution,
        }


@dataclass(frozen=True, slots=True)
class HFDatasetsLoader:
    """Load a HuggingFace ``datasets`` repo as ``{split: EvalSlice}``.

    Soft dependency on the ``datasets`` package: an :class:`ImportError` is
    raised at :meth:`load_splits` time with a clear install hint. This is
    intentional — eval-toolkit's core deps are numpy / scipy / sklearn only.

    Since v0.41.0, :meth:`describe` enriches its output with per-file
    ``sha256`` hashes fetched from the HF Hub tree API (the ``lfs.oid``
    field), plus Croissant metadata fetched from HF Hub's Croissant
    endpoint. The dual-source design is documented in
    ``methodology/reproducibility.md`` §"Croissant interoperability";
    in short: HF Hub's Croissant emitter currently punts the
    ``distribution[].sha256`` field per MLCommons Croissant issue #80
    (open), so we read the authoritative sha256 from the tree API's
    ``lfs.oid`` instead. When #80 resolves and HF Hub starts populating
    Croissant ``sha256`` with real values, the implementation collapses
    to a single source.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repo, e.g. ``"stanfordnlp/sst2"``.
    splits : sequence of str or None, optional
        Subset of HF splits to load. ``None`` = every split the repo defines.
    feature_col : str, optional
        Column name in the HF dataset. Default ``"text"``.
    label_col : str, optional
        Column name. Default ``"label"``.
    strata_col : str or None, optional
        Column carrying per-row stratum labels, propagated to each
        :class:`EvalSlice` for stratified slicing. ``None`` (default)
        disables strata propagation.
    config_name : str or None, optional
        HF dataset config name (some datasets have multiple configs).
    feature_cols : sequence of str or None, optional
        Multiple feature columns stringified and joined (in order) into one
        synthesized feature column, e.g. a system+user prompt pair. When
        set, takes precedence over ``feature_col``. Any missing/NaN cell
        raises ``ValueError`` at :meth:`load_splits`. Added in v1.5.0.
    feature_join : str, optional
        Separator joining the ``feature_cols`` values. Default: two
        newlines. Added in v1.5.0.
    label_map : mapping or None, optional
        Remap raw label values to ints (e.g. ``{"benign": 0,
        "injection": 1}``); any uncovered value raises ``ValueError``.
        Added in v1.5.0.
    revision : str or None, optional
        Git revision (tag, branch, or commit sha) forwarded to
        ``datasets.load_dataset``; echoed in :meth:`describe` output —
        see its Notes for the tree-hash limitation. Added in v1.5.0.
    name, description, cite_as, license, url : str, optional
        Croissant metadata overrides. If empty, :meth:`describe` will
        fall back to fetching from HF Hub's Croissant endpoint.
    fetch_remote_metadata : bool, optional
        If ``True`` (default), :meth:`describe` fetches Croissant + tree
        metadata from HF Hub. Set ``False`` to disable network calls
        (useful for offline / unit testing).
    """

    repo_id: str
    splits: Sequence[str] | None = None
    feature_col: str = "text"
    label_col: str = "label"
    strata_col: str | None = None
    config_name: str | None = None
    feature_cols: Sequence[str] | None = None
    feature_join: str = "\n\n"
    label_map: Mapping[Any, int] | None = None
    revision: str | None = None
    name: str = ""
    description: str = ""
    cite_as: str = ""
    license: str = ""
    url: str = ""
    fetch_remote_metadata: bool = True

    def _load_dataset(self) -> Mapping[str, Any]:
        """Soft-import ``datasets`` and return the loaded DatasetDict.

        Returns a ``Mapping[str, Any]`` (HF ``DatasetDict`` is dict-like —
        keys are split names, values are HF ``Dataset`` objects exposing
        ``.to_pandas()``). Annotated as ``Mapping`` rather than concrete
        ``DatasetDict`` so consumers don't need to install ``datasets`` to
        type-check downstream code.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "HFDatasetsLoader requires datasets. Install with: pip install datasets"
            ) from exc
        # Only pass optional kwargs when set, so the default call signature is
        # unchanged (backward-compatible: load_dataset(repo_id) / (repo_id, name=...)).
        kwargs: dict[str, Any] = {}
        if self.config_name is not None:
            kwargs["name"] = self.config_name
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return cast(Mapping[str, Any], load_dataset(self.repo_id, **kwargs))

    def load_splits(self) -> dict[str, EvalSlice]:
        """Convert each requested HF split to an :class:`EvalSlice`.

        When ``feature_cols`` is given, those columns are stringified and joined
        with ``feature_join`` into a single synthesized feature column (e.g. a
        system+user prompt pair). When ``label_map`` is given, raw label values
        are remapped through it (e.g. ``{"benign": 0, "injection": 1}``); any
        value the map does not cover raises ``ValueError`` (fail-fast, no silent
        NaN labels).

        Raises
        ------
        KeyError
            If any split's DataFrame is missing a required feature/label column
            (the error lists the observed columns).
        ValueError
            If ``label_map`` leaves any label value unmapped, or if any
            ``feature_cols`` cell is missing/NaN — the join never fabricates
            empty strings; the error carries the split name and per-column
            NaN counts.
        ImportError
            If the optional ``datasets`` package is not installed (raised by
            the soft import with an install hint).
        """
        ds = self._load_dataset()
        ds_splits = list(ds.keys()) if self.splits is None else list(self.splits)
        feat_cols = list(self.feature_cols) if self.feature_cols else [self.feature_col]
        out: dict[str, EvalSlice] = {}
        for split_name in ds_splits:
            df = ds[split_name].to_pandas()
            _check_required_columns(
                df,
                (*feat_cols, self.label_col),
                context=f"HFDatasetsLoader: split {split_name!r}",
            )
            if self.feature_cols:
                df = df.copy()
                synth = "\x00".join(feat_cols)  # collision-proof synthesized name
                # Fail-fast on missing prompt parts (#98, STYLE §1): pre-#98 a
                # fillna("") here silently fabricated empty strings, shipping
                # truncated prompts while the label path eight lines down
                # fail-fasted. Valid data keeps the pandas-3.0-safe per-column
                # astype(str) before the join.
                na_counts = df[feat_cols].isna().sum()
                if int(na_counts.sum()) > 0:
                    per_col = {str(c): int(n) for c, n in na_counts.items() if int(n) > 0}
                    raise ValueError(
                        f"HFDatasetsLoader: split {split_name!r} has missing values in "
                        f"feature_cols; NaN counts per column: {per_col}. The join does "
                        f"not fabricate empty strings — fix or drop these rows upstream, "
                        f"or exclude the sparse column from feature_cols."
                    )
                cols = df[feat_cols].apply(lambda s: s.astype(str))
                df[synth] = cols.apply(lambda row: self.feature_join.join(row.tolist()), axis=1)
                feature_col = synth
            else:
                feature_col = self.feature_col
            if self.label_map is not None:
                df = df.copy()
                mapped = df[self.label_col].map(self.label_map)
                if mapped.isna().any():
                    unmapped = sorted(set(df[self.label_col][mapped.isna()].unique()), key=str)
                    raise ValueError(
                        f"HFDatasetsLoader: split {split_name!r} label_map did not cover "
                        f"values {unmapped}; map keys: {sorted(self.label_map, key=str)}"
                    )
                df[self.label_col] = mapped
            out[split_name] = EvalSlice(
                name=split_name,
                df=df,
                description=self.description,
                feature_col=feature_col,
                label_col=self.label_col,
                strata_col=self.strata_col,
            )
        return out

    def describe(self) -> dict[str, object]:
        """Croissant-compatible metadata + per-file sha256 from HF Hub.

        When ``fetch_remote_metadata=True`` (default), enriches the
        baseline metadata with two HF Hub API fetches:

        - **Croissant endpoint** (``/api/datasets/{repo}/croissant``) —
          provides ``name``, ``description``, ``citeAs``, ``license``,
          ``url`` defaults when the loader's fields are empty.
        - **Tree API** (``/api/datasets/{repo}/tree/...?recursive=true``) —
          provides per-file ``sha256`` (from ``lfs.oid``) and
          ``contentSize`` for each parquet shard under the
          ``refs/convert/parquet`` branch.

        Network failures degrade gracefully (warning emitted; sha256
        empty as in pre-v0.41 behavior). See class docstring for the
        dual-source rationale (MLCommons Croissant issue #80).

        Notes
        -----
        ``revision`` is echoed verbatim into the output (``None`` when
        unpinned). Limitation: the tree-API fetch hardcodes the
        ``refs/convert/parquet`` ref (HF Hub's auto-converted parquet
        branch), so the per-file ``sha256`` entries always describe the
        *latest* parquet conversion — when ``revision`` pins an older
        snapshot, those hashes may not correspond to the pinned
        revision's data. The ``revision`` key makes a pinned load
        distinguishable from an unpinned one in downstream manifests.
        """
        remote_meta: dict[str, object] = {}
        distribution: list[dict[str, object]] = []
        if self.fetch_remote_metadata:
            remote_meta = self._fetch_croissant_metadata_safe()
            distribution = self._fetch_tree_distribution_safe()

        # Caller-provided fields win; Croissant fills gaps.
        def _pick(local: str, key: str) -> str:
            if local:
                return local
            val = remote_meta.get(key)
            return val if isinstance(val, str) else ""

        if not distribution:
            distribution = [
                {
                    "name": f"hf:{self.repo_id}",
                    "contentUrl": f"https://huggingface.co/datasets/{self.repo_id}",
                    "sha256": "",
                    "contentSize": 0,
                }
            ]

        return {
            "name": _pick(self.name, "name") or self.repo_id,
            "description": _pick(self.description, "description"),
            "citeAs": _pick(self.cite_as, "citeAs"),
            "license": _pick(self.license, "license"),
            "url": self.url or f"https://huggingface.co/datasets/{self.repo_id}",
            "distribution": distribution,
            "config_name": self.config_name,
            "revision": self.revision,
        }

    def _fetch_croissant_metadata_safe(self) -> dict[str, object]:
        """Fetch HF Hub Croissant JSON-LD; return empty dict on any failure."""
        try:
            data = _hf_get_json(f"/api/datasets/{self.repo_id}/croissant")
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:  # urllib.URLError, JSONDecodeError, etc.
            warnings.warn(
                f"HFDatasetsLoader {self.repo_id}: Croissant fetch failed ({exc}); "
                "proceeding without",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

    def _fetch_tree_distribution_safe(self) -> list[dict[str, object]]:
        """Fetch HF Hub tree API for the parquet-convert branch; return ``cr:FileObject`` entries.

        Each entry carries ``sha256`` (from ``lfs.oid`` — the git-LFS
        content hash, equal to ``sha256sum`` of the file content) and
        ``contentSize`` (from the tree response's ``size`` field).

        Falls back to an empty list on any failure — callers should
        treat empty distribution as "no remote provenance available."
        """
        # HF stores native parquet (or auto-converts) under
        # refs/convert/parquet; that's the canonical hash target.
        path = f"/api/datasets/{self.repo_id}/tree/refs%2Fconvert%2Fparquet?recursive=true"
        try:
            entries = _hf_get_json(path)
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"HFDatasetsLoader {self.repo_id}: tree-API fetch failed ({exc}); "
                "sha256 unavailable",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        if not isinstance(entries, list):
            return []
        out: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "file":
                continue
            path_val = entry.get("path", "")
            if not isinstance(path_val, str) or not path_val.endswith(".parquet"):
                continue
            lfs = entry.get("lfs")
            sha = ""
            if isinstance(lfs, dict):
                oid = lfs.get("oid")
                if isinstance(oid, str) and len(oid) == 64:  # sha256 hex
                    sha = f"sha256:{oid}"
            size = entry.get("size", 0)
            out.append(
                {
                    "name": path_val,
                    "contentUrl": (
                        f"https://huggingface.co/datasets/{self.repo_id}"
                        f"/resolve/refs%2Fconvert%2Fparquet/{path_val}"
                    ),
                    "sha256": sha,
                    "contentSize": int(size) if isinstance(size, (int, float)) else 0,
                }
            )
        return out


# ----------------------------------------------------------------------------
# OOD manifest loader (v0.43.0; closes #48)
# ----------------------------------------------------------------------------

_OOD_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "eval-toolkit" / "ood"
_OOD_DOWNLOAD_TIMEOUT_SEC = 60


def _load_yaml_manifest(yaml_path: str | Path) -> dict[str, Any]:
    """Read and parse a YAML manifest file with explicit UTF-8 encoding.

    Soft-imports ``yaml`` (PyYAML); raises ``ImportError`` with install
    hint if the optional ``[yaml]`` extra is not installed. Raises
    ``FileNotFoundError`` if the path does not exist, and ``ValueError``
    on YAML parse error (honoring the contract documented in
    :func:`ood_dataset_from_manifest`) or on a structurally invalid
    manifest.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ood_dataset_from_manifest requires pyyaml. "
            "Install with: pip install eval-toolkit[yaml]"
        ) from exc
    path = Path(yaml_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"OOD manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            # Honor the documented contract: ood_dataset_from_manifest's
            # Raises section promises ValueError on YAML parse error, and
            # callers should not need to import third-party exception types.
            raise ValueError(f"OOD manifest {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"OOD manifest {path} must be a YAML mapping at the top level; got {type(data).__name__}"
        )
    if "slices" not in data or not isinstance(data["slices"], dict):
        raise ValueError(f"OOD manifest {path} missing required top-level 'slices' mapping")
    return data


def _sha256_of_bytes(data: bytes) -> str:
    """Return ``sha256:<hexdigest>`` of bytes — matches Croissant + HF Hub convention."""
    import hashlib

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_of_text(text: str) -> str:
    """Return ``sha256:<hexdigest>`` of UTF-8-encoded text (for ``row_id`` column)."""
    return _sha256_of_bytes(text.encode("utf-8"))


def _download_with_sha_verify(
    url: str,
    expected_sha: str,
    cache_dir: Path,
    *,
    slice_id: str,
) -> Path:
    """Download ``url`` to ``cache_dir/{sha-suffix}.bin``; verify sha256.

    Cache key is the *expected* sha256 (so two slices with the same
    expected hash share storage). On cache hit, re-verifies the cached
    file's sha against the expected value to defend against on-disk
    corruption. On cache miss, downloads via stdlib ``urllib``, verifies,
    and atomically renames into place.

    Parameters
    ----------
    url : str
        HTTP(S) URL to fetch. ``file://`` URLs and bare local paths are
        also supported for offline manifests.
    expected_sha : str
        ``sha256:<hex>`` or bare 64-char hex digest. The downloaded bytes
        must hash to this value or a :class:`ValueError` is raised.
    cache_dir : pathlib.Path
        Directory to store the verified file. Created if absent.
    slice_id : str
        Slice id from the manifest, used only in error messages.

    Returns
    -------
    pathlib.Path
        Path to the verified cached file.

    Raises
    ------
    ValueError
        If the downloaded (or cached) content's sha256 does not match
        ``expected_sha``. Message includes expected vs actual + the
        remediation hint to update the manifest sha.
    OSError
        On network / filesystem failure during download.
    """
    # Normalize expected sha
    expected_hex = (
        expected_sha[len("sha256:") :] if expected_sha.startswith("sha256:") else expected_sha
    )
    if len(expected_hex) != 64:
        raise ValueError(
            f"OOD slice {slice_id!r}: expected_sha must be 64-char hex (with optional 'sha256:' prefix); "
            f"got {expected_sha!r} (len={len(expected_hex)})"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{expected_hex}.bin"

    if cached.exists():
        actual = _sha256_of_bytes(cached.read_bytes())
        if actual == f"sha256:{expected_hex}":
            _logger.debug("OOD slice %s: cache hit at %s", slice_id, cached)
            return cached
        # Corrupted cache — log and re-download
        warnings.warn(
            f"OOD slice {slice_id}: cached file {cached} sha mismatch "
            f"(expected {expected_hex}, got {actual}); re-downloading",
            RuntimeWarning,
            stacklevel=2,
        )
        cached.unlink()

    # Cache miss — download
    if url.startswith(("http://", "https://")):
        req = _urlrequest.Request(url, headers={"User-Agent": "eval-toolkit"})
        with _urlrequest.urlopen(req, timeout=_OOD_DOWNLOAD_TIMEOUT_SEC) as resp:
            content = resp.read()
    else:
        # Treat as local path (bare or file://)
        local_path = url[len("file://") :] if url.startswith("file://") else url
        content = Path(local_path).expanduser().read_bytes()

    actual = _sha256_of_bytes(content)
    if actual != f"sha256:{expected_hex}":
        raise ValueError(
            f"OOD slice {slice_id!r}: sha256 mismatch for {url}\n"
            f"  expected: sha256:{expected_hex}\n"
            f"  actual:   {actual}\n"
            f"  remediation: re-run with --update-manifest to refresh the manifest sha "
            f"after auditing the upstream change"
        )
    # Atomic rename via temp file
    tmp = cached.with_suffix(".bin.tmp")
    tmp.write_bytes(content)
    tmp.rename(cached)
    return cached


def _infer_format_from_url(url: str) -> str | None:
    """Infer file format from a URL or path's extension; ``None`` if unknown."""
    # Strip query string / fragment
    path = url.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(path).suffix.lstrip(".").lower()
    if suffix in ("parquet", "pq", "jsonl", "json", "csv"):
        return suffix
    return None


def _read_slice_dataframe(
    cached_path: Path,
    *,
    fmt: str,
    slice_id: str,
) -> pd.DataFrame:
    """Read a cached file into a pandas DataFrame using an explicit format hint.

    Supports parquet, jsonl, json, and csv. Caller resolves the format
    (manifest field, then URL extension fallback) before calling.
    """
    import pandas as pd

    fmt = fmt.lower()
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(cached_path)
    if fmt in ("jsonl", "json"):
        return pd.read_json(cached_path, lines=(fmt == "jsonl"))
    if fmt == "csv":
        return pd.read_csv(cached_path, encoding="utf-8")
    raise ValueError(
        f"OOD slice {slice_id!r}: unsupported format {fmt!r}; "
        f"set 'format' in the manifest to one of: parquet, jsonl, json, csv"
    )


def _apply_label_map(
    series: pd.Series,
    label_map: Mapping[Any, int] | None,
    *,
    slice_id: str,
) -> pd.Series:
    """Apply ``label_map`` (or pass-through if None) and cast to int.

    With ``label_map=None``, the pass-through path verifies every value is
    finite and exactly integral before the int cast — a bare ``astype(int)``
    silently truncates numeric non-integers (``0.7 -> 0``), the same
    silent-truncation class fixed in ``analysis.load_prediction_arrays`` at
    v1.9.0. Loader labels are generic ints (not necessarily binary), so the
    check is integrality + finiteness, not {0, 1} membership.

    Raises
    ------
    ValueError
        If ``label_map`` is None and the label column contains non-numeric
        values (strings etc.), missing / non-finite values (NaN, inf,
        ``pd.NA``), numeric values with a fractional part, or float values
        too large for an exact float64 integrality check (>= 2**53); or if
        ``label_map`` is given and any value is missing from the map. Every
        message carries the slice id and diagnostic context (offending
        values, or count + first row index for missing values).
    """
    if label_map is None:
        import numpy as np
        import pandas as pd

        if pd.api.types.is_integer_dtype(series) and not series.isna().any():
            # Exact path: integer dtypes (incl. nullable Int64) skip the
            # float64 round-trip below, which would silently corrupt labels
            # >= 2**53 (float64 exactness limit). Nullable ints WITH NA fall
            # through to the float path so they hit the missing-value raise.
            return series.astype(int)
        try:
            as_float = series.astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OOD slice {slice_id!r}: label column has non-integer values and no "
                f"label_map provided; unique values: "
                f"{sorted(set(series.unique()), key=str)[:10]}"
            ) from exc
        values = np.asarray(as_float, dtype=np.float64)
        non_finite = ~np.isfinite(values)
        if non_finite.any():
            n_bad = int(non_finite.sum())
            first_bad = int(np.flatnonzero(non_finite)[0])
            raise ValueError(
                f"OOD slice {slice_id!r}: label column contains {n_bad} missing/non-finite "
                f"value(s) (NaN/inf/NA) and no label_map provided; first at row index "
                f"{first_bad}. Fix the source data or remap via label_map."
            )
        too_large = np.abs(values) >= 2.0**53
        if too_large.any():
            first_bad = int(np.flatnonzero(too_large)[0])
            raise ValueError(
                f"OOD slice {slice_id!r}: label column contains value(s) with magnitude "
                f">= 2**53, which cannot be integrality-checked exactly in float64; "
                f"first at row index {first_bad}. Use an integer dtype for the label column."
            )
        non_integral = values != np.trunc(values)
        if non_integral.any():
            offending = sorted(set(values[non_integral].tolist()))[:10]
            raise ValueError(
                f"OOD slice {slice_id!r}: label column contains non-integer numeric "
                f"value(s) {offending} and no label_map provided; an int cast would "
                f"silently truncate (0.7 -> 0). Provide a label_map or fix the labels."
            )
        return as_float.astype(int)
    mapped = series.map(label_map)
    if mapped.isna().any():
        missing = sorted(set(series[mapped.isna()].unique()), key=str)
        raise ValueError(
            f"OOD slice {slice_id!r}: label_map missing keys {missing!r}; "
            f"map covers {sorted(label_map.keys(), key=str)!r}"
        )
    return mapped.astype(int)


def ood_dataset_from_manifest(
    yaml_path: str | Path,
    *,
    slices: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load multiple OOD eval slates declaratively from a YAML manifest.

    Returns a single concatenated :class:`pandas.DataFrame` with the
    columns ``text``, ``label``, ``source``, ``row_id``, ``sha`` — one
    row per example across every slice. Each source's bytes are fetched
    from the URL, sha256-verified against the manifest, and cached
    on-disk keyed by content hash (second call hits cache).

    Designed as a metadata-driven primitive for portfolio / submission
    consumers that load BIPIA, AgentDojo, InjecAgent, NotInject, PINT,
    LLMail-Inject and similar OOD prompt-injection slates side-by-side
    without open-coding per-source loaders.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to the YAML manifest. See :doc:`/ood_loader` for the schema
        (validated by ``ood_manifest.v1.json``).
    slices : sequence of str or None, optional
        Subset of slice ids to load. ``None`` (default) loads every
        slice in the manifest. Unknown ids raise :class:`KeyError`.
    cache_dir : str or pathlib.Path or None, optional
        Directory for verified downloads. Default
        ``~/.cache/eval-toolkit/ood/``. Cache key is the expected
        sha256 so two slices with the same content share storage.

    Returns
    -------
    pandas.DataFrame
        Columns: ``text`` (str), ``label`` (int — usually
        ``0=benign / 1=injected``), ``source`` (str, slice id from the
        manifest), ``row_id`` (str, ``sha256:`` of UTF-8-encoded text),
        ``sha`` (str, the manifest sha256 for the slice — for
        reproducibility). Row order is the manifest's slice declaration
        order; per-slice row order is preserved from the source file
        (or from ``sample(n=sample_size, random_state=seed)`` if
        ``sample_size`` is set).

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` does not exist.
    KeyError
        If ``slices`` references an id not in the manifest, or a slice's
        ``text_field`` / ``label_field`` is missing from the loaded data.
    ValueError
        On YAML parse error, sha256 mismatch (with expected vs actual +
        remediation hint), unsupported ``format``, ``sample_size`` without
        ``seed``, or label_map missing keys. Also — when a slice omits
        ``label_map`` — on label values that are non-numeric, non-finite
        (NaN/inf), non-integral (a bare int cast would silently truncate
        ``0.7 -> 0``), or too large for an exact float64 integrality check.
    ImportError
        If the optional ``[yaml]`` extra is not installed (PyYAML).

    Examples
    --------
    >>> # Not executable in doctest (requires real manifest + network).
    >>> # Equivalent:
    >>> # df = ood_dataset_from_manifest("configs/data/source_manifest.yaml",
    >>> #                                slices=["bipia", "agentdojo"])
    >>> # df.columns.tolist()
    >>> # ['text', 'label', 'source', 'row_id', 'sha']

    See Also
    --------
    OodManifestLoader : Protocol-compliant wrapper exposing the same
        function as a :class:`DatasetLoader` for harness pipelines.

    Notes
    -----
    Manifest schema (v1):

    .. code-block:: yaml

        name: my-ood-slate     # optional
        description: ...       # optional
        license: MIT           # optional (SPDX id)
        slices:
          bipia:
            url: https://example.com/bipia.parquet
            sha256: abc123...          # 64-char hex or sha256:<hex>
            text_field: prompt          # source column → output 'text'
            label_field: label          # source column → output 'label'
            label_map: {clean: 0, injected: 1}   # optional
            format: parquet             # parquet | jsonl | json | csv
            sample_size: 1000           # optional; null = all rows
            seed: 42                    # optional; required if sample_size set
          agentdojo:
            ...

    The manifest is intentionally minimal — for richer Croissant
    metadata or HF Hub auto-conversion, use :class:`HFDatasetsLoader`
    directly. This factory targets the **portfolio + submission
    library-first** pattern: a single YAML drives a single function
    call that returns a unified DataFrame.
    """
    import pandas as pd

    manifest = _load_yaml_manifest(yaml_path)
    all_slices: dict[str, dict[str, Any]] = manifest["slices"]
    cache_path = Path(cache_dir).expanduser() if cache_dir else _OOD_DEFAULT_CACHE_DIR

    # Resolve which slices to load (preserve manifest order; ignore caller order).
    if slices is None:
        slice_ids = list(all_slices.keys())
    else:
        unknown = sorted(set(slices) - set(all_slices.keys()))
        if unknown:
            raise KeyError(
                f"OOD manifest: unknown slice ids {unknown!r}; "
                f"available: {sorted(all_slices.keys())!r}"
            )
        slice_ids = [s for s in all_slices if s in set(slices)]

    parts: list[pd.DataFrame] = []
    for slice_id in slice_ids:
        spec = all_slices[slice_id]
        if not isinstance(spec, dict):
            raise ValueError(
                f"OOD slice {slice_id!r}: spec must be a YAML mapping; got {type(spec).__name__}"
            )
        url = spec.get("url")
        sha = spec.get("sha256")
        if not isinstance(url, str) or not isinstance(sha, str):
            raise ValueError(f"OOD slice {slice_id!r}: 'url' and 'sha256' are required strings")
        text_field = spec.get("text_field", "text")
        label_field = spec.get("label_field", "label")
        label_map = spec.get("label_map")
        sample_size = spec.get("sample_size")
        seed = spec.get("seed")

        fmt = spec.get("format") or _infer_format_from_url(url)
        if not fmt:
            raise ValueError(
                f"OOD slice {slice_id!r}: cannot infer format from URL "
                f"{url!r}; set 'format' explicitly in the manifest "
                f"(parquet, jsonl, json, csv)"
            )

        cached = _download_with_sha_verify(url, sha, cache_path, slice_id=slice_id)
        raw_df = _read_slice_dataframe(cached, fmt=fmt, slice_id=slice_id)
        _check_required_columns(
            raw_df,
            (text_field, label_field),
            context=f"OOD slice {slice_id!r}",
        )

        text_series = raw_df[text_field].astype(str)
        label_series = _apply_label_map(raw_df[label_field], label_map, slice_id=slice_id)

        out_df = pd.DataFrame(
            {
                "text": text_series.values,
                "label": label_series.values,
                "source": slice_id,
                "row_id": [_sha256_of_text(t) for t in text_series],
                "sha": f"sha256:{sha[len('sha256:') :] if sha.startswith('sha256:') else sha}",
            }
        )

        if sample_size is not None:
            if seed is None:
                raise ValueError(
                    f"OOD slice {slice_id!r}: sample_size set without seed; "
                    f"reproducibility requires both"
                )
            n = min(int(sample_size), len(out_df))
            out_df = out_df.sample(n=n, random_state=int(seed)).reset_index(drop=True)

        parts.append(out_df)

    if not parts:
        # Edge case: empty slices= filter that resolves to nothing (only happens
        # when slices=[] explicitly; unknown ids already raised KeyError).
        return pd.DataFrame(columns=["text", "label", "source", "row_id", "sha"])
    return pd.concat(parts, axis=0, ignore_index=True)


@dataclass(frozen=True, slots=True)
class OodManifestLoader:
    """Wrap :func:`ood_dataset_from_manifest` as a :class:`DatasetLoader`.

    Exposes the OOD-manifest loading pattern as a Protocol-compliant
    loader so harness pipelines (``evaluate`` / ``evaluate_folded``) can
    consume it via the same interface as :class:`DataFrameLoader` and
    :class:`HFDatasetsLoader`. :meth:`load_splits` returns
    ``{"all": <EvalSlice>}`` carrying the concatenated DataFrame
    (with ``source`` as the strata column by default).

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to the OOD manifest YAML. See
        :func:`ood_dataset_from_manifest` for schema.
    slices : sequence of str or None, optional
        Subset of slice ids. ``None`` (default) loads every slice.
    cache_dir : str or pathlib.Path or None, optional
        Override default cache directory.
    name, description, cite_as, license, url : str, optional
        Croissant metadata for :meth:`describe`. The manifest's own
        top-level ``name`` / ``description`` / ``license`` fields are
        used as fallback defaults.
    """

    yaml_path: str | Path
    slices: Sequence[str] | None = None
    cache_dir: str | Path | None = None
    name: str = ""
    description: str = ""
    cite_as: str = ""
    license: str = ""
    url: str = ""

    def load_splits(self) -> dict[str, EvalSlice]:
        """Load the manifest and return ``{"all": EvalSlice}``.

        The single ``all`` split carries every selected slice
        concatenated; use the ``source`` column to filter or stratify
        downstream.
        """
        df = ood_dataset_from_manifest(self.yaml_path, slices=self.slices, cache_dir=self.cache_dir)
        return {
            "all": EvalSlice(
                name="all",
                df=df,
                description=self.description,
                feature_col="text",
                label_col="label",
                strata_col="source",
            )
        }

    def describe(self) -> dict[str, object]:
        """Croissant-subset metadata with per-slice URI + sha256 in ``distribution``."""
        manifest = _load_yaml_manifest(self.yaml_path)
        all_slices: dict[str, dict[str, Any]] = manifest["slices"]
        ids = list(all_slices.keys()) if self.slices is None else list(self.slices)
        distribution: list[dict[str, object]] = []
        for slice_id in ids:
            spec = all_slices.get(slice_id, {})
            distribution.append(
                {
                    "name": slice_id,
                    "contentUrl": spec.get("url", ""),
                    "sha256": spec.get("sha256", ""),
                    "contentSize": 0,  # not pre-known from manifest
                }
            )

        # Caller-supplied fields win; manifest top-level fills gaps.
        def _pick(local: str, key: str) -> str:
            if local:
                return local
            val = manifest.get(key, "")
            return val if isinstance(val, str) else ""

        return {
            "name": _pick(self.name, "name") or f"OOD:{Path(self.yaml_path).name}",
            "description": _pick(self.description, "description"),
            "citeAs": self.cite_as,
            "license": _pick(self.license, "license"),
            "url": self.url,
            "distribution": distribution,
        }
