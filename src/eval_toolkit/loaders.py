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
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from eval_toolkit.harness import EvalSlice
from eval_toolkit.provenance import file_sha256

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DataFrameLoader",
    "DatasetLoader",
    "HFDatasetsLoader",
    "ParquetGlobLoader",
    "SingleSliceLoader",
]


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
        for col in (self.split_col, self.feature_col, self.label_col):
            if col not in self.df.columns:
                raise KeyError(
                    f"DataFrameLoader: missing column {col!r}; available: "
                    f"{list(self.df.columns)}"
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
        """
        import pandas as pd  # function-local: pandas is in the [parquet]/[dataframe] extra

        files_by_split = self._resolve_files()
        out: dict[str, EvalSlice] = {}
        for split_name, files in files_by_split.items():
            parts = [pd.read_parquet(p) for p in files]
            df = pd.concat(parts, axis=0, ignore_index=True)
            for col in (self.feature_col, self.label_col):
                if col not in df.columns:
                    raise KeyError(
                        f"ParquetGlobLoader: split {split_name!r}: missing column "
                        f"{col!r}; available: {list(df.columns)}"
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
        """Croissant-subset metadata with per-file SHA-256 in ``distribution``."""
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

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repo, e.g. ``"deepset/prompt-injections"``.
    splits : sequence of str or None, optional
        Subset of HF splits to load. ``None`` = every split the repo defines.
    feature_col : str, optional
        Column name in the HF dataset. Default ``"text"``.
    label_col : str, optional
        Column name. Default ``"label"``.
    strata_col : str or None, optional
    config_name : str or None, optional
        HF dataset config name (some datasets have multiple configs).
    name, description, cite_as, license, url : str, optional
        Croissant metadata fields.
    """

    repo_id: str
    splits: Sequence[str] | None = None
    feature_col: str = "text"
    label_col: str = "label"
    strata_col: str | None = None
    config_name: str | None = None
    name: str = ""
    description: str = ""
    cite_as: str = ""
    license: str = ""
    url: str = ""

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
                "HFDatasetsLoader requires the optional 'datasets' package. "
                "Install with: pip install datasets"
            ) from exc
        if self.config_name is not None:
            return cast(Mapping[str, Any], load_dataset(self.repo_id, name=self.config_name))
        return cast(Mapping[str, Any], load_dataset(self.repo_id))

    def load_splits(self) -> dict[str, EvalSlice]:
        """Convert each requested HF split to an :class:`EvalSlice`.

        Raises
        ------
        KeyError
            If any split's pandas DataFrame is missing ``feature_col`` or
            ``label_col``.
        """
        ds = self._load_dataset()
        ds_splits = list(ds.keys()) if self.splits is None else list(self.splits)
        out: dict[str, EvalSlice] = {}
        for split_name in ds_splits:
            sub = ds[split_name]
            df = sub.to_pandas()
            for col in (self.feature_col, self.label_col):
                if col not in df.columns:
                    raise KeyError(
                        f"HFDatasetsLoader: split {split_name!r}: missing column "
                        f"{col!r}; available: {list(df.columns)}"
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
        """Croissant-subset metadata pointing at the HF repo (no file hashes — HF caches)."""
        return {
            "name": self.name or self.repo_id,
            "description": self.description,
            "citeAs": self.cite_as,
            "license": self.license,
            "url": self.url or f"https://huggingface.co/datasets/{self.repo_id}",
            "distribution": [
                {
                    "name": f"hf:{self.repo_id}",
                    "contentUrl": f"https://huggingface.co/datasets/{self.repo_id}",
                    "sha256": "",  # HF cache hash not exposed via the public API
                    "contentSize": 0,
                }
            ],
            "config_name": self.config_name,
        }
