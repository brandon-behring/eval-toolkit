"""Lightweight public Protocols with minimal dependency surface.

These contracts are intentionally kept out of the heavier implementation
modules so consumers can type adapters without importing pandas, matplotlib,
or filesystem-oriented helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "EvalSliceLike",
    "PredictionReader",
    "Scorer",
    "SliceAwareScorer",
    "TextTransform",
    "Versioned",
]


@runtime_checkable
class Scorer(Protocol):
    """Anything exposing ``predict_proba(X) -> np.ndarray of P(positive)``.

    Accepts ``list[str]``, ``np.ndarray``, or ``pd.Series`` of features.
    Pandas is imported under ``TYPE_CHECKING`` only, so this Protocol
    has no runtime pandas dependency.

    Notes
    -----
    When passed to a parallel-capable harness call (``n_jobs > 1``), Scorer
    instances MUST be picklable — joblib's loky backend serializes the entire
    delayed call (function plus bound arguments) before worker dispatch.
    Closures, lambdas, local-scope classes, and attributes holding live
    sockets / file handles break pickling. See
    ``docs/source/methodology/parallelism.md#scorer-picklability`` for the
    full contract and worked examples.
    """

    def predict_proba(  # pragma: no cover
        self, X: Sequence[str] | np.ndarray | pd.Series
    ) -> np.ndarray:
        """Return one P(positive) score per input row."""
        ...


@runtime_checkable
class SliceAwareScorer(Scorer, Protocol):
    """Optional scorer contract for cost-controlled slice skipping."""

    def should_score_slice(self, slice_name: str) -> bool:  # pragma: no cover
        """Return whether this scorer should run on the named slice."""
        ...


@runtime_checkable
class Versioned(Protocol):
    """Anything exposing a stable version string."""

    @property
    def version(self) -> str:  # pragma: no cover
        """Stable version string for this implementation."""
        ...


@runtime_checkable
class TextTransform(Protocol):
    """Uniform text-transform shape for sweep + defence + attack strategies.

    Decision K (plan §4B) + Audit R5-F3 (Codex Round 5): the v0.47 sweep
    consolidation requires a single Protocol that both preprocessing
    (Spotlighting) variants and adversarial (character-injection) variants
    structurally satisfy. Concrete classes anywhere in eval-toolkit that
    expose ``name: str`` + ``transform(text: str) -> str`` will type-check
    as :class:`TextTransform` instances without inheriting from this class.

    The 9th strict Tier-2 Protocol per ADR 0003 (Decision M); method shape
    frozen at v1.0 modulo SemVer-major bumps. Additive subprotocols (e.g.,
    a future ``BatchTextTransform`` with ``transform_batch(...)``) are
    permitted per the additive-only contract.

    Concrete implementations in v0.47:

    - :class:`eval_toolkit.preprocessing.DelimitVariant` /
      :class:`~eval_toolkit.preprocessing.DatamarkVariant` /
      :class:`~eval_toolkit.preprocessing.EncodeVariant` — defence side.
    - :class:`eval_toolkit.adversarial.ZeroWidthSpaceInjection` and the
      other 5 core + 6 advanced character-injection dataclasses —
      attack side. All satisfy structurally without source changes.
    """

    name: str
    """Stable identifier; participates in sweep row labels + result keys."""

    def transform(self, text: str) -> str:  # pragma: no cover
        """Return the transformed text. Deterministic + side-effect-free."""
        ...


@runtime_checkable
class EvalSliceLike(Protocol):
    """Pandas-free slice surface needed by evaluation contracts."""

    @property
    def name(self) -> str:  # pragma: no cover
        """Stable slice identifier."""
        ...

    @property
    def y_true(self) -> np.ndarray:  # pragma: no cover
        """Binary labels as a 1-D array."""
        ...

    @property
    def features(self) -> Sequence[str]:  # pragma: no cover
        """Feature values passed to a scorer."""
        ...

    @property
    def strata(self) -> np.ndarray | None:  # pragma: no cover
        """Optional stratum labels for slice-aware reporting."""
        ...


@runtime_checkable
class PredictionReader(Protocol):
    """Reads manifest-referenced prediction artifacts into column arrays."""

    def read_predictions(  # pragma: no cover
        self,
        uri: str,
        *,
        columns: Mapping[str, str],
    ) -> Mapping[str, Sequence[object]]:
        """Return a column-oriented table for the requested artifact URI."""
        ...
