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
    "Versioned",
]


@runtime_checkable
class Scorer(Protocol):
    """Anything exposing ``predict_proba(X) -> np.ndarray of P(positive)``.

    Accepts ``list[str]``, ``np.ndarray``, or ``pd.Series`` of features.
    Pandas is imported under ``TYPE_CHECKING`` only, so this Protocol
    has no runtime pandas dependency.
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
