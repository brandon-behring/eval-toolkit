"""Shared Hypothesis strategies for eval_toolkit property tests.

Hypothesis docs explicitly recommend a `tests/strategies.py` module for
strategies used across multiple test files. v0.3.0 consolidates the
previously-duplicated `_balanced_binary_array` and `_score_array` from
`test_metrics_props.py`, `test_bootstrap_props.py`, and
`test_calibration_props.py` into one place.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def balanced_binary_array(n: int, *, min_per_class: int = 5) -> st.SearchStrategy[np.ndarray]:
    """Hypothesis strategy: 1-D binary array with at least ``min_per_class``
    of each class.

    Filters out single-class draws which would crash sklearn metrics.

    Parameters
    ----------
    n : int
        Array length.
    min_per_class : int, optional
        Minimum count for each of {0, 1}. Default 5.

    Returns
    -------
    SearchStrategy[np.ndarray]
        Hypothesis strategy yielding arrays of dtype int64, shape (n,).
    """
    return arrays(
        dtype=np.int64,
        shape=n,
        elements=st.integers(0, 1),
    ).filter(lambda y: min_per_class <= int(y.sum()) <= n - min_per_class)


def score_array(n: int) -> st.SearchStrategy[np.ndarray]:
    """Hypothesis strategy: 1-D float array of probabilities in [0, 1].

    Parameters
    ----------
    n : int
        Array length.

    Returns
    -------
    SearchStrategy[np.ndarray]
        Hypothesis strategy yielding arrays of dtype float64, shape (n,).
    """
    return arrays(
        dtype=np.float64,
        shape=n,
        elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
