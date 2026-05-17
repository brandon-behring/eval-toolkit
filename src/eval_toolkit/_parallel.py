"""Toolkit-internal parallel-map helper. Single source of truth for parallelism.

This module is *internal* (not exported via ``__all__`` or
``__init__._EXPORTS``); future per-function ``n_jobs`` additions across the
toolkit call into this helper rather than each inventing their own
parallelism backend.

See ``docs/source/methodology/parallelism.md`` for the design rationale +
caller contract (reproducibility, picklability, smart defaults).
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable, Iterable, Sized

_logger = logging.getLogger(__name__)

_GUIDANCE_THRESHOLD = 1000

_GUIDANCE_EMITTED = False


def parallel_map[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    n_jobs: int = 1,
    description: str = "work",
) -> list[R]:
    """Map ``fn`` over ``items``; parallel when ``n_jobs != 1``.

    Design contract (see ``docs/source/methodology/parallelism.md``):

    - ``n_jobs == 1`` (default) — pure-Python serial; preserves tracebacks.
      If ``len(items) >= 1000``, emits an INFO log on the first qualifying
      call per Python process suggesting ``n_jobs > 1`` (silent thereafter).
    - ``n_jobs == -1`` — joblib loky with all cores.
    - ``n_jobs > 1`` — joblib loky; values exceeding ``os.cpu_count()``
      are silently capped (with a WARNING log) to avoid CPU-frying.
    - ``n_jobs == 0`` — raises ``ValueError`` (likely a typo for 1 or -1).
    - ``fn`` MUST be picklable when ``n_jobs != 1`` (lambdas and closures
      over local state are rejected at call time with a helpful
      ``TypeError``).
    - Reproducibility: caller is responsible for deterministic per-item
      state (use ``np.random.SeedSequence(seed).spawn(n)`` for resample
      loops so ``n_jobs > 1`` produces identical results to ``n_jobs == 1``
      for the same seed).

    Parameters
    ----------
    fn : Callable[[T], R]
        Picklable callable to apply to each item. Lambdas and closures
        over local state are rejected when ``n_jobs != 1``.
    items : Iterable[T]
        Work items. Materialised internally; pass any iterable.
    n_jobs : int, optional
        Default 1 (sequential). Set to -1 for all cores, or a positive int.
        ``n_jobs=0`` is rejected (use 1 or -1).
    description : str, optional
        Used in log messages and error messages for context (e.g.,
        ``"paired bootstrap"``).

    Returns
    -------
    list[R]
        Results in the order of ``items``.

    Raises
    ------
    ValueError
        If ``n_jobs == 0``.
    TypeError
        If ``n_jobs != 1`` and ``fn`` is not picklable.

    Examples
    --------
    >>> def square(x): return x * x
    >>> parallel_map(square, [1, 2, 3], n_jobs=1)
    [1, 4, 9]
    """
    if n_jobs == 0:
        raise ValueError(
            f"n_jobs=0 is not allowed for {description}; use 1 (sequential), "
            "-1 (all cores), or a positive integer."
        )

    items_list = items if isinstance(items, list) else list(items)
    n_items = len(items_list) if isinstance(items_list, Sized) else 0

    if n_jobs == 1:
        global _GUIDANCE_EMITTED
        if n_items >= _GUIDANCE_THRESHOLD and not _GUIDANCE_EMITTED:
            _logger.info(
                "%s: running %d items sequentially (n_jobs=1). For parallel "
                "speedup set n_jobs > 1 (typical wall-clock 3-5x on 8 cores). "
                "(Shown once per process.)",
                description,
                n_items,
            )
            _GUIDANCE_EMITTED = True
        return [fn(item) for item in items_list]

    if n_jobs > 0:
        cpu_count = os.cpu_count() or 1
        if n_jobs > cpu_count:
            _logger.warning(
                "%s: capping n_jobs from %d to %d (os.cpu_count()).",
                description,
                n_jobs,
                cpu_count,
            )
            n_jobs = cpu_count

    try:
        pickle.dumps(fn)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise TypeError(
            f"parallel_map of {description}: callable is not picklable "
            f"(lambdas and closures over local state are not supported "
            f"with n_jobs != 1). Define a named top-level function. "
            f"Underlying error: {e}"
        ) from e

    from joblib import Parallel, delayed  # noqa: PLC0415

    return list(Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(item) for item in items_list))
