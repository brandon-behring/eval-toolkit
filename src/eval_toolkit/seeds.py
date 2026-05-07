"""Global seed-setting for reproducibility.

Public API: :func:`set_global_seeds`. Sets ``random``, ``numpy``, and
optionally ``torch`` (with ``cudnn.deterministic=True``). Torch is a soft
dependency — toolkit installs without it.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = ["set_global_seeds"]


def set_global_seeds(seed: int) -> None:
    """Pin random/numpy/torch seeds + enable deterministic CUDA where possible.

    Parameters
    ----------
    seed : int
        Non-negative seed value.

    Notes
    -----
    Full determinism on CUDA is not always achievable — some kernels lack
    deterministic implementations. Documented residual variance can be
    handled by reporting multi-seed CIs rather than relying on bit-exact
    reproducibility.

    Torch is imported lazily inside the function. If ``torch`` is not
    installed, the random + numpy seeds are still set.

    Examples
    --------
    >>> import numpy as np
    >>> set_global_seeds(42)
    >>> a1 = np.random.rand(5)
    >>> set_global_seeds(42)
    >>> a2 = np.random.rand(5)
    >>> bool(np.all(a1 == a2))
    True
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
