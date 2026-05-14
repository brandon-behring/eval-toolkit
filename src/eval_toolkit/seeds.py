"""Global seed-setting for reproducibility.

Public API: :func:`set_global_seeds`. Sets ``PYTHONHASHSEED``, ``random``,
``numpy``, and optionally ``torch`` (with ``cudnn.deterministic=True``).
Torch is a soft dependency — toolkit installs without it.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np

__all__ = ["set_global_seeds"]


def set_global_seeds(seed: int, *, strict_torch_determinism: bool = False) -> None:
    """Pin random / numpy / torch seeds + enable deterministic CUDA where possible.

    Sets the four sources of nondeterminism that any Python-flavored ML
    pipeline can practically control: ``PYTHONHASHSEED`` (string-hash
    salt), the stdlib ``random`` module, ``numpy``'s legacy global RNG,
    and ``torch`` (CPU + CUDA + cuDNN flags) when torch is importable.

    Matches PyTorch Lightning's :func:`seed_everything` discipline.

    Parameters
    ----------
    seed : int
        Non-negative seed value.
    strict_torch_determinism : bool, optional
        If ``True``, also call ``torch.use_deterministic_algorithms(True)``
        to make every torch op raise on nondeterministic kernels (vs. the
        default cuDNN-only flags). Useful when bit-exact reproducibility
        is required across runs. Has no effect if torch is not installed.
        Default ``False``.

    Raises
    ------
    TypeError
        If ``seed`` is not an ``int``.
    ValueError
        If ``seed`` is negative.
    RuntimeError
        If ``strict_torch_determinism=True`` is requested but ``torch`` is
        not installed.

    Notes
    -----
    Full determinism on CUDA is not always achievable — some kernels lack
    deterministic implementations. Documented residual variance can be
    handled by reporting multi-seed CIs rather than relying on bit-exact
    reproducibility.

    ``PYTHONHASHSEED`` is set in the *current* process's environment but
    only takes effect for child processes (the current interpreter has
    already initialized its hash randomization). To get deterministic
    string hashing in the current run, set ``PYTHONHASHSEED`` from the
    shell *before* launching Python; this function emits a UserWarning
    when the variable is already set to a different value.

    Torch is imported lazily inside the function. If ``torch`` is not
    installed, the random + numpy seeds + ``PYTHONHASHSEED`` are still
    set.

    Examples
    --------
    >>> import numpy as np
    >>> set_global_seeds(42)
    >>> a1 = np.random.rand(5)
    >>> set_global_seeds(42)
    >>> a2 = np.random.rand(5)
    >>> bool(np.all(a1 == a2))
    True

    See Also
    --------
    `pytorch-lightning.seed_everything <https://lightning.ai/docs/pytorch/stable/common/seed.html>`_ :
        Reference for the strict_torch_determinism semantics.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    target = str(seed)
    existing = os.environ.get("PYTHONHASHSEED")
    if existing is not None and existing != target:
        warnings.warn(
            f"PYTHONHASHSEED was already set to {existing!r}; overwriting to "
            f"{target!r}. The current Python process's hash randomization is "
            f"unaffected — set PYTHONHASHSEED before launching Python for "
            f"in-process determinism.",
            UserWarning,
            stacklevel=2,
        )
    os.environ["PYTHONHASHSEED"] = target

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        if strict_torch_determinism:
            raise RuntimeError(
                "strict_torch_determinism=True requested but torch is not installed"
            ) from None
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_torch_determinism:
        torch.use_deterministic_algorithms(True)
