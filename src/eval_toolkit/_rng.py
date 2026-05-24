"""Private RNG-parameter type aliases per Scientific-Python SPEC 7.

This module centralizes the type aliases used to annotate user-facing RNG
parameters across the toolkit. Per `SPEC 7 — Seeding PRNG
<https://scientific-python.org/specs/spec-0007/>`_ (Endorsed) eval-toolkit
exposes a single canonical parameter name ``rng`` typed as
``RNGLike | SeedLike | None`` on every function that consumes a NumPy
``Generator``. Bodies normalize via ``np.random.default_rng(rng)``.

This module is private (underscore prefix) so the aliases stay an
implementation detail — public symbols use them only in their annotations.
If a Tier-2 consumer ever needs them exposed for their own callsite type
annotations, promote them via ``eval_toolkit.protocols`` per the
asymmetric-promotion principle in ADR 0001 + STYLE.md §3d.

Exceptions to the SPEC 7 convention — documented in STYLE.md §3a:

- ``seeds.set_global_seeds(seed: int)`` — global-state setter, not a
  per-function RNG parameter; SPEC 7 is scoped to per-function RNG inputs.
- ``adversarial.*Injection`` / ``*Substitution`` / ``CaseInjection``
  dataclass fields — they use Python's stdlib ``random.Random(seed)``,
  not NumPy. SPEC 7's typing (``RNGLike = np.random.Generator | ...``) is
  strictly NumPy-scoped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

type SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
"""Anything that can seed a NumPy bit generator.

Per SPEC 7, ``np.random.default_rng`` accepts any of these as a seed
without further conversion. ``Sequence[int]`` is the entropy-vector form
used by ``np.random.SeedSequence``.
"""

type RNGLike = np.random.Generator | np.random.BitGenerator
"""An already-instantiated NumPy bit generator or generator wrapper.

``np.random.default_rng(rng)`` is the identity function on
``Generator`` inputs and lifts ``BitGenerator`` inputs into a
``Generator`` — both forms compose cleanly.
"""


def spawn_seed_sequences(rng: RNGLike | SeedLike | None, n: int) -> list[np.random.SeedSequence]:
    """Spawn ``n`` independent SeedSequences from any SPEC 7 ``rng`` input.

    Normalizes the input to a ``Generator``, then draws ``n`` random
    64-bit entropy values FROM the generator and wraps each in a fresh
    ``SeedSequence``. The draws advance the generator's internal state
    so subsequent calls to ``spawn_seed_sequences`` on the same
    ``Generator`` instance produce different children.

    Used by the bootstrap parallel workers (which take spawned
    ``SeedSequence`` objects to seed their internal ``default_rng()`` calls).

    Notes
    -----
    Prior to v0.51, this function extracted the
    ``bit_generator.seed_seq`` from the input and called ``.spawn(n)``
    on it. That implementation IGNORED Generator state — a ``Generator``
    that had been advanced by prior draws produced the same children as
    a fresh ``Generator`` with the same construction seed. Two callers
    sharing one ``Generator`` therefore got identical bootstrap streams,
    silently violating bootstrap independence across (slice, scorer)
    pairs in ``harness.evaluate`` and across resamples in
    ``bootstrap.bootstrap_ci``. The v0.51 implementation draws fresh
    entropy from the generator on every call so spawning is
    state-respecting; this also restores the SPEC 7 bit-for-bit identity
    contract between sequential and parallel modes when the seeds are
    fanned out at a single batch boundary.
    """
    gen = np.random.default_rng(rng)
    # Draw n independent 64-bit unsigned entropy values FROM the generator.
    # Each draw advances generator state, so repeated calls on the same
    # Generator instance yield different children. Each entropy value seeds
    # an independent SeedSequence (SeedSequence's bit-mixing guarantees
    # downstream independence across the n children).
    seeds = gen.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    return [np.random.SeedSequence(int(s)) for s in seeds]
