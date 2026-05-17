# Parallelism

> **Background** *(skip if you've internalized this)*. Several toolkit
> primitives — especially the bootstrap family — spend the bulk of their
> wall-clock time in Python-level resample loops. For high
> ``n_resamples`` (5 000+) on cheap metrics or any ``n_resamples`` on
> expensive metrics (operating-point refit, ECE recomputation), parallel
> execution across CPU cores gives 3–10× speedup.

This chapter codifies eval-toolkit's *unified* parallelism design — six
principles + one internal helper that every parallel-capable public API
calls into. The goal: contributors adding ``n_jobs`` to a new function
inherit a consistent backend, reproducibility contract, and error-surface
shape without re-inventing them.

## Design principles (the six)

### 1. Single backend

All parallelism in the toolkit uses **joblib with the ``loky`` backend**.
Reasons:

- ``joblib`` is already transitive via sklearn — no new dependency.
- ``loky`` is process-based (sidesteps the GIL for CPU-bound metric calls).
- Helpful pickling-error messages save user time vs raw
  ``multiprocessing``.
- Matches the sklearn-eval ecosystem convention; callers' mental model
  carries over.

Toolkit does **not** use ``concurrent.futures``, raw ``multiprocessing``,
``threading``, or ``asyncio`` for CPU-bound parallelism.

### 2. Single helper

The single source of truth is
{func}`eval_toolkit._parallel.parallel_map` (internal — not exported in
``__all__``). Every public function that accepts ``n_jobs`` calls into
this helper rather than inlining its own ``joblib.Parallel`` invocation.

Why centralised:

- One place to update if joblib's API shifts
- Consistent error messages across the toolkit
- Smart-default semantics (auto-cap, guidance log, picklability sniff)
  apply uniformly

### 3. Opt-in per function

``n_jobs`` is an **explicit keyword-only parameter** on each
parallel-capable public function, defaulting to ``1`` (sequential).

```python
def paired_bootstrap_diff(..., *, n_jobs: int = 1) -> PairedBootstrapCI: ...
```

No global config (no ``EVAL_TOOLKIT_N_JOBS`` env var, no
``joblib.parallel_backend`` context-manager support). Reasoning:

- Tests stay deterministic in wall-clock (no environment-dependent timing)
- Callsite is auditable — readers can see at a glance whether a call is
  parallel
- Matches Python stdlib convention ("explicit opt-in" for parallelism)

### 4. Default sequential

``n_jobs=1`` (default) runs a pure-Python ``for``-loop, preserving:

- Traceback fidelity (no joblib worker-process indirection in stack traces)
- Reproducibility (no race conditions; deterministic execution order)
- Zero startup overhead (no loky pool spin-up)

For ``n_jobs == 1`` AND ``len(items) >= 1000``, the helper emits a one-
shot INFO log per Python process suggesting the caller might want
``n_jobs > 1``. This is the *only* INFO-level log site in the toolkit
(everything else is DEBUG); the threshold + once-per-process semantics
keep the noise floor near-zero for typical workloads.

### 5. Reproducibility contract

When a parallel-capable function performs random resampling, it MUST use
{func}`numpy.random.SeedSequence.spawn` to derive per-item RNG seeds so
``n_jobs > 1`` produces **bit-for-bit-identical** output to ``n_jobs == 1``
for the same caller-supplied ``seed``:

```python
seed_seq = np.random.SeedSequence(seed)
spawned = seed_seq.spawn(n_resamples)
deltas = parallel_map(_resample_step, spawned, n_jobs=n_jobs)
```

This is the *invariant* that callers rely on when they tune ``n_jobs``
for speed. Every parallel-capable function in the toolkit has a
``test_*_n_jobs_reproducibility`` test asserting this equivalence.

### 6. Picklability surface

The helper does a ``pickle.dumps(fn)`` sniff-test up front and raises a
helpful ``TypeError`` when ``fn`` is unpicklable (lambdas, closures over
local state):

```text
TypeError: parallel_map of paired bootstrap: callable is not picklable
(lambdas and closures over local state are not supported with n_jobs != 1).
Define a named top-level function. Underlying error: ...
```

This avoids the cryptic ``_pickle.PicklingError`` deep in a joblib worker
traceback.

## Smart defaults summary

| ``n_jobs`` | Behavior |
|---|---|
| ``1`` (default) | Sequential pure-Python loop. INFO guidance log if ``items >= 1000``, once per process. |
| ``0`` | Rejected — ``ValueError`` ("use 1 or -1"). Catches the common typo. |
| ``-1`` | joblib convention: all available cores. Not capped. |
| ``2..cpu_count()`` | joblib loky with ``n_jobs`` workers. |
| ``> cpu_count()`` | Capped with WARNING log; prevents CPU-frying when caller passes ``n_jobs=64`` on an 8-core box. |

## When to add ``n_jobs`` to a new function

Add the kwarg when **all** of these are true:

1. The function has a Python-level loop over independent work units
2. Per-item cost is medium or higher (>10ms typical)
3. Typical iteration count is ≥1000 OR per-item cost is expensive enough
   that smaller counts still win (e.g., scorer.predict_proba per slice)
4. The loop body is picklable (no lambdas; doesn't close over thread-local
   state)

Then:

1. Refactor the loop body into a module-level ``_step_fn`` helper
2. Use ``np.random.SeedSequence(seed).spawn(n)`` to seed each step (if
   random)
3. Replace the loop with ``parallel_map(_step_fn, items, n_jobs=n_jobs,
   description="...")``
4. Add a ``test_*_n_jobs_reproducibility`` test asserting equivalence
5. Update the function's docstring ``Parameters`` section with the
   standard ``n_jobs`` blurb (see existing bootstrap-fn docstrings as
   the template)

## Currently parallel-capable functions

As of v0.34.0:

| Function | n_jobs threshold | Notes |
|---|---|---|
| {func}`~eval_toolkit.bootstrap.bootstrap_ci` | ``n_resamples >= 1000`` | All resamples are independent |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_diff` | ``n_resamples >= 1000`` | Original issue #17 ask |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_ece_diff` | ``n_resamples >= 1000`` | |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_op_point_diff` | ``n_resamples >= 1000`` | Per-resample threshold refit benefits most from parallelism |
| {func}`~eval_toolkit.bootstrap.paired_mde` | Pass-through to internal ``paired_bootstrap_diff`` | |

**Not yet parallelised** (filed as follow-up issues):

- ``harness._score_all_slices`` and ``harness.evaluate_folded`` (slice ×
  scorer loop) — Scorer picklability is the gatekeeper
- ``harness._attach_transferred_operating_points`` (OperatingPointSpec
  loops)
- ``text_dedup.MinHashLSHStrategy`` hash-bucket construction — race-
  condition design needed for per-worker accumulation

## See also

- {mod}`eval_toolkit.bootstrap` — primary callers
- [`docs/source/methodology/bootstrap.md`](bootstrap.md) — algorithmic
  background on bootstrap resampling
- ``joblib`` documentation —
  <https://joblib.readthedocs.io/en/latest/parallel.html>
