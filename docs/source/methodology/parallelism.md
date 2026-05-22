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

```text
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

```text
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

As of v0.36.0 the unified `n_jobs` pattern covers both the bootstrap
family and the harness scoring loop:

| Function | n_jobs threshold | Notes |
|---|---|---|
| {func}`~eval_toolkit.bootstrap.bootstrap_ci` | ``n_resamples >= 1000`` | Effective ONLY for `method="studentized"` — `BCa` and `percentile` raise when `n_jobs != 1` (the BCa jackknife is sequential by construction; `percentile` is fast enough that parallel overhead dominates). `BCa` is the default `bootstrap_ci` method. |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_diff` | ``n_resamples >= 1000`` | Original issue #17 ask |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_ece_diff` | ``n_resamples >= 1000`` | |
| {func}`~eval_toolkit.bootstrap.paired_bootstrap_op_point_diff` | ``n_resamples >= 1000`` | Per-resample threshold refit benefits most from parallelism |
| {func}`~eval_toolkit.bootstrap.paired_mde` | Pass-through to internal ``paired_bootstrap_diff`` | |
| {func}`~eval_toolkit.harness.evaluate` | ``n_jobs`` kwarg on the scorer × slice loop (v0.36, closes #29) | Scorer must be picklable — see [Scorer picklability](#scorer-picklability) below. |
| {func}`~eval_toolkit.harness.evaluate_folded` | ``n_jobs`` kwarg on the (scorer × slice × fold) loop (v0.36, closes #30) | Same Scorer picklability requirement. |

**Not yet parallelised** (filed as follow-up issues):

- ``harness._attach_transferred_operating_points`` (OperatingPointSpec
  loops) — same Scorer gatekeeper.
- ``text_dedup.MinHashLSHStrategy`` hash-bucket construction — race-
  condition design needed for per-worker accumulation.

## Scorer picklability

With harness parallelization wired in v0.36
([#29](https://github.com/brandon-behring/eval-toolkit/issues/29),
[#30](https://github.com/brandon-behring/eval-toolkit/issues/30) — both
closed), the work-unit dispatched to each loky worker bundles both a step
function AND a {class}`~eval_toolkit.protocols.Scorer` instance.
``joblib`` pickles the *entire* delayed call — function plus bound
arguments — so an unpicklable ``Scorer`` fails at dispatch even when the
step function itself is fine.

The existing ``parallel_map`` sniff (Principle #6 above) covers the function;
this section establishes the parallel contract for the Scorer surface.

**Rule.** Any ``Scorer`` passed to a parallel-capable harness call
(``evaluate(..., n_jobs > 1)`` and ``evaluate_folded(..., n_jobs > 1)``
since v0.36) MUST be picklable. The toolkit's existing helper raises a
clean ``TypeError`` with the underlying pickle error attached — no
special exception subclass.

**Picklable** (works — top-level class, picklable state):

```text
class ThresholdScorer:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict_proba(self, X):
        # bound numpy / torch state is fine — both pickle cleanly
        ...
```

**Not picklable** (raises ``TypeError`` at dispatch):

```text
def make_scorer(threshold):
    def predict_proba(X):
        # closure over `threshold` — joblib cannot pickle the inner fn
        ...
    return predict_proba   # ← returned closure is not picklable
```

**Fix.** Promote the closure to a top-level class (as in the picklable
example above). Bound instance attributes pickle naturally; closures over
local state do not.

**Common non-picklable cases** to watch for in user-supplied Scorers:

- Closures (above) — most common
- ``lambda`` expressions assigned to instance attributes
- Local-scope class definitions (defined inside a function or method)
- Attributes holding open file handles, sockets, or live model-server
  connections — re-establish these inside ``predict_proba`` instead of
  caching on ``self``

For deeper joblib pickling guidance see the joblib docs (link below).

(worker-copy-memory)=
## Memory model: worker-copy semantics

When `n_jobs != 1`, joblib's loky backend forks a worker pool and
**each worker receives a full copy of every argument bound at
`delayed(fn)(arg)` call time**. There is no shared-memory channel for
caller objects — loky's worker-to-parent IPC is pickle-based, so the
parent's memory is duplicated, not aliased.

The implication for memory-bounded parallelism:

- **A `pd.DataFrame` of size N MB held in each work item's spec
  produces `n_jobs × N` MB of resident memory across the worker
  pool.** On a 128-core / 247 GB machine, an `n_jobs=-1` sweep where
  each cell carries a 30 GB working set (BCa bootstrap intermediates
  on a multi-million-row corpus) projects to ~3.8 TB peak — the OOM
  killer fires before any cell finishes. This is **not** a joblib bug;
  it's the cost of the picklability contract (Principle #6).
- **NumPy arrays** are subject to the same rule but loky transparently
  uses joblib's memmap-based shared-memory fast path for arrays above
  a threshold (~1 MB by default). This means small NumPy arrays
  duplicate; large NumPy arrays auto-share via `/dev/shm`. **No such
  fast path exists for DataFrames** — pandas objects always duplicate.
- **Module-level / process-global state** is NOT copied per call.
  Workers inherit the parent's globals at fork time, then operate
  independently. State written to a module-level singleton inside one
  worker is invisible to others.

### Pattern: shared-state via file-path + reload

The standard workaround for "DataFrame shared across many cells" is
file-path indirection: persist the DataFrame to disk in the parent,
pass the path (a small string) to workers, and reload inside `fn`.
The OS page cache means each worker's `read_parquet()` hits warm cache
after the first; resident memory grows by the size of the materialised
DataFrame per worker, but that's bounded by `min(n_jobs, n_useful_workers)`
rather than `n_jobs × spec_size`.

```text
# Parent:
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    parquet_path = Path(tmp) / "predictions.parquet"
    df.to_parquet(parquet_path)
    specs = [(parquet_path, cell_config) for cell_config in cells]
    results = parallel_map(_compute_cell, specs, n_jobs=8, description="marginal CI")


def _compute_cell(spec: tuple[Path, CellConfig]) -> CellResult:
    parquet_path, cfg = spec
    df = pd.read_parquet(parquet_path)  # OS page cache amortises after first call
    return _bootstrap_one_cell(df, cfg)
```

For numerical-only workloads where the DataFrame's content is convertible
to a structured NumPy array, casting before parallelisation lets joblib's
memmap fast-path kick in:

```text
arr = df[["score", "label"]].to_numpy()  # joblib will memmap this above ~1 MB
results = parallel_map(_bootstrap_one_cell, [(arr, cfg) for cfg in cells], n_jobs=8)
```

### Recommended ceiling for DataFrame-bearing workloads

When in doubt, cap `n_jobs` at `min(8, available_RAM_GB / spec_size_GB)`.
For the 30 GB / cell case above, a 247 GB machine supports
`min(8, 247/30) ≈ 8` workers — far below the 128 cores available, but
the limit is RAM, not compute. The `parallel_map` helper does not
auto-detect this; it's a caller responsibility to size `n_jobs`
against memory headroom.

## See also

- {mod}`eval_toolkit.bootstrap` — primary callers
- [`docs/source/methodology/bootstrap.md`](bootstrap.md) — algorithmic
  background on bootstrap resampling
- ``joblib`` documentation —
  <https://joblib.readthedocs.io/en/latest/parallel.html>
