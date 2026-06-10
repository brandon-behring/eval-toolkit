# eval-toolkit — Coding Standards

Self-contained quick reference for this repository. The ADRs
(`docs/source/adr/`) are the authoritative source for the decisions summarized
here; everything needed for day-to-day contribution lives in this file.

## 1. Foundational principles

1. **Never fail silently.** Every validation failure raises a stdlib exception
   with a diagnostic message. No silent defaults, no fake/fallback data.
2. **Fail fast.** Validate at public API entry points; trust internals
   downstream. Diagnostic messages explain what was expected, what was found,
   and how to fix the input.
3. **Immutability by default.** Functions return new data structures; never
   mutate caller-supplied arguments. Mutating functions return `None` and say
   so in the docstring.
4. **Pure-vs-IO separation.** Pure functions (math, transformation) are
   separated from IO (filesystem, network) so consumers can test without side
   effects.
5. **Anti-overengineering.** Don't add an abstraction unless a second concrete
   use exists or is concretely planned.

## 2. Tooling

| Tool | Setting |
|---|---|
| Formatter | `black`, line length 100 |
| Linter | `ruff` with `select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "C4"]`, ignore `E501` (Black handles), `N803`/`N806` (math identifiers) |
| Type checker | `mypy` strict (`disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`, `no_implicit_optional`, `warn_redundant_casts`, `warn_unused_ignores`, `warn_no_return`, `strict_equality`, `warn_return_any`) |
| Test runner | `pytest` with markers `unit`, `property`, `smoke`, `golden`; coverage floor `92%` |
| Build backend | `hatchling` |
| Env manager | `uv` (`uv venv` → `.venv/`; `uv pip install -e .[dev]`) |
| Python | `>=3.13` (RunPod parity floor; py313 tool targets in pyproject.toml) |

Run via `make lint` (= `ruff check + black --check + mypy`) and `make test`.

## 3. Naming

For the full decision record + industry-citations, see
[ADR 0004 — Naming conventions](docs/source/adr/0004-naming-conventions.md).
This section is the day-to-day quick reference; the ADR is the
authoritative source.

- Module names: `snake_case`, lowercase package (`eval_toolkit`).
- Class names: `PascalCase`. Suffixes used in this repo:
  - `*Config` — frozen dataclass for settings
  - `*CI` — frozen dataclass holding a confidence interval
  - `*Result` / `*Estimate` / `*Report` — frozen dataclasses for analysis outputs
- Function names: `snake_case`. Sklearn-compatible (`pr_auc`, not
  `calculate_pr_auc`). No mandatory verb prefix.
- Constants: `UPPER_SNAKE_CASE` at module level.
- Private helpers: leading `_`; not exported via `__all__`.
- **Math identifiers**: Unicode allowed (`π`, `θ`, `μ`, `σ`, `α`, `β`) with a
  required English-comment alias on first appearance per scope. Example:
  ```python
  def bayes_optimal_threshold(π: float, c_fp: float, c_fn: float) -> float:
      # π = prior; c_fp/c_fn = false-positive/false-negative costs
      ...
  ```
- Mutation marking: not used. Mutating functions return `None` (Pythonic over
  Julia's `_inplace` suffix).

### 3a. Parameter naming (canonical list, locked at v1.0)

These names mean these things, everywhere. Future functions MUST use
them; deviations need justification in the PR description.

| Parameter | Meaning |
|---|---|
| `y_true` | Ground-truth labels (binary, shape `(n,)`) |
| `y_score` | Continuous score / probability (shape `(n,)`) |
| `y_pred` | Discrete prediction (threshold-dependent) |
| `n_resamples` | Bootstrap iteration count |
| `confidence` | Two-sided confidence level (0.95 default) |
| `n_bins` | Binning count for calibration / ECE |
| `n_jobs` | Parallelism (joblib + sklearn convention) |
| `ax` | Matplotlib axis (matplotlib convention) |
| `metric` | Callable `(y_true, y_score) -> float` |
| `rng` | RNG argument per [SPEC 7](https://scientific-python.org/specs/spec-0007/) — **canonical** convention (adopted v0.50.0). Accepts `int`, `np.random.Generator`, `BitGenerator`, `SeedSequence`, or `None`. |

The v0.50.0 SPEC 7 adoption preserves two `seed: int` exceptions:
`set_global_seeds(seed: int)` (global-state setter, not per-function
RNG; SPEC 7 doesn't apply) and adversarial dataclass fields (use Python
`random.Random(seed)`; not NumPy-RNG, so SPEC 7's typing doesn't fit).

### 3b. Class suffixes by domain

Each suffix maps to a Protocol contract. Stay within the pattern:

| Suffix | Domain | Protocol |
|---|---|---|
| `*Selector` | Threshold selection | `ThresholdSelector` |
| `*Splitter` | Cross-validation splits | `Splitter` |
| `*Check` | Leakage detection | `LeakageCheck` |
| `*Loader` | Dataset loading | `DatasetLoader` |
| `*Reader` | Prediction artifact reading | `PredictionReader` |
| `*Variant` | Preprocessing variant | (functional API) |
| `*Strategy` | Dedup similarity backend | `SimilarityStrategy` |
| `*Injection` / `*Substitution` | Adversarial char-injection / -substitution | `TextTransform` |

### 3c. Module naming (singular vs plural)

- **Plural noun** for collection-of-types modules: `metrics`,
  `loaders`, `protocols`, `losses`, `probes`, `splits`, `paths`,
  `seeds`, `thresholds`, `artifacts`, `claims`, `embeddings`,
  `scorecards`.
- **Singular noun** for domain-concept modules: `harness`,
  `bootstrap`, `manifest`, `calibration`, `leakage`, `analysis`,
  `provenance`, `evidence`, `stacking`, `text_dedup`.
- **Gerund** for process-domain modules: `preprocessing`.

### 3d. Asymmetric module promotion (private → public)

Collection-of-types private modules MAY be promoted to plural-public
when they hold ≥2 user-relevant types. Single-function private
modules SHOULD stay underscore. See
[ADR 0001](docs/source/adr/0001-flat-module-layout.md) for the trigger
analysis.

Examples:

- `_scorecard.py` (4 public exports) → `scorecards.py` at v0.49.0. ✓ promote.
- `_sweep.py` (1 public function `sweep`) → stays `_sweep.py`. ✓ keep private.

## 4. Type hints

- Every public function has fully typed parameters and return.
  `disallow_untyped_defs = True` in mypy enforces this.
- Modern syntax: `list[T]`, `X | None`. Use `Optional` only when stylistically
  required.
- `from __future__ import annotations` only when forward refs require it.
- `Protocol` only at "real seams" — where two or more concrete implementations
  exist or are planned. The authoritative Tier-2-stable set is `_TIER2_PROTOCOLS` in
`tests/test_public_api.py` plus
[ADR 0003](docs/source/adr/0003-stability-contract-and-gate3-methodology.md):
the nine strict Tier-2 Protocols are `Scorer`, `LeakageCheck`, `Splitter`,
`ThresholdSelector`, `DatasetLoader`, `MetricSpec`, `MetaLearner`, `Probe`,
`TextTransform`. The seams below are illustrative detail —
`SliceAwareScorer` is an opt-in subprotocol of `Scorer`, and
`SimilarityStrategy` / `Versioned` are real seams that are **not** in the
Tier-2 frozenset:
  - `Scorer` + `SliceAwareScorer` (`harness.py`) — anything with
    `predict_proba(X) -> np.ndarray`. `SliceAwareScorer` adds opt-in
    `should_score_slice(name)` for cost-controlled skipping.
  - `LeakageCheck` (`leakage.py`) — uniform `validate(splits) -> LeakageFinding`
    contract for 7 reference impls (exact / near / encoding-obfuscated /
    cross-split / label-conflict / group / temporal).
  - `Splitter` (`splits.py`) — `iter_folds(slice) -> Iterator[dict[str, EvalSlice]]`
    + `get_n_splits` for 5 reference impls.
  - `ThresholdSelector` (`thresholds.py`) — `select(y_true, y_score) ->
    ThresholdResult` for 6 reference impls.
  - `DatasetLoader` (`loaders.py`) — HF-`DatasetDict`-shaped
    `load_splits() -> dict[str, EvalSlice]` + Croissant-compatible `describe()`
    for 4 reference impls.
  - `SimilarityStrategy` (`text_dedup.py`) — pluggable similarity backend for
    `near_dedup` / `cross_dedup` / `NearDuplicateCheck` / `CrossSplitLeakageCheck`.
  - `Versioned` (`protocols.py`) — opt-in single-attribute Protocol; any
    Tier-2 implementation may expose `version: str`.
    `RunManifest.versioned_objects` auto-collects them. Mirrors the
    `lm-evaluation-harness` task `VERSION` pattern. See
    `docs/methodology/versioning.md`. (Single source of truth at
    `protocols.py:64` since v0.49.0; the duplicate previously in
    `leakage.py:82` was removed.)
- All seams are `@runtime_checkable` so callers can `isinstance(obj, Protocol)`.
- Reference impls are `@dataclass(frozen=True, slots=True)` with config in the
  constructor (`TargetRecallSelector(recall=0.90)`) and the Protocol method as
  the only behavior.
- `NamedTuple` for stable public records that benefit from positional access;
  frozen dataclasses with `slots=True` otherwise.

### 4a. Fitted-attribute trailing underscore (sklearn convention)

Estimator-style classes (`fit`/`predict` pattern) that store
**learned-from-data attributes** use trailing underscore per scikit-learn
convention: `coef_`, `classes_`, `n_features_in_`, `feature_importances_`.
These attributes MUST NOT be set in `__init__` — set them only in `fit()`.

Frozen reference-impl dataclasses (`@dataclass(frozen=True, slots=True)`)
are **exempt** — they hold config, not fitted state.

Current canonical example: `stacking.LogisticStacker`.

### 4b. TypeVar naming

Internal (private) `TypeVar`s use a leading underscore per Google Python
Style Guide §3.19.10: `_T = TypeVar("_T")`. Public, constrained `TypeVar`s
without the underscore are allowed only when explicitly part of an
exported generic API.

## 5. Dataclasses

1. **`slots=True` always** on repo-owned dataclasses. Catches typos at
   attribute-set time and trims memory.
2. **`frozen=True`** for value/config/result types. Mutable only for
   runtime-state wrappers.
3. **Validation in `__post_init__`** using stdlib exceptions. Fail fast on
   invalid configurations.
4. Mutable defaults via `field(default_factory=...)`. Never `field([])`.

## 6. Errors

- **Always `raise`. Always stdlib.** `ValueError` for bad-data inputs,
  `TypeError` for wrong-type inputs, `RuntimeError` for state errors,
  `FileNotFoundError` for missing artifacts, `KeyError` for missing entries.
  **No custom exception hierarchy.** **No `Result[T, Error]` pattern.**
- **No `assert` in `src/`.** It is stripped under `python -O`. Use
  `raise ValueError(...)` instead, even for "this should be impossible" cases.
  `assert` in `tests/` is fine and idiomatic.
- **Diagnostic-message rule.** A caller should be able to fix the input
  without reading the function's internals:
  ```python
  raise ValueError(f"max_length must be > 0, got {self.max_length}")
  raise TypeError(f"texts must be list or Series, got {type(texts).__name__}")
  ```

## 7. Validation boundary

Validate at:
- Public API entry points
- Config loaders (YAML → typed object)
- Before resource-heavy operations (file writes, network calls)

Do NOT re-validate in helpers downstream of those boundaries. Trust is faster.

## 8. Function design

- **Single responsibility.** Two functions doing 80% the same thing → factor a
  helper.
- **Soft 20–50 line guideline.** Long functions are fine when they're cohesive
  math kernels; longer is allowed with a docstring rationale. No formal cap.
- **Pure helpers preferred** for parsing, formatting, rendering.
- **Don't over-extract.** Three-line helpers used once add cost without adding
  clarity. Inline.

## 9. DataFrames

**Never mutate.** Always return new:

- ✅ `df = df.assign(col=value)`
- ❌ `df["col"] = value`
- ✅ `df = df.rename(columns={"old": "new"})`
- ❌ `df.rename(columns={"old": "new"}, inplace=True)`

`.copy()` is fine when defensive copying is the goal.

## 10. Imports

Order (enforced by ruff `I`):

1. `__future__` imports (only when needed)
2. Stdlib
3. Third-party
4. First-party (`eval_toolkit.*`)

Local imports inside functions are allowed for:
- Lazy heavy imports (matplotlib, sklearn fitters)
- Optional-dep imports (pyyaml, torch)

## 11. Logging

Use `logging` (library context — consumers configure handlers). Do not use
`print` in `src/eval_toolkit/`. Log levels: `DEBUG` for internal events; `INFO`
only for the rare user-relevant harness progress signal; **`WARNING` is reserved
for `warnings.warn(...)`, not `logger.warning(...)`**; and **`ERROR` must not
appear in library code — raise an exception instead**. See CONTRIBUTING.md
§Logging for the full rationale.

## 12. Docstrings

NumPy-style required for every public symbol with these sections (where
applicable):

```python
def fit_temperature(val_logits, val_labels, bounds=(0.05, 20.0)):
    r"""Single-parameter temperature scaling [1]_.

    Parameters
    ----------
    val_logits : np.ndarray, shape (n, 2)
        Validation logits.
    val_labels : np.ndarray, shape (n,)
        Binary labels in {0, 1}.

    Returns
    -------
    dict
        With keys 'temperature', 'nll_pre', 'nll_post', 'improvement'.

    Raises
    ------
    ValueError
        If val_logits.shape[1] != 2.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> logits = rng.normal(size=(100, 2))
    >>> labels = (logits[:, 1] > logits[:, 0]).astype(int)
    >>> result = fit_temperature(logits, labels)
    >>> 0.05 <= result['temperature'] <= 20.0
    True

    Notes
    -----
    Minimizes NLL of softmax(logits / T):

    .. math:: T^* = \arg\min_T - \frac{1}{n}\sum_i \log p_{y_i}(x_i / T)

    References
    ----------
    .. [1] Guo et al., "On Calibration of Modern Neural Networks,"
           ICML 2017. arXiv:1706.04599
    """
```

- **Examples** are doctest-runnable for math/algorithmic kernels (`metrics`,
  `bootstrap`, `calibration`, `text_dedup` hash/normalize). Enforced via
  `pytest --doctest-modules`.
- **Notes** carries LaTeX (`.. math::`) for mathematical content.
- **References** cites arXiv IDs / DOIs / journal cites.
- For modules where doctests would be contrived (`plotting`, `harness`,
  `provenance`), Examples are optional.
- **Docstring prose wraps at 75 cols** (numpydoc convention) so that
  `help()` is readable in a terminal. Doctest code blocks inside the
  docstring follow the 100-col Black rule (code stays comfortable in an
  editor even though prose around it is narrower).

## 13. Comments

Default: none. Comment only when intent is non-obvious from code/types. Never
restate what the code says.

## 14. Tests

- **File naming**: `tests/test_<module>.py` mirrors
  `src/eval_toolkit/<module>.py`. Auxiliary tests per module use
  suffixes (`test_<module>_props.py`, `test_<module>_validation.py`,
  `test_<module>_golden.py`).
- **Function naming**: `test_<thing_under_test>_<scenario>`. No
  class-based test grouping unless fixtures truly demand it (rare).
- **Markers**: `unit`, `property`, `smoke`, `golden`.
- **Sklearn-reference + analytical** as the unit-test oracle where available.
- **Hypothesis** required for math/stat invariants. Strategies use
  `hypothesis.extra.numpy` for arrays.
- **Golden tests** only for `docs.py`, where the output is the contract.
- **Doctests** for math/algorithmic kernels.
- **Coverage floor**: 92%.
- **`assert` is fine in tests.**

## 15. Packaging

- **Semver from v0.1.0.** Public API breaking changes require major-version
  bump.
- **CHANGELOG.md** in Keep-a-Changelog format.
- **Optional extras**: `core` (default), `dataframe`, `plotting`, `property`,
  `yaml`, `all`, `dev`. Users install only what they need.
- **PEP 561**: ships `py.typed` marker.

## 16. No-go list

- ❌ Pydantic, LangChain, config frameworks
- ❌ Custom exception hierarchies
- ❌ `Result[T, Error]` patterns (use raises)
- ❌ Thin dependency wrappers
- ❌ `_inplace` mutation suffix (not Pythonic)
- ❌ `from __future__ import annotations` everywhere (only when needed)
- ❌ Mathematical Unicode without an English-comment alias

## 17. Public API discipline

- Every module declares `__all__`.
- The package's `__init__.py` re-exports the public surface so both
  `from eval_toolkit import scorecard` and
  `from eval_toolkit.scorecards import scorecard` work — matches
  sklearn/pandas/scipy convention. (Threshold-dependent scalar metrics
  such as `pr_auc` left the top level at v0.46 Decision L — import
  them from `eval_toolkit.metrics`.)
- Private helpers are prefixed with `_` and not re-exported.
