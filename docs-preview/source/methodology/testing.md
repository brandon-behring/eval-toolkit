# Testing your evaluation code

> **Background** *(skip if you've internalized this)*. Evaluation code is
> *also* code, and bugs in it produce subtler errors than bugs in
> models — a metric that's silently wrong looks plausible because
> "metric numbers are just numbers". Four test patterns catch the
> failure modes that matter: (1) property tests catch mathematical
> invariant violations (e.g., ROC-AUC + ROC-AUC of inverted scores
> should sum to 1); (2) reference-equivalence tests catch divergence
> from canonical implementations (sklearn, scipy); (3) golden tests
> catch regressions in deterministic outputs; (4) visual-regression
> tests catch plot drift.

eval-toolkit's own test suite uses all four. This chapter describes the
patterns so consumer projects (and downstream agents) can copy them.

## The 4 patterns at a glance

| Pattern | What it catches | Library | When to use |
|---|---|---|---|
| Property | Math-invariant violations | hypothesis | Every public function with a known invariant |
| Reference-equivalence | Divergence from canonical impls | pytest + sklearn | Every wrapped function (one per kernel) |
| Golden | Regressions in deterministic outputs | pytest + JSON snapshot | Anchor-based docs, JSON serializers |
| Visual regression | Plot rendering drift | pytest-mpl | Every plotting function |

eval-toolkit's `tests/` mirrors this taxonomy:

```
tests/
  test_*_unit.py        ← reference-equivalence + smoke
  test_*_props.py       ← property tests (hypothesis)
  test_docs_golden.py   ← golden tests (snapshot)
  test_plotting_visual.py ← pytest-mpl
  test_schemas.py       ← jsonschema validation (golden-ish)
  strategies.py         ← shared hypothesis strategies
```

(property-tests)=
## Property tests
A *property* is a mathematical statement that should hold for every
valid input. Hypothesis generates inputs and checks the property; if
it finds a counterexample, it shrinks it to a minimal failing case.

The toolkit's `tests/strategies.py` exports two reusable strategies:

```python
# tests/strategies.py
# from hypothesis import strategies as st
# def balanced_binary_array(size): ...     # generates (n,) binary arrays with both classes
# def score_array(size): ...                # generates (n,) float arrays in [0, 1]
```

A property test has the shape:

```python
import numpy as np
import pytest
from eval_toolkit import roc_auc

# Real test (mirrors tests/test_metrics_props.py:test_auroc_inversion):
@pytest.mark.unit
def test_auroc_inversion_demo() -> None:
    """ROC-AUC is anti-symmetric in score sign: roc_auc(y, -s) = 1 - roc_auc(y, s)."""
    rng = np.random.default_rng(0)
    for _ in range(10):  # mini sweep instead of hypothesis for the doc-block
        y = rng.integers(0, 2, size=80)
        if len(set(y.tolist())) < 2:
            continue
        s = rng.uniform(-1, 1, size=80)
        assert roc_auc(y, s) == pytest.approx(1.0 - roc_auc(y, -s), abs=1e-9)

test_auroc_inversion_demo()  # smoke-call so Sybil verifies it
print("AUROC inversion property holds")
```

The full property-test version uses Hypothesis decorators —
`@given(...)` + `@settings(deadline=None, max_examples=30)` — to
generate hundreds of cases per run. See
`tests/test_metrics_props.py` for the canonical templates.

### Properties worth testing

For every metric:

- **Monotone-transform invariance.** ROC-AUC is invariant to monotone
  transformations of the score: `roc_auc(y, s) == roc_auc(y, f(s))` for
  any monotone f.
- **Boundedness.** PR-AUC ∈ [0, 1]; ECE ∈ [0, 1].
- **Inversion symmetry.** `roc_auc(y, -s) = 1 - roc_auc(y, s)`.
- **Single-class behavior.** Returns a `"skipped"` marker (toolkit
  convention), not a silent NaN or zero.
- **Empty / size-1 input.** Raises `ValueError`, not a confusing
  `IndexError`.

For threshold selectors:

- **`MaxF1Selector` returns F1 ≥ F1 at any other threshold on the
  PR curve.** This is the optimality property from Lipton-Elkan 2014.
- **`TargetRecallSelector(p)` returns a threshold whose recall ≥ p.**

For leakage checks:

- **`run_leakage_checks([], splits)` returns an empty
  LeakageReport.has_errors() == False.**
- **A clean fixture produces no error-severity findings with
  n_affected > 0.**

(reference-equivalence)=
## Reference-equivalence tests
When the toolkit *wraps* a canonical library function, an equivalence
test pins the wrapping faithful: it runs the wrapper and the canonical
on the same input and asserts numerical equality (within tolerance).

```python
import numpy as np
from sklearn.metrics import average_precision_score
from eval_toolkit import pr_auc

# Real test pattern (tests/test_metrics_unit.py:test_pr_auc_matches_sklearn):
rng = np.random.default_rng(42)
y = rng.integers(0, 2, size=200)
s = rng.uniform(0, 1, size=200)

assert pr_auc(y, s) == np.float64(average_precision_score(y, s)).astype(float)
print("pr_auc matches sklearn.average_precision_score exactly")
```

The toolkit's policy: every kernel that wraps a canonical implementation
gets one equivalence test. From the v0.3 research audit:

> *Reference-impl tests are sparse — only `pr_auc` is value-equality-tested
> against its sklearn equivalent. For a "wraps and validates" library,
> one such test per wrapped function is the table-stakes contract.*

This is a known v0.7 gap closing in PR 1.5 alongside the property
tests for the new modules.

(golden-tests)=
## Golden tests
A golden test pins a deterministic output to a snapshot file. Running
the test re-runs the deterministic computation and compares it to the
snapshot. If a code change unintentionally changes the output, the test
fails loudly.

The toolkit uses golden tests for:

1. The anchor-based markdown rendering in
   [`eval_toolkit.docs`](../api/docs.md).
2. **JSON schema validation** of `results.json` /
   `results_full.json` / `manifest.json` outputs against the v1
   schemas in `src/eval_toolkit/schemas/` (see `tests/test_schemas.py`).

Pattern:

```python
# Sketch — see tests/test_schemas.py for the real one.
import json
import tempfile
from pathlib import Path
import pandas as pd
from jsonschema import Draft202012Validator

import eval_toolkit
from eval_toolkit import EvalSlice, evaluate, write_run_result

class _Scorer:
    def predict_proba(self, X):
        import numpy as np
        return np.full(len(X), 0.5)

# Run an evaluation (deterministic output for fixed seed).
df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
slice_ = EvalSlice(name="test", df=df)
result = evaluate({"s": _Scorer()}, [slice_], run_id="demo")

with tempfile.TemporaryDirectory() as d:
    compact_path, _ = write_run_result(result, Path(d))
    loaded = json.loads(compact_path.read_text())

# Schema lives at src/eval_toolkit/schemas/results.v1.json.
schema_path = Path(eval_toolkit.__file__).parent / "schemas" / "results.v1.json"
schema = json.loads(schema_path.read_text())
Draft202012Validator(schema).validate(loaded)
print("results.json validates against v1 schema")
```

Schema validation is more robust than literal JSON snapshots — it
allows additive changes (adding new optional fields) without breaking
the test, while still catching breaking schema changes.

(visual-regression)=
## Visual regression
Plotting code is hard to test with assertions — "the plot looks right"
isn't a numeric predicate. `pytest-mpl` saves baseline PNGs and
pixel-compares on each run.

```
# tests/test_plotting_visual.py
@pytest.mark.mpl_image_compare(baseline_dir="baseline")
def test_pr_curve_matches_baseline():
    fig = plot_pr_curve(...)
    return fig
```

Run with `pytest --mpl --mpl-baseline-path=tests/baseline`. Failures
write a diff PNG; the developer inspects, accepts (regenerate
baseline) or fixes the plot code.

The pattern is mature enough that the toolkit's CI runs it on every
commit. Consumer projects with their own plots should adopt the same
pattern (it's free coverage for hard-to-assert code).

## Putting it all together

A well-tested evaluation module has all four:

```
my_eval/
  metrics.py
tests/
  test_metrics_unit.py     # reference-equivalence vs sklearn
  test_metrics_props.py    # hypothesis @given invariants
  test_outputs_golden.py   # JSON schema validation on serialized results
  test_plots_visual.py     # pytest-mpl on plotting functions
  baseline/
    test_*.png             # pytest-mpl baselines
```

This is the structure eval-toolkit's own `tests/` follows. Consumer
projects copying this pattern get the same defensive depth: math-
invariant correctness, behavioral parity with canonical libraries,
serialization stability, and visual stability.

(pitfalls)=
## Pitfalls / Common mistakes
- **Property tests with no actual invariants.** "PR-AUC is a number" is
  not a property. The property tests in
  `tests/test_metrics_props.py` are templates of the *real* invariants
  (inversion, boundedness, monotone-invariance) — pattern-match those.
- **Reference tests with NaN tolerance.** sklearn / scipy versions
  drift; allow `pytest.approx(..., rel=1e-6)`-level tolerance, not bit
  equality. Bit equality breaks across numpy / blas updates.
- **Golden snapshots that include non-deterministic fields.** A
  `RunResult` snapshot that includes `git_sha` or `wall_clock_seconds`
  fails on every commit. Use schema validation (allows additive
  fields) or strip non-deterministic fields before snapshotting.
- **Property tests that hit external services.** Hypothesis generates
  hundreds of inputs per run; if your test calls an LLM API per
  case, the bill is real. Use the smoke-test pattern (10 hand-picked
  inputs) for expensive predictors.
- **Visual regression baselines committed without review.** A
  pytest-mpl failure can be silenced by re-generating the baseline.
  Treat baseline regeneration as a code change subject to PR review,
  not a "fix the test" reflex.

## Further reading

- Hypothesis docs: https://hypothesis.readthedocs.io/
- pytest-mpl docs: https://pytest-mpl.readthedocs.io/
- jsonschema docs: https://python-jsonschema.readthedocs.io/
- *Property-Based Testing for Stats Code,* PyData 2018 (Christopher
  Armstrong) — taxonomy of stats-code-relevant properties.
- The v0.3 research audit (`docs/v0.3_research_audit.md`) — catalogs
  the toolkit's own kernel-level reference-equivalence gaps.

See also: [reproducibility.md](reproducibility.md) (golden tests need
deterministic outputs).
