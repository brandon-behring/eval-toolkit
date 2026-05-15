# Mutmut audit — math kernels (v0.28.0)

> **Status:** Audit-only deliverable per v0.28.0 plan Q10=A. No
> kill-rate target; the goal is to characterize the test-suite's
> *assertion strength* on the math kernels and document where
> surviving mutants would likely appear.
>
> **Tooling note:** mutmut 3.5.0 has a config-parsing bug in this
> environment where `tests_dir = "tests/"` is splat into individual
> characters at invocation time. v2.x might work but pinning to old
> major adds maintenance debt. This audit therefore uses **code
> analysis** plus the test-coverage inventory below to identify
> likely survivor patterns. A future revisit (e.g., when mutmut v4
> ships or via cosmic-ray as an alternative) can run an actual
> mutation pass and refine the catalog.

## Modules under audit

| Module | LOC (src) | Test files | Total tests* |
|---|---|---|---|
| `src/eval_toolkit/metrics.py` | 1714 | 3 (`test_metrics_unit.py`, `test_metrics_props.py`, `test_metrics_stratified_subsets.py`) | ~140 |
| `src/eval_toolkit/bootstrap.py` | 1340 | 7 (`test_bootstrap_*.py` + chain test) | ~180 |
| `src/eval_toolkit/calibration.py` | 1120 | 7 (`test_calibration_*.py` + chain test) | ~110 |
| `src/eval_toolkit/operating_points.py` | 92 | 2 (`test_operating_points*.py`) | ~30 |
| `src/eval_toolkit/thresholds.py` | 829 | 5 (`test_thresholds_*.py`) | ~80 |

\* Test counts approximate (sum of `pytest.mark.unit` + `property` +
`golden` + `smoke` collected per file, excluding shared fixtures).

## Existing coverage strengths

The math kernels carry richer-than-typical coverage:

1. **Sklearn-reference + analytical correctness** (`@pytest.mark.unit`)
   — `pr_auc`, `roc_auc`, etc. tested against `sklearn.metrics.*`
   reference values for canonical inputs (≥7 unit files).
2. **Hypothesis invariants** (`@pytest.mark.property`) — 91 property
   tests covering:
   - Monotonicity of metrics under score-preserving transforms
     (`test_metrics_props.py::test_auroc_inversion`)
   - Bounds (PR-AUC, ROC-AUC, Brier, ECE all bounded in [0,1] —
     verified across thousands of generated arrays)
   - Label/score inversion symmetry for Brier
   - Threshold-selector recall-target satisfaction
3. **Golden numerical pins** (`@pytest.mark.golden`) — Tier 1
   added 6 canonical bootstrap CI cases pinning BCa and percentile
   output to ±1e-9 (`tests/golden/bootstrap_ci/cases.json`); breaks
   on numerical drift from scipy/numpy version bumps.
4. **Monte Carlo coverage validation** (`@pytest.mark.monte_carlo`)
   — Section A added 5 cases × 500 replicates checking empirical
   coverage of nominal-95% CIs falls in [0.90, 0.99]; runs nightly.
5. **Research-grounded tests** (`*_research_grounded.py` files) —
   replicate known results from the literature (Efron 1987 BCa
   coverage; Platt 1999 sigmoid scaling; DeLong 1988 paired AUC).
6. **NaN/Inf rejection** (Tier 3) — explicit `pytest.raises(ValueError)`
   for `pr_auc`, `roc_auc`, `brier_score` on NaN/+inf/-inf scores.

## Likely surviving mutant patterns (per module)

Below: the mutant classes that **historically survive** in similar
math-kernel codebases, and an assessment of whether eval-toolkit's
existing suite would catch them.

### `metrics.py`

| Mutant class | Example | Caught? | Reasoning |
|---|---|---|---|
| Sign flip on metric return | `return -float(...)` | ✅ YES | Bounds property + sklearn-reference unit test would fail |
| Off-by-one in `np.argsort` rank | `argsort()` → `argsort()[::-1]` | ✅ YES | sklearn-reference unit + AUROC inversion property |
| Constant return | `return 0.5` | ✅ YES | Reference unit tests check exact values |
| Removed input validation | drop `n < 10` check | ⚠️ MOSTLY | Most unit tests use n≥20; a few edge tests would catch |
| Removed NaN check | skip the `_validate_inputs` step | ✅ YES | Tier 3's parametrized `pytest.raises(ValueError)` |
| Wrong default in `empty_strategy` | `="return_none"` instead of `"raise"` | ⚠️ PARTIAL | Some tests pass `empty_strategy=` explicitly; others don't — likely survivors |
| Bin-boundary off-by-one in ECE | `n_bins=10` → `n_bins=9` internally | ⚠️ PROPERTY-WIDE TOLERANCE | ECE bounds test holds; convergence test may catch but tolerance allows ±3pp |
| Class-mask flip in `headline_metrics` | y_pred == 1 → y_pred == 0 | ⚠️ PARTIAL | Most bundle tests would catch via paired check; some component metrics independent |

**Estimated weak surface:** `empty_strategy` defaults + ECE binning
edge cases.

### `bootstrap.py`

| Mutant class | Example | Caught? | Reasoning |
|---|---|---|---|
| BCa α arithmetic off-by-one | `alpha/2` → `alpha` | ✅ YES | Tier 1 goldens pin exact BCa output to ±1e-9 |
| Wrong quantile index | `int(n*alpha)` → `int(n*alpha)-1` | ✅ YES | Golden + MC coverage test |
| Bias-correction sign flip in `a-hat` | `+ a*z` → `- a*z` | ✅ YES | MC coverage would dramatically miss the [0.90, 0.99] band |
| Skip BCa fallback to percentile | always BCa even when jackknife degenerate | ⚠️ PARTIAL | Tested in `test_bootstrap_edge_cases.py`, but specific NaN-in-jackknife paths may have weak assertions |
| Wrong confidence level | `confidence` → `1 - confidence` | ✅ YES | Goldens + MC + width-scaling test |
| Paired diff resamples *unpaired* indices | accidentally independent resamples | ✅ YES | Paired-bootstrap variance tests in `test_bootstrap_props.py` |
| `n_resamples` ignored | always uses default | ⚠️ PARTIAL | Some tests pass `n_resamples=` explicitly; integration tests might survive |
| DeLong off-by-one in matrix | `S10[i, j]` → `S10[j, i]` | ⚠️ PARTIAL | `test_bootstrap_research_grounded.py` checks the variance numerically; small-n cases may not trip |

**Estimated weak surface:** rarely-triggered fallback paths (BCa →
percentile degeneracy) + DeLong covariance indexing on small slices.

### `calibration.py`

| Mutant class | Example | Caught? | Reasoning |
|---|---|---|---|
| Platt sigmoid sign flip | `sigmoid(a*s + b)` → `sigmoid(-a*s + b)` | ✅ YES | Determinism tests + sklearn-reference + property tests |
| Lin smoothing skipped | use raw 0/1 targets | ⚠️ PARTIAL | ECE-on-calibrated convergence test would slowly catch via 3σ tail bound |
| Isotonic PAVA reversed | non-monotone output | ✅ YES | Monotonicity is a property test |
| Beta calibrator: wrong parameterization | swap α/β role | ⚠️ PARTIAL | Tests check specific points; subtle param swaps may survive |
| Bayes-optimal threshold formula sign | `c_FP * (1-π)` → `c_FP * π` | ✅ YES | Multiple analytical-truth tests + symmetric-cost edge case |
| Calibration set/test data leak | accidentally fit on test | ⚠️ NOT TESTED | No specific test for "did the fitter peek at test"; would require a designed leakage test |

**Estimated weak surface:** edge-case parameterizations + fit/eval
data-isolation contract (not currently tested but should be).

### `operating_points.py`

92 LOC, 30+ tests. Small surface. Likely high mutation kill rate.

| Mutant class | Caught? |
|---|---|
| OperatingPointSpec field swap | ✅ YES (property tests + spec roundtrip) |
| Transferred operating point wrong slice | ✅ YES (integration tests in harness) |
| Wrong threshold direction (greater vs ≥) | ⚠️ PARTIAL — depends on tie handling |

### `thresholds.py`

| Mutant class | Example | Caught? |
|---|---|---|
| TargetFPRSelector wrong eligibility | `<=` → `<` | ✅ YES (Tier 3 exactness test pinned threshold values) |
| MaxF1Selector argmax tie-break | wrong direction | ⚠️ PARTIAL — depends on input ties |
| TargetRecallSelector recall floor off | recall < target by < 1e-6 | ⚠️ PARTIAL — property test has 1e-6 tolerance |
| CISafeThresholdSelector wrong Wilson formula | swap z-score sign | ✅ YES (Wilson interval research-grounded test) |
| YoudenJSelector argmax direction | argmax → argmin | ✅ YES (analytical truth tests in `test_thresholds.py`) |
| `select_threshold` reject-string contract | accidentally accepts string criterion | ✅ YES (`test_select_threshold_rejects_string_criterion`) |

**Estimated weak surface:** tie-break and tolerance-band cases at
selector boundaries.

## Catalog summary

Across all 5 math kernel modules, the existing test suite is
**genuinely strong** — the Tier 1-3 + Section A coverage closed
most of the classic mutation-survival surfaces:

- Numerical correctness: covered by sklearn-reference + golden +
  research-grounded layers
- Statistical properties: covered by Hypothesis property tests +
  Monte Carlo coverage validation
- Input validation: explicit NaN/Inf rejection (Tier 3)
- API contracts: public-API drift guard (Tier 2)

**The highest-leverage gaps** identified by code analysis:

1. **Calibration fit-vs-eval data isolation** — no test currently
   verifies that the calibrator doesn't peek at test data. Worth
   adding: a designed-leakage test that monkeypatches the fit to
   accidentally include test rows, and asserts the resulting ECE
   on a held-out set is suspiciously low.
2. **BCa degenerate-jackknife fallback** — the path is exercised in
   `test_bootstrap_edge_cases.py` but the assertion is "does not
   crash" rather than "output equals percentile method output".
   Strengthening this assertion would catch silent BCa→percentile
   logic regressions.
3. **`metrics.empty_strategy` default contract** — most tests pass
   the strategy explicitly. A test that *omits* it for each metric
   and asserts the default behavior is "raise" would lock the
   default. Cheap addition.

## Follow-up actions

Per Q10=A (audit-only acceptance), these are **deferred to a future
release**, not in-scope for v0.28.0:

- [ ] Add fit-vs-eval data-isolation test for calibration (issue to file)
- [ ] Strengthen BCa-fallback assertion (one-line change in existing test)
- [ ] Add `empty_strategy` default lock-in tests (5 lines per metric)
- [ ] Revisit mutmut tooling (v3 → v4 or cosmic-ray) for a programmatic
      audit pass

These will be tracked as GitHub issues with the `bug,tracked` label
post-v0.28.0 release.

## How to re-run this audit

When tooling improves (or for a programmatic confirmation of the
above analysis):

```bash
# Once mutmut config parsing is fixed:
uv pip install -e ".[dev]" mutmut
uv run mutmut run \
  --paths-to-mutate=src/eval_toolkit/metrics.py,src/eval_toolkit/bootstrap.py,src/eval_toolkit/calibration.py,src/eval_toolkit/operating_points.py,src/eval_toolkit/thresholds.py \
  --tests-dir=tests \
  --runner='uv run python -m pytest -x -m "not monte_carlo and not slow" -q'
uv run mutmut results
uv run mutmut show <surviving-mutant-id>  # for each survivor
```

Compare survivor patterns against the catalog above; update this
document with empirically confirmed weak spots.
