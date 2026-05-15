# What's new

A digestible summary of recent releases — for the full per-line
detail, see [CHANGELOG.md](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md).

## 0.29.0 — best-practice infrastructure (May 2026)

**Theme**: forward-compatibility + observability + perf-regression
detection. No new user-facing API surface; better support for the
existing one.

### Highlights

- **`@deprecated` decorator** for forward-compat planning.
  Mark public symbols with a string deadline:
  `@deprecated("0.31.0", reason="…", use_instead="…")`. Validated
  at decoration time; emits `DeprecationWarning` on every call.
  Policy in [`docs/DEPRECATION.md`](DEPRECATION.md): minimum
  two-minor-versions warning before removal.
- **Structured library logging.** `eval_toolkit.{harness,leakage,
  bootstrap,loaders}` each emit DEBUG records (silent by default
  via NullHandler at the root). Filter granularly:
  `logging.getLogger("eval_toolkit.harness").setLevel(logging.DEBUG)`.
- **pytest-benchmark on math kernels.** Weekly cron in
  `nightly-benchmarks.yml` records per-kernel timings; catches
  perf regressions before release. Excluded from PR CI to keep
  iteration fast.
- **`docs/RELEASING.md` runbook.** Concrete release recipe with
  every gotcha + recovery from the v0.27.x and v0.28.x cycles.
- **`.editorconfig` + `CODEOWNERS`.** Repository hygiene.

### Migration

No migration needed — v0.29.0 is fully backward-compatible with
v0.28.x. Logging is opt-in; deprecation decorator is internal
tooling, not yet applied to any public symbols.

## 0.28.1 — security patch (May 2026)

**Theme**: supply-chain security signal.

- **CodeQL workflow** (push/PR/weekly cron) — static security
  analysis with the `security-extended` query suite. Findings
  populate the Security → Code scanning tab.
- **pip-audit gate in `test-base-install`** — fails PR CI on any
  known CVE in runtime deps (`numpy` / `scipy` / `scikit-learn` /
  `jsonschema`). Dev-extras vulns are surfaced via Dependabot.

No source-code behavior change.

## 0.28.0 — temporalcv cross-pollination bundle (May 2026)

**Theme**: cross-pollinate from sibling `temporalcv` + hosted docs.

- **`PurgedKFoldSplitter`** (`from eval_toolkit import PurgedKFoldSplitter`)
  — time-aware k-fold with explicit purge gap + post-test embargo,
  preventing label-window leakage when labels have a forward
  horizon. Per López de Prado 2018 Ch. 7.
- **`compute_label_overlap(t_train, t_test, horizon)`** — standalone
  helper for auditing arbitrary train/test label overlap.
- **Nightly Monte Carlo bootstrap CI calibration** — validates
  empirical coverage of nominal 95% CIs across pr_auc / roc_auc /
  imbalanced cases. Catches "math is wrong but self-consistent"
  bugs that goldens can't.
- **6-example documentation gallery** — minimal sybil-validated
  Markdown examples covering metrics + bootstrap, evaluate harness,
  calibration, leakage detection, claims/gates, paired comparison.
- **Hosted mkdocs-material docs site** at
  [brandon-behring.github.io/eval-toolkit](https://brandon-behring.github.io/eval-toolkit/).
  Auto-generated API reference from NumPy-style docstrings via
  mkdocstrings. Full LaTeX + TikZ rendering via MathJax v3 +
  tikzjax.
- **Public-repo polish**: `SECURITY.md` disclosure policy,
  `CITATION.cff` academic citation, 5 README badges.

## 0.27.2 — base-install fix (May 2026)

**Theme**: silent bug fix — `pip install eval-toolkit` was broken
for users who didn't also install `[dataframe]`.

- Fixed `import pandas as pd` at module top level in four modules
  (`harness`, `loaders`, `leakage`, `splits`) so base install
  works for the headline `from eval_toolkit import evaluate` path.
- Added a regression-guard CI job (`test-base-install`) that
  installs without extras and verifies imports.

## 0.27.1 — first PyPI release (May 2026)

**Theme**: eval-toolkit goes public.

- First release on PyPI: `pip install eval-toolkit`.
- Auto-publish on `v*` tag via PyPI Trusted Publishing (OIDC).
- Bumped from v0.27.0 because the internal milestone tag of that
  version pre-dated the PyPI publishing infrastructure.

---

## Prior releases

For the full v0.26 → v0.27 → v0.28 progression and individual
release CHANGELOG entries, see
[CHANGELOG.md](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md).
