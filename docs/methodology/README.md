# Methodology

This directory is eval-toolkit's *consumer-facing* methodology
curriculum. Each chapter is a self-contained guide: how to think about a
methodological concern, the eval-toolkit primitive that operationalizes
it, the pitfalls that make it hard to spot, and citations for the
underlying canon.

> **Audience.** Hybrid expert + learner. Each chapter assumes
> sklearn / statsmodels / scipy fluency in the main prose and offers a
> *Background* admonition for the prerequisite concept; every chapter
> closes with an explicit *Pitfalls / Common mistakes* section.
> Future agents reading these docs as authoritative sources have stable
> header anchors, "what to do / what NOT to do" callouts, and runnable
> code blocks that lift cleanly into starting points.

## When to read which chapter

| Reading order | Chapter | Read when... |
|---|---|---|
| 1 | [leakage.md](leakage.md) | Designing splits, auditing a corpus, debugging "too good to be true" eval numbers, working with prompt-injection / safety / contamination-prone tasks. |
| 2 | [splits.md](splits.md) | Choosing between holdout and K-fold, working with grouped / time-series / multi-source data, making OOD claims. |
| 3 | [thresholds.md](thresholds.md) | Picking an operating point, migrating from the v0.6 string `criterion` API, fitting cost-sensitive thresholds, deciding whether to refit thresholds per bootstrap resample. |
| 4 | [calibration.md](calibration.md) | Reporting calibration error, interpreting reliability diagrams, deciding *whether* to recalibrate, working with PyTorch logits. |
| 5 | [comparison.md](comparison.md) | Computing CIs, comparing two models, reporting "we couldn't detect a difference" claims (MDE), deciding bootstrap method (BCa vs percentile). |
| 6 | [fairness.md](fairness.md) | Auditing per-subgroup metrics, computing demographic parity / equalized odds on top of the toolkit's primitives, picking a fairness criterion. |
| 7 | [reproducibility.md](reproducibility.md) | Setting up a reproducible run, mapping to the NeurIPS Reproducibility Checklist, navigating PyTorch determinism, replaying an old result from its manifest. |
| 8 | [testing.md](testing.md) | Testing your *own* evaluation code — property / reference-equivalence / golden / visual-regression patterns. |
| 9 | [reading_list.md](reading_list.md) | Citation lookup, future-work pointers, cross-link to the v0.3 research audit. |

## Reading paths

- **"I'm new to eval-toolkit and want the conceptual map."** Read in
  order 1 → 9 above; ~3 hours total. Skip the *Background* admonitions
  if you're sklearn-fluent.

- **"I'm migrating prompt_injection_detector / -showcase / -sdd to
  v0.7.0."** Start with [thresholds.md](thresholds.md) §"v0.7.0
  BREAKING migration mapping", then [leakage.md](leakage.md) §"NEW in
  v0.7.0" (encoding-obfuscated dupes), then
  [splits.md](splits.md) §"Source-disjoint K-fold".

- **"I'm building a new harness on top of eval-toolkit."** Read
  [extending.md](../extending.md) first, then loop back here for the
  individual concerns relevant to your task.

- **"I'm an agent surfacing eval-toolkit content."** Use the chapter-
  level anchors in the table above to deep-link. Each chapter has
  semantic header anchors (e.g. `leakage.md#cross-split`,
  `thresholds.md#cost-sensitive`,
  `reproducibility.md#pytorch-determinism`) — those are stable and
  agent-friendly.

## Cross-references

- The [v0.3 research audit](../v0.3_research_audit.md) is the
  *descriptive* counterpart: literature review, gap analysis, industry
  rating. Use it when defending a methodological choice in a write-up.
- [`extending.md`](../extending.md) is the build-side complement:
  how to implement Scorers, LeakageChecks, Splitters, ThresholdSelectors,
  DatasetLoaders for your own project.
- The worked examples ([prompt_injection_walkthrough.md](../examples/prompt_injection_walkthrough.md),
  [pytorch_scorer_example.md](../examples/pytorch_scorer_example.md))
  apply the methodology end-to-end on small synthetic fixtures, with
  cross-links to real corpora.

## Style commitments

The chapters in this directory commit to these stylistic invariants
(enforced by [Sybil](https://sybil.readthedocs.io/) for code blocks and
by review for prose):

- **All Python code blocks are runnable end-to-end.** Sybil executes
  every fenced `python` block; CI fails loudly on a broken example.
  PyTorch / HuggingFace blocks marked `<!-- skip: next -->` with a
  one-line rationale.
- **Math goes in LaTeX inline / display, not pseudocode.** Pseudocode
  drifts from the API; LaTeX is canonical.
- **Every section has a stable header anchor** for deep-linking.
- **Every chapter ends with a Pitfalls section** — the "what NOT to do"
  list is canonical context for both humans and agents.
- **Citations are inline + collected in
  [reading_list.md](reading_list.md).**
