# Reading list

Canonical references for the methodology this toolkit operationalizes.
Listed in rough order of operational relevance: the things you'll find
yourself citing in production-eval write-ups go first.

## Core methodology

- **Kapoor, S. & Narayanan, A.** *Leakage and the reproducibility crisis
  in machine-learning-based science.* Patterns 4(9), 2023.
  [arXiv:2207.07048](https://arxiv.org/abs/2207.07048).
  *The 8-type leakage taxonomy adopted by [leakage.md](leakage.md).
  294 papers across 17 fields where leakage was the cause of
  non-replication.*

- **Efron, B. & Tibshirani, R.** *An Introduction to the Bootstrap.*
  Chapman & Hall / CRC, 1993. *§14 derives BCa; the canonical reference
  for [comparison.md](comparison.md)'s bootstrap CIs.*

- **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** *On Calibration
  of Modern Neural Networks.* ICML 2017.
  [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
  *Temperature scaling — the canonical post-hoc calibration method;
  see [calibration.md](calibration.md).*

- **Naeini, M. P., Cooper, G. F., & Hauskrecht, M.** *Obtaining Well
  Calibrated Probabilities Using Bayesian Binning.* AAAI 2015.
  [arXiv:1411.0760](https://arxiv.org/abs/1411.0760).
  *ECE definition + the equal-mass-binning rationale used in
  `expected_calibration_error_equal_mass`.*

- **Kumar, A., Liang, P., & Ma, T.** *Verified Uncertainty Calibration.*
  NeurIPS 2019.
  [arXiv:1909.10155](https://arxiv.org/abs/1909.10155).
  *Debiased ECE estimators — the basis for
  `expected_calibration_error_debiased` and the L2 variant.*

## Threshold selection & decision rules

- **Lipton, Z., Elkan, C., & Naryanaswamy, B.** *Optimal thresholding of
  classifiers to maximize F1 measure.* ECML PKDD 2014.
  [arXiv:1402.1892](https://arxiv.org/abs/1402.1892).
  *Optimality proof for [`MaxF1Selector`](../api/thresholds.md).*

- **Elkan, C.** *The foundations of cost-sensitive learning.* IJCAI 2001.
  *Bayes-optimal threshold derivation used by
  [`CostSensitiveSelector`](../api/thresholds.md).*

- **Youden, W. J.** *Index for rating diagnostic tests.* Cancer 3(1),
  1950. *Original Youden's J statistic
  ([`YoudenJSelector`](../api/thresholds.md)).*

## Splits & cross-validation

- **Bates, S., Hastie, T., & Tibshirani, R.** *Cross-validation: what
  does it estimate and how well does it do it?* JASA 2024. *The CLT-
  corrected K-fold CI underlying
  [`cv_clt_ci`](../api/bootstrap.md).*

- **Hastie, T., Tibshirani, R., & Friedman, J.** *The Elements of
  Statistical Learning.* §7.10. *Cross-validation methodology canon.*

- **Yan, X. et al.** *Hidden Leaks in Time Series Forecasting.* arXiv,
  2025. [arXiv:2512.06932](https://arxiv.org/html/2512.06932v1).
  *Validation-strategy leakage in time-series settings; relevant to
  [`TimeSeriesSplitter`](../api/splits.md) and
  [`TemporalLeakageCheck`](../api/leakage.md).*

- **Pellizzoni, S. et al.** *Don't push the button! Data leakage risks
  in ML and transfer learning.* Springer AI Review, 2025.
  [DOI](https://link.springer.com/article/10.1007/s10462-025-11326-3).
  *Modern leakage taxonomy extending Kapoor & Narayanan; introduces
  transfer-learning leakage as an explicit class.*

- **Recht, B., Roelofs, R., Schmidt, L., & Shankar, V.** *Do ImageNet
  classifiers generalize to ImageNet?* ICML 2019. *Empirical CV-vs-
  final-holdout divergence — the case study for why CV alone isn't
  an OOD claim.*

## Reproducibility

- **NeurIPS Paper Checklist.**
  [neurips.cc/public/guides/PaperChecklist](https://neurips.cc/public/guides/PaperChecklist).
  *Manifest field alignment; see
  [reproducibility.md](reproducibility.md) §"NeurIPS mapping".*

- **PyTorch 2.8 reproducibility notes.**
  [docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html).
  *Canonical citation for the four sharp edges in
  [reproducibility.md](reproducibility.md) §"PyTorch determinism".*

- **Croissant: A Metadata Format for ML-Ready Datasets.** MLCommons,
  2024. [arXiv:2403.19546](https://arxiv.org/abs/2403.19546).
  *Croissant-compatible metadata in
  [`DatasetLoader.describe()`](../api/loaders.md).*

- **Pineau, J. et al.** *Improving reproducibility in machine learning
  research.* JMLR 22, 2021. *Practical guide alongside the NeurIPS
  checklist.*

## Prompt-injection eval (consumer-relevant)

- **OWASP.** *LLM01:2025 Prompt Injection.*
  [genai.owasp.org](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
  *Slice taxonomy used in
  [`prompt_injection_walkthrough.md`](../examples/prompt_injection_walkthrough.md):
  direct, indirect, encoded/obfuscated, system-prompt-leak, multi-stage.*

- **PI_HackAPrompt_SQuAD analysis (2025).**
  [arXiv:2505.04806](https://arxiv.org/html/2505.04806v1).
  *21.3 % naive-dedup detection vs 76.2 % attack-success-rate finding
  motivating
  [`NormalizedFormLeakageCheck`](../api/leakage.md).*

- **DataSentinel + PromptLocate.** [arXiv:2511.15759](https://arxiv.org/abs/2511.15759).
  *Strict-normalization contamination checks for prompt-injection
  benchmarks.*

- **Lakera PINT benchmark.** [github.com/lakeraai/pint-benchmark](https://github.com/lakeraai/pint-benchmark).
  *Canonical detection benchmark — the dataset used by the worked
  example's "full walkthrough" cross-link.*

- **Open-Prompt-Injection (Liu et al.).**
  [github.com/liu00222/Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
  *Reference dataset of attack prompts.*

## Eval harness ecosystem

- **EleutherAI lm-evaluation-harness.**
  [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
  *Source of the per-task `VERSION` field pattern adopted by
  [`Versioned`](../api/leakage.md).*

- **UK AISI Inspect AI.** [inspect.aisi.org.uk](https://inspect.aisi.org.uk/).
  *Reference architecture for safety-eval harness Scorer/Solver
  separation; cross-link from
  [`extending.md`](../extending.md).*

- **Stanford HELM.** [github.com/stanford-crfm/helm](https://github.com/stanford-crfm/helm).
  *Reference for benchmark-schema-as-versioned-artifact pattern.*

## Fairness

- **Hardt, M., Price, E., & Srebro, N.** *Equality of Opportunity in
  Supervised Learning.* NeurIPS 2016. *Equalized odds —
  [fairness.md](fairness.md) §"Equalized odds".*

- **Kleinberg, J., Mullainathan, S., & Raghavan, M.** *Inherent
  Trade-offs in the Fair Determination of Risk Scores.* ITCS 2017.
  [arXiv:1609.05807](https://arxiv.org/abs/1609.05807).
  *Incompatibility of fairness criteria.*

- **Mitchell, M. et al.** *Model Cards for Model Reporting.* FAccT 2019.
  *Documentation pattern that consumes per-subgroup metrics.*

## Statistical comparison (deferred from v0.7.0)

- **DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L.** *Comparing
  the areas under two or more correlated ROC curves: a nonparametric
  approach.* Biometrics 44, 1988.
  *Out-of-scope alternative to bootstrap CI on ROC-AUC differences;
  use [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
  + a manual implementation if required.*

- **DiCiccio, T. J. & Efron, B.** *Bootstrap confidence intervals.*
  Statistical Science 11(3), 1996. *Comparison of CI methods.*

## Future work for eval-toolkit

Items called out as out-of-scope in the v0.7.0 plan, with citations
that motivate them:

- **Inline bootstrap CI on every metric.** Inspect AI / lm-eval-harness
  pattern; appropriate for scorecard-oriented harnesses.
- **McNemar / DeLong as named functions.** Currently consumers compute
  these on top of the bootstrap framework; see
  [comparison.md §"What's NOT in eval-toolkit"](comparison.md#out-of-scope).
- **Full Croissant production by `DatasetLoader`.** Currently the
  `describe()` output is a Croissant-compatible *subset*; full
  production requires JSON-LD generation and schema validation against
  the Croissant spec.
- **Native fairness metrics.** Demographic parity, equalized odds,
  calibration parity. Pointers to `fairlearn` and `aequitas` instead;
  see [fairness.md](fairness.md).
- **Property tests for new v0.7.0 modules** — restoring the 90 %
  coverage gate. Tracked for v0.7.1.

## Cross-link

The v0.3 research audit (`docs/v0.3_research_audit.md`) is the
defensive review of every public method — literature review, gap
analysis, B+ industry rating against sklearn / scipy / statsmodels.
Read it once when you need to defend a methodological choice; the
chapters in this directory are the *prescriptive* counterpart to the
audit's *descriptive* layer.
