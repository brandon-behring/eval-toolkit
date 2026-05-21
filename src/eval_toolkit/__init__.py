"""eval-toolkit — reusable evaluation contracts for binary classification.

Public API remains available from ``eval_toolkit`` and from submodules:

    from eval_toolkit import pr_auc, bootstrap_ci, BootstrapCI
    from eval_toolkit.metrics import pr_auc

The package root uses lazy exports so importing ``eval_toolkit`` does not
eagerly import optional-heavy modules such as plotting, loaders, or harnesses.
"""

from __future__ import annotations

import logging as _logging
from importlib import import_module
from typing import Any

from eval_toolkit._version import __version__

# Library-friendly logging discipline (per v0.29.0 plan Q4=A): attach a
# NullHandler to the package root so eval_toolkit is silent by default.
# Consumers configure handlers explicitly; per-module loggers
# (`logging.getLogger(__name__)`) inherit through this NullHandler.
# Matches numpy / scikit-learn / requests convention.
_logging.getLogger("eval_toolkit").addHandler(_logging.NullHandler())

# Entries are grouped by source module (matches the value column). Keys
# within each group keep their existing order — roughly constants → types
# → functions — for IDE outline + import-grep ergonomics. The section
# dividers below are informational only; the snapshot in
# tests/golden/public_api/ reads dict keys + values, not comments.
_EXPORTS: dict[str, str] = {
    # --- adversarial ---
    "CORE_TECHNIQUES": "eval_toolkit.adversarial",
    "CaseRandomization": "eval_toolkit.adversarial",
    "CharacterInjectionStrategy": "eval_toolkit.adversarial",
    "DiacriticInjection": "eval_toolkit.adversarial",
    "HomoglyphSubstitution": "eval_toolkit.adversarial",
    "PunctuationInjection": "eval_toolkit.adversarial",
    "WhitespaceInjection": "eval_toolkit.adversarial",
    "ZeroWidthSpaceInjection": "eval_toolkit.adversarial",
    "character_injection": "eval_toolkit.adversarial",
    # --- losses ---
    "RecallAtLowFPR": "eval_toolkit.losses",
    # --- preprocessing ---
    "DatamarkVariant": "eval_toolkit.preprocessing",
    "DelimitVariant": "eval_toolkit.preprocessing",
    "EncodeVariant": "eval_toolkit.preprocessing",
    "datamark": "eval_toolkit.preprocessing",
    "delimit": "eval_toolkit.preprocessing",
    "encode": "eval_toolkit.preprocessing",
    "spotlighting": "eval_toolkit.preprocessing",
    # --- probes ---
    "ActivationDeltaProbe": "eval_toolkit.probes",
    "ActivationExtractor": "eval_toolkit.probes",
    "Probe": "eval_toolkit.probes",
    # --- analysis ---
    "CsvPredictionReader": "eval_toolkit.analysis",
    "JsonlPredictionReader": "eval_toolkit.analysis",
    "PredictionArrays": "eval_toolkit.analysis",
    "bootstrap_metric_from_predictions": "eval_toolkit.analysis",
    "load_prediction_arrays": "eval_toolkit.analysis",
    "paired_diff_from_prediction_refs": "eval_toolkit.analysis",
    # --- artifacts ---
    "MetricState": "eval_toolkit.artifacts",
    "PredictionArtifactRef": "eval_toolkit.artifacts",
    "PredictionColumns": "eval_toolkit.artifacts",
    "error_metric": "eval_toolkit.artifacts",
    "sanitize_for_json": "eval_toolkit.artifacts",
    "skipped_metric": "eval_toolkit.artifacts",
    "validate_manifest": "eval_toolkit.artifacts",
    "validate_payload": "eval_toolkit.artifacts",
    "validate_prediction_artifact_ref": "eval_toolkit.artifacts",
    "validate_results": "eval_toolkit.artifacts",
    "write_json_strict": "eval_toolkit.artifacts",
    # --- bootstrap ---
    "DEFAULT_CONFIDENCE": "eval_toolkit.bootstrap",
    "DEFAULT_METHOD": "eval_toolkit.bootstrap",
    "DEFAULT_N_RESAMPLES": "eval_toolkit.bootstrap",
    "DEFAULT_SEED": "eval_toolkit.bootstrap",
    "BootstrapCI": "eval_toolkit.bootstrap",
    "CorrectionMethod": "eval_toolkit.bootstrap",
    "DeLongResult": "eval_toolkit.bootstrap",
    "MDEEstimate": "eval_toolkit.bootstrap",
    "MetricFn": "eval_toolkit.bootstrap",
    "PairedBootstrapCI": "eval_toolkit.bootstrap",
    "ThresholdedMetricFn": "eval_toolkit.bootstrap",
    "ThresholdFn": "eval_toolkit.bootstrap",
    "block_bootstrap_on_folds": "eval_toolkit.bootstrap",
    "bonferroni_correct": "eval_toolkit.bootstrap",
    "bootstrap_ci": "eval_toolkit.bootstrap",
    "correct_p_values": "eval_toolkit.bootstrap",
    "cross_validate_metric": "eval_toolkit.bootstrap",
    "cv_clt_ci": "eval_toolkit.bootstrap",
    "delong_roc_variance": "eval_toolkit.bootstrap",
    "fdr_bh_correct": "eval_toolkit.bootstrap",
    "mde_from_ci": "eval_toolkit.bootstrap",
    "paired_bootstrap_diff": "eval_toolkit.bootstrap",
    "paired_bootstrap_ece_diff": "eval_toolkit.bootstrap",
    "paired_bootstrap_op_point_diff": "eval_toolkit.bootstrap",
    "paired_mde": "eval_toolkit.bootstrap",
    # --- calibration ---
    "DEFAULT_FN_COST": "eval_toolkit.calibration",
    "DEFAULT_FP_COST": "eval_toolkit.calibration",
    "DEFAULT_N_BINS": "eval_toolkit.calibration",
    "DEFAULT_PRIOR": "eval_toolkit.calibration",
    "DEFAULT_STRATEGY": "eval_toolkit.calibration",
    "CostMatrix": "eval_toolkit.calibration",
    "bayes_optimal_threshold": "eval_toolkit.calibration",
    "fit_beta_binary": "eval_toolkit.calibration",
    "fit_beta_calibrator": "eval_toolkit.calibration",
    "fit_isotonic_binary": "eval_toolkit.calibration",
    "fit_isotonic_calibrator": "eval_toolkit.calibration",
    "fit_platt_binary": "eval_toolkit.calibration",
    "fit_platt_calibrator": "eval_toolkit.calibration",
    "fit_temperature": "eval_toolkit.calibration",
    "fit_temperature_binary": "eval_toolkit.calibration",
    "fit_temperature_oracle": "eval_toolkit.calibration",
    "reliability_curve": "eval_toolkit.calibration",
    "reliability_diagram_data": "eval_toolkit.calibration",
    # --- claims ---
    "ClaimReport": "eval_toolkit.claims",
    "ClaimSpec": "eval_toolkit.claims",
    "EvidenceGate": "eval_toolkit.claims",
    "GateResult": "eval_toolkit.claims",
    "evaluate_claims": "eval_toolkit.claims",
    "external_diagnostic_gate": "eval_toolkit.claims",
    "headline_present_gate": "eval_toolkit.claims",
    "low_fpr_feasibility_gate": "eval_toolkit.claims",
    "metric_threshold_gate": "eval_toolkit.claims",
    "minimum_slice_size_gate": "eval_toolkit.claims",
    "no_leakage_errors_gate": "eval_toolkit.claims",
    "no_scorer_errors_gate": "eval_toolkit.claims",
    "paired_diff_present_gate": "eval_toolkit.claims",
    "required_metric_gate": "eval_toolkit.claims",
    "required_scorer_gate": "eval_toolkit.claims",
    "required_slice_gate": "eval_toolkit.claims",
    "source_role_gate": "eval_toolkit.claims",
    "strict_artifact_gate": "eval_toolkit.claims",
    # --- config ---
    "from_yaml": "eval_toolkit.config",
    "frozen_config": "eval_toolkit.config",
    # --- docs ---
    "ANCHOR_RE": "eval_toolkit.docs",
    "DEFAULT_FORMATTERS": "eval_toolkit.docs",
    "render_files": "eval_toolkit.docs",
    "render_text": "eval_toolkit.docs",
    "walk_path": "eval_toolkit.docs",
    # --- embeddings ---
    "make_minilm_embedder": "eval_toolkit.embeddings",
    # --- evidence ---
    "AggregateEvidence": "eval_toolkit.evidence",
    "EvidenceAxis": "eval_toolkit.evidence",
    "PairingMetadata": "eval_toolkit.evidence",
    "RECOMMENDED_SOURCE_ROLES": "eval_toolkit.evidence",
    # --- harness ---
    "DEFAULT_BOOTSTRAP_RESAMPLES": "eval_toolkit.harness",
    "RUN_RESULT_SCHEMA_VERSION": "eval_toolkit.harness",
    "EvalSlice": "eval_toolkit.harness",
    "RunResult": "eval_toolkit.harness",
    "evaluate": "eval_toolkit.harness",
    "evaluate_folded": "eval_toolkit.harness",
    "evaluate_scorer_on_slice": "eval_toolkit.harness",
    "with_claim_report": "eval_toolkit.harness",
    "write_run_result": "eval_toolkit.harness",
    # --- leakage ---
    "CrossSplitLeakageCheck": "eval_toolkit.leakage",
    "ExactDuplicateCheck": "eval_toolkit.leakage",
    "GroupLeakageCheck": "eval_toolkit.leakage",
    "LabelConflictCheck": "eval_toolkit.leakage",
    "LeakageCheck": "eval_toolkit.leakage",
    "LeakageFinding": "eval_toolkit.leakage",
    "LeakageReport": "eval_toolkit.leakage",
    "NearDuplicateCheck": "eval_toolkit.leakage",
    "NormalizedFormLeakageCheck": "eval_toolkit.leakage",
    "TemporalLeakageCheck": "eval_toolkit.leakage",
    "TokenizationLeakageCheck": "eval_toolkit.leakage",
    "run_leakage_checks": "eval_toolkit.leakage",
    # --- loaders ---
    "DataFrameLoader": "eval_toolkit.loaders",
    "DatasetLoader": "eval_toolkit.loaders",
    "HFDatasetsLoader": "eval_toolkit.loaders",
    "OodManifestLoader": "eval_toolkit.loaders",
    "ParquetGlobLoader": "eval_toolkit.loaders",
    "SingleSliceLoader": "eval_toolkit.loaders",
    "ood_dataset_from_manifest": "eval_toolkit.loaders",
    # --- manifest ---
    "MANIFEST_SCHEMA_VERSION": "eval_toolkit.manifest",
    "RunManifest": "eval_toolkit.manifest",
    "SourceRoleRecord": "eval_toolkit.manifest",
    "build_manifest": "eval_toolkit.manifest",
    "validate_source_roles": "eval_toolkit.manifest",
    "write_manifest": "eval_toolkit.manifest",
    # --- metrics ---
    "DEFAULT_ASSUMED_PRIORS": "eval_toolkit.metrics",
    "SINGLE_CLASS_INCOMPATIBLE_METRICS": "eval_toolkit.metrics",
    "ThresholdResult": "eval_toolkit.metrics",
    "brier_decomposition": "eval_toolkit.metrics",
    # `brier_score`, `pr_auc`, `roc_auc`, and the 5 ECE variants removed from
    # `_EXPORTS` at v0.46 (Decision L). They remain reachable at the top
    # level via the `__getattr__` deprecation branch (emits
    # `DeprecationWarning`; branch removed at v0.47) and via the metrics
    # submodule (`from eval_toolkit.metrics import pr_auc` — internal API
    # per ADR 0002, not part of the v1.0 stability contract).
    "headline_metrics": "eval_toolkit.metrics",
    "is_metric_defined_for_slice": "eval_toolkit.metrics",
    "metrics_at_threshold": "eval_toolkit.metrics",
    "precision_at_prior": "eval_toolkit.metrics",
    "quantile_stratified_pr_auc": "eval_toolkit.metrics",
    "quantile_stratified_report": "eval_toolkit.metrics",
    "score_distribution_summary": "eval_toolkit.metrics",
    "single_class_threshold_metrics": "eval_toolkit.metrics",
    "stratified_recall": "eval_toolkit.metrics",
    # --- operating_points ---
    "FittedOperatingPoint": "eval_toolkit.operating_points",
    "OperatingPointSpec": "eval_toolkit.operating_points",
    "apply_operating_points": "eval_toolkit.operating_points",
    "fit_operating_points": "eval_toolkit.operating_points",
    # --- paths ---
    "path_for_config": "eval_toolkit.paths",
    "resolve_repo_path": "eval_toolkit.paths",
    "split_provenance_config": "eval_toolkit.paths",
    # --- plotting ---
    "DEFAULT_FIGSIZE": "eval_toolkit.plotting",
    "PALETTE": "eval_toolkit.plotting",
    "PLOT_STYLE": "eval_toolkit.plotting",
    "make_palette": "eval_toolkit.plotting",
    "plot_bootstrap_distribution": "eval_toolkit.plotting",
    "plot_confusion_matrix_grid": "eval_toolkit.plotting",
    "plot_lift_ci": "eval_toolkit.plotting",
    "plot_metric_bars": "eval_toolkit.plotting",
    "plot_pareto_frontier": "eval_toolkit.plotting",
    "plot_pr_curve": "eval_toolkit.plotting",
    "plot_reliability_diagram": "eval_toolkit.plotting",
    "plot_roc_curve": "eval_toolkit.plotting",
    "plot_score_histograms": "eval_toolkit.plotting",
    "plot_slice_metric_heatmap": "eval_toolkit.plotting",
    "save_figure": "eval_toolkit.plotting",
    "set_plot_style": "eval_toolkit.plotting",
    # --- provenance ---
    "FileHash": "eval_toolkit.provenance",
    "FileHashMissing": "eval_toolkit.provenance",
    "capture_git_sha": "eval_toolkit.provenance",
    "compute_file_hash": "eval_toolkit.provenance",
    "figure_metadata": "eval_toolkit.provenance",
    "file_sha256": "eval_toolkit.provenance",
    "make_run_dir": "eval_toolkit.provenance",
    # --- protocols ---
    "EvalSliceLike": "eval_toolkit.protocols",
    "PredictionReader": "eval_toolkit.protocols",
    "Scorer": "eval_toolkit.protocols",
    "SliceAwareScorer": "eval_toolkit.protocols",
    "TextTransform": "eval_toolkit.protocols",
    "Versioned": "eval_toolkit.protocols",
    # --- seeds ---
    "set_global_seeds": "eval_toolkit.seeds",
    # --- splits ---
    "GroupKFoldSplitter": "eval_toolkit.splits",
    "HoldoutSplitter": "eval_toolkit.splits",
    "PoolBuilder": "eval_toolkit.splits",
    "PurgedKFoldSplitter": "eval_toolkit.splits",
    "SourceDisjointKFoldSplitter": "eval_toolkit.splits",
    "Splitter": "eval_toolkit.splits",
    "StratifiedKFoldSplitter": "eval_toolkit.splits",
    "TimeSeriesSplitter": "eval_toolkit.splits",
    "compute_label_overlap": "eval_toolkit.splits",
    "iter_folds_with_pool": "eval_toolkit.splits",
    # --- text_dedup ---
    "DEFAULT_DEDUP_THRESHOLD": "eval_toolkit.text_dedup",
    "DedupReport": "eval_toolkit.text_dedup",
    "EmbeddingCosineStrategy": "eval_toolkit.text_dedup",
    "ExactNormalizedHashStrategy": "eval_toolkit.text_dedup",
    "JaccardNgramStrategy": "eval_toolkit.text_dedup",
    "MinHashLSHStrategy": "eval_toolkit.text_dedup",
    "SimilarityAuditFinding": "eval_toolkit.text_dedup",
    "SimilarityAuditReport": "eval_toolkit.text_dedup",
    "SimilarityStrategy": "eval_toolkit.text_dedup",
    "TfidfCosineStrategy": "eval_toolkit.text_dedup",
    "audit_source_label_similarity": "eval_toolkit.text_dedup",
    "cross_dedup": "eval_toolkit.text_dedup",
    "near_dedup": "eval_toolkit.text_dedup",
    "normalize_text_for_dedup": "eval_toolkit.text_dedup",
    "sha256_text": "eval_toolkit.text_dedup",
    # --- thresholds ---
    "CISafeThresholdSelector": "eval_toolkit.thresholds",
    "CostSensitiveSelector": "eval_toolkit.thresholds",
    "MaxF1Selector": "eval_toolkit.thresholds",
    "RecallAtFprResult": "eval_toolkit.thresholds",
    "TargetFPRSelector": "eval_toolkit.thresholds",
    "TargetPrecisionSelector": "eval_toolkit.thresholds",
    "TargetRecallSelector": "eval_toolkit.thresholds",
    "ThresholdPolicyMetadata": "eval_toolkit.thresholds",
    "ThresholdSelector": "eval_toolkit.thresholds",
    "WilsonInterval": "eval_toolkit.thresholds",
    "YoudenJSelector": "eval_toolkit.thresholds",
    "recall_at_fpr": "eval_toolkit.thresholds",
    "select_threshold": "eval_toolkit.thresholds",
    "wilson_interval": "eval_toolkit.thresholds",
    "LogisticStacker": "eval_toolkit.stacking",
    "MetaLearner": "eval_toolkit.stacking",
    "MetricResult": "eval_toolkit._scorecard",
    "MetricSpec": "eval_toolkit._scorecard",
    "Scorecard": "eval_toolkit._scorecard",
    "scorecard": "eval_toolkit._scorecard",
}

__all__ = ["__version__", *_EXPORTS.keys()]


# ── BEGIN TRANSITIONAL DEPRECATION BRANCH (Decision L; REMOVE AT v0.47) ──
# At v0.46 the scalar metric functions left the top-level `_EXPORTS` map (above)
# in favor of the `scorecard()` surface (Decision A). To give the consumer one
# release of overlap before the hard removal at v0.47, the names below remain
# reachable via the package-level `__getattr__` (which delegates to the
# `eval_toolkit.metrics` submodule) but emit a `DeprecationWarning` on first
# lookup pointing at the new API.
#
# WHY THIS IS A BRANCH, NOT A REPLACEMENT (Audit F4 — Round 5):
# `__getattr__` below is the load-bearing lazy export resolver for every name
# in `_EXPORTS`. The deprecation branch is a discrete `if name in
# _DEPRECATED_SCALARS` check ABOVE the resolver — the resolver's existing
# behavior for non-deprecated names is unchanged. At v0.47 we delete this
# transitional block and the resolver continues to work for every remaining
# `_EXPORTS` entry.
_DEPRECATED_SCALARS: frozenset[str] = frozenset(
    {
        "pr_auc",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "expected_calibration_error_debiased",
        "expected_calibration_error_equal_mass",
        "expected_calibration_error_l2",
        "expected_calibration_error_l2_debiased",
    }
)
# ── END TRANSITIONAL DEPRECATION (Decision L; REMOVE AT v0.47) ──


def __getattr__(name: str) -> Any:
    """Resolve public symbols lazily."""
    if name == "__version__":
        return __version__
    # ── BEGIN TRANSITIONAL DEPRECATION BRANCH (Decision L; REMOVE AT v0.47) ──
    if name in _DEPRECATED_SCALARS:
        import warnings

        warnings.warn(
            _deprecation_warning_for(name),
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module("eval_toolkit.metrics")
        value = getattr(module, name)
        # Do NOT cache in globals() — repeated lookups should keep re-warning
        # (one warning per call site, modulo Python's default
        # DeprecationWarning de-duplication).
        return value
    # ── END TRANSITIONAL DEPRECATION (Decision L; REMOVE AT v0.47) ──
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'eval_toolkit' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


# ── BEGIN TRANSITIONAL DEPRECATION HELPER (Decision L; REMOVE AT v0.47) ──
#
# Per Round 6 audit (Codex R6-F2 + Gemini R6-F2; Decisions R6-F + R6-G):
# - For deprecated scalars with a first-party `metric_specs` equivalent, the
#   warning emits an EXECUTABLE scorecard snippet (factory expression + the
#   correct encoded scorecard key, not the factory call string).
# - For the 3 ECE variants without a `metric_specs` equivalent
#   (expected_calibration_error_debiased / _l2 / _l2_debiased), the warning
#   instead points at the submodule path per Decision R6-G — no first-party
#   replacement is shipped at v0.47.
# - ECE `n_bins=10` preserves the pre-v0.46 default (verified at
#   `metrics.py:730-734`) — Decision R6-F. A migration note explains that
#   the v0.46+ `metric_specs.ece()` factory defaults to `n_bins=15` (matching
#   Hines et al.) and how to opt in.
_FirstParty = tuple[str, str]  # (factory_expression, scorecard_key)
"""Type alias for a deprecated-scalar that has a metric_specs replacement.

The factory expression is what the user types after ``metric_specs.``; the
scorecard key is the literal string that indexes ``Scorecard``.
"""


_FIRST_PARTY_REPLACEMENTS: dict[str, _FirstParty] = {
    "pr_auc": ("pr_auc", "pr_auc"),
    "roc_auc": ("roc_auc", "roc_auc"),
    "brier_score": ("brier", "brier"),
    # ECE variants: use n_bins=10 (pre-v0.46 default per Decision R6-F).
    # The migration note in the warning text explains how to switch to
    # n_bins=15 if the user wants the v0.46+ metric_specs.ece() default.
    "expected_calibration_error": (
        "ece(n_bins=10)",
        "ece_n_bins_10_strategy_uniform",
    ),
    "expected_calibration_error_equal_mass": (
        'ece(n_bins=10, strategy="quantile")',
        "ece_n_bins_10_strategy_quantile",
    ),
}
"""Names that have a first-party metric_specs replacement at v0.46.

The 3 ECE variants NOT in this map (_debiased, _l2, _l2_debiased) get the
submodule-path warning template instead (Decision R6-G).
"""


def _deprecation_warning_for(name: str) -> str:
    """Render the DeprecationWarning message for a deprecated scalar name.

    Branches on whether ``name`` has a first-party `metric_specs` replacement
    (Decision R6-G):

    - First-party (5 names): scorecard snippet with the correct encoded key
      (Decision R6-F).
    - Submodule-only (3 ECE variants): point at the submodule path per
      Decision R6-G.

    The first-party variants for ECE include a migration note explaining the
    new ``metric_specs.ece()`` factory default of ``n_bins=15`` so users can
    opt in to the new convention; the snippet itself uses ``n_bins=10`` for
    bit-identical pre-v0.46 math (Decision R6-F).

    Parameters
    ----------
    name : str
        A name in ``_DEPRECATED_SCALARS``.

    Returns
    -------
    str
        The warning message, ready to pass to ``warnings.warn``.
    """
    first_party = _FIRST_PARTY_REPLACEMENTS.get(name)
    if first_party is not None:
        factory_expr, scorecard_key = first_party
        msg = (
            f"eval_toolkit.{name} is deprecated and will be removed in v0.47. "
            f"For the same math, use:\n"
            f"    scorecard(y, s, metrics=[metric_specs.{factory_expr}])"
            f'["{scorecard_key}"].value\n'
            f"Or import from the eval_toolkit.metrics submodule directly "
            f"(internal API per ADR 0002 — stable across v1.x, subject to "
            f"refactor in major versions)."
        )
        # ECE-specific migration note about the n_bins default change.
        if name.startswith("expected_calibration_error"):
            msg += (
                "\nNote: the v0.46+ metric_specs.ece() factory defaults to "
                "n_bins=15 (matching Hines et al.); the n_bins=10 in this "
                "snippet preserves the pre-v0.46 math. Pass n_bins=15 to use "
                "the new convention."
            )
        return msg
    # Decision R6-G: 3 ECE variants without first-party replacements →
    # submodule path only.
    return (
        f"eval_toolkit.{name} is deprecated and will be removed in v0.47. "
        f"This variant is NOT in v0.46+ metric_specs. Use:\n"
        f"    from eval_toolkit.metrics import {name}\n"
        f"(internal API per ADR 0002 — stable across v1.x, subject to "
        f"refactor in major versions). Or contribute the variant to "
        f"metric_specs if you use it regularly."
    )


# ── END TRANSITIONAL DEPRECATION HELPER (Decision L; REMOVE AT v0.47) ──


def __dir__() -> list[str]:
    """Expose lazy public symbols to introspection."""
    return sorted(__all__)
