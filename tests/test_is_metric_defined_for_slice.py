"""Tests for ``is_metric_defined_for_slice`` (v0.39.0, closes #39)."""

from __future__ import annotations

import pytest

from eval_toolkit import (
    SINGLE_CLASS_INCOMPATIBLE_METRICS,
    is_metric_defined_for_slice,
)


@pytest.mark.unit
def test_auroc_undefined_on_single_class() -> None:
    assert not is_metric_defined_for_slice("auroc", is_single_class=True)


@pytest.mark.unit
def test_auprc_undefined_on_single_class() -> None:
    assert not is_metric_defined_for_slice("auprc", is_single_class=True)


@pytest.mark.unit
def test_auroc_defined_on_mixed_class() -> None:
    assert is_metric_defined_for_slice("auroc", is_single_class=False)


@pytest.mark.unit
def test_auprc_defined_on_mixed_class() -> None:
    assert is_metric_defined_for_slice("auprc", is_single_class=False)


@pytest.mark.unit
@pytest.mark.parametrize("metric", ["ece", "brier", "recall", "precision"])
def test_threshold_metrics_defined_on_single_class(metric: str) -> None:
    """Calibration + count-based metrics ARE defined on single-class slices."""
    assert is_metric_defined_for_slice(metric, is_single_class=True)


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["AUROC", "AuRoc", "auroc", "AUROC "])
def test_case_insensitive_lookup(variant: str) -> None:
    """Lowercase comparison so callers can pass any casing.

    Note: trailing whitespace is NOT stripped — the primitive doesn't
    do general string normalization, only case folding.
    """
    if variant == "AUROC ":
        # Whitespace variant: caller's job to strip before passing.
        assert is_metric_defined_for_slice(variant, is_single_class=True)
    else:
        assert not is_metric_defined_for_slice(variant, is_single_class=True)


@pytest.mark.unit
def test_incompatible_metrics_override_widens() -> None:
    """Override the incompatible set to include recall_at_fpr."""
    custom = frozenset({"auroc", "auprc", "recall_at_fpr"})
    assert not is_metric_defined_for_slice(
        "recall_at_fpr", is_single_class=True, incompatible_metrics=custom
    )
    # The default would say recall_at_fpr is fine
    assert is_metric_defined_for_slice("recall_at_fpr", is_single_class=True)


@pytest.mark.unit
def test_incompatible_metrics_override_narrows() -> None:
    """Override to exclude only AUROC; AUPRC becomes 'defined'."""
    custom = frozenset({"auroc"})
    assert not is_metric_defined_for_slice(
        "auroc", is_single_class=True, incompatible_metrics=custom
    )
    assert is_metric_defined_for_slice("auprc", is_single_class=True, incompatible_metrics=custom)


@pytest.mark.unit
def test_unknown_metric_defaults_to_defined() -> None:
    """Metrics not in the incompatible set are assumed defined."""
    assert is_metric_defined_for_slice("custom_score", is_single_class=True)
    assert is_metric_defined_for_slice("custom_score", is_single_class=False)


@pytest.mark.unit
def test_constant_is_frozenset() -> None:
    """SINGLE_CLASS_INCOMPATIBLE_METRICS should be immutable."""
    assert isinstance(SINGLE_CLASS_INCOMPATIBLE_METRICS, frozenset)
    assert {"auroc", "auprc"} == SINGLE_CLASS_INCOMPATIBLE_METRICS


@pytest.mark.unit
def test_empty_incompatible_set_means_everything_defined() -> None:
    """If the override is empty, no metric is filtered."""
    assert is_metric_defined_for_slice(
        "auroc", is_single_class=True, incompatible_metrics=frozenset()
    )


@pytest.mark.unit
def test_accepts_arbitrary_collection_not_just_frozenset() -> None:
    """incompatible_metrics is typed Collection[str] — list / tuple / set all work."""
    assert not is_metric_defined_for_slice(
        "auroc", is_single_class=True, incompatible_metrics=["auroc"]
    )
    assert not is_metric_defined_for_slice(
        "auprc", is_single_class=True, incompatible_metrics=("auprc",)
    )
    assert not is_metric_defined_for_slice(
        "auroc", is_single_class=True, incompatible_metrics={"auroc"}
    )
