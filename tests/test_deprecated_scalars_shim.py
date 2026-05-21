"""Post-removal smoke tests for the v0.47 hard-removal of scalar deprecation shim.

At v0.46, 8 scalar metric symbols (``pr_auc``, ``roc_auc``, ``brier_score``, and
5 ECE variants) remained reachable at the top level via a transitional
``__getattr__`` deprecation branch (emitting ``DeprecationWarning``). At v0.47
that branch — together with the ``_DEPRECATED_SCALARS`` frozenset and the
``_deprecation_warning_for`` helper — is **hard-removed**. The deprecated names
now raise ``AttributeError`` cleanly at the top level.

This module verifies that contract:

1. Each removed name raises ``AttributeError`` when accessed via
   ``from eval_toolkit import <name>`` / ``getattr(eval_toolkit, <name>)``.
2. The submodule path (``from eval_toolkit.metrics import <name>``) — the
   documented internal-API escape hatch per Decision C / ADR 0002 —
   continues to work without warnings.
3. The base lazy resolver still resolves every entry in ``__all__`` /
   ``_EXPORTS`` (Audit F4 invariant — hard-removal must not break the base
   ``__getattr__`` path).
"""

from __future__ import annotations

import warnings
from typing import Final

import pytest

import eval_toolkit

REMOVED_SCALARS: Final[frozenset[str]] = frozenset(
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


# ─────────────────────────────────────────────────────────────────────────────
# v0.47 hard-removal: each removed name raises AttributeError at top level
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pr_auc_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``eval_toolkit.pr_auc`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.pr_auc


@pytest.mark.unit
def test_roc_auc_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``eval_toolkit.roc_auc`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.roc_auc


@pytest.mark.unit
def test_brier_score_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``eval_toolkit.brier_score`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.brier_score


@pytest.mark.unit
def test_expected_calibration_error_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``eval_toolkit.expected_calibration_error`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.expected_calibration_error


@pytest.mark.unit
def test_expected_calibration_error_debiased_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``expected_calibration_error_debiased`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.expected_calibration_error_debiased


@pytest.mark.unit
def test_expected_calibration_error_equal_mass_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``expected_calibration_error_equal_mass`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.expected_calibration_error_equal_mass


@pytest.mark.unit
def test_expected_calibration_error_l2_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``expected_calibration_error_l2`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.expected_calibration_error_l2


@pytest.mark.unit
def test_expected_calibration_error_l2_debiased_raises_attribute_error_at_top_level() -> None:
    """v0.47 hard-removal: ``expected_calibration_error_l2_debiased`` raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.expected_calibration_error_l2_debiased


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(REMOVED_SCALARS))
def test_removed_name_not_in_exports_or_all(name: str) -> None:
    """Removed names appear in neither ``_EXPORTS`` nor ``__all__``."""
    assert name not in eval_toolkit._EXPORTS
    assert name not in eval_toolkit.__all__


# ─────────────────────────────────────────────────────────────────────────────
# Submodule path (Decision C / ADR 0002) still works without warnings
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_submodule_imports_still_work_without_warning() -> None:
    """``from eval_toolkit.metrics import <name>`` still resolves for all 8 names.

    The submodule path is the documented internal-API escape hatch per
    Decision C / ADR 0002 — it never emitted a warning under v0.46's shim and
    must continue to work cleanly post hard-removal.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from eval_toolkit.metrics import (  # noqa: F401
            brier_score,
            expected_calibration_error,
            expected_calibration_error_debiased,
            expected_calibration_error_equal_mass,
            expected_calibration_error_l2,
            expected_calibration_error_l2_debiased,
            pr_auc,
            roc_auc,
        )

        relevant = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "eval_toolkit" in str(w.message)
        ]
        assert relevant == [], f"unexpected eval-toolkit deprecation warnings: {relevant}"


# ─────────────────────────────────────────────────────────────────────────────
# Audit F4: lazy resolver still works for ALL non-deprecated _EXPORTS entries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_base_resolver_unbroken_for_non_removed_symbols() -> None:
    """The hard-removal must not break the base lazy resolver.

    Round 5 Audit F4: ``__getattr__`` is the lazy export resolver for every
    name in ``_EXPORTS``. The v0.47 cleanup removes ONLY the transitional
    BEGIN/END deprecation block; the base resolver below it must continue
    to resolve every surviving public symbol.
    """
    from eval_toolkit import (
        BootstrapCI,
        LogisticStacker,
        MaxF1Selector,
        MetricSpec,
        Scorecard,
        bootstrap_ci,
        metrics_at_threshold,
        scorecard,
    )

    assert all(
        x is not None
        for x in (
            BootstrapCI,
            LogisticStacker,
            MaxF1Selector,
            MetricSpec,
            Scorecard,
            bootstrap_ci,
            metrics_at_threshold,
            scorecard,
        )
    )


@pytest.mark.unit
def test_full_all_resolves_without_attribute_error() -> None:
    """Every symbol in ``__all__`` resolves through ``__getattr__`` without raising."""
    for name in eval_toolkit.__all__:
        getattr(eval_toolkit, name)  # raises AttributeError if resolver is broken


@pytest.mark.unit
def test_unknown_name_still_raises_attribute_error() -> None:
    """Unknown names still raise ``AttributeError`` (and not, e.g., a stale ``DeprecationWarning``)."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.nonexistent_symbol_xyz


@pytest.mark.unit
def test_deprecation_shim_internals_are_gone() -> None:
    """The internal v0.46 shim attributes are removed (``_DEPRECATED_SCALARS``, etc.).

    Guards against partial hard-removal where the branch is deleted but a
    stale module-level constant lingers as cruft.
    """
    assert not hasattr(eval_toolkit, "_DEPRECATED_SCALARS")
    assert not hasattr(eval_toolkit, "_FIRST_PARTY_REPLACEMENTS")
    assert not hasattr(eval_toolkit, "_deprecation_warning_for")
