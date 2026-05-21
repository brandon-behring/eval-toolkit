"""Tests for the v0.46 scalar-metric deprecation shim (closes #36 v0.46 leg, Decision L).

At v0.46, 8 scalar metric symbols (`pr_auc`, `roc_auc`, `brier_score`, and 5 ECE
variants) are removed from `eval_toolkit._EXPORTS` but kept accessible at the top
level via a transitional branch in `eval_toolkit.__getattr__`. The branch emits
`DeprecationWarning` on lookup and delegates to `eval_toolkit.metrics`. At v0.47
the branch is deleted (the deprecated names raise `AttributeError` cleanly).

This test suite covers v0.46 behavior. The v0.47 hard-removal will replace these
tests with assertions that lookup raises `AttributeError`.

Per Audit F4 (Round 5): the transitional branch must NOT break the base lazy
resolver. Every existing _EXPORTS symbol must continue to resolve.
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import pytest

import eval_toolkit

DEPRECATED_SCALARS: Final[frozenset[str]] = frozenset(
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
# Deprecated names removed from `_EXPORTS`
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(DEPRECATED_SCALARS))
def test_deprecated_names_not_in_exports(name: str) -> None:
    """Deprecated names no longer appear in `_EXPORTS` (and thus not in `__all__`)."""
    assert name not in eval_toolkit._EXPORTS
    assert name not in eval_toolkit.__all__


# ─────────────────────────────────────────────────────────────────────────────
# Deprecated names still importable at top level with DeprecationWarning
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(DEPRECATED_SCALARS))
def test_deprecated_name_emits_warning(name: str) -> None:
    """Looking up a deprecated name at the top level emits DeprecationWarning.

    Updated v0.46.1 per Decision R6-G: the 3 ECE variants without first-party
    `metric_specs` equivalents point at the submodule path
    (`from eval_toolkit.metrics import ...`) rather than a scorecard snippet.
    The other 5 first-party-replaceable names use the scorecard snippet.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = getattr(eval_toolkit, name)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert (
            len(deprecations) >= 1
        ), f"expected DeprecationWarning for {name}; got {[w.category.__name__ for w in caught]}"
        msg = str(deprecations[0].message)
        # Universal assertions for ALL deprecated names:
        assert name in msg
        assert "v0.47" in msg
        # Per-name-class assertions: scorecard for first-party, submodule for the rest.
        if name in _EXPECTED_SUBMODULE_ONLY:
            assert "eval_toolkit.metrics" in msg
            assert "NOT in v0.46+ metric_specs" in msg
        else:
            assert "scorecard" in msg


@pytest.mark.unit
def test_deprecated_pr_auc_still_functional() -> None:
    """The returned function still works — only the WAY it's imported is deprecated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pr_auc = eval_toolkit.pr_auc  # type: ignore[attr-defined]
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    s = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6, 0.4])
    assert 0.0 <= pr_auc(y, s) <= 1.0


@pytest.mark.unit
def test_deprecated_brier_score_still_functional() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        brier_score = eval_toolkit.brier_score  # type: ignore[attr-defined]
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= brier_score(y, s) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Submodule path still works WITHOUT warning (internal API per Decision C / ADR 0002)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_submodule_import_no_warning() -> None:
    """`from eval_toolkit.metrics import pr_auc` does NOT warn — submodule is the
    documented internal-API escape hatch per Decision C / ADR 0002."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from eval_toolkit.metrics import brier_score, pr_auc, roc_auc  # noqa: F401

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        # Filter out unrelated deprecations (e.g., scipy / sklearn) by message
        relevant = [
            d
            for d in deprecations
            if "eval_toolkit" in str(d.message) and "scorecard" in str(d.message)
        ]
        assert relevant == [], f"unexpected eval-toolkit deprecation warnings: {relevant}"


# ─────────────────────────────────────────────────────────────────────────────
# Audit F4: lazy resolver still works for ALL non-deprecated _EXPORTS entries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_base_resolver_unbroken_for_non_deprecated_symbols() -> None:
    """The deprecation branch must not break the base lazy resolver.

    Round 5 Audit F4: the original plan said "add a shim then delete the
    whole __getattr__ block at v0.47." That would have shattered all root
    imports because __getattr__ is the lazy export resolver for every
    name in _EXPORTS. The corrected plan extends the resolver with a
    discrete BEGIN/END branch; this test guards that the base path
    survives the v0.46 edit.
    """
    # Sample a few non-deprecated names from across modules — if the resolver
    # is broken, these would AttributeError.
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
    """Every symbol in `__all__` resolves through __getattr__ without raising."""
    for name in eval_toolkit.__all__:
        getattr(eval_toolkit, name)  # raises AttributeError if resolver is broken


@pytest.mark.unit
def test_unknown_name_still_raises_attribute_error() -> None:
    """The deprecation branch must not swallow unknown-name errors."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_toolkit.nonexistent_symbol_xyz  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Constant introspection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_deprecated_scalars_set_matches() -> None:
    """The internal `_DEPRECATED_SCALARS` set lines up with this test's expectations."""
    assert eval_toolkit._DEPRECATED_SCALARS == DEPRECATED_SCALARS


# ─────────────────────────────────────────────────────────────────────────────
# v0.46.1 — Round 6 R6-F2 + R6-F + R6-G: warning snippet content & executability
# ─────────────────────────────────────────────────────────────────────────────


# First-party replacements that should appear in warning snippets verbatim.
# (factory_expression, scorecard_key) per deprecated name. Matches
# `eval_toolkit._FIRST_PARTY_REPLACEMENTS`.
_EXPECTED_FIRST_PARTY: dict[str, tuple[str, str]] = {
    "pr_auc": ("pr_auc", "pr_auc"),
    "roc_auc": ("roc_auc", "roc_auc"),
    "brier_score": ("brier", "brier"),
    "expected_calibration_error": ("ece(n_bins=10)", "ece_n_bins_10_strategy_uniform"),
    "expected_calibration_error_equal_mass": (
        'ece(n_bins=10, strategy="quantile")',
        "ece_n_bins_10_strategy_quantile",
    ),
}

# ECE variants without first-party metric_specs equivalents (Decision R6-G).
_EXPECTED_SUBMODULE_ONLY: frozenset[str] = frozenset(
    {
        "expected_calibration_error_debiased",
        "expected_calibration_error_l2",
        "expected_calibration_error_l2_debiased",
    }
)


def _capture_warning_message(name: str) -> str:
    """Trigger the deprecation shim for `name` and return the rendered message."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(eval_toolkit, name)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, f"no DeprecationWarning emitted for {name}"
        return str(deprecations[0].message)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED_FIRST_PARTY))
def test_first_party_warning_contains_correct_snippet(name: str) -> None:
    """First-party replacements emit scorecard snippet with the encoded key.

    Round 6 R6-F2: prior warnings used the factory-call expression
    (e.g. ``"ece(n_bins=10)"``) as the scorecard lookup key. The shipped
    Scorecard is a Mapping keyed by the encoded spec name
    (e.g. ``"ece_n_bins_10_strategy_uniform"``). The v0.46.1 fix uses the
    correct encoded key inline so blindly-copied snippets actually work.
    """
    factory_expr, scorecard_key = _EXPECTED_FIRST_PARTY[name]
    msg = _capture_warning_message(name)
    assert f"metric_specs.{factory_expr}" in msg
    assert f'["{scorecard_key}"]' in msg


@pytest.mark.unit
def test_ece_first_party_warnings_carry_n_bins_10_migration_note() -> None:
    """ECE first-party warnings preserve pre-v0.46 default (n_bins=10) + nudge.

    Per Decision R6-F: pre-v0.46 `expected_calibration_error` defaulted to
    n_bins=10; v0.46+ `metric_specs.ece()` defaults to n_bins=15. The
    warning snippet uses n_bins=10 for bit-identical math; an appended note
    explains the new convention.
    """
    for name in ("expected_calibration_error", "expected_calibration_error_equal_mass"):
        msg = _capture_warning_message(name)
        assert "n_bins=10" in msg
        # Migration note about the new default:
        assert "n_bins=15" in msg
        assert "Hines" in msg or "new convention" in msg


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED_SUBMODULE_ONLY))
def test_submodule_only_warning_points_at_submodule_path(name: str) -> None:
    """The 3 ECE variants without first-party specs route users to the submodule.

    Per Decision R6-G: `expected_calibration_error_debiased` / `_l2` /
    `_l2_debiased` are research-completeness primitives without
    `metric_specs` equivalents at v0.46. Their warnings cite
    `eval_toolkit.metrics.<name>` rather than a scorecard snippet.
    """
    msg = _capture_warning_message(name)
    assert f"from eval_toolkit.metrics import {name}" in msg
    assert "NOT in v0.46+ metric_specs" in msg


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED_FIRST_PARTY))
def test_first_party_warning_snippet_is_executable(name: str) -> None:
    """The scorecard snippet in the warning produces a usable MetricResult.

    Parses the snippet, executes it against a synthetic balanced slice, and
    asserts that the resulting `MetricResult` has `status="ok"` and finite
    `value`. This is the user-facing migration contract: copy the snippet,
    run it, get a number.
    """

    from eval_toolkit import metric_specs as ms
    from eval_toolkit import scorecard  # noqa: F401

    msg = _capture_warning_message(name)
    factory_expr, scorecard_key = _EXPECTED_FIRST_PARTY[name]

    # Build the snippet that the warning instructs the user to use:
    #   scorecard(y, s, metrics=[metric_specs.<factory_expr>])["<key>"].value
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)

    snippet = (
        f"scorecard(y, s, metrics=[ms.{factory_expr}], bootstrap=False)" f'["{scorecard_key}"]'
    )
    # Confirm the warning actually contains the snippet shape it promises:
    assert f"metric_specs.{factory_expr}" in msg
    # Evaluate (safe — we constructed factory_expr from the known mapping):
    cell = eval(snippet, {"scorecard": scorecard, "ms": ms, "y": y, "s": s})  # noqa: S307
    assert cell.status == "ok", f"snippet for {name}: {cell.status} (reason: {cell.reason})"
    assert cell.value is not None
    assert isinstance(cell.value, float)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED_SUBMODULE_ONLY))
def test_submodule_only_snippet_is_importable(name: str) -> None:
    """The submodule-import snippet in the warning actually imports something callable."""
    import importlib

    metrics_mod = importlib.import_module("eval_toolkit.metrics")
    assert hasattr(metrics_mod, name), (
        f"warning for {name} promises `from eval_toolkit.metrics import {name}` "
        f"but the symbol isn't present in the submodule"
    )
    assert callable(getattr(metrics_mod, name))
