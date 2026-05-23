"""Canonical ImportError message contract for every lazy-extras surface (§5C, v0.48).

Style invariant 1 (no silent failures) + 3 (API-level errors with context) demand
that every ImportError raised by ``src/eval_toolkit/**.py`` when an optional
dependency is missing follows the canonical template:

    "<feature> requires <pkg>. Install with: pip install eval-toolkit[<extra>]"

When the dependency has no corresponding ``[<extra>]`` (e.g., the ``datasets``
package, which we depend on directly), the trailing install hint drops the
``[<extra>]`` segment:

    "<feature> requires <pkg>. Install with: pip install <pkg>"

Each test below pins the EXACT canonical string for the surface it covers.
Migration drift (someone retypes one of these messages and "Install with:"
becomes "install via:") will trip the assertion and force the doc-honest fix.

The 8 surfaces audited (one assertion per surface):

  1. ``embeddings.make_minilm_embedder``   ([embeddings] extra)
  2. ``config.from_yaml``                   ([yaml] extra)
  3. ``_sweep.sweep``                       ([dataframe] extra)
  4. ``losses.RecallAtLowFPR``              ([losses] extra)
  5. ``loaders.HFDatasetsLoader``           (no extra — direct ``datasets`` install)
  6. ``loaders.ood_dataset_from_manifest``  ([yaml] extra)
  7. ``probes.ActivationDeltaProbe``        ([probes] extra)
  8. ``scorecards.Scorecard.to_pandas``     ([dataframe] extra)
"""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# Canonical message bank — single source of truth for the §5C audit.
#
# When a future contributor adds a new lazy-extras surface, the canonical
# string goes here AND the new surface gets its own test below. The bank
# being authoritative is what gives us a single grep-target for the
# canonical wording across the codebase.
# ---------------------------------------------------------------------------

CANONICAL_MESSAGES: dict[str, str] = {
    "make_minilm_embedder": (
        "make_minilm_embedder requires sentence-transformers. "
        "Install with: pip install eval-toolkit[embeddings]"
    ),
    "from_yaml": ("from_yaml requires pyyaml. Install with: pip install eval-toolkit[yaml]"),
    "sweep": ("sweep() requires pandas. Install with: pip install eval-toolkit[dataframe]"),
    "RecallAtLowFPR": (
        "RecallAtLowFPR requires torch. Install with: pip install eval-toolkit[losses]"
    ),
    "HFDatasetsLoader": ("HFDatasetsLoader requires datasets. Install with: pip install datasets"),
    "ood_dataset_from_manifest": (
        "ood_dataset_from_manifest requires pyyaml. " "Install with: pip install eval-toolkit[yaml]"
    ),
    "ActivationDeltaProbe": (
        "ActivationDeltaProbe requires torch + transformers. "
        "Install with: pip install eval-toolkit[probes]"
    ),
    "Scorecard.to_pandas": (
        "Scorecard.to_pandas() requires pandas. "
        "Install with: pip install eval-toolkit[dataframe]"
    ),
}


def _exact_match(expected: str) -> str:
    """Compile expected ImportError message into an exact pytest.raises ``match`` regex."""
    import re

    return f"^{re.escape(expected)}$"


# ---------------------------------------------------------------------------
# Surface 1: embeddings.make_minilm_embedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_make_minilm_embedder_canonical_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When sentence-transformers is missing, raise the exact canonical message."""
    from eval_toolkit.embeddings import make_minilm_embedder

    make_minilm_embedder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    expected = CANONICAL_MESSAGES["make_minilm_embedder"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        make_minilm_embedder(model_id="any/model")
    make_minilm_embedder.cache_clear()


# ---------------------------------------------------------------------------
# Surface 2: config.from_yaml
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_yaml_canonical_import_error(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pyyaml is missing, ``from_yaml`` raises the exact canonical message."""
    from dataclasses import dataclass

    from eval_toolkit.config import from_yaml

    @dataclass
    class _Cfg:
        x: int = 1

    # Path doesn't need to exist — the ImportError is raised BEFORE the path is read.
    monkeypatch.setitem(sys.modules, "yaml", None)
    expected = CANONICAL_MESSAGES["from_yaml"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        from_yaml(_Cfg, str(tmp_path / "missing.yaml"))


# ---------------------------------------------------------------------------
# Surface 3: _sweep.sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sweep_canonical_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pandas is missing, ``sweep()`` raises the exact canonical message.

    ``sweep()`` requires pandas (lazy-imported inside the function body),
    so we must drop it from ``sys.modules`` to drive the failure branch.
    """
    from eval_toolkit import sweep

    monkeypatch.setitem(sys.modules, "pandas", None)
    expected = CANONICAL_MESSAGES["sweep"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        sweep([], [])


# ---------------------------------------------------------------------------
# Surface 4: losses.RecallAtLowFPR
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recall_at_low_fpr_canonical_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When torch is missing, ``RecallAtLowFPR()`` raises the exact canonical message."""
    from eval_toolkit import losses
    from eval_toolkit.losses import RecallAtLowFPR

    losses._CLASS_CACHE.clear()
    monkeypatch.setitem(sys.modules, "torch", None)
    expected = CANONICAL_MESSAGES["RecallAtLowFPR"]
    try:
        with pytest.raises(ImportError, match=_exact_match(expected)):
            RecallAtLowFPR()
    finally:
        losses._CLASS_CACHE.clear()


# ---------------------------------------------------------------------------
# Surface 5: loaders.HFDatasetsLoader (no [extra] — direct `pip install datasets`)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hf_datasets_loader_canonical_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``datasets`` is missing, ``HFDatasetsLoader.load_splits`` raises the canonical message.

    Note: ``datasets`` is NOT advertised as an eval-toolkit extra; the install
    hint refers to the package directly. The canonical-message contract still
    applies (same template, ``[<extra>]`` segment dropped).
    """
    from eval_toolkit.loaders import HFDatasetsLoader

    monkeypatch.setitem(sys.modules, "datasets", None)
    loader = HFDatasetsLoader(repo_id="any/repo", feature_col="text", label_col="label")
    expected = CANONICAL_MESSAGES["HFDatasetsLoader"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        loader._load_dataset()


# ---------------------------------------------------------------------------
# Surface 6: loaders.ood_dataset_from_manifest
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ood_dataset_from_manifest_canonical_import_error(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pyyaml is missing, ``_load_yaml_manifest`` raises the exact canonical message.

    Drives the manifest loader (which calls ``_load_yaml_manifest`` internally).
    """
    from eval_toolkit.loaders import _load_yaml_manifest

    monkeypatch.setitem(sys.modules, "yaml", None)
    expected = CANONICAL_MESSAGES["ood_dataset_from_manifest"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        _load_yaml_manifest(str(tmp_path / "missing.yaml"))


# ---------------------------------------------------------------------------
# Surface 7: probes.ActivationDeltaProbe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_activation_delta_probe_canonical_import_error(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When torch/transformers are missing, default-extractor construction raises canonical."""
    from eval_toolkit import ActivationDeltaProbe

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    probe = ActivationDeltaProbe(backbone="any/model", cache_dir=str(tmp_path))
    expected = CANONICAL_MESSAGES["ActivationDeltaProbe"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        probe.fit(["a"], ["b"])


# ---------------------------------------------------------------------------
# Surface 8: scorecards.Scorecard.to_pandas
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scorecard_to_pandas_canonical_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pandas is missing, ``Scorecard.to_pandas()`` raises the exact canonical message."""
    import numpy as np

    from eval_toolkit import metric_specs as ms
    from eval_toolkit import scorecard

    # Build a real Scorecard first (BEFORE patching pandas out, since the
    # scorecard() call itself doesn't need pandas — only .to_pandas() does).
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    s = rng.random(50)
    card = scorecard(y, s, metrics=[ms.pr_auc], bootstrap=False)

    monkeypatch.setitem(sys.modules, "pandas", None)
    expected = CANONICAL_MESSAGES["Scorecard.to_pandas"]
    with pytest.raises(ImportError, match=_exact_match(expected)):
        card.to_pandas()


# ---------------------------------------------------------------------------
# Template consistency: spot-check every canonical message matches the schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_canonical_messages_follow_template() -> None:
    """All canonical messages start with ``<feature> requires <pkg>. Install with:``.

    Catches future drift (e.g., a contributor adds a new surface and writes
    "install via:" or skips the period). The message bank is the single
    source of truth for the §5C audit.
    """
    for feature, msg in CANONICAL_MESSAGES.items():
        # Strip whitespace from line continuations.
        flat = " ".join(msg.split())
        assert " requires " in flat, f"{feature}: missing ' requires '"
        assert ". Install with: " in flat, f"{feature}: missing '. Install with: '"
        assert flat.startswith(
            feature.replace("Scorecard.to_pandas", "Scorecard.to_pandas()")
        ) or flat.startswith(feature), f"{feature}: message must lead with the feature name"
        assert "pip install" in flat, f"{feature}: missing 'pip install'"
