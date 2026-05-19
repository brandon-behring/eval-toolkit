"""Tests for ``eval_toolkit.probes`` (v0.43.0; closes #53).

Uses a deterministic mock :class:`ActivationExtractor` so the suite runs
without torch / transformers / a downloaded HF backbone. Real-backbone
verification lives in the integration suite (skipped when ``[probes]``
extra isn't installed; not exercised in this PR).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.probes import (
    ActivationDeltaProbe,
    ActivationExtractor,
    Probe,
)

# ----------------------------------------------------------------------------
# Mock extractor — deterministic, no torch / transformers
# ----------------------------------------------------------------------------


@dataclass
class _MockExtractor:
    """Deterministic test extractor: hash text → fixed-length pseudo-activation.

    Implements :class:`~eval_toolkit.probes.ActivationExtractor` without
    needing torch. Same text always returns the same vector. Different
    seeds (per-instance) produce different vectors.
    """

    hidden_size: int = 8
    backbone_id: str = "mock/test"
    seed: int = 0
    call_count: int = field(default=0, init=False)

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        self.call_count += 1
        out = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        for i, t in enumerate(texts):
            # Deterministic per-text vector: seed RNG with (instance seed + text hash).
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            text_seed = int.from_bytes(digest[:4], "big") ^ self.seed
            rng = np.random.default_rng(text_seed)
            out[i] = rng.standard_normal(self.hidden_size).astype(np.float32)
        return out


@dataclass
class _SeparableExtractor:
    """Mock extractor that yields linearly-separable features for fit tests.

    'clean' / 'injected' presence in the text shifts the activation vector
    by a fixed offset → a logistic regression learns the decision boundary
    perfectly.
    """

    hidden_size: int = 4
    backbone_id: str = "mock/separable"

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        for i, t in enumerate(texts):
            if "injected" in t:
                out[i] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            else:
                out[i] = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        return out


# ----------------------------------------------------------------------------
# Protocol compliance
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_extractor_is_protocol_compliant() -> None:
    extractor = _MockExtractor()
    assert isinstance(extractor, ActivationExtractor)


@pytest.mark.unit
def test_fitted_probe_is_protocol_compliant(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(
        backbone="mock/test",
        extractor=_SeparableExtractor(),
        cache_dir=tmp_path,
    )
    probe.fit(
        clean_texts=["a clean text", "another clean one"],
        injected_texts=["this is injected", "totally injected"],
    )
    assert isinstance(probe, Probe)


# ----------------------------------------------------------------------------
# Fit + predict
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_fit_returns_self_for_chaining(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_SeparableExtractor(), cache_dir=tmp_path)
    result = probe.fit(["clean a"], ["injected a"])
    assert result is probe


@pytest.mark.unit
def test_fit_then_predict_shape(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_SeparableExtractor(), cache_dir=tmp_path)
    probe.fit(
        clean_texts=["clean 1", "clean 2", "clean 3"],
        injected_texts=["injected 1", "injected 2", "injected 3"],
    )
    preds = probe.predict(["new clean", "new injected"])
    assert preds.shape == (2,)
    assert set(preds.tolist()).issubset({0, 1})


@pytest.mark.unit
def test_fit_then_predict_proba_shape_and_normalization(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_SeparableExtractor(), cache_dir=tmp_path)
    probe.fit(
        clean_texts=["clean 1", "clean 2", "clean 3"],
        injected_texts=["injected 1", "injected 2", "injected 3"],
    )
    proba = probe.predict_proba(["new clean", "new injected"])
    assert proba.shape == (2, 2)
    # Rows sum to 1 (probability normalization).
    np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0], rtol=1e-5)


@pytest.mark.unit
def test_separable_extractor_yields_perfect_classifier(tmp_path: Path) -> None:
    """Linearly-separable mock should reach 100% accuracy on training data."""
    probe = ActivationDeltaProbe(backbone="x", extractor=_SeparableExtractor(), cache_dir=tmp_path)
    clean = ["clean text " + str(i) for i in range(8)]
    injected = ["injected text " + str(i) for i in range(8)]
    probe.fit(clean, injected)
    train_preds = probe.predict(clean + injected)
    expected = np.array([0] * 8 + [1] * 8)
    np.testing.assert_array_equal(train_preds, expected)


@pytest.mark.unit
def test_coef_and_classes_after_fit(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_SeparableExtractor(), cache_dir=tmp_path)
    probe.fit(["clean 1"], ["injected 1"])
    assert probe.coef_.shape == (1, 4)  # (1 class boundary, hidden_size=4)
    np.testing.assert_array_equal(probe.classes_, np.array([0, 1]))


# ----------------------------------------------------------------------------
# Pre-fit error handling
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_predict_before_fit_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        probe.predict(["a"])


@pytest.mark.unit
def test_predict_proba_before_fit_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        probe.predict_proba(["a"])


@pytest.mark.unit
def test_coef_before_fit_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        _ = probe.coef_


@pytest.mark.unit
def test_classes_before_fit_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        _ = probe.classes_


@pytest.mark.unit
def test_fit_with_empty_clean_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="clean_texts"):
        probe.fit([], ["injected"])


@pytest.mark.unit
def test_fit_with_empty_injected_raises(tmp_path: Path) -> None:
    probe = ActivationDeltaProbe(backbone="x", extractor=_MockExtractor(), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="injected_texts"):
        probe.fit(["clean"], [])


# ----------------------------------------------------------------------------
# Activation cache
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_hit_avoids_second_extractor_call(tmp_path: Path) -> None:
    """Second extract for the same text reads from .npy cache, not the extractor."""
    extractor = _MockExtractor(seed=99)
    probe = ActivationDeltaProbe(backbone="x", extractor=extractor, cache_dir=tmp_path)
    probe.fit(["clean A", "clean B"], ["injected A", "injected B"])
    first_call_count = extractor.call_count
    # Second .predict on already-seen texts should hit cache for all of them
    _ = probe.predict(["clean A", "clean B", "injected A"])
    # call_count went up by 0 because every text was cached; baseline also cached
    assert extractor.call_count == first_call_count


@pytest.mark.unit
def test_cache_files_keyed_by_text_and_config(tmp_path: Path) -> None:
    extractor = _MockExtractor()
    probe = ActivationDeltaProbe(
        backbone="x",
        layer_index=-2,
        aggregate="max",
        extractor=extractor,
        cache_dir=tmp_path,
    )
    probe.fit(["c1"], ["i1"])
    # Expect cached files: baseline + 2 texts × suffix L-2_max
    cached = list(tmp_path.rglob("*_L-2_max.npy"))
    # 1 baseline + 1 clean + 1 injected = 3 files
    assert len(cached) == 3


@pytest.mark.unit
def test_cache_separates_by_aggregate_mode(tmp_path: Path) -> None:
    """Same text + different aggregate → different cache files."""
    extractor = _MockExtractor()
    probe_mean = ActivationDeltaProbe(
        backbone="x", aggregate="mean", extractor=extractor, cache_dir=tmp_path
    )
    probe_max = ActivationDeltaProbe(
        backbone="x", aggregate="max", extractor=extractor, cache_dir=tmp_path
    )
    probe_mean.fit(["clean 1"], ["injected 1"])
    probe_max.fit(["clean 1"], ["injected 1"])
    mean_files = list(tmp_path.rglob("*_L-1_mean.npy"))
    max_files = list(tmp_path.rglob("*_L-1_max.npy"))
    assert len(mean_files) == 3  # baseline + clean + injected
    assert len(max_files) == 3


# ----------------------------------------------------------------------------
# Deterministic mock extractor
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_extractor_is_deterministic() -> None:
    a = _MockExtractor(seed=42).extract(["hello"])
    b = _MockExtractor(seed=42).extract(["hello"])
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_mock_extractor_different_seeds_differ() -> None:
    a = _MockExtractor(seed=1).extract(["hello world"])
    b = _MockExtractor(seed=2).extract(["hello world"])
    assert not np.array_equal(a, b)


# ----------------------------------------------------------------------------
# Missing-extra ImportError surface
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_default_extractor_raises_helpful_import_error_when_deps_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without [probes], fitting w/o an injected extractor raises ImportError.

    Asserts the install-hint text mentions ``[probes]`` so the user can fix.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    probe = ActivationDeltaProbe(backbone="x", cache_dir=tmp_path)
    with pytest.raises(ImportError, match=r"\[probes\]"):
        probe.fit(["a"], ["b"])


# ----------------------------------------------------------------------------
# Custom classifier injection
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_custom_classifier_is_used(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression

    custom = LogisticRegression(C=0.01, max_iter=100, class_weight=None)
    probe = ActivationDeltaProbe(
        backbone="x",
        extractor=_SeparableExtractor(),
        cache_dir=tmp_path,
        classifier=custom,
    )
    probe.fit(["clean a"], ["injected a"])
    # The injected classifier should have been .fit'd on the deltas.
    assert hasattr(probe.classifier, "coef_")
    assert probe.classifier.C == 0.01  # config preserved
