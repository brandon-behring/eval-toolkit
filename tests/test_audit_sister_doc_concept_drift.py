"""Tests for eval_toolkit.audit_sister_doc_concept_drift.

Uses a deterministic mock embedder (no sentence_transformers dep). Real
MiniLM is exercised by consumer adoption + the existing leakage test
suite; here we focus on the parsing + clustering + reporting logic.

Seed motivating case (from prompt-injection-detection-prototype, T1
contradiction): docs/REPRODUCIBILITY.md:85 says T1 = full canonical
re-eval (GPU); WRITEUP/reproducibility.md:33 says T1 = laptop smoke.
The validator must flag T1 as a drift cluster with 2 distinct claims.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.audit_sister_doc_concept_drift import (
    DriftCluster,
    SisterDocDriftReport,
    validate_sister_doc_concept_drift,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _keyword_embedder(keywords: dict[str, np.ndarray]):
    """Build a deterministic embedder: assign each unique keyword in a
    snippet its preset vector; default to zero vector if no keyword fires.

    Lets tests control which snippets cluster together by which keywords
    they contain. Keys are case-sensitive whole-word matches.
    """

    def embedder(snippets: Sequence[str]) -> np.ndarray:
        dim = next(iter(keywords.values())).shape[0]
        out = np.zeros((len(snippets), dim), dtype=np.float64)
        for i, snip in enumerate(snippets):
            for kw, vec in keywords.items():
                if kw in snip:
                    out[i] = vec
                    break
        return out

    return embedder


# Two orthogonal embeddings to force clear cluster split.
GPU_VEC = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
LAPTOP_VEC = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
# CLOUD_VEC tilted enough to give cos(GPU, CLOUD) ≈ 0.707 after L2 norm.
# Tests use this for threshold-sensitivity probes (cluster at 0.5; split at 0.95).
CLOUD_VEC = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)


@pytest.mark.unit
def test_seed_case_t1_drift(tmp_path: Path) -> None:
    """T1 contradiction across two sister docs → drift cluster."""
    doc1 = _write(
        tmp_path,
        "docs_REPRODUCIBILITY.md",
        "## Tier ladder\n\nT1 means full canonical re-eval on GPU (A100 80GB).",
    )
    doc2 = _write(
        tmp_path,
        "WRITEUP_reproducibility.md",
        "## Reproducing results\n\nT1 means laptop smoke; make smoke verifies code.",
    )
    embedder = _keyword_embedder({"GPU": GPU_VEC, "laptop": LAPTOP_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.5,
    )

    assert isinstance(report, SisterDocDriftReport)
    assert len(report.drift_clusters) == 1
    drift = report.drift_clusters[0]
    assert drift.token == "T1"
    assert len(drift.sentences) == 2
    files_in_drift = {s[0].name for s in drift.sentences}
    assert files_in_drift == {"docs_REPRODUCIBILITY.md", "WRITEUP_reproducibility.md"}
    # Orthogonal embeddings (cos=0) → divergence ≈ 1.0
    assert drift.divergence_score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_consistent_definition_across_files(tmp_path: Path) -> None:
    """T1 with the same definition across files → consistent_tokens, no drift."""
    doc1 = _write(tmp_path, "a.md", "T1 means full GPU re-eval.")
    doc2 = _write(tmp_path, "b.md", "T1 covers the full GPU re-eval cycle.")
    embedder = _keyword_embedder({"GPU": GPU_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.5,
    )
    assert report.drift_clusters == ()
    assert "T1" in report.consistent_tokens


@pytest.mark.unit
def test_single_occurrence_consistent(tmp_path: Path) -> None:
    """Token mentioned in exactly one file (or one place) → consistent (nothing to compare)."""
    doc = _write(tmp_path, "a.md", "T1 means full GPU re-eval.")
    embedder = _keyword_embedder({"GPU": GPU_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc],
        concept_tokens=["T1"],
        embedder=embedder,
    )
    assert report.drift_clusters == ()
    assert "T1" in report.consistent_tokens


@pytest.mark.unit
def test_unreferenced_token_excluded_from_coverage(tmp_path: Path) -> None:
    """Token never mentioned → not in consistent_tokens; coverage reflects it."""
    doc = _write(tmp_path, "a.md", "T1 means full GPU re-eval.")
    embedder = _keyword_embedder({"GPU": GPU_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc],
        concept_tokens=["T1", "T99_never_mentioned"],
        embedder=embedder,
    )
    assert "T1" in report.consistent_tokens
    assert "T99_never_mentioned" not in report.consistent_tokens
    assert report.coverage == 0.5


@pytest.mark.unit
def test_multiple_tokens_mixed_consistency(tmp_path: Path) -> None:
    """T0 consistent + T1 drift + T3 consistent."""
    doc1 = _write(
        tmp_path,
        "a.md",
        "T0 means unit tests only. T1 means full GPU re-eval. T3 means manual review.",
    )
    doc2 = _write(
        tmp_path,
        "b.md",
        "T0 means unit tests only. T1 means laptop smoke. T3 means manual review.",
    )
    # Vectors: unit→V0, GPU→V_GPU, laptop→V_LAP, manual→V_MAN.
    V0 = np.array([1.0, 0.0, 0.0, 0.0])
    V_GPU = np.array([0.0, 1.0, 0.0, 0.0])
    V_LAP = np.array([0.0, 0.0, 1.0, 0.0])
    V_MAN = np.array([0.0, 0.0, 0.0, 1.0])

    embedder = _keyword_embedder({"unit": V0, "GPU": V_GPU, "laptop": V_LAP, "manual": V_MAN})

    # context_window_sentences=0 keeps each T_n occurrence's embedding
    # snippet to just the sentence containing it (avoids cross-token
    # contamination from the keyword-based mock embedder).
    report = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T0", "T1", "T3"],
        embedder=embedder,
        similarity_threshold=0.5,
        context_window_sentences=0,
    )
    drift_tokens = {dc.token for dc in report.drift_clusters}
    assert drift_tokens == {"T1"}
    assert set(report.consistent_tokens) == {"T0", "T3"}


@pytest.mark.unit
def test_threshold_sensitivity(tmp_path: Path) -> None:
    """Two near-identical occurrences cluster at high threshold; not at very high."""
    doc1 = _write(tmp_path, "a.md", "T1 means GPU re-eval.")
    doc2 = _write(tmp_path, "b.md", "T1 means cloud re-eval.")  # very similar prose
    # GPU and CLOUD vectors are close (cos≈0.9) — they should cluster at threshold 0.5
    embedder = _keyword_embedder({"GPU": GPU_VEC, "cloud": CLOUD_VEC})

    # At threshold 0.5: identical-enough, consistent.
    report_loose = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.5,
    )
    assert "T1" in report_loose.consistent_tokens
    assert report_loose.drift_clusters == ()

    # At threshold 0.95: stricter, they split.
    report_strict = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.95,
    )
    assert len(report_strict.drift_clusters) == 1


@pytest.mark.unit
def test_whole_word_boundary(tmp_path: Path) -> None:
    """T1 does NOT match T10 or t1."""
    doc1 = _write(tmp_path, "a.md", "T1 means GPU. T10 is a different tier. t1 is lowercase.")
    doc2 = _write(tmp_path, "b.md", "T1 means laptop.")
    embedder = _keyword_embedder({"GPU": GPU_VEC, "laptop": LAPTOP_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc1, doc2],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.5,
    )
    assert len(report.drift_clusters) == 1
    drift = report.drift_clusters[0]
    # Only 2 occurrences (a.md "T1 means GPU" + b.md "T1 means laptop"),
    # NOT 4 (T10 and t1 excluded).
    assert len(drift.sentences) == 2


@pytest.mark.unit
def test_context_window_zero_keeps_token_sentence_only(tmp_path: Path) -> None:
    """context_window_sentences=0 → only the sentence containing the token."""
    doc = _write(
        tmp_path,
        "a.md",
        "Background prose. T1 means GPU. Future work.",
    )
    embedder = _keyword_embedder({"GPU": GPU_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc],
        concept_tokens=["T1"],
        embedder=embedder,
        context_window_sentences=0,
    )
    # Single occurrence → consistent.
    assert "T1" in report.consistent_tokens


@pytest.mark.unit
def test_empty_concept_tokens(tmp_path: Path) -> None:
    """Empty concept_tokens → empty report, coverage=0.0."""
    doc = _write(tmp_path, "a.md", "T1 means GPU.")
    embedder = _keyword_embedder({"GPU": GPU_VEC})

    report = validate_sister_doc_concept_drift(
        files=[doc],
        concept_tokens=[],
        embedder=embedder,
    )
    assert report.drift_clusters == ()
    assert report.consistent_tokens == ()
    assert report.coverage == 0.0


@pytest.mark.unit
def test_coverage_100_percent(tmp_path: Path) -> None:
    """All tokens referenced → coverage=1.0."""
    doc = _write(tmp_path, "a.md", "T0 unit. T1 GPU. T3 manual.")
    V_U = np.array([1.0, 0.0, 0.0])
    V_G = np.array([0.0, 1.0, 0.0])
    V_M = np.array([0.0, 0.0, 1.0])
    embedder = _keyword_embedder({"unit": V_U, "GPU": V_G, "manual": V_M})

    report = validate_sister_doc_concept_drift(
        files=[doc],
        concept_tokens=["T0", "T1", "T3"],
        embedder=embedder,
    )
    assert report.coverage == 1.0


@pytest.mark.unit
def test_three_way_drift(tmp_path: Path) -> None:
    """3 distinct definitions → DriftCluster with 3 sentences across clusters."""
    doc1 = _write(tmp_path, "a.md", "T1 means GPU.")
    doc2 = _write(tmp_path, "b.md", "T1 means laptop.")
    V_3 = np.array([0.0, 0.0, 1.0, 0.0])
    doc3 = _write(tmp_path, "c.md", "T1 means TPU.")
    embedder = _keyword_embedder({"GPU": GPU_VEC, "laptop": LAPTOP_VEC, "TPU": V_3})

    report = validate_sister_doc_concept_drift(
        files=[doc1, doc2, doc3],
        concept_tokens=["T1"],
        embedder=embedder,
        similarity_threshold=0.5,
    )
    assert len(report.drift_clusters) == 1
    drift = report.drift_clusters[0]
    assert len(drift.sentences) == 3


@pytest.mark.unit
def test_frozen_dataclass_invariants() -> None:
    """DriftCluster + SisterDocDriftReport are immutable."""
    dc = DriftCluster(
        token="T1",
        sentences=((Path("a.md"), 1, "T1 means GPU."),),
        divergence_score=0.5,
    )
    r = SisterDocDriftReport(drift_clusters=(dc,), consistent_tokens=(), coverage=0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        dc.divergence_score = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.coverage = 1.0  # type: ignore[misc]


@pytest.mark.unit
def test_default_embedder_lazy_import() -> None:
    """embedder=None triggers lazy MiniLM resolution; raises ImportError if extras missing."""
    # If sentence_transformers IS installed (CI with [embeddings] extra),
    # this should succeed loading; otherwise it raises ImportError. Both are
    # acceptable — we just want to verify the validator's lazy-resolution path
    # doesn't crash with an AttributeError or similar.
    try:
        # Empty files + tokens means the embedder is never actually called.
        # This exercises the validator's "lazy resolve embedder only when
        # there's something to embed" semantics — but we still want to
        # at least import the audit-module path successfully.
        from eval_toolkit.audit_sister_doc_concept_drift import (
            _default_embedder,  # noqa: PLC0415
        )

        _ = _default_embedder  # the callable itself; don't call it
    except ImportError as exc:
        # If sentence_transformers absent, the docstring promised hint.
        assert "embeddings" in str(exc).lower() or "sentence_transformers" in str(exc)


@pytest.mark.unit
def test_non_utf8_file_skipped_with_warning(tmp_path: Path) -> None:
    """A non-UTF-8 file is skipped (not a crash) and the skip is warned, not silent."""
    good = _write(tmp_path, "good.md", "T1 is full GPU re-eval.")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"T1 is \xff laptop smoke.\n")
    embedder = _keyword_embedder({"GPU": GPU_VEC, "laptop": LAPTOP_VEC})
    with pytest.warns(UserWarning, match="skipping unreadable"):
        report = validate_sister_doc_concept_drift(
            files=[good, bad],
            concept_tokens=["T1"],
            embedder=embedder,
            similarity_threshold=0.5,
        )
    # bad.md dropped → only one T1 occurrence → no drift cluster, still consistent.
    assert report.drift_clusters == ()
    assert "T1" in report.consistent_tokens


@pytest.mark.unit
def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A nonexistent path in `files` raises — explicit list, not glob discovery (#99 V10)."""
    good = _write(tmp_path, "good.md", "T1 is full GPU re-eval.")
    embedder = _keyword_embedder({"GPU": GPU_VEC})
    with pytest.raises(FileNotFoundError):
        validate_sister_doc_concept_drift(
            files=[good, tmp_path / "missing.md"],
            concept_tokens=["T1"],
            embedder=embedder,
        )
