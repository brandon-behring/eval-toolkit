"""Tests for `eval_toolkit._parallel.parallel_map`.

The helper is internal but is the single source of truth for parallelism
across the toolkit's public APIs. Tests cover:
- Sequential equivalence (n_jobs=1 path)
- Parallel correctness (n_jobs > 1 path; loky backend)
- Reproducibility (SeedSequence-based caller pattern)
- Smart-default semantics (Q8 + Q11): reject 0, cap > cpu_count, once-per-process INFO log
- Lambda rejection with helpful TypeError
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from eval_toolkit._parallel import parallel_map


def _square(x: int) -> int:
    """Module-level helper (picklable for n_jobs > 1)."""
    return x * x


def _seeded_normal(seed_seq: np.random.SeedSequence) -> float:
    """Module-level helper used to test SeedSequence-based reproducibility."""
    rng = np.random.default_rng(seed_seq)
    return float(rng.normal())


@pytest.fixture(autouse=True)
def _reset_guidance_flag():
    """Reset the module-level once-per-process flag so each test sees a clean state."""
    import eval_toolkit._parallel as p

    p._GUIDANCE_EMITTED = False
    yield
    p._GUIDANCE_EMITTED = False


@pytest.mark.unit
def test_sequential_path_matches_list_comprehension() -> None:
    """n_jobs=1 path matches a plain list comprehension."""
    result = parallel_map(_square, [1, 2, 3, 4], n_jobs=1)
    assert result == [1, 4, 9, 16]


@pytest.mark.unit
def test_parallel_path_preserves_order() -> None:
    """n_jobs=2 with loky backend preserves item order."""
    result = parallel_map(_square, [1, 2, 3, 4], n_jobs=2)
    assert result == [1, 4, 9, 16]


@pytest.mark.unit
@pytest.mark.slow
def test_parallel_path_with_all_cores_runs() -> None:
    """n_jobs=-1 (all cores) completes without error; no speedup assertion."""
    result = parallel_map(_square, list(range(50)), n_jobs=-1)
    assert result == [x * x for x in range(50)]


@pytest.mark.unit
def test_seed_sequence_reproducibility_across_n_jobs() -> None:
    """Same SeedSequence.spawn(n) seeds produce identical results regardless of n_jobs.

    This is the reproducibility contract documented in
    ``docs/source/methodology/parallelism.md``: callers use
    ``np.random.SeedSequence(seed).spawn(n)`` to derive per-item RNG state,
    making ``n_jobs > 1`` give bit-for-bit-identical output to ``n_jobs == 1``
    for the same caller-supplied seed.
    """
    seed_seqs = np.random.SeedSequence(42).spawn(20)
    serial = parallel_map(_seeded_normal, seed_seqs, n_jobs=1)
    parallel = parallel_map(_seeded_normal, seed_seqs, n_jobs=2)
    assert serial == parallel


@pytest.mark.unit
def test_n_jobs_zero_raises_value_error() -> None:
    """n_jobs=0 is rejected with a helpful error pointing at 1 / -1."""
    with pytest.raises(ValueError, match=r"n_jobs=0 is not allowed.*1.*-1"):
        parallel_map(_square, [1, 2], n_jobs=0)


@pytest.mark.unit
def test_lambda_rejected_with_helpful_type_error() -> None:
    """Lambdas fail picklability sniff; helpful TypeError surfaces."""
    with pytest.raises(TypeError, match=r"not picklable.*top-level function"):
        parallel_map(lambda x: x, [1, 2], n_jobs=2)


@pytest.mark.unit
def test_n_jobs_caps_at_cpu_count_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """n_jobs exceeding os.cpu_count() is capped (with WARNING log)."""
    cpu_count = os.cpu_count() or 1
    excessive = cpu_count + 100
    with caplog.at_level(logging.WARNING, logger="eval_toolkit._parallel"):
        result = parallel_map(_square, [1, 2, 3], n_jobs=excessive)
    assert result == [1, 4, 9]
    assert any(
        "capping n_jobs" in rec.message for rec in caplog.records
    ), f"expected capping WARNING; got: {[r.message for r in caplog.records]}"


@pytest.mark.unit
def test_n_jobs_minus_one_does_not_trigger_cap_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """n_jobs=-1 is joblib's all-cores convention; no cap warning."""
    with caplog.at_level(logging.WARNING, logger="eval_toolkit._parallel"):
        result = parallel_map(_square, [1, 2, 3], n_jobs=-1)
    assert result == [1, 4, 9]
    assert not any("capping n_jobs" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_guidance_log_emits_when_threshold_met(caplog: pytest.LogCaptureFixture) -> None:
    """n_jobs=1 with >= 1000 items emits an INFO guidance log on first call."""
    items = list(range(1000))
    with caplog.at_level(logging.INFO, logger="eval_toolkit._parallel"):
        parallel_map(_square, items, n_jobs=1, description="bootstrap")
    assert any(
        "sequentially (n_jobs=1)" in rec.message for rec in caplog.records
    ), f"expected guidance INFO log; got: {[r.message for r in caplog.records]}"


@pytest.mark.unit
def test_guidance_log_quiet_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """n_jobs=1 with < 1000 items emits NO guidance log (quiet for small calls)."""
    with caplog.at_level(logging.INFO, logger="eval_toolkit._parallel"):
        parallel_map(_square, list(range(100)), n_jobs=1)
    assert not any("sequentially (n_jobs=1)" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_guidance_log_emits_only_once_per_process(caplog: pytest.LogCaptureFixture) -> None:
    """Per Q11: once-per-process semantics. Two qualifying calls → one log."""
    items = list(range(1000))
    with caplog.at_level(logging.INFO, logger="eval_toolkit._parallel"):
        parallel_map(_square, items, n_jobs=1, description="first call")
        parallel_map(_square, items, n_jobs=1, description="second call")
    guidance_logs = [r for r in caplog.records if "sequentially (n_jobs=1)" in r.message]
    assert len(guidance_logs) == 1, (
        f"expected exactly 1 guidance log across 2 qualifying calls; "
        f"got {len(guidance_logs)}: {[r.message for r in guidance_logs]}"
    )
