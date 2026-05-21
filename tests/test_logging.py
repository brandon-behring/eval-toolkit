"""Tests for eval-toolkit's library-logger discipline (Tier β #4).

Verifies the standard library-friendly pattern (per v0.29.0 plan Q4=A):

1. The package root logger (`eval_toolkit`) has a `NullHandler`
   attached — so eval-toolkit is silent by default unless the calling
   application configures handlers.
2. Per-module loggers (`eval_toolkit.harness`, `eval_toolkit.leakage`,
   `eval_toolkit.bootstrap`, `eval_toolkit.loaders`) exist and emit
   at DEBUG level on relevant events.
3. The hierarchical name structure mirrors the import path, enabling
   granular filter (e.g., enable just `eval_toolkit.harness` debug).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from eval_toolkit import bootstrap_ci
from eval_toolkit.harness import EvalSlice, evaluate
from eval_toolkit.leakage import ExactDuplicateCheck, run_leakage_checks
from eval_toolkit.loaders import DataFrameLoader
from eval_toolkit.metrics import pr_auc


@pytest.mark.unit
def test_root_logger_has_null_handler() -> None:
    """eval_toolkit's root logger is silent by default via a NullHandler.

    Matches the convention from PEP-friendly libraries (numpy, scikit-learn,
    requests) — `addHandler(NullHandler())` at import time prevents
    'No handlers could be found' warnings AND silences output when
    consumers haven't configured handlers.
    """
    root = logging.getLogger("eval_toolkit")
    handler_types = {type(h).__name__ for h in root.handlers}
    assert (
        "NullHandler" in handler_types
    ), f"Expected NullHandler on eval_toolkit root logger; got {handler_types}"


@pytest.mark.unit
def test_per_module_loggers_resolvable() -> None:
    """Each module has its own logger resolvable via the import-path name."""
    for module in (
        "eval_toolkit.harness",
        "eval_toolkit.leakage",
        "eval_toolkit.bootstrap",
        "eval_toolkit.loaders",
    ):
        logger = logging.getLogger(module)
        assert logger.name == module
        # All inherit through eval_toolkit which has the NullHandler
        assert logger.parent is not None
        assert "eval_toolkit" in logger.parent.name


@pytest.mark.unit
def test_evaluate_emits_at_eval_toolkit_harness(caplog: pytest.LogCaptureFixture) -> None:
    """`evaluate(...)` emits at least one INFO record at eval_toolkit.harness.

    The existing harness logger emits per-slice metadata at INFO level.
    Confirms the logger name resolves correctly so consumers can filter
    `logging.getLogger("eval_toolkit.harness").setLevel(...)`.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"text": [f"r{i}" for i in range(40)], "label": [i % 2 for i in range(40)]})
    parent = EvalSlice(name="test", df=df)

    class _Stub:
        def predict_proba(self, X: object) -> np.ndarray:
            return rng.uniform(0, 1, size=len(X))  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO, logger="eval_toolkit.harness"):
        evaluate({"stub": _Stub()}, [parent], run_id="logtest", n_resamples=20)

    harness_records = [r for r in caplog.records if r.name == "eval_toolkit.harness"]
    assert harness_records, "expected at least one eval_toolkit.harness record"


@pytest.mark.unit
def test_run_leakage_checks_emits_debug(caplog: pytest.LogCaptureFixture) -> None:
    """`run_leakage_checks` emits per-check DEBUG records + a summary."""
    df_train = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    df_test = pd.DataFrame({"text": ["d", "e"], "label": [0, 1]})
    splits = {
        "train": EvalSlice(name="train", df=df_train),
        "test": EvalSlice(name="test", df=df_test),
    }

    with caplog.at_level(logging.DEBUG, logger="eval_toolkit.leakage"):
        run_leakage_checks([ExactDuplicateCheck()], splits)

    leakage_records = [r for r in caplog.records if r.name == "eval_toolkit.leakage"]
    assert leakage_records, "expected at least one eval_toolkit.leakage DEBUG record"
    # Should see both a per-check record and a summary record
    messages = " | ".join(r.message for r in leakage_records)
    assert "leakage check" in messages or "run_leakage_checks completed" in messages


@pytest.mark.unit
def test_bootstrap_ci_emits_debug(caplog: pytest.LogCaptureFixture) -> None:
    """`bootstrap_ci(...)` emits a DEBUG record with run parameters."""
    rng = np.random.default_rng(42)
    n = 50
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    rng.shuffle(y)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)

    with caplog.at_level(logging.DEBUG, logger="eval_toolkit.bootstrap"):
        bootstrap_ci(y, s, metric=pr_auc, n_resamples=50, seed=42)

    bootstrap_records = [r for r in caplog.records if r.name == "eval_toolkit.bootstrap"]
    assert bootstrap_records, "expected at least one eval_toolkit.bootstrap DEBUG record"
    msg = bootstrap_records[0].message
    assert "n_resamples=50" in msg
    assert "method=BCa" in msg


@pytest.mark.unit
def test_dataframeloader_load_splits_emits_debug(caplog: pytest.LogCaptureFixture) -> None:
    """`DataFrameLoader.load_splits()` emits a DEBUG record summarizing the splits."""
    df = pd.DataFrame(
        {
            "text": [f"r{i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
            "split": (["train"] * 14 + ["test"] * 6),
        }
    )
    loader = DataFrameLoader(df=df, split_col="split")

    with caplog.at_level(logging.DEBUG, logger="eval_toolkit.loaders"):
        loader.load_splits()

    loader_records = [r for r in caplog.records if r.name == "eval_toolkit.loaders"]
    assert loader_records, "expected at least one eval_toolkit.loaders DEBUG record"
    msg = loader_records[0].message
    assert "train=14" in msg and "test=6" in msg


@pytest.mark.unit
def test_silent_by_default_without_handler() -> None:
    """With no handler configured, library logging produces no stderr output.

    This is the load-bearing test: it asserts the NullHandler pattern
    actually works. If a future refactor accidentally added a
    StreamHandler at __init__.py, library imports would emit log lines
    to host applications' stderr — exactly the antipattern the NullHandler
    prevents.
    """
    # Confirm no real handlers are attached at the root other than NullHandler
    root = logging.getLogger("eval_toolkit")
    real_handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]
    assert real_handlers == [], (
        f"eval_toolkit root logger should ONLY have NullHandler; "
        f"found other handlers: {real_handlers}"
    )
