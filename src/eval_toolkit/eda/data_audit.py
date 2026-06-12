"""Job-1 dataset integrity gate: thin per-split profiling + composable gates.

This module is the **integrity gate** layer of an EDA-first research
program. It answers a narrow question before any modelling: *is this
dataset's split structure sound enough to trust downstream metrics?* It
computes per-split shape (row counts, class balance, text-length quantiles)
and runs the reused :mod:`eval_toolkit.leakage` dedup / cross-split checks,
then folds the results into machine-readable
:class:`~eval_toolkit.claims.GateResult` gates.

Scope is deliberately thin and torch-free. There are **no embeddings, no
semantic similarity, no contamination scoring, and no UMAP** here — those
distribution-shift concerns are deferred to a later ``distribution_shift``
module. The near-duplicate / cross-split checks default to the lexical,
torch-free :class:`~eval_toolkit.text_dedup.TfidfCosineStrategy`.

Public access is ``eval_toolkit.eda.*`` (Tier-2 per ADR 0003): this layer
is intentionally evolvable and is **not** part of the v2.0-frozen top-level
surface.

Reused v1.4.0 primitives
------------------------
- :class:`eval_toolkit.loaders.DataFrameLoader` (``.load_splits()``) +
  :class:`eval_toolkit.harness.EvalSlice` (``.features`` / ``.y_true``).
- :class:`eval_toolkit.leakage.ExactDuplicateCheck`,
  :class:`~eval_toolkit.leakage.NearDuplicateCheck`,
  :class:`~eval_toolkit.leakage.CrossSplitLeakageCheck`, and
  :func:`~eval_toolkit.leakage.run_leakage_checks` →
  :class:`~eval_toolkit.leakage.LeakageReport`.
- :class:`eval_toolkit.text_dedup.TfidfCosineStrategy` (torch-free backend).
- :class:`eval_toolkit.claims.GateResult` (one per integrity gate).
- :func:`eval_toolkit.artifacts.write_json_strict` /
  :func:`~eval_toolkit.artifacts.sanitize_for_json`.
- :func:`eval_toolkit.metrics.is_metric_defined_for_slice` (single-class flag).

Examples
--------
>>> import pandas as pd
>>> from eval_toolkit.loaders import DataFrameLoader
>>> from eval_toolkit.eda import audit_dataset
>>> df = pd.DataFrame(
...     {
...         "text": ["benign one", "benign two", "attack a", "attack b"],
...         "label": [0, 0, 1, 1],
...         "split": ["train", "test", "train", "test"],
...     }
... )
>>> loader = DataFrameLoader(df, split_col="split", name="toy")
>>> audit = audit_dataset(loader, near=False, cross_split=False)
>>> audit.gate_passed
True
>>> sorted(audit.split_summaries)
['test', 'train']
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from eval_toolkit.artifacts import sanitize_for_json, write_json_strict
from eval_toolkit.claims import GateResult
from eval_toolkit.eda.obfuscation import ObfuscationProfile, analyze_obfuscation
from eval_toolkit.leakage import (
    CrossSplitLeakageCheck,
    ExactDuplicateCheck,
    LeakageCheck,
    LeakageReport,
    NearDuplicateCheck,
    run_leakage_checks,
)
from eval_toolkit.metrics import is_metric_defined_for_slice
from eval_toolkit.text_dedup import TfidfCosineStrategy

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from eval_toolkit.harness import EvalSlice
    from eval_toolkit.loaders import DatasetLoader

_logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_NEG_POS_RATIO",
    "DEFAULT_MIN_NEG_POS_RATIO",
    "DEFAULT_PCT_OVER_CONTEXT_THRESHOLD",
    "EDA_AUDIT_SCHEMA_VERSION",
    "DataAudit",
    "SplitSummary",
    "Tokenizer",
    "audit_dataset",
    "class_balance",
    "length_quantiles",
    "summarize_split",
]

# Schema version for the emitted DataAudit artifact. Bump on any
# backward-incompatible change to DataAudit.to_dict() (mirrors the
# RUN_RESULT_SCHEMA_VERSION / MANIFEST_SCHEMA_VERSION discipline).
# Authored as v2 from the outset: the gate is fully deterministic, so the
# common ``seed: int`` field is omitted (STYLE.md §3a forbids it outside the
# two declared exceptions).
EDA_AUDIT_SCHEMA_VERSION: Final[str] = "v2"

# Default class-balance gate bounds, expressed as neg:pos ratio. A balanced
# binary set has ratio 1.0; [0.1, 10] tolerates up to a 10:1 imbalance in
# either direction before flagging. Picked as a sane, symmetric default for
# prompt-injection corpora (which skew benign-heavy); override per call.
DEFAULT_MIN_NEG_POS_RATIO: Final[float] = 0.1
DEFAULT_MAX_NEG_POS_RATIO: Final[float] = 10.0

# Default ceiling for the fraction of rows whose token length exceeds the
# model context window. Above this, truncation would silently corrupt a
# material share of the eval set. Only evaluated when both a tokenizer and a
# context_window are supplied.
DEFAULT_PCT_OVER_CONTEXT_THRESHOLD: Final[float] = 0.05

# A caller-supplied tokenizer: maps one string to a sequence of token ids.
# Mirrors the "caller passes a tokenizer callable" contract of
# TokenizationLeakageCheck — this module never imports transformers.
Tokenizer = Callable[[str], Sequence[int]]


def class_balance(slice_: EvalSlice) -> dict[str, object]:
    """Summarize the binary class distribution of one slice.

    Parameters
    ----------
    slice_ : EvalSlice
        A single slice exposing a binary ``.y_true`` array in ``{0, 1}``.

    Returns
    -------
    dict
        Keys: ``n_rows`` , ``n_positive`` , ``n_negative`` , ``pos_pct``
        (positive fraction in ``[0, 1]`` , ``0.0`` for an empty slice),
        ``neg_pos_ratio`` (negatives / positives; ``None`` when there are
        no positives), and ``is_single_class`` (only one observed label).

    Examples
    --------
    >>> import pandas as pd
    >>> from eval_toolkit.harness import EvalSlice
    >>> sl = EvalSlice(name="s", df=pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 1]}))
    >>> bal = class_balance(sl)
    >>> bal["n_positive"], bal["n_negative"], round(bal["pos_pct"], 4)  # type: ignore[index]
    (2, 1, 0.6667)
    """
    y_true = slice_.y_true
    n_rows = int(y_true.shape[0])
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    pos_pct = (n_positive / n_rows) if n_rows else 0.0
    neg_pos_ratio = (n_negative / n_positive) if n_positive else None
    is_single_class = n_rows > 0 and (n_positive == 0 or n_negative == 0)
    return {
        "n_rows": n_rows,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pos_pct": pos_pct,
        "neg_pos_ratio": neg_pos_ratio,
        "is_single_class": is_single_class,
    }


def _quantiles(values: Sequence[int]) -> dict[str, float | int | None]:
    """Return p50 / p95 / max of a non-negative integer length sample.

    Empty input yields ``None`` for every quantile (no fabricated zeros).
    """
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)

    def _percentile(pct: float) -> float:
        # Linear interpolation between closest ranks (numpy "linear" method),
        # kept dependency-free so the helper stays import-light.
        if n == 1:
            return float(ordered[0])
        rank = pct / 100.0 * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac

    return {
        "p50": _percentile(50.0),
        "p95": _percentile(95.0),
        "max": int(ordered[-1]),
    }


def length_quantiles(
    texts: Sequence[str],
    tokenizer: Tokenizer | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Compute char / word (and optionally token) length quantiles.

    Character and word length quantiles (p50 / p95 / max) are always
    computed. Token-length quantiles are computed **only** when a
    ``tokenizer`` callable is supplied — this module never imports
    ``transformers`` (mirrors the caller-supplied-tokenizer contract of
    :class:`eval_toolkit.leakage.TokenizationLeakageCheck` ).

    Parameters
    ----------
    texts : sequence of str
        The text samples to measure. An empty sequence yields ``None`` for
        every quantile (no fabricated zeros).
    tokenizer : Callable[[str], Sequence[int]] or None, optional
        Maps one string to a sequence of token ids. When supplied, a
        ``"token"`` block is added with the same p50 / p95 / max keys. The
        callable's length is measured via ``len(tokenizer(text))``.

    Returns
    -------
    dict
        ``{"char": {...}, "word": {...}}`` always, plus ``"token": {...}``
        when a tokenizer is given. Each inner dict has ``p50`` / ``p95`` /
        ``max`` keys.

    Examples
    --------
    >>> q = length_quantiles(["a b c", "d e", "f"], tokenizer=lambda s: s.split())
    >>> q["word"]["max"], q["token"]["max"]
    (3, 3)
    >>> length_quantiles([])["char"]["p50"] is None
    True
    """
    char_lengths = [len(t) for t in texts]
    word_lengths = [len(t.split()) for t in texts]
    out: dict[str, dict[str, float | int | None]] = {
        "char": _quantiles(char_lengths),
        "word": _quantiles(word_lengths),
    }
    if tokenizer is not None:
        token_lengths = [len(tokenizer(t)) for t in texts]
        out["token"] = _quantiles(token_lengths)
    return out


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Per-split shape + class-balance + length profile.

    Token-length fields are populated only when a tokenizer was supplied to
    the producing call; ``pct_over_context_window`` additionally requires a
    ``context_window`` and is ``None`` otherwise.

    Parameters
    ----------
    split : str
        Split name (e.g. ``"train"`` / ``"test"``).
    n_rows : int
        Total rows in the split.
    n_positive, n_negative : int
        Positive / negative label counts.
    pos_pct : float
        Positive fraction in ``[0, 1]`` (``0.0`` for an empty split).
    neg_pos_ratio : float or None
        Negatives / positives; ``None`` when there are no positives.
    is_single_class : bool
        Whether only one label is observed in the split.
    char_lengths, word_lengths : dict
        ``{"p50", "p95", "max"}`` character / whitespace-word length
        quantiles.
    token_lengths : dict or None
        Same shape as ``char_lengths`` when a tokenizer was supplied; else
        ``None``.
    pct_over_context_window : float or None
        Fraction of rows whose token length exceeds the supplied
        ``context_window``; ``None`` when a tokenizer or context window was
        not supplied.
    obfuscation : ObfuscationProfile or None
        Corpus-level obfuscation / encoding / invisible-Unicode prevalence
        per :func:`eval_toolkit.eda.obfuscation.analyze_obfuscation`.
        Profile-only — does not gate ``DataAudit.gate_passed``. ``None``
        when the caller passed ``compute_obfuscation=False``.
    """

    split: str
    n_rows: int
    n_positive: int
    n_negative: int
    pos_pct: float
    neg_pos_ratio: float | None
    is_single_class: bool
    char_lengths: dict[str, float | int | None]
    word_lengths: dict[str, float | int | None]
    token_lengths: dict[str, float | int | None] | None = None
    pct_over_context_window: float | None = None
    obfuscation: ObfuscationProfile | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this summary."""
        return {
            "split": self.split,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "pos_pct": self.pos_pct,
            "neg_pos_ratio": self.neg_pos_ratio,
            "is_single_class": self.is_single_class,
            "char_lengths": dict(self.char_lengths),
            "word_lengths": dict(self.word_lengths),
            "token_lengths": dict(self.token_lengths) if self.token_lengths is not None else None,
            "pct_over_context_window": self.pct_over_context_window,
            "obfuscation": self.obfuscation.to_dict() if self.obfuscation is not None else None,
        }


def summarize_split(
    slice_: EvalSlice,
    *,
    tokenizer: Tokenizer | None = None,
    context_window: int | None = None,
    compute_obfuscation: bool = True,
) -> SplitSummary:
    """Build a :class:`SplitSummary` for one slice.

    Parameters
    ----------
    slice_ : EvalSlice
        The split to profile.
    tokenizer : Callable[[str], Sequence[int]] or None, optional
        Caller-supplied tokenizer. When given, token-length quantiles are
        computed. This module never imports ``transformers``.
    context_window : int or None, optional
        Model context window in tokens. When given **with** a tokenizer,
        ``pct_over_context_window`` is computed.
    compute_obfuscation : bool, optional
        Run :func:`eval_toolkit.eda.obfuscation.analyze_obfuscation` over
        the slice and attach the resulting :class:`~eval_toolkit.eda.
        obfuscation.ObfuscationProfile`. Default ``True``. Cheap relative
        to the leakage checks (single pass over chars + token regex).

    Returns
    -------
    SplitSummary

    Raises
    ------
    ValueError
        If ``context_window`` is supplied without a tokenizer (the fraction
        is undefined without token lengths), or if ``context_window <= 0``.
    """
    if context_window is not None:
        if tokenizer is None:
            raise ValueError(
                "context_window requires a tokenizer; token lengths are "
                "undefined without one (pass tokenizer=...)"
            )
        if context_window <= 0:
            raise ValueError(f"context_window must be positive; got {context_window}")

    # Compute counts directly from the typed label array (avoids round-tripping
    # through the object-typed class_balance() dict). class_balance() remains
    # the public helper for callers wanting the dict shape.
    y_true = slice_.y_true
    n_rows = int(y_true.shape[0])
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    pos_pct = (n_positive / n_rows) if n_rows else 0.0
    neg_pos_ratio = (n_negative / n_positive) if n_positive else None
    is_single_class = n_rows > 0 and (n_positive == 0 or n_negative == 0)

    texts = slice_.features
    lengths = length_quantiles(texts, tokenizer=tokenizer)
    token_lengths = lengths.get("token")

    pct_over: float | None = None
    if tokenizer is not None and context_window is not None:
        if texts:
            n_over = sum(1 for t in texts if len(tokenizer(t)) > context_window)
            pct_over = n_over / len(texts)
        else:
            pct_over = 0.0

    obfuscation = analyze_obfuscation(texts) if compute_obfuscation else None

    return SplitSummary(
        split=slice_.name,
        n_rows=n_rows,
        n_positive=n_positive,
        n_negative=n_negative,
        pos_pct=pos_pct,
        neg_pos_ratio=neg_pos_ratio,
        is_single_class=is_single_class,
        char_lengths=lengths["char"],
        word_lengths=lengths["word"],
        token_lengths=token_lengths,
        pct_over_context_window=pct_over,
        obfuscation=obfuscation,
    )


@dataclass(frozen=True, slots=True)
class DataAudit:
    """Aggregate Job-1 integrity-gate report for a dataset.

    Bundles per-split summaries, the reused dedup / leakage report(s), and
    the integrity gates (as :class:`~eval_toolkit.claims.GateResult` ) into
    a single auditable, JSON-serializable artifact.

    Parameters
    ----------
    schema_version : str
        Artifact schema version (see :data:`EDA_AUDIT_SCHEMA_VERSION`).
    dataset_name : str
        Dataset identifier (from the loader's ``name`` when available).
    run_timestamp : str
        ISO-8601 UTC timestamp (``YYYY-MM-DDTHH:MM:SSZ``).
    split_summaries : dict[str, SplitSummary]
        One entry per split.
    leakage_reports : dict[str, LeakageReport]
        Reused dedup / leakage report(s), keyed by phase
        (``"within_split"`` / ``"cross_split"``).
    gates : list[GateResult]
        One :class:`~eval_toolkit.claims.GateResult` per integrity gate.
    gate_passed : bool
        ``True`` iff every error-severity gate passed.
    gate_summary : str
        Human-readable one-line summary.
    """

    schema_version: str
    dataset_name: str
    run_timestamp: str
    split_summaries: dict[str, SplitSummary]
    leakage_reports: dict[str, LeakageReport]
    gates: list[GateResult]
    gate_passed: bool
    gate_summary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the full audit.

        The mapping routes cleanly through
        :func:`eval_toolkit.artifacts.sanitize_for_json` /
        :func:`~eval_toolkit.artifacts.write_json_strict`; see
        :meth:`write`.
        """
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "run_timestamp": self.run_timestamp,
            "split_summaries": {
                name: summary.to_dict() for name, summary in self.split_summaries.items()
            },
            "leakage_reports": {
                phase: report.to_dict() for phase, report in self.leakage_reports.items()
            },
            "gates": [gate.to_dict() for gate in self.gates],
            "gate_passed": self.gate_passed,
            "gate_summary": self.gate_summary,
        }

    def write(self, path: Path | str) -> Path:
        """Write the audit to ``path`` as strict RFC 8259 JSON.

        Delegates to :func:`eval_toolkit.artifacts.write_json_strict` ,
        which sanitizes any non-finite floats before serialization.

        Parameters
        ----------
        path : Path or str
            Destination file. Parent directories are created as needed.

        Returns
        -------
        Path
            The written path.
        """
        return write_json_strict(self.to_dict(), path)


def _utc_now_iso8601() -> str:
    """Return current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` (1-second resolution).

    Mirrors :func:`eval_toolkit.manifest._utc_now_iso8601` for
    cross-artifact timestamp consistency.
    """
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_train_split(splits: Mapping[str, EvalSlice]) -> str | None:
    """Pick the reference split for cross-split leakage.

    Prefers a split literally named ``"train"``; otherwise returns ``None``
    (cross-split leakage cannot be defined without a reference corpus).
    """
    if "train" in splits:
        return "train"
    return None


def _class_balance_gate(
    summaries: Mapping[str, SplitSummary],
    *,
    min_ratio: float,
    max_ratio: float,
) -> GateResult:
    """Error gate: every non-single-class split's neg:pos ratio is in range.

    Single-class splits cannot satisfy a balance bound (ratio is undefined
    or infinite); they fail this gate with an explicit reason so the
    degeneracy is surfaced rather than silently ignored.
    """
    offenders: dict[str, object] = {}
    for name, summary in summaries.items():
        # A balance ratio is only well-defined when both classes are present.
        # Reuse is_metric_defined_for_slice (the canonical single-class signal):
        # it returns False for single-class slices, so a False here means the
        # ratio bound cannot be satisfied and the split fails the gate.
        balance_defined = is_metric_defined_for_slice(
            "auroc", is_single_class=summary.is_single_class
        )
        if not balance_defined:
            offenders[name] = {
                "reason": "single_class",
                "n_positive": summary.n_positive,
                "n_negative": summary.n_negative,
            }
            continue
        ratio = summary.neg_pos_ratio
        if ratio is None or not (min_ratio <= ratio <= max_ratio):
            offenders[name] = {
                "reason": "out_of_range",
                "neg_pos_ratio": ratio,
            }
    passed = not offenders
    return GateResult(
        name="class_balance",
        passed=passed,
        severity="error",
        message=(
            f"all splits within neg:pos ratio [{min_ratio}, {max_ratio}]"
            if passed
            else f"{len(offenders)} split(s) outside neg:pos ratio "
            f"[{min_ratio}, {max_ratio}] (or single-class)"
        ),
        evidence={
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "offending_splits": offenders,
        },
    )


def _no_cross_split_leakage_gate(report: LeakageReport | None) -> GateResult:
    """Error gate: zero error-severity cross-split leakage findings.

    When cross-split checking was skipped (``report is None``), the gate
    passes with an explanatory message (nothing was asserted to leak).
    """
    if report is None:
        return GateResult(
            name="no_cross_split_leakage",
            passed=True,
            severity="error",
            message="cross-split leakage check skipped (no train split / disabled)",
            evidence={"skipped": True},
        )
    errors = report.errors()
    n_affected = sum(f.n_affected for f in errors)
    passed = not errors
    return GateResult(
        name="no_cross_split_leakage",
        passed=passed,
        severity="error",
        message=(
            "no cross-split leakage errors"
            if passed
            else f"{n_affected} eval row(s) leak across splits " f"({len(errors)} error finding(s))"
        ),
        evidence={
            "n_error_findings": len(errors),
            "n_affected": n_affected,
            "findings": [f.to_dict() for f in errors],
        },
    )


def _context_window_gate(
    summaries: Mapping[str, SplitSummary],
    *,
    threshold: float,
) -> GateResult:
    """Error gate: every split keeps ``pct_over_context_window`` below threshold.

    Only meaningful when token lengths were computed (tokenizer +
    context_window supplied); callers gate on its presence before adding
    it.
    """
    offenders: dict[str, float] = {}
    for name, summary in summaries.items():
        pct = summary.pct_over_context_window
        if pct is not None and pct > threshold:
            offenders[name] = pct
    passed = not offenders
    return GateResult(
        name="context_window_fit",
        passed=passed,
        severity="error",
        message=(
            f"all splits keep <{threshold:.0%} of rows over the context window"
            if passed
            else f"{len(offenders)} split(s) exceed {threshold:.0%} of rows "
            "over the context window"
        ),
        evidence={
            "threshold": threshold,
            "offending_splits": offenders,
        },
    )


def audit_dataset(
    loader: DatasetLoader,
    *,
    tokenizer: Tokenizer | None = None,
    context_window: int | None = None,
    exact: bool = True,
    near: bool = True,
    near_threshold: float = 0.9,
    cross_split: bool = True,
    cross_split_threshold: float = 0.9,
    obfuscation: bool = True,
    min_neg_pos_ratio: float = DEFAULT_MIN_NEG_POS_RATIO,
    max_neg_pos_ratio: float = DEFAULT_MAX_NEG_POS_RATIO,
    pct_over_context_threshold: float = DEFAULT_PCT_OVER_CONTEXT_THRESHOLD,
) -> DataAudit:
    """Run the Job-1 dataset integrity gate.

    Thin orchestrator: loads splits via ``loader.load_splits()`` , computes
    a :class:`SplitSummary` per split, runs the reused dedup / leakage
    checks (torch-free TF-IDF backend), evaluates integrity gates, and
    returns a :class:`DataAudit` . Pure except for the gate work; performs
    no IO (call :meth:`DataAudit.write` to persist).

    Parameters
    ----------
    loader : DatasetLoader
        Anything exposing ``load_splits() -> dict[str, EvalSlice]`` (e.g.
        :class:`eval_toolkit.loaders.DataFrameLoader`).
    tokenizer : Callable[[str], Sequence[int]] or None, optional
        Caller-supplied tokenizer for token-length quantiles. This module
        never imports ``transformers``.
    context_window : int or None, optional
        Model context window in tokens. When supplied **with** a tokenizer,
        enables ``pct_over_context_window`` + the ``context_window_fit``
        gate.
    exact : bool, optional
        Run :class:`~eval_toolkit.leakage.ExactDuplicateCheck`
        (within-split). Default ``True`` . Recorded as ``"warning"``
        severity (informational — does not gate ``gate_passed`` ).
    near : bool, optional
        Run :class:`~eval_toolkit.leakage.NearDuplicateCheck`
        (within-split, TF-IDF). Default ``True`` . ``"warning"`` severity
        (informational).
    near_threshold : float, optional
        Similarity threshold for the near-duplicate check. Default ``0.9``.
    cross_split : bool, optional
        Run :class:`~eval_toolkit.leakage.CrossSplitLeakageCheck` . Default
        ``True`` . Requires a split named ``"train"`` ; otherwise skipped
        (the ``no_cross_split_leakage`` gate then passes with a "skipped"
        note).
    cross_split_threshold : float, optional
        Similarity threshold for cross-split leakage. Default ``0.9``.
    obfuscation : bool, optional
        Compute the per-split :class:`~eval_toolkit.eda.obfuscation.
        ObfuscationProfile` (invisible-Unicode / NFKC delta / high-entropy
        runs / ROT13 markers / leetspeak prevalence). Default ``True``.
        Profile-only — does not contribute to ``gate_passed``. Set
        ``False`` to skip (e.g. on very large corpora where the single
        char-walk pass per detector is undesirable).
    min_neg_pos_ratio, max_neg_pos_ratio : float, optional
        Inclusive bounds for the class-balance gate's neg:pos ratio.
        Defaults ``0.1`` / ``10.0`` .
    pct_over_context_threshold : float, optional
        Ceiling for the fraction of rows over the context window. Default
        ``0.05``. Only used when a tokenizer + context window are supplied.

    Returns
    -------
    DataAudit

    Raises
    ------
    ValueError
        If ``context_window`` is supplied without a tokenizer.

    Notes
    -----
    Only **error** -severity gates contribute to ``gate_passed`` . The
    exact / near within-split dedup checks are recorded at ``"warning"``
    severity because exact / lexical near-dupes are common in real corpora
    and rarely invalidate a split *structure*; the genuinely dangerous
    signal — train↔eval leakage — gates via ``no_cross_split_leakage`` .
    """
    splits = loader.load_splits()
    dataset_name = getattr(loader, "name", "") or ""

    summaries: dict[str, SplitSummary] = {
        name: summarize_split(
            slice_,
            tokenizer=tokenizer,
            context_window=context_window,
            compute_obfuscation=obfuscation,
        )
        for name, slice_ in splits.items()
    }

    # --- reused dedup / leakage checks (torch-free TF-IDF backend) ---
    within_checks: list[LeakageCheck] = []
    if exact:
        within_checks.append(ExactDuplicateCheck(severity="warning"))
    if near:
        within_checks.append(
            NearDuplicateCheck(
                threshold=near_threshold,
                strategy=TfidfCosineStrategy(),
                severity="warning",
            )
        )
    leakage_reports: dict[str, LeakageReport] = {}
    if within_checks:
        leakage_reports["within_split"] = run_leakage_checks(within_checks, splits)

    cross_report: LeakageReport | None = None
    train_split = _resolve_train_split(splits) if cross_split else None
    if cross_split and train_split is not None:
        cross_report = run_leakage_checks(
            [
                CrossSplitLeakageCheck(
                    train_split=train_split,
                    threshold=cross_split_threshold,
                    strategy=TfidfCosineStrategy(),
                    severity="error",
                )
            ],
            splits,
        )
        leakage_reports["cross_split"] = cross_report
    elif cross_split and train_split is None:
        _logger.debug(
            "audit_dataset: cross_split requested but no 'train' split found "
            "in %s; skipping cross-split leakage check",
            sorted(splits),
        )

    # --- integrity gates (GateResult per gate) ---
    gates: list[GateResult] = [
        _class_balance_gate(summaries, min_ratio=min_neg_pos_ratio, max_ratio=max_neg_pos_ratio),
        _no_cross_split_leakage_gate(cross_report),
    ]
    if tokenizer is not None and context_window is not None:
        gates.append(_context_window_gate(summaries, threshold=pct_over_context_threshold))

    gate_passed = all(g.passed for g in gates if g.severity == "error")
    n_failed = sum(1 for g in gates if g.severity == "error" and not g.passed)
    gate_summary = (
        f"{len(gates)} gate(s) evaluated; all passed"
        if gate_passed
        else f"{len(gates)} gate(s) evaluated; {n_failed} error gate(s) FAILED"
    )

    audit = DataAudit(
        schema_version=EDA_AUDIT_SCHEMA_VERSION,
        dataset_name=dataset_name,
        run_timestamp=_utc_now_iso8601(),
        split_summaries=summaries,
        leakage_reports=leakage_reports,
        gates=gates,
        gate_passed=gate_passed,
        gate_summary=gate_summary,
    )
    # Defensive: confirm the artifact is strict-JSON-safe at construction
    # time (cheap; surfaces any non-serializable evidence immediately).
    _ = sanitize_for_json(audit.to_dict())
    _logger.debug(
        "audit_dataset(%r): %d split(s), gate_passed=%s",
        dataset_name,
        len(summaries),
        gate_passed,
    )
    return audit
