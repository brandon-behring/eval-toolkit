"""Leakage detection: pluggable :class:`LeakageCheck` Protocol + reference impls.

A :class:`LeakageCheck` validates a dict of :class:`~eval_toolkit.harness.EvalSlice`
(the shape produced by :class:`~eval_toolkit.loaders.DatasetLoader.load_splits`)
and returns a :class:`LeakageFinding` describing rows / pairs to drop and a
severity. Multiple checks compose into a :class:`LeakageReport`.

Within-split checks (:class:`ExactDuplicateCheck`, :class:`NearDuplicateCheck`,
:class:`NormalizedFormLeakageCheck`, :class:`LabelConflictCheck`) read one
split at a time. Cross-split checks (:class:`CrossSplitLeakageCheck`,
:class:`GroupLeakageCheck`, :class:`TemporalLeakageCheck`) read multiple
splits at once. The contract is uniform — every check takes a
``Mapping[str, EvalSlice]``.

Two integration paths:

- Standalone :func:`run_leakage_checks` for offline / CI use.
- Pass ``leakage_checks=[...]`` directly to
  :func:`eval_toolkit.harness.evaluate` for inline enforcement
  (``on_leakage="raise"`` by default).

Either way the report is captured in :class:`~eval_toolkit.manifest.RunManifest`
so even non-failing runs are auditable.

References
----------
.. [1] Kapoor, S. & Narayanan, A. "Leakage and the reproducibility crisis in
       machine-learning-based science." Patterns 4(9), 2023.
       https://arxiv.org/abs/2207.07048
.. [2] PI_HackAPrompt_SQuAD (2025): encoding-obfuscated dupes detect at 21.3 %
       under naive dedup but achieve 76.2 % attack success rate, motivating
       the :class:`NormalizedFormLeakageCheck` reference impl.
       https://arxiv.org/html/2505.04806v1
"""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from eval_toolkit.harness import EvalSlice
from eval_toolkit.text_dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    ExactNormalizedHashStrategy,
    SimilarityStrategy,
    TfidfCosineStrategy,
    cross_dedup,
    cross_dedup_pairs,
    near_dedup,
    sha256_text,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "CrossSplitLeakageCheck",
    "ExactDuplicateCheck",
    "GroupLeakageCheck",
    "LabelConflictCheck",
    "LeakageCheck",
    "LeakageFinding",
    "LeakageReport",
    "NearDuplicateCheck",
    "NormalizedFormLeakageCheck",
    "Severity",
    "TemporalLeakageCheck",
    "Versioned",
    "run_leakage_checks",
]

Severity = Literal["error", "warning", "info"]


@runtime_checkable
class Versioned(Protocol):
    """Anything exposing a ``version: str`` attribute.

    Used by :class:`~eval_toolkit.manifest.RunManifest` to capture per-object
    versions of any Tier-2 implementation (Scorer, LeakageCheck, Splitter,
    ThresholdSelector, DatasetLoader). Mirrors the lm-evaluation-harness
    ``VERSION`` field pattern, which invalidates cross-version metric
    comparisons. Opt-in: implementations are not required to set ``version``.
    """

    @property
    def version(self) -> str:  # pragma: no cover
        """Stable version string for this implementation."""
        ...


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """A single :class:`LeakageCheck`'s finding.

    Mirrors the audit-trail discipline of
    :class:`eval_toolkit.text_dedup.DedupReport` but is composable across
    multiple checks via :class:`LeakageReport`.

    Parameters
    ----------
    check_name : str
        Stable identifier for the check (e.g. ``"ExactDuplicateCheck"``).
    severity : {"error", "warning", "info"}
        Gating semantics for ``on_leakage="raise"``: only ``"error"``
        findings raise; warnings and info are recorded but don't fail.
    drop_indices : dict[str, list[int]] or None
        Per-split row indices the check recommends dropping. ``None``
        (v0.18.0) signals a pair-count finding that has not localized
        rows to drop — useful for emitters that only know
        ``n_affected`` (e.g. V4's SHA256 disjointness audit and other
        pair-tally audits). Distinguishes "no rows to drop because the
        check found nothing" (``{}``) from "this check doesn't track
        rows" (``None``).
    evidence : dict[str, object]
        Check-specific structured evidence (e.g. dropped-pairs lists,
        violating group ids, time-boundary intervals).
    message : str
        Human-readable summary; appears in error messages and logs.
    n_affected : int
        Total rows affected (sum across splits) — quick-scan field.
    """

    check_name: str
    severity: Severity
    drop_indices: dict[str, list[int]] | None
    evidence: dict[str, object]
    message: str
    n_affected: int

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation for the manifest."""
        drop: object = dict(self.drop_indices) if self.drop_indices is not None else None
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "drop_indices": drop,
            "evidence": dict(self.evidence),
            "message": self.message,
            "n_affected": self.n_affected,
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Aggregate of one or more :class:`LeakageCheck` results.

    Parameters
    ----------
    findings : list[LeakageFinding]
        One entry per check that ran. Empty findings list = clean run.
    """

    findings: list[LeakageFinding] = field(default_factory=list)

    def has_errors(self) -> bool:
        """True iff any finding is ``"error"`` severity AND actually flagged rows.

        A clean run produces error-severity findings with ``n_affected=0``
        ("the check ran and found nothing") — those don't gate the run.
        """
        return any(f.severity == "error" and f.n_affected > 0 for f in self.findings)

    def errors(self) -> list[LeakageFinding]:
        """Subset of findings with severity ``"error"`` AND ``n_affected > 0``."""
        return [f for f in self.findings if f.severity == "error" and f.n_affected > 0]

    def warnings(self) -> list[LeakageFinding]:
        """Subset of findings with severity ``"warning"`` AND ``n_affected > 0``."""
        return [f for f in self.findings if f.severity == "warning" and f.n_affected > 0]

    def merged_drop_indices(self) -> dict[str, list[int]]:
        """Union of all findings' ``drop_indices``, sorted, deduped per split.

        Skips findings whose ``drop_indices`` is ``None`` (pair-tally
        findings that don't localize rows). See v0.18.0 changelog.
        """
        merged: dict[str, set[int]] = defaultdict(set)
        for f in self.findings:
            if f.drop_indices is None:
                continue
            for split_name, idxs in f.drop_indices.items():
                merged[split_name].update(idxs)
        return {name: sorted(idxs) for name, idxs in merged.items()}

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation for the manifest."""
        return {"findings": [f.to_dict() for f in self.findings]}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LeakageCheck(Protocol):
    """A pluggable validator over a dict of splits.

    All checks take the same shape — a ``Mapping[str, EvalSlice]``, the
    output of :class:`~eval_toolkit.loaders.DatasetLoader.load_splits`.
    Within-split checks ignore non-target keys; cross-split checks read
    multiple. Pure: no IO, no mutation.

    Attributes
    ----------
    name : str
        Stable identifier for the check, written into the
        :class:`LeakageFinding` and used for downstream filtering.
    """

    name: str

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:  # pragma: no cover
        """Validate the splits and return one :class:`LeakageFinding`."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggressive_normalize(text: str) -> str:
    """NFKC + zero-width / control / Symbol-Other strip + lowercase.

    Used by :class:`NormalizedFormLeakageCheck` to detect encoding-obfuscated
    duplicates that naive lowercase + whitespace dedup misses (the dominant
    unfixed leakage class in prompt-injection corpora — see
    [PI_HackAPrompt_SQuAD 2025]).
    """
    nfkc = unicodedata.normalize("NFKC", text)
    # Drop control chars (Cc, Cf — zero-width formatters), surrogate (Cs),
    # private-use (Co), and Symbol-Other (So — most emoji + decorative pads).
    drop_categories = {"Cc", "Cf", "Cs", "Co", "So"}
    cleaned = "".join(c for c in nfkc if unicodedata.category(c) not in drop_categories)
    # Collapse whitespace + lowercase
    return " ".join(cleaned.split()).lower()


def _select_targets(
    splits: Mapping[str, EvalSlice], target_splits: Sequence[str] | None
) -> list[str]:
    """Return the split names to scan; default = all keys in insertion order."""
    if target_splits is None:
        return list(splits.keys())
    missing = [name for name in target_splits if name not in splits]
    if missing:
        raise KeyError(f"target_splits not in splits: {missing}")
    return list(target_splits)


# ---------------------------------------------------------------------------
# Within-split checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactDuplicateCheck:
    """Within-split exact-duplicate detection.

    Wraps :class:`eval_toolkit.text_dedup.ExactNormalizedHashStrategy`
    (whitespace-normalized SHA-256 buckets). Severity defaults to
    ``"warning"`` because exact dupes are common in real corpora; opt into
    ``"error"`` for strict-mode CI gates.

    Parameters
    ----------
    target_splits : sequence of str or None, optional
        Splits to scan. ``None`` = all splits.
    severity : {"error", "warning", "info"}, optional
        Gating severity. Default ``"warning"``.
    """

    target_splits: Sequence[str] | None = None
    severity: Severity = "warning"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "ExactDuplicateCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Drop exact-normalized duplicates within each target split."""
        targets = _select_targets(splits, self.target_splits)
        drop: dict[str, list[int]] = {}
        evidence_pairs: dict[str, list[tuple[int, int, float]]] = {}
        n_affected = 0
        for split_name in targets:
            slice_ = splits[split_name]
            texts = slice_.features
            if len(texts) <= 1:
                continue
            report = near_dedup(
                texts,
                threshold=0.5,  # any positive threshold; hash strategy returns 0.0/1.0
                strategy=ExactNormalizedHashStrategy(),
            )
            dropped = [pair[0] for pair in report.dropped_pairs]
            if dropped:
                drop[split_name] = sorted(set(dropped))
                evidence_pairs[split_name] = list(report.dropped_pairs)
                n_affected += len(drop[split_name])
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices=drop,
            evidence={"dropped_pairs_by_split": evidence_pairs},
            message=(
                f"exact-duplicate dedup affected {n_affected} rows across " f"{len(drop)} split(s)"
                if n_affected
                else "no exact duplicates found"
            ),
            n_affected=n_affected,
        )


@dataclass(frozen=True, slots=True)
class NearDuplicateCheck:
    """Within-split near-duplicate detection via a pluggable similarity strategy.

    Default strategy is :class:`eval_toolkit.text_dedup.TfidfCosineStrategy`
    (lexical n-gram TF-IDF). Pass an :class:`EmbeddingCosineStrategy` for
    semantic dedup or :class:`MinHashLSHStrategy` for cheap approximate dedup.

    Parameters
    ----------
    threshold : float, optional
        Similarity threshold in (0, 1). Default 0.9.
    strategy : SimilarityStrategy or None, optional
        Backend. ``None`` instantiates :class:`TfidfCosineStrategy`.
    target_splits : sequence of str or None, optional
        Splits to scan. ``None`` = all.
    severity : {"error", "warning", "info"}, optional
        Default ``"warning"``.
    label_aware : bool, optional
        When ``True``, :meth:`validate_label_split` emits two findings per
        check — one for same-label pairs and one for cross-label pairs —
        with their own severities (see ``severity_same_label`` /
        ``severity_cross_label``). Cross-label near-duplicates within a
        split are a label-noise signal (the same text carries conflicting
        supervision) and typically warrant a stricter severity. Default
        ``False`` preserves the single-finding contract.
    severity_same_label : {"error", "warning", "info"}, optional
        Severity for the same-label finding when ``label_aware=True``.
        Default ``"warning"``.
    severity_cross_label : {"error", "warning", "info"}, optional
        Severity for the cross-label finding when ``label_aware=True``.
        Default ``"error"``.
    """

    threshold: float = DEFAULT_DEDUP_THRESHOLD
    strategy: SimilarityStrategy | None = None
    target_splits: Sequence[str] | None = None
    severity: Severity = "warning"
    label_aware: bool = False
    severity_same_label: Severity = "warning"
    severity_cross_label: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "NearDuplicateCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Drop near-duplicates per the active strategy within each target split."""
        targets = _select_targets(splits, self.target_splits)
        active = self.strategy if self.strategy is not None else TfidfCosineStrategy()
        drop: dict[str, list[int]] = {}
        evidence_pairs: dict[str, list[tuple[int, int, float]]] = {}
        n_affected = 0
        for split_name in targets:
            slice_ = splits[split_name]
            texts = slice_.features
            if len(texts) <= 1:
                continue
            report = near_dedup(texts, threshold=self.threshold, strategy=active)
            dropped = [pair[0] for pair in report.dropped_pairs]
            if dropped:
                drop[split_name] = sorted(set(dropped))
                evidence_pairs[split_name] = list(report.dropped_pairs)
                n_affected += len(drop[split_name])
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices=drop,
            evidence={
                "threshold": self.threshold,
                "strategy": type(active).__name__,
                "dropped_pairs_by_split": evidence_pairs,
            },
            message=(
                f"near-duplicate dedup affected {n_affected} rows "
                f"(threshold={self.threshold:.2f})"
                if n_affected
                else f"no near-duplicates above threshold={self.threshold:.2f}"
            ),
            n_affected=n_affected,
        )

    def validate_label_split(
        self, splits: Mapping[str, EvalSlice]
    ) -> tuple[LeakageFinding, LeakageFinding]:
        """Emit (same_label, cross_label) findings for the label-aware mode.

        Pairs each near-duplicate pair within a split by whether the two
        ends share the slice's ``y_true`` label. Requires the slice to
        carry binary labels (eval-toolkit's standard EvalSlice contract).
        Always available regardless of ``label_aware`` value — callers
        opt in by invoking this method rather than :meth:`validate`.
        """
        targets = _select_targets(splits, self.target_splits)
        active = self.strategy if self.strategy is not None else TfidfCosineStrategy()
        same_drop: dict[str, list[int]] = {}
        cross_drop: dict[str, list[int]] = {}
        same_pairs_by_split: dict[str, list[tuple[int, int, float]]] = {}
        cross_pairs_by_split: dict[str, list[tuple[int, int, float]]] = {}
        n_same = 0
        n_cross = 0
        for split_name in targets:
            slice_ = splits[split_name]
            texts = slice_.features
            if len(texts) <= 1:
                continue
            labels = slice_.y_true
            report = near_dedup(texts, threshold=self.threshold, strategy=active)
            same_pairs: list[tuple[int, int, float]] = []
            cross_pairs: list[tuple[int, int, float]] = []
            for i, j, sim in report.dropped_pairs:
                if labels[i] == labels[j]:
                    same_pairs.append((i, j, sim))
                else:
                    cross_pairs.append((i, j, sim))
            if same_pairs:
                same_drop[split_name] = sorted({p[0] for p in same_pairs})
                same_pairs_by_split[split_name] = same_pairs
                n_same += len(same_drop[split_name])
            if cross_pairs:
                cross_drop[split_name] = sorted({p[0] for p in cross_pairs})
                cross_pairs_by_split[split_name] = cross_pairs
                n_cross += len(cross_drop[split_name])
        same_finding = LeakageFinding(
            check_name=f"{self.name}.same_label",
            severity=self.severity_same_label,
            drop_indices=same_drop,
            evidence={
                "threshold": self.threshold,
                "strategy": type(active).__name__,
                "dropped_pairs_by_split": same_pairs_by_split,
                "label_polarity": "same",
            },
            message=(
                f"same-label near-duplicates affected {n_same} rows "
                f"(threshold={self.threshold:.2f})"
                if n_same
                else f"no same-label near-duplicates above threshold={self.threshold:.2f}"
            ),
            n_affected=n_same,
        )
        cross_finding = LeakageFinding(
            check_name=f"{self.name}.cross_label",
            severity=self.severity_cross_label,
            drop_indices=cross_drop,
            evidence={
                "threshold": self.threshold,
                "strategy": type(active).__name__,
                "dropped_pairs_by_split": cross_pairs_by_split,
                "label_polarity": "cross",
            },
            message=(
                f"cross-label near-duplicates affected {n_cross} rows "
                f"(threshold={self.threshold:.2f}) — label-conflict signal"
                if n_cross
                else f"no cross-label near-duplicates above threshold={self.threshold:.2f}"
            ),
            n_affected=n_cross,
        )
        return same_finding, cross_finding


@dataclass(frozen=True, slots=True)
class NormalizedFormLeakageCheck:
    """Encoding-obfuscated duplicate detection.

    NFKC + zero-width / Symbol-Other strip + whitespace collapse + lowercase
    before hashing. Catches the dominant unfixed leakage class in
    prompt-injection corpora (encoding-obfuscated dupes detect at 21.3 %
    under naive dedup but achieve 76.2 % attack success rate per
    [PI_HackAPrompt_SQuAD 2025]). Severity defaults to ``"error"`` —
    encoding-obfuscated overlap is dangerous.

    Parameters
    ----------
    target_splits : sequence of str or None, optional
        Splits to scan. ``None`` = all.
    severity : {"error", "warning", "info"}, optional
        Default ``"error"`` (this is the obfuscation-aware variant — opt
        out via ``"warning"`` only when consumer trusts upstream cleaning).
    """

    target_splits: Sequence[str] | None = None
    severity: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "NormalizedFormLeakageCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Drop rows whose aggressively-normalized text collides with another."""
        targets = _select_targets(splits, self.target_splits)
        drop: dict[str, list[int]] = {}
        collisions: dict[str, list[tuple[int, int]]] = {}
        n_affected = 0
        for split_name in targets:
            slice_ = splits[split_name]
            texts = slice_.features
            if len(texts) <= 1:
                continue
            seen: dict[str, int] = {}
            split_drops: list[int] = []
            split_collisions: list[tuple[int, int]] = []
            for i, text in enumerate(texts):
                key = sha256_text(_aggressive_normalize(text), normalize=False)
                if key in seen:
                    split_drops.append(i)
                    split_collisions.append((i, seen[key]))
                else:
                    seen[key] = i
            if split_drops:
                drop[split_name] = sorted(set(split_drops))
                collisions[split_name] = split_collisions
                n_affected += len(drop[split_name])
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices=drop,
            evidence={"collisions_by_split": collisions},
            message=(
                f"encoding-obfuscated duplicates: {n_affected} rows collide "
                f"after NFKC / zero-width / Symbol-Other strip"
                if n_affected
                else "no encoding-obfuscated duplicates found"
            ),
            n_affected=n_affected,
        )


@dataclass(frozen=True, slots=True)
class LabelConflictCheck:
    """Cross-source label conflicts: same text, different labels across splits.

    The conflict resolution that ``prompt-injection-sdd`` and
    ``prompt_injection_detector`` reimplement today (~50 LOC each). Severity
    defaults to ``"error"``: the same prompt receiving different labels in
    train vs. test poisons evaluation regardless of which label is "correct".

    Parameters
    ----------
    target_splits : sequence of str or None, optional
        Splits to compare. ``None`` = all keys; pairs (a, b) with a < b are
        scanned. Use ``("train", "test")`` for the most common case.
    severity : {"error", "warning", "info"}, optional
        Default ``"error"``.
    """

    target_splits: Sequence[str] | None = None
    severity: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "LabelConflictCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Find rows whose normalized text appears with different labels."""
        targets = _select_targets(splits, self.target_splits)
        # Build text -> {(split, idx, label)} lookup
        text_to_rows: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for split_name in targets:
            slice_ = splits[split_name]
            texts = slice_.features
            labels = slice_.y_true
            for i, text in enumerate(texts):
                key = sha256_text(text, normalize=True)
                text_to_rows[key].append((split_name, i, int(labels[i])))
        # Collect rows where the same key carries multiple distinct labels
        drop: dict[str, list[int]] = defaultdict(list)
        conflicts: list[dict[str, object]] = []
        for rows in text_to_rows.values():
            distinct_labels = {label for _, _, label in rows}
            if len(distinct_labels) > 1:
                conflicts.append(
                    {
                        "rows": [{"split": s, "index": i, "label": lbl} for s, i, lbl in rows],
                        "labels": sorted(distinct_labels),
                    }
                )
                for split_name, idx, _ in rows:
                    drop[split_name].append(idx)
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices={k: sorted(set(v)) for k, v in drop.items()},
            evidence={"conflicts": conflicts},
            message=(
                f"{len(conflicts)} text(s) carry conflicting labels across splits"
                if conflicts
                else "no cross-split label conflicts"
            ),
            n_affected=sum(len(v) for v in drop.values()),
        )


# ---------------------------------------------------------------------------
# Cross-split checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossSplitLeakageCheck:
    """Train↔eval near-duplicate leakage (the genuinely dangerous one).

    Wraps :func:`eval_toolkit.text_dedup.cross_dedup` to drop rows in any
    eval split that are near-duplicate to any row in the training split.
    Severity is ``"error"``: cross-split dup is the canonical leakage that
    inflates eval metrics.

    Parameters
    ----------
    train_split : str, optional
        Reference split (the corpus the eval rows must NOT echo). Default ``"train"``.
    eval_splits : sequence of str or None, optional
        Splits to scrub. ``None`` = every key except ``train_split``.
    threshold : float, optional
        Similarity threshold in (0, 1). Default 0.9.
    strategy : SimilarityStrategy or None, optional
        Backend. ``None`` instantiates :class:`TfidfCosineStrategy`.
    severity : {"error", "warning", "info"}, optional
        Default ``"error"``.
    label_aware : bool, optional
        When ``True``, :meth:`validate_label_split` emits two findings — one
        for eval rows whose matched train neighbor shares the eval row's
        label (same-label leakage = memorization risk) and one for
        cross-label matches (supervision conflict + memorization). Default
        ``False`` preserves the single-finding contract.
    severity_same_label : {"error", "warning", "info"}, optional
        Severity for the same-label finding when ``label_aware=True``.
        Default ``"warning"``.
    severity_cross_label : {"error", "warning", "info"}, optional
        Severity for the cross-label finding when ``label_aware=True``.
        Default ``"error"``.
    """

    train_split: str = "train"
    eval_splits: Sequence[str] | None = None
    threshold: float = DEFAULT_DEDUP_THRESHOLD
    strategy: SimilarityStrategy | None = None
    severity: Severity = "error"
    label_aware: bool = False
    severity_same_label: Severity = "warning"
    severity_cross_label: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "CrossSplitLeakageCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Find eval-side rows that near-duplicate any train-side row.

        Raises
        ------
        KeyError
            If ``self.train_split`` is missing from ``splits``, or if any
            entry in ``self.eval_splits`` is missing.
        """
        if self.train_split not in splits:
            raise KeyError(f"train_split {self.train_split!r} not in splits")
        eval_targets = (
            list(self.eval_splits)
            if self.eval_splits is not None
            else [k for k in splits if k != self.train_split]
        )
        train_texts = splits[self.train_split].features
        active = self.strategy if self.strategy is not None else TfidfCosineStrategy()
        drop: dict[str, list[int]] = {}
        n_affected = 0
        for eval_name in eval_targets:
            if eval_name not in splits:
                raise KeyError(f"eval split {eval_name!r} not in splits")
            eval_texts = splits[eval_name].features
            if not train_texts or not eval_texts:
                continue
            kept = cross_dedup(
                train_texts,
                eval_texts,
                threshold=self.threshold,
                strategy=active,
            )
            kept_set = set(kept)
            dropped = [i for i in range(len(eval_texts)) if i not in kept_set]
            if dropped:
                drop[eval_name] = dropped
                n_affected += len(dropped)
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices=drop,
            evidence={
                "train_split": self.train_split,
                "eval_splits": eval_targets,
                "threshold": self.threshold,
                "strategy": type(active).__name__,
            },
            message=(
                f"cross-split leakage: {n_affected} eval rows near-duplicate "
                f"to {self.train_split!r} (threshold={self.threshold:.2f})"
                if n_affected
                else f"no cross-split leakage above threshold={self.threshold:.2f}"
            ),
            n_affected=n_affected,
        )

    def validate_label_split(
        self, splits: Mapping[str, EvalSlice]
    ) -> tuple[LeakageFinding, LeakageFinding]:
        """Emit (same_label, cross_label) findings for cross-split leakage.

        Pairs each cross-split near-duplicate by whether the matched train
        and eval rows carry the same label. Uses
        :func:`cross_dedup_pairs` (v0.17.0) so the train-side index of each
        match is preserved. Same-label = memorization risk; cross-label =
        supervision conflict + memorization.

        Raises
        ------
        KeyError
            If ``self.train_split`` is missing from ``splits``, or if any
            entry in ``self.eval_splits`` is missing.
        """
        if self.train_split not in splits:
            raise KeyError(f"train_split {self.train_split!r} not in splits")
        eval_targets = (
            list(self.eval_splits)
            if self.eval_splits is not None
            else [k for k in splits if k != self.train_split]
        )
        train_slice = splits[self.train_split]
        train_texts = train_slice.features
        train_labels = train_slice.y_true
        active = self.strategy if self.strategy is not None else TfidfCosineStrategy()
        same_drop: dict[str, list[int]] = {}
        cross_drop: dict[str, list[int]] = {}
        same_pairs_by_split: dict[str, list[tuple[int, int, float]]] = {}
        cross_pairs_by_split: dict[str, list[tuple[int, int, float]]] = {}
        n_same = 0
        n_cross = 0
        for eval_name in eval_targets:
            if eval_name not in splits:
                raise KeyError(f"eval split {eval_name!r} not in splits")
            eval_slice = splits[eval_name]
            eval_texts = eval_slice.features
            eval_labels = eval_slice.y_true
            if not train_texts or not eval_texts:
                continue
            pairs = cross_dedup_pairs(
                train_texts,
                eval_texts,
                threshold=self.threshold,
                strategy=active,
            )
            same_pairs: list[tuple[int, int, float]] = []
            cross_pairs: list[tuple[int, int, float]] = []
            for eval_idx, train_idx, sim in pairs:
                if eval_labels[eval_idx] == train_labels[train_idx]:
                    same_pairs.append((eval_idx, train_idx, sim))
                else:
                    cross_pairs.append((eval_idx, train_idx, sim))
            if same_pairs:
                same_drop[eval_name] = sorted({p[0] for p in same_pairs})
                same_pairs_by_split[eval_name] = same_pairs
                n_same += len(same_drop[eval_name])
            if cross_pairs:
                cross_drop[eval_name] = sorted({p[0] for p in cross_pairs})
                cross_pairs_by_split[eval_name] = cross_pairs
                n_cross += len(cross_drop[eval_name])
        evidence_base: dict[str, object] = {
            "train_split": self.train_split,
            "eval_splits": eval_targets,
            "threshold": self.threshold,
            "strategy": type(active).__name__,
        }
        same_finding = LeakageFinding(
            check_name=f"{self.name}.same_label",
            severity=self.severity_same_label,
            drop_indices=same_drop,
            evidence={
                **evidence_base,
                "pairs_by_split": same_pairs_by_split,
                "label_polarity": "same",
            },
            message=(
                f"same-label cross-split leakage: {n_same} eval rows near-duplicate "
                f"to {self.train_split!r} sharing label (threshold={self.threshold:.2f})"
                if n_same
                else f"no same-label cross-split leakage above threshold={self.threshold:.2f}"
            ),
            n_affected=n_same,
        )
        cross_finding = LeakageFinding(
            check_name=f"{self.name}.cross_label",
            severity=self.severity_cross_label,
            drop_indices=cross_drop,
            evidence={
                **evidence_base,
                "pairs_by_split": cross_pairs_by_split,
                "label_polarity": "cross",
            },
            message=(
                f"cross-label cross-split leakage: {n_cross} eval rows near-duplicate "
                f"to {self.train_split!r} with opposing label "
                f"(threshold={self.threshold:.2f}) — supervision conflict"
                if n_cross
                else f"no cross-label cross-split leakage above threshold={self.threshold:.2f}"
            ),
            n_affected=n_cross,
        )
        return same_finding, cross_finding


@dataclass(frozen=True, slots=True)
class GroupLeakageCheck:
    """Group-id leakage: a single group spans multiple splits.

    Common failure mode in clinical / user-level / source-level evals where
    rows from the same patient / user / source need to stay in the same fold.
    Requires every target split to carry a ``group_col``-named column.
    Severity ``"error"``.

    Parameters
    ----------
    group_col : str
        Column name carrying the group id in every target slice's dataframe.
    target_splits : sequence of str or None, optional
        Splits to compare. ``None`` = all keys.
    severity : {"error", "warning", "info"}, optional
        Default ``"error"``.
    """

    group_col: str
    target_splits: Sequence[str] | None = None
    severity: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "GroupLeakageCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Find group ids appearing in more than one target split.

        Raises
        ------
        KeyError
            If any target split's dataframe lacks ``self.group_col``.
        """
        targets = _select_targets(splits, self.target_splits)
        group_to_splits: dict[object, set[str]] = defaultdict(set)
        group_rows: dict[tuple[object, str], list[int]] = defaultdict(list)
        for split_name in targets:
            slice_ = splits[split_name]
            if self.group_col not in slice_.df.columns:
                raise KeyError(f"split {split_name!r}: missing group column {self.group_col!r}")
            for i, gid in enumerate(slice_.df[self.group_col].tolist()):
                group_to_splits[gid].add(split_name)
                group_rows[(gid, split_name)].append(i)
        offending = {gid: spls for gid, spls in group_to_splits.items() if len(spls) > 1}
        drop: dict[str, list[int]] = defaultdict(list)
        for (gid, split_name), idxs in group_rows.items():
            if gid in offending:
                drop[split_name].extend(idxs)
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices={k: sorted(set(v)) for k, v in drop.items()},
            evidence={
                "group_col": self.group_col,
                "violating_groups": {str(gid): sorted(spls) for gid, spls in offending.items()},
            },
            message=(
                f"{len(offending)} group(s) span multiple splits " f"(group_col={self.group_col!r})"
                if offending
                else f"no group-id leakage on {self.group_col!r}"
            ),
            n_affected=sum(len(v) for v in drop.values()),
        )


@dataclass(frozen=True, slots=True)
class TemporalLeakageCheck:
    """Temporal ordering invariant: every earlier split's max(time) ≤ next's min(time).

    Required for honest time-series eval — failure means the model could
    train on data from the future and "validate" on data from the past.
    Severity ``"error"``.

    Parameters
    ----------
    time_col : str
        Column name carrying a sortable timestamp / date / index in every
        target split's dataframe.
    split_order : sequence of str
        Required temporal order, e.g. ``("train", "val", "test")``. The
        check asserts that for every adjacent pair ``(a, b)``,
        ``max(a[time_col]) <= min(b[time_col])``.
    severity : {"error", "warning", "info"}, optional
        Default ``"error"``.
    """

    time_col: str
    split_order: Sequence[str]
    severity: Severity = "error"

    @property
    def name(self) -> str:
        """Stable check identifier."""
        return "TemporalLeakageCheck"

    def validate(self, splits: Mapping[str, EvalSlice]) -> LeakageFinding:
        """Check the temporal ordering invariant pairwise.

        Raises
        ------
        KeyError
            If any entry in ``self.split_order`` is missing from ``splits``,
            or if any target split's dataframe lacks ``self.time_col``.
        """
        for split_name in self.split_order:
            if split_name not in splits:
                raise KeyError(f"split {split_name!r} (in split_order) not in splits")
            if self.time_col not in splits[split_name].df.columns:
                raise KeyError(f"split {split_name!r}: missing time column {self.time_col!r}")
        violations: list[dict[str, object]] = []
        drop: dict[str, list[int]] = defaultdict(list)
        for earlier, later in zip(self.split_order, self.split_order[1:], strict=False):
            t_earlier = splits[earlier].df[self.time_col]
            t_later = splits[later].df[self.time_col]
            if t_earlier.empty or t_later.empty:
                continue
            max_earlier = t_earlier.max()
            min_later = t_later.min()
            if max_earlier > min_later:
                # Drop offending rows from the LATER split (those that come
                # before max_earlier).
                bad = t_later[t_later <= max_earlier].index.tolist()
                # Map dataframe index -> positional index in slice features
                later_df = splits[later].df.reset_index(drop=False)
                positional = [int(p) for p, dfi in enumerate(later_df["index"]) if dfi in bad]
                drop[later].extend(positional)
                violations.append(
                    {
                        "earlier_split": earlier,
                        "later_split": later,
                        "max_earlier": str(max_earlier),
                        "min_later": str(min_later),
                        "n_offending_in_later": len(positional),
                    }
                )
        return LeakageFinding(
            check_name=self.name,
            severity=self.severity,
            drop_indices={k: sorted(set(v)) for k, v in drop.items()},
            evidence={
                "time_col": self.time_col,
                "split_order": list(self.split_order),
                "violations": violations,
            },
            message=(
                f"temporal ordering violated at {len(violations)} boundary/boundaries"
                if violations
                else f"temporal ordering OK on {self.time_col!r}"
            ),
            n_affected=sum(len(v) for v in drop.values()),
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_leakage_checks(
    checks: Sequence[LeakageCheck], splits: Mapping[str, EvalSlice]
) -> LeakageReport:
    """Run a sequence of :class:`LeakageCheck` over ``splits`` and aggregate.

    Pure: no IO, no mutation. Each check is run sequentially in the order
    given. Use this standalone for offline / CI audits, or pass the same
    ``checks`` list to :func:`eval_toolkit.harness.evaluate` for inline
    enforcement.

    Parameters
    ----------
    checks : sequence of LeakageCheck
    splits : Mapping[str, EvalSlice]
        Output of :class:`~eval_toolkit.loaders.DatasetLoader.load_splits`.

    Returns
    -------
    LeakageReport

    Raises
    ------
    KeyError
        If a check references a split / column not present in ``splits``.

    Examples
    --------
    >>> import pandas as pd
    >>> from eval_toolkit.harness import EvalSlice
    >>> from eval_toolkit.leakage import (
    ...     ExactDuplicateCheck, run_leakage_checks,
    ... )
    >>> df_train = pd.DataFrame({"text": ["a", "a", "b"], "label": [0, 0, 1]})
    >>> df_test = pd.DataFrame({"text": ["c", "d"], "label": [0, 1]})
    >>> splits = {
    ...     "train": EvalSlice(name="train", df=df_train),
    ...     "test": EvalSlice(name="test", df=df_test),
    ... }
    >>> report = run_leakage_checks([ExactDuplicateCheck()], splits)
    >>> report.has_errors()
    False
    >>> sum(f.n_affected for f in report.findings)
    1
    """
    findings: list[LeakageFinding] = []
    for check in checks:
        finding = check.validate(splits)
        findings.append(finding)
        _logger.debug(
            "leakage check %s: severity=%s n_affected=%d",
            finding.check_name,
            finding.severity,
            finding.n_affected,
        )
    _logger.debug(
        "run_leakage_checks completed: %d checks, %d findings (%d errors)",
        len(checks),
        len(findings),
        sum(1 for f in findings if f.severity == "error"),
    )
    return LeakageReport(findings=findings)


# Suppress unused-import warning: np is referenced in Examples blocks.
# pandas is imported lazily inside the doctest itself (see Examples above).
_ = (np,)
