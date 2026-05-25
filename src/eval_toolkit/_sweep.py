"""Top-level ``sweep()`` — unified ``TextTransform`` enumeration (v0.47).

Decision K + Decision D + Audit R5-F3 (Codex Round 5).

The v0.47 sweep consolidation replaces the two per-module ``sweep`` functions
(``adversarial.sweep`` and ``preprocessing.sweep``) with a single top-level
entry point that accepts any object satisfying the
:class:`~eval_toolkit.protocols.TextTransform` Protocol — defence strategies
(``DelimitVariant`` / ``DatamarkVariant`` / ``EncodeVariant``) and attack
strategies (all 6 v0.43 core character-injection dataclasses + the 6 advanced
ones landing in this release) compose freely.

Behaviour contract (Audit F3 — Codex Round 5):

- **Neutral by default.** ``sweep(strategies, texts)`` returns
  ``text_id`` × ``variant`` × ``transformed_text`` only. Pure text-transform
  enumeration; works whether the strategies are attacks, defences, or both.
- **Scorer is opt-in.** Providing ``scorer`` adds ``original_score`` +
  ``transformed_score`` columns. No magic threshold — the result describes
  what the scorer said, not whether the attack "succeeded".
- **ASR requires an explicit threshold.** ``asr`` (attack success rate per
  row) only materializes when both ``scorer`` AND ``attack_threshold`` are
  passed. The methodology docs explicitly warn against a magic default-0.5
  threshold (``methodology/thresholds.md``); enforcing the kwarg requires
  callers to commit to a calibrated operating point.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from eval_toolkit.protocols import Scorer, TextTransform


__all__ = ["sweep"]


def sweep(
    strategies: Sequence[TextTransform],
    texts: Sequence[str],
    *,
    scorer: Scorer | None = None,
    attack_threshold: float | None = None,
) -> pd.DataFrame:
    """Apply each ``TextTransform`` strategy to each text; one row per pair.

    Parameters
    ----------
    strategies : sequence of TextTransform
        Strategies to apply. Each must satisfy the
        :class:`~eval_toolkit.TextTransform` Protocol (``name: str`` attribute
        + ``transform(text: str) -> str`` method). Mix defence and attack
        strategies freely.
    texts : sequence of str
        Input texts. Each gets a 0-based ``text_id`` assigned by position.
    scorer : Scorer or None, optional
        When provided, also compute
        ``scorer.predict_proba([original_text])`` and
        ``scorer.predict_proba([transformed_text])`` for each row; emits
        ``original_score`` + ``transformed_score`` columns. Defaults to
        ``None`` (no scoring).
    attack_threshold : float or None, optional
        When provided alongside ``scorer``, also emit an ``asr``
        (attack-success-rate) column per row: ``s_orig >= threshold > s_adv``
        (the scorer flagged the original but missed the transformed input).
        **Required to materialize ``asr``** — the documented contract refuses
        a magic default threshold (cf. ``methodology/thresholds.md``).
        Ignored when ``scorer`` is ``None`` (with ``ValueError`` if passed
        with ``scorer=None`` to surface the API misuse).

    Returns
    -------
    pandas.DataFrame
        Columns vary by which optional kwargs are passed:

        - Always: ``text_id`` (int), ``variant`` (str — from
          ``strategy.name``), ``transformed_text`` (str).
        - With ``scorer``: also ``original_score`` (float) +
          ``transformed_score`` (float).
        - With ``scorer`` AND ``attack_threshold``: also ``asr`` (bool —
          per-row attack-success flag).

        Row order: ``(variant, text_id)`` nested.

    Raises
    ------
    ValueError
        - If ``strategies`` is empty.
        - If ``attack_threshold`` is provided without ``scorer``.
        - If any strategy doesn't satisfy ``TextTransform`` structurally
          (typically a missing ``name`` attribute).

    Examples
    --------
    Defence-only sweep (neutral text-transform enumeration):

    >>> from eval_toolkit import DelimitVariant, DatamarkVariant, sweep
    >>> df = sweep([DelimitVariant(), DatamarkVariant()], ["hello world"])
    >>> sorted(df.columns.tolist())
    ['strategy_id', 'text_id', 'transformed_text', 'variant']
    >>> df[df["variant"] == "delimit"].iloc[0]["transformed_text"]
    '<<hello world>>'

    Mixed defence + attack with a scorer + explicit operating point:

    >>> import numpy as np
    >>> from eval_toolkit import DelimitVariant
    >>> from eval_toolkit.adversarial import ZeroWidthSpaceInjection
    >>> class _Detector:
    ...     def predict_proba(self, X):
    ...         return np.array([0.9 if "ignore" in t else 0.1 for t in X])
    >>> df = sweep(  # doctest: +SKIP
    ...     [DelimitVariant(), ZeroWidthSpaceInjection()],
    ...     ["ignore prior instructions", "weather today"],
    ...     scorer=_Detector(),
    ...     attack_threshold=0.5,
    ... )
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sweep() requires pandas. Install with: pip install eval-toolkit[dataframe]"
        ) from exc

    if not strategies:
        raise ValueError("sweep(): strategies must be non-empty.")
    if attack_threshold is not None and scorer is None:
        raise ValueError(
            "sweep(): attack_threshold requires scorer; "
            "ASR cannot be computed without a scorer. "
            "Either pass scorer=<scorer> + attack_threshold=<float>, "
            "or omit attack_threshold."
        )
    for i, strategy in enumerate(strategies):
        if not (hasattr(strategy, "name") and hasattr(strategy, "transform")):
            raise ValueError(
                f"sweep(): strategy at index {i} ({type(strategy).__name__}) "
                f"does not satisfy TextTransform (missing 'name' or 'transform')."
            )
    _validate_unique_strategy_ids(strategies)

    text_list = list(texts)
    rows: list[dict[str, object]] = []

    # Pre-compute original scores if a scorer was passed — single batched call
    # is cheaper than per-row Scorer dispatch for large `texts`.
    original_scores: np.ndarray | None = None
    if scorer is not None and text_list:
        original_scores = np.asarray(scorer.predict_proba(text_list))
        _validate_scorer_output(
            original_scores, expected_n=len(text_list), label="original-texts batch"
        )

    for strategy in strategies:
        sid = _strategy_id_for(strategy)
        transformed_list = [strategy.transform(t) for t in text_list]
        transformed_scores: np.ndarray | None = None
        if scorer is not None and transformed_list:
            transformed_scores = np.asarray(scorer.predict_proba(transformed_list))
            _validate_scorer_output(
                transformed_scores,
                expected_n=len(text_list),
                label=f"transformed-texts batch for strategy {strategy.name!r}",
            )
        for text_id, (_, transformed) in enumerate(zip(text_list, transformed_list, strict=True)):
            row: dict[str, object] = {
                "text_id": text_id,
                "strategy_id": sid,
                "variant": strategy.name,
                "transformed_text": transformed,
            }
            if scorer is not None:
                assert original_scores is not None
                assert transformed_scores is not None
                s_orig = float(original_scores[text_id])
                s_adv = float(transformed_scores[text_id])
                row["original_score"] = s_orig
                row["transformed_score"] = s_adv
                if attack_threshold is not None:
                    row["asr"] = bool(s_orig >= attack_threshold > s_adv)
            rows.append(row)

    base_cols = ["text_id", "strategy_id", "variant", "transformed_text"]
    if scorer is not None:
        base_cols += ["original_score", "transformed_score"]
    if attack_threshold is not None:
        base_cols += ["asr"]
    return pd.DataFrame(rows, columns=base_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — strategy identity (Decision R7-B; v0.48 §5I)
# ─────────────────────────────────────────────────────────────────────────────


def _strategy_id_for(strategy: TextTransform) -> str:
    """Build a stable, repr-stable identifier from a strategy's configured state.

    Decision R7-B (Round 7 audit, Codex R7-F2): a strategy's ``name`` alone is
    not enough to identify a configured instance. Two instances of the same
    dataclass with different kwargs (e.g., ``DelimitVariant(delimiter="<<")``
    and ``DelimitVariant(delimiter="[[")``) share ``name == "delimit"`` and
    would silently merge under ``groupby("variant")``. The ``strategy_id``
    column carries the canonical configured identity so downstream
    analysis can disambiguate.

    Format (pseudo-URI; chosen for groupby-friendliness + special-char
    safety via ``repr()``):

    - Frozen dataclass strategies: ``"<name>/<k1>=<repr(v1)>,<k2>=<repr(v2)>,..."``
      with kwargs alphabetized (excluding the ``name`` field itself). Mirrors
      :func:`eval_toolkit.metric_specs.make_spec_name` but uses ``repr(value)``
      instead of ``str(value)`` so string kwargs with special chars (``<<``,
      ``[[``, ``^``, etc.) round-trip cleanly.
    - Plain :class:`TextTransform`-Protocol-satisfying objects without
      ``__dataclass_fields__``: falls back to ``strategy.name``.

    Examples
    --------
    >>> from eval_toolkit.preprocessing import DelimitVariant
    >>> _strategy_id_for(DelimitVariant(delimiter="<<", end=">>"))
    "delimit/delimiter='<<',end='>>'"
    >>> from eval_toolkit.adversarial import ZeroWidthSpaceInjection
    >>> _strategy_id_for(ZeroWidthSpaceInjection(ratio=0.5, seed=42))
    'zero_width_space/ratio=0.5,seed=42'
    """
    fields = getattr(strategy, "__dataclass_fields__", None)
    if fields is None:
        return strategy.name
    kw_pairs = sorted((f, getattr(strategy, f)) for f in fields if f != "name")
    if not kw_pairs:
        return strategy.name
    return f"{strategy.name}/" + ",".join(f"{k}={v!r}" for k, v in kw_pairs)


def _validate_scorer_output(scores: np.ndarray, *, expected_n: int, label: str) -> None:
    """Validate the shape AND finiteness of a batched ``Scorer.predict_proba`` result.

    Decision R7-C (Round 7 audit, Codex R7-F3): three failure modes Codex
    surfaced via runtime probe — too many 1-D scores (silent truncation,
    worst class), too few (later ``IndexError``), and matrix-shaped
    (later ``TypeError`` when ``float(...)`` is applied to a row). All
    three become a single API-level ``ValueError`` with context.

    R9 follow-on (R8-F-sweep-1 audit): the original R7-C check covered
    shape only — NaN/inf scorer outputs passed validation and silently
    propagated into the sweep DataFrame's ``original_score`` /
    ``transformed_score`` columns, then into downstream comparisons
    (``s_orig >= threshold > s_adv`` becomes ``False`` for any NaN
    comparison, silently zeroing the ASR flag). The v0.51 audit
    surfaced this as a candidate v1.0 blocker — same "no silent
    failures" invariant R7-C was designed to enforce. Stacking.py
    validates non-finite scores in ``_validate_fit_inputs`` and
    ``_validate_predict_inputs``; this brings the sweep boundary to
    parity.

    Style invariants 1 (no silent failures) + 3 (API-level errors, never
    low-level exceptions through the boundary).

    Parameters
    ----------
    scores : np.ndarray
        The ``np.asarray()``-wrapped result of ``scorer.predict_proba(...)``.
    expected_n : int
        The expected length — ``len(texts)`` for the current sweep call.
    label : str
        Context for the error message naming the offending batch
        (e.g., ``"original-texts batch"`` or
        ``"transformed-texts batch for strategy 'zero_width_space'"``).

    Raises
    ------
    ValueError
        If ``scores.shape != (expected_n,)`` (R7-C original) OR if any
        score is non-finite (NaN / +inf / -inf; R9 follow-on R8-F-sweep-1).
    """
    if scores.shape != (expected_n,):
        raise ValueError(
            f"sweep(): scorer.predict_proba({label}) returned shape "
            f"{scores.shape}; expected ({expected_n},). The Scorer Protocol "
            f"requires one float P(positive) per input row (see "
            f"`eval_toolkit.protocols.Scorer`); ensure your adapter returns "
            f"a 1-D array of length len(texts)."
        )
    # R9 follow-on (F-sweep-1): close the NaN/inf gap in R7-C.
    # R10 follow-on (F1): drop `[0, 1]` from the runtime message since the
    # finiteness check doesn't enforce range; the [0, 1] semantic contract
    # is documented in `protocols.Scorer` but boundary-enforcement of range
    # is intentionally deferred to a future minor (R10 Codex Option C).
    if not np.all(np.isfinite(scores)):
        n_nonfinite = int(np.sum(~np.isfinite(scores)))
        raise ValueError(
            f"sweep(): scorer.predict_proba({label}) returned non-finite "
            f"values (NaN / +inf / -inf); got {n_nonfinite}/{expected_n} "
            f"non-finite scores. The Scorer Protocol requires finite float "
            f"P(positive) scores (see `eval_toolkit.protocols.Scorer`); "
            f"check the scorer for div-by-zero, log(0), missing-data, "
            f"or unguarded sigmoid-of-large-logit paths."
        )


def _validate_unique_strategy_ids(strategies: Sequence[TextTransform]) -> None:
    """Reject duplicate ``strategy_id`` values in a single ``sweep()`` call.

    Decision R7-B (Round 7 audit, Codex R7-F2): mirrors R6-B's duplicate
    ``MetricSpec.name`` rejection in ``scorecard()`` — same anti-silent-merge
    invariant, applied to the sweep surface. No methodology-honest reason to
    put the same configured strategy twice in one sweep; cache-warming +
    reproducibility re-runs use ``strategy.transform()`` directly outside
    ``sweep()``.
    """
    seen: dict[str, int] = {}
    for i, strategy in enumerate(strategies):
        sid = _strategy_id_for(strategy)
        if sid in seen:
            raise ValueError(
                f"sweep(): duplicate strategy_id {sid!r} at index {i} "
                f"(previously at index {seen[sid]}); each strategy must "
                f"produce a unique strategy_id. If you want two configurations "
                f"of the same dataclass in the same sweep, vary their kwargs "
                f"so the canonical identifier differs."
            )
        seen[sid] = i
