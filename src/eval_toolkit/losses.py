"""Differentiable losses for prompt-injection detector training.

Implements :class:`RecallAtLowFPR` — the Meta Prompt Guard 2 (PG2) training
recipe, a differentiable approximation of recall-at-fixed-FPR. Optimizes
detector ranking at a constrained operating point (e.g. FPR ≤ 0.01)
rather than the implicit FPR-agnostic posture of cross-entropy.

This module is base-install safe: ``torch`` is soft-imported inside the
class methods. ``pip install eval-toolkit[losses]`` installs torch.
The lazy-import pattern matches the ``[probes]`` precedent (separate
extra so callers wanting only the loss don't have to install
transformers).

The formulation follows the soft-rank approximation described in
Meta's PG2 release notes and similar metric-learning losses (Liu et al.
NeurIPS 2020 family):

1. Compute the empirical FPR-target threshold from the negative-class
   scores in the batch via the ``fpr_target``-th percentile.
2. Smooth the indicator ``I(s_i >= threshold)`` with
   ``sigmoid(beta * (s_i - threshold))`` so gradients flow.
3. Recall@FPR ≈ ``Σ approx_indicator * y / Σ y``; the loss returned is
   ``1 - Recall@FPR``.

References
----------
.. [1] Meta. 2024. "Prompt Guard 2 — release notes & training recipe."
.. [2] Liu, X., et al. 2020. "Black-box ranking under FPR constraints."
       NeurIPS 2020.
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = [
    "RecallAtLowFPR",
]


ReductionMode = Literal["mean", "sum", "none"]


def _require_torch() -> Any:
    """Import torch with a copy-paste install hint if [losses] is missing."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "RecallAtLowFPR requires torch. Install with: pip install eval-toolkit[losses]"
        ) from exc
    return torch


def _build_module_class() -> Any:
    """Build the :class:`RecallAtLowFPR` ``nn.Module`` lazily.

    Defined as a factory so importing :mod:`eval_toolkit.losses` does not
    pull torch at module-import time. The class itself is built on first
    instantiation; the factory caches the class on the module so repeated
    construction is constant-time after the first call.
    """
    torch = _require_torch()
    nn = torch.nn

    # ``nn.Module`` is a runtime-constructed base; mypy can't follow the dynamic
    # class creation. The runtime behavior is correct (nn.Module API + autograd).
    class _RecallAtLowFPR(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(
            self,
            fpr_target: float = 0.01,
            fpr_smoothing_beta: float = 10.0,
            pos_weight: float = 1.0,
            reduction: ReductionMode = "mean",
        ) -> None:
            super().__init__()
            if not 0.0 < fpr_target <= 1.0:
                raise ValueError(f"RecallAtLowFPR: fpr_target must be in (0, 1]; got {fpr_target}")
            if fpr_smoothing_beta <= 0:
                raise ValueError(
                    f"RecallAtLowFPR: fpr_smoothing_beta must be > 0; got {fpr_smoothing_beta}"
                )
            if reduction not in ("mean", "sum", "none"):
                raise ValueError(
                    f"RecallAtLowFPR: reduction must be 'mean'|'sum'|'none'; got {reduction!r}"
                )
            self.fpr_target = float(fpr_target)
            self.fpr_smoothing_beta = float(fpr_smoothing_beta)
            self.pos_weight = float(pos_weight)
            self.reduction = reduction

        def forward(
            self,
            logits: Any,
            labels: Any,
        ) -> Any:
            """Compute the (differentiable) 1 - Recall@FPR loss.

            Parameters
            ----------
            logits : torch.Tensor
                Predicted scores, shape ``(B,)`` or ``(B, 1)``. Higher
                value → higher probability of positive class.
            labels : torch.Tensor
                Binary labels in ``{0, 1}``, shape ``(B,)``.

            Returns
            -------
            torch.Tensor
                Scalar (``reduction="mean"`` or ``"sum"``) or
                per-positive-sample loss (``reduction="none"``).
            """
            scores = logits.squeeze(-1) if logits.dim() == 2 else logits
            if scores.shape != labels.shape:
                raise ValueError(
                    f"RecallAtLowFPR: logits shape {tuple(scores.shape)} != "
                    f"labels shape {tuple(labels.shape)}"
                )

            labels_f = labels.float()
            neg_mask = labels_f < 0.5
            pos_mask = labels_f >= 0.5

            if not torch.any(pos_mask):
                # No positives → recall is undefined; return zero loss with grad.
                return scores.sum() * 0.0

            # Threshold = (1 - fpr_target)-th quantile of negative scores.
            # quantile is straight-through differentiable through neg_scores in PyTorch.
            neg_scores = scores[neg_mask]
            if neg_scores.numel() == 0:
                # No negatives → no FPR constraint binds; threshold at -inf so
                # everything ranks above it (recall = 1 → loss = 0).
                threshold = scores.min().detach() - 1.0
            else:
                # quantile q = 1 - fpr_target means we want the score above which
                # exactly fpr_target fraction of negatives sit.
                q = 1.0 - self.fpr_target
                threshold = torch.quantile(neg_scores, q)

            # Soft indicator: sigmoid(beta * (s - t)) → near-step function as beta → ∞.
            approx_above = torch.sigmoid(self.fpr_smoothing_beta * (scores - threshold))
            # Recall@FPR = (Σ I(s_i ≥ t) * y_i * pos_weight) / (Σ y_i * pos_weight)
            tp_weighted = approx_above * labels_f * self.pos_weight
            denom = labels_f.sum() * self.pos_weight
            recall_at_fpr = tp_weighted.sum() / denom.clamp(min=1e-9)
            per_pos = 1.0 - approx_above[pos_mask]  # per-positive contribution

            if self.reduction == "mean":
                return torch.tensor(1.0, device=scores.device) - recall_at_fpr
            if self.reduction == "sum":
                return per_pos.sum()
            return per_pos  # "none"

    return _RecallAtLowFPR


_CLASS_CACHE: dict[str, Any] = {}


def RecallAtLowFPR(  # noqa: N802 — matches issue spec PascalCase class-like name
    fpr_target: float = 0.01,
    fpr_smoothing_beta: float = 10.0,
    pos_weight: float = 1.0,
    reduction: ReductionMode = "mean",
) -> Any:
    """Construct a Recall@LowFPR loss module.

    Differentiable approximation of recall at a constrained false-positive
    rate, per the Meta Prompt Guard 2 training recipe. Optimizes
    detector ranking at a specific operating point (e.g. ``fpr_target=0.01``
    → "maximize recall while keeping FPR ≤ 1%").

    Parameters
    ----------
    fpr_target : float, optional
        Target false-positive rate (operating point constraint).
        Must be in ``(0, 1]``. Default ``0.01`` (1% FPR).
    fpr_smoothing_beta : float, optional
        Temperature of the soft-indicator approximation; higher values
        make the loss sharper (closer to the hard step function) but
        produce smaller gradients away from the threshold. Default ``10.0``.
        Increase toward training convergence; start low for stable
        gradient flow.
    pos_weight : float, optional
        Per-positive-sample weight applied to the recall numerator and
        denominator. Default ``1.0`` (unweighted).
    reduction : {"mean", "sum", "none"}, optional
        How to reduce the per-positive loss. Default ``"mean"``.
        ``"mean"`` returns the scalar ``1 - Recall@FPR`` (the canonical
        training objective). ``"sum"`` returns the sum of per-positive
        ``1 - approx_indicator``. ``"none"`` returns the per-positive
        ``1 - approx_indicator`` tensor for custom downstream weighting.

    Returns
    -------
    torch.nn.Module
        The constructed loss module. Drop into any standard PyTorch
        training loop.

    Raises
    ------
    ImportError
        If the ``[losses]`` extra is not installed.
    ValueError
        On invalid ``fpr_target`` / ``fpr_smoothing_beta`` / ``reduction``.

    Examples
    --------
    >>> # Requires the [losses] extra.
    >>> # import torch
    >>> # loss = RecallAtLowFPR(fpr_target=0.01)
    >>> # logits = torch.randn(32, requires_grad=True)
    >>> # labels = torch.randint(0, 2, (32,))
    >>> # loss(logits, labels).backward()
    """
    if "cls" not in _CLASS_CACHE:
        _CLASS_CACHE["cls"] = _build_module_class()
    cls = _CLASS_CACHE["cls"]
    return cls(
        fpr_target=fpr_target,
        fpr_smoothing_beta=fpr_smoothing_beta,
        pos_weight=pos_weight,
        reduction=reduction,
    )
