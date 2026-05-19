"""Tests for ``eval_toolkit.losses.RecallAtLowFPR`` (v0.44.0; closes #50).

Requires the ``[losses]`` extra (torch); all tests use
``pytest.importorskip`` so the test file is silent when torch isn't
installed (matching the consumer experience).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")  # noqa: F841 — needed for skip behavior at module load

from eval_toolkit.losses import RecallAtLowFPR  # noqa: E402

# ----------------------------------------------------------------------------
# Construction + validation
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_construct_with_defaults() -> None:
    loss = RecallAtLowFPR()
    assert loss.fpr_target == 0.01
    assert loss.fpr_smoothing_beta == 10.0
    assert loss.reduction == "mean"


@pytest.mark.unit
@pytest.mark.parametrize("fpr_target", [-0.1, 0.0, 1.5])
def test_invalid_fpr_target_raises(fpr_target: float) -> None:
    with pytest.raises(ValueError, match="fpr_target"):
        RecallAtLowFPR(fpr_target=fpr_target)


@pytest.mark.unit
@pytest.mark.parametrize("beta", [-1.0, 0.0])
def test_invalid_smoothing_beta_raises(beta: float) -> None:
    with pytest.raises(ValueError, match="smoothing_beta"):
        RecallAtLowFPR(fpr_smoothing_beta=beta)


@pytest.mark.unit
def test_invalid_reduction_raises() -> None:
    with pytest.raises(ValueError, match="reduction"):
        RecallAtLowFPR(reduction="median")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Forward shapes
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_forward_returns_scalar_under_mean_reduction() -> None:
    loss = RecallAtLowFPR(fpr_target=0.1, reduction="mean")
    logits = torch.randn(16)
    labels = torch.randint(0, 2, (16,))
    out = loss(logits, labels)
    assert out.dim() == 0  # scalar


@pytest.mark.unit
def test_forward_returns_scalar_under_sum_reduction() -> None:
    loss = RecallAtLowFPR(fpr_target=0.1, reduction="sum")
    logits = torch.randn(16)
    labels = torch.cat([torch.zeros(8), torch.ones(8)]).int()
    out = loss(logits, labels)
    assert out.dim() == 0


@pytest.mark.unit
def test_forward_returns_per_positive_vector_under_none_reduction() -> None:
    loss = RecallAtLowFPR(fpr_target=0.1, reduction="none")
    n_pos = 5
    n_neg = 15
    logits = torch.randn(n_pos + n_neg)
    labels = torch.cat([torch.zeros(n_neg), torch.ones(n_pos)]).int()
    out = loss(logits, labels)
    assert out.shape == (n_pos,)


@pytest.mark.unit
def test_forward_accepts_2d_logits_via_squeeze() -> None:
    loss = RecallAtLowFPR(fpr_target=0.1)
    logits = torch.randn(8, 1)
    labels = torch.randint(0, 2, (8,))
    _ = loss(logits, labels)  # should not raise


@pytest.mark.unit
def test_forward_rejects_mismatched_shapes() -> None:
    loss = RecallAtLowFPR(fpr_target=0.1)
    logits = torch.randn(8)
    labels = torch.randint(0, 2, (4,))  # wrong size
    with pytest.raises(ValueError, match="shape"):
        loss(logits, labels)


# ----------------------------------------------------------------------------
# Differentiability
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_forward_is_differentiable_wrt_logits() -> None:
    """Backprop returns a finite grad of correct shape."""
    loss = RecallAtLowFPR(fpr_target=0.1, fpr_smoothing_beta=5.0)
    torch.manual_seed(0)
    logits = torch.randn(16, requires_grad=True)
    labels = torch.cat([torch.zeros(8), torch.ones(8)]).int()
    out = loss(logits, labels)
    out.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


@pytest.mark.unit
def test_loss_descends_under_gradient_steps() -> None:
    """A few SGD steps reduce loss on a well-separated toy problem."""
    torch.manual_seed(42)
    # Construct logits where positives start UNDER the FPR-target threshold;
    # SGD on RecallAtLowFPR should shift positive logits upward.
    logits = torch.cat([torch.randn(32) - 0.5, torch.randn(8) - 0.5])
    logits = logits.detach().clone().requires_grad_(True)
    labels = torch.cat([torch.zeros(32), torch.ones(8)]).int()

    loss = RecallAtLowFPR(fpr_target=0.1, fpr_smoothing_beta=5.0)
    optimizer = torch.optim.SGD([logits], lr=1.0)

    initial_loss = loss(logits, labels).item()
    for _ in range(50):
        optimizer.zero_grad()
        out = loss(logits, labels)
        out.backward()
        optimizer.step()
    final_loss = loss(logits, labels).item()
    assert final_loss < initial_loss, f"Loss did not descend: {initial_loss} → {final_loss}"


@pytest.mark.unit
def test_positives_only_batch_returns_zero_with_grad() -> None:
    """No negatives → FPR constraint doesn't bind → loss should be ~0."""
    loss = RecallAtLowFPR(fpr_target=0.1)
    logits = torch.randn(8, requires_grad=True)
    labels = torch.ones(8).int()
    out = loss(logits, labels)
    # Loss should be very small (recall is approx 1 because no threshold binds).
    assert out.item() < 0.1


@pytest.mark.unit
def test_negatives_only_batch_returns_zero_loss() -> None:
    """No positives → recall undefined → loss is zero (with grad path)."""
    loss = RecallAtLowFPR(fpr_target=0.1)
    logits = torch.randn(8, requires_grad=True)
    labels = torch.zeros(8).int()
    out = loss(logits, labels)
    assert out.item() == 0.0
    # backprop still works without error (graph-trivial grad).
    out.backward()


# ----------------------------------------------------------------------------
# Missing-extra ImportError surface
# ----------------------------------------------------------------------------


@pytest.mark.unit
def test_factory_raises_helpful_import_error_when_torch_missing() -> None:
    """If torch is somehow absent from sys.modules, RecallAtLowFPR raises ImportError."""
    # Save and clear the class cache + simulate torch removal
    from eval_toolkit import losses

    losses._CLASS_CACHE.clear()
    saved_torch = sys.modules.get("torch")
    with (
        patch.dict(sys.modules, {"torch": None}),
        pytest.raises(ImportError, match=r"\[losses\]"),
    ):
        RecallAtLowFPR()
    # Restore
    if saved_torch is not None:
        sys.modules["torch"] = saved_torch
    losses._CLASS_CACHE.clear()
