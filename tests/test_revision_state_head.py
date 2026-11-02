from __future__ import annotations

import torch

from floorflow.heads.revision_state_head import (
    RevisionStateHead,
    revision_state_loss,
)


def test_revision_state_is_learned_from_edit_mask_and_trainable() -> None:
    head = RevisionStateHead(16, hidden_dim=32, operations=4, families=4)
    features = torch.randn(2, 16, 16, 16)
    edit_mask = torch.zeros(2, 1, 16, 16)
    edit_mask[:, :, 3:9, 4:10] = 1.0
    prediction = head(features, edit_mask)
    assert prediction["operation_logits"].shape == (2, 4)
    assert prediction["family_logits"].shape == (2, 4)
    assert prediction["region_logits"].shape == edit_mask.shape
    loss, components = revision_state_loss(
        prediction,
        target_operation=torch.tensor([1, 2]),
        target_family=torch.tensor([0, 2]),
        target_region=edit_mask,
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "revision_operation_loss",
        "revision_family_loss",
        "revision_region_loss",
        "revision_gate_loss",
    }
    loss.backward()
    assert head.operation_queries.grad is not None
    assert head.region[-1].weight.grad is not None
    assert head.gate[-1].weight.grad is not None
    assert head.gate[-1].weight.grad.abs().sum() > 0


def test_nonempty_region_supervises_revision_gate_toward_admission() -> None:
    head = RevisionStateHead(16, hidden_dim=32, operations=4, families=4)
    features = torch.randn(1, 16, 16, 16)
    region = torch.zeros(1, 1, 16, 16)
    region[:, :, 3:9, 4:10] = 1.0
    prediction = head(features, region)
    with torch.no_grad():
        head.gate[-1].bias.fill_(-2.0)
    loss, components = revision_state_loss(
        prediction,
        target_operation=torch.tensor([1]),
        target_family=torch.tensor([0]),
        target_region=region,
    )
    assert components["revision_gate_loss"] > 0.1
    assert torch.isfinite(loss)
