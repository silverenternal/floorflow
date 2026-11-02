from __future__ import annotations

import torch

from floorflow.objectives.control_repair import masked_room_loss


def test_masked_room_loss_accepts_channel_mask() -> None:
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    valid = torch.zeros(2, 1, 8, 8)
    valid[:, :, 2:6, 2:6] = 1.0
    loss, terms = masked_room_loss(logits, target, valid)
    assert torch.isfinite(loss)
    assert set(terms) == {"room_ce", "room_dice"}
    loss.backward()
    assert logits.grad is not None
