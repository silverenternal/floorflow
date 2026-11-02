"""Selective wall closure on low-confidence room seams."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def low_confidence_gap_closure_loss(
    wall_logits: torch.Tensor,
    target_labels: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    *,
    band: int = 3,
) -> torch.Tensor:
    """Close room seams only where the wall prediction is uncertain."""

    if wall_logits.ndim == 4:
        wall_logits = wall_logits[:, 0]
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    if target_labels.ndim == 4:
        target_labels = target_labels[:, 0]
    if wall_logits.shape != target_labels.shape:
        raise ValueError("wall logits and labels must share shape")
    kernel = max(1, int(band))
    room_tensor = torch.tensor(room_ids, device=target_labels.device)
    room_masks = torch.stack(
        [
            (target_labels == class_id).float()
            for class_id in room_tensor
        ],
        dim=1,
    )
    near = F.max_pool2d(
        room_masks,
        kernel,
        stride=1,
        padding=kernel // 2,
    )
    seam = (near.sum(dim=1) > 1.0).float()
    wall_probability = wall_logits.sigmoid()
    uncertainty = 1.0 - (2.0 * wall_probability - 1.0).abs()
    missing = 1.0 - wall_probability
    valid = footprint.bool()
    weighted = seam * uncertainty * missing * valid
    return weighted.sum() / (
        seam * valid
    ).sum().clamp_min(1.0)


def room_transition_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dilation: int = 1,
) -> torch.Tensor:
    """Cross-entropy on internal room-transition bands."""

    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("expected [B,C,H,W] logits and [B,H,W] target")
    valid = target >= 0
    valid_x = valid[:, :, 1:] & valid[:, :, :-1]
    boundary_x = valid_x & (
        target[:, :, 1:] != target[:, :, :-1]
    )
    boundary = F.pad(boundary_x, (0, 1)) | F.pad(boundary_x, (1, 0))
    valid_y = valid[:, 1:, :] & valid[:, :-1, :]
    boundary_y = valid_y & (
        target[:, 1:, :] != target[:, :-1, :]
    )
    boundary = boundary | (
        F.pad(boundary_y, (0, 0, 0, 1))
        | F.pad(boundary_y, (0, 0, 1, 0))
    )
    if dilation > 0:
        boundary = (
            F.max_pool2d(
                boundary[:, None].float(),
                kernel_size=2 * int(dilation) + 1,
                stride=1,
                padding=int(dilation),
            )[:, 0]
            > 0.5
        )
    if not boundary.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(
        logits,
        target.clamp_min(0),
        reduction="none",
    )
    return loss[boundary].mean()


__all__ = ["low_confidence_gap_closure_loss", "room_transition_loss"]
