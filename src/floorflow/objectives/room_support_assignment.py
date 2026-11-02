"""Dense room-support supervision for query-owner assignment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.objectives.gpu_assignment import exact_assignment_batch


def _mask_assignment_cost(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    predicted_flat = predicted.flatten(1)
    target_flat = target.flatten(1)
    intersection = predicted_flat @ target_flat.T
    denominator = (
        predicted_flat.sum(dim=1)[:, None]
        + target_flat.sum(dim=1)[None]
        - intersection
    )
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def _match_owner_targets(
    owner_probability: torch.Tensor,
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = [
        int(value) for value in target_valid.sum(dim=1).tolist()
    ]
    class_probability = class_logits.float().softmax(dim=-1)
    costs = []
    populated = []
    for batch, count in enumerate(counts):
        if count == 0:
            continue
        class_cost = -class_probability[batch][
            :, target_classes[batch, :count]
        ]
        mask_cost = _mask_assignment_cost(
            owner_probability[batch],
            target_masks[batch, :count],
        )
        costs.append(0.5 * class_cost + 2.0 * mask_cost)
        populated.append(batch)
    solved = exact_assignment_batch(costs)
    by_batch = dict(zip(populated, solved))
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    return [by_batch.get(batch, empty) for batch in range(len(counts))]


def room_support_assignment_loss(
    owner_probability: torch.Tensor,
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    boundary_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign every target-room pixel to its matched query owner."""

    if owner_probability.ndim != 4:
        raise ValueError("owner_probability must be [B,Q,H,W]")
    if target_masks.ndim != 4:
        raise ValueError("target_masks must be [B,T,H,W]")
    if target_masks.shape[0] != owner_probability.shape[0]:
        raise ValueError("owner and target batches must match")
    if target_masks.shape[-2:] != owner_probability.shape[-2:]:
        target_masks = F.interpolate(
            target_masks.float(),
            size=owner_probability.shape[-2:],
            mode="nearest",
        )
    assignments = _match_owner_targets(
        owner_probability,
        class_logits,
        target_classes,
        target_masks,
        target_valid,
    )
    target_owner = torch.full(
        target_masks.shape[0:1] + target_masks.shape[-2:],
        -100,
        dtype=torch.long,
        device=target_masks.device,
    )
    pixel_weight = torch.zeros(
        target_owner.shape,
        dtype=owner_probability.dtype,
        device=owner_probability.device,
    )
    for batch, (rows, columns) in enumerate(assignments):
        for row, column in zip(rows, columns):
            target_mask = target_masks[batch, int(column)] > 0.5
            target_owner[batch, target_mask] = int(row)
            eroded = -F.max_pool2d(
                -target_mask.float()[None, None],
                3,
                stride=1,
                padding=1,
            )[0, 0]
            boundary = (
                target_mask.float() - eroded
            ).clamp_min(0.0) > 0.0
            pixel_weight[batch, target_mask] = 1.0
            pixel_weight[batch][boundary] = 1.0 + float(boundary_weight)
    valid = target_owner >= 0
    normalized_owner = owner_probability.float()
    normalized_owner = normalized_owner / normalized_owner.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    log_probability = normalized_owner.clamp_min(1.0e-8).log()
    cross_entropy = F.nll_loss(
        log_probability,
        target_owner.clamp_min(0),
        reduction="none",
    )
    weights = pixel_weight * valid.to(pixel_weight.dtype)
    loss = (cross_entropy * weights).sum() / weights.sum().clamp_min(1.0)
    exact = (
        normalized_owner.argmax(dim=1).eq(target_owner) & valid
    ).to(weights.dtype)
    exact = (exact * weights).sum() / weights.sum().clamp_min(1.0)
    boundary_fraction = (
        (pixel_weight > 1.0) & valid
    ).to(weights.dtype).sum() / valid.sum().clamp_min(1)
    return loss, {
        "room_support_assignment": loss.detach(),
        "room_support_owner_exact": exact.detach(),
        "room_support_boundary_fraction": boundary_fraction.detach(),
        "room_support_valid_pixels": valid.sum().detach(),
        "room_support_weighted_pixels": weights.sum().detach(),
    }


__all__ = ["room_support_assignment_loss"]
