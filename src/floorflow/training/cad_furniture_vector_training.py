"""Targets and losses for room-slot furniture vector primitives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_object_instance import object_instance_targets
from floorflow.training.cad_furniture_assignment import (
    room_assignment_targets,
)


def _room_boxes(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    height, width = labels.shape
    boxes = np.zeros((len(taxonomy.room_ids), 4), dtype=np.float32)
    for index, room_id in enumerate(taxonomy.room_ids):
        ys, xs = np.nonzero(labels == room_id)
        if ys.size == 0:
            continue
        boxes[index] = (
            xs.min() / max(1, width - 1),
            ys.min() / max(1, height - 1),
            xs.max() / max(1, width - 1),
            ys.max() / max(1, height - 1),
        )
    return boxes


def furniture_vector_targets(
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_per_room: int = 6,
) -> dict[str, torch.Tensor]:
    """Convert instance masks into room-local center, size, and class targets."""

    device = target.device
    classes, masks, valid = object_instance_targets(
        target,
        taxonomy.furniture_ids,
        max_instances=128,
    )
    assignment = room_assignment_targets(
        classes,
        masks,
        valid,
        target,
        taxonomy,
    )
    batch, height, width = target.shape
    room_count = len(taxonomy.room_ids)
    target_exists = torch.zeros(
        batch,
        room_count,
        max_per_room,
        dtype=torch.float32,
        device=device,
    )
    target_class = torch.zeros(
        batch,
        room_count,
        max_per_room,
        dtype=torch.long,
        device=device,
    )
    target_center = torch.zeros(
        batch,
        room_count,
        max_per_room,
        2,
        device=device,
    )
    target_size = torch.zeros_like(target_center)
    masks_np = masks.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    assignment_np = assignment.detach().cpu().numpy()
    rooms = []
    for batch_index in range(batch):
        labels = target[batch_index].detach().cpu().numpy()
        boxes = _room_boxes(labels, taxonomy)
        room_items: list[list[tuple[float, int, int]]] = [
            [] for _ in range(room_count)
        ]
        for query in range(masks_np.shape[1]):
            if not valid_np[batch_index, query]:
                continue
            room_index = int(assignment_np[batch_index, query])
            if not 0 <= room_index < room_count:
                continue
            mask = masks_np[batch_index, query] > 0.5
            area = float(mask.sum())
            if area <= 0.0:
                continue
            room_items[room_index].append(
                (area, query, room_index)
            )
        for room_index, items in enumerate(room_items):
            items = sorted(items, reverse=True)[:max_per_room]
            box = boxes[room_index]
            box_size = np.maximum(box[2:] - box[:2], 1.0e-4)
            for slot, (_area, query, _room) in enumerate(items):
                ys, xs = np.nonzero(masks_np[batch_index, query] > 0.5)
                center = np.asarray(
                    [
                        xs.mean() / max(1, width - 1),
                        ys.mean() / max(1, height - 1),
                    ],
                    dtype=np.float32,
                )
                item_box = np.asarray(
                    [
                        xs.min() / max(1, width - 1),
                        ys.min() / max(1, height - 1),
                        xs.max() / max(1, width - 1),
                        ys.max() / max(1, height - 1),
                    ],
                    dtype=np.float32,
                )
                relative_center = (center - box[:2]) / box_size
                relative_size = (
                    item_box[2:] - item_box[:2]
                ) / box_size
                target_exists[batch_index, room_index, slot] = 1.0
                target_class[batch_index, room_index, slot] = classes[
                    batch_index,
                    query,
                ]
                target_center[batch_index, room_index, slot] = (
                    torch.from_numpy(relative_center).to(device)
                )
                target_size[batch_index, room_index, slot] = (
                    torch.from_numpy(relative_size).to(device)
                )
        rooms.append(boxes)
    target_mask = torch.isin(
        target,
        torch.tensor(taxonomy.furniture_ids, device=device),
    )
    return {
        "exists": target_exists,
        "class": target_class,
        "center": target_center,
        "size": target_size,
        "target_mask": target_mask,
        "room_boxes": torch.from_numpy(np.stack(rooms)).to(device),
    }


def furniture_vector_target_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    render_grid: int = 64,
    render_weight: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hungarian room-slot loss plus differentiable rectangle rendering."""

    predicted_exists = predictions["furniture_exists_logits"]
    predicted_class = predictions["furniture_class_logits"]
    predicted_center = predictions["furniture_center"]
    predicted_size = predictions["furniture_size"]
    target_exists = targets["exists"]
    target_class = targets["class"]
    target_center = targets["center"]
    target_size = targets["size"]
    batch, rooms, _slots = predicted_exists.shape
    class_losses = []
    center_losses = []
    size_losses = []
    matched_exists = torch.zeros_like(predicted_exists)
    for batch_index in range(batch):
        for room_index in range(rooms):
            count = int(target_exists[batch_index, room_index].sum())
            if count == 0:
                continue
            class_cost = -predicted_class[
                batch_index,
                room_index,
            ].softmax(dim=-1)[:, target_class[batch_index, room_index, :count]]
            center_cost = (
                predicted_center[batch_index, room_index, :, None]
                - target_center[
                    batch_index,
                    room_index,
                    None,
                    :count,
                ]
            ).abs().mean(dim=-1)
            rows, columns = linear_sum_assignment(
                (class_cost + 2.0 * center_cost)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            row_tensor = torch.as_tensor(rows, device=predicted_exists.device)
            column_tensor = torch.as_tensor(
                columns,
                device=predicted_exists.device,
            )
            matched_exists[batch_index, room_index, row_tensor] = 1.0
            class_losses.append(
                F.cross_entropy(
                    predicted_class[batch_index, room_index, row_tensor],
                    target_class[
                        batch_index,
                        room_index,
                        column_tensor,
                    ],
                    weight=class_weights,
                )
            )
            center_losses.append(
                F.smooth_l1_loss(
                    predicted_center[batch_index, room_index, row_tensor],
                    target_center[
                        batch_index,
                        room_index,
                        column_tensor,
                    ],
                )
            )
            size_losses.append(
                F.smooth_l1_loss(
                    predicted_size[batch_index, room_index, row_tensor],
                    target_size[
                        batch_index,
                        room_index,
                        column_tensor,
                    ],
                )
            )
    existence_loss = F.binary_cross_entropy_with_logits(
        predicted_exists,
        matched_exists,
        pos_weight=predicted_exists.new_tensor(4.0),
    )
    class_loss = (
        torch.stack(class_losses).mean()
        if class_losses
        else predicted_class.sum() * 0.0
    )
    center_loss = (
        torch.stack(center_losses).mean()
        if center_losses
        else predicted_center.sum() * 0.0
    )
    size_loss = (
        torch.stack(size_losses).mean()
        if size_losses
        else predicted_size.sum() * 0.0
    )
    target_mask = F.interpolate(
        targets["target_mask"].float()[:, None],
        size=(render_grid, render_grid),
        mode="nearest",
    )[:, 0]
    occupancy = (
        predictions["furniture_render"]
        * predicted_exists.sigmoid()[..., None, None]
    ).amax(dim=(1, 2))
    intersection = (occupancy * target_mask).sum(dim=(1, 2))
    denominator = occupancy.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))
    dice = (
        1.0
        - (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    ).mean()
    render_loss = dice + F.binary_cross_entropy(
        occupancy.clamp(1.0e-5, 1.0 - 1.0e-5),
        target_mask,
    )
    total = (
        existence_loss
        + class_loss
        + 2.0 * center_loss
        + 2.0 * size_loss
        + float(render_weight) * render_loss
    )
    return total, {
        "vector_exists_loss": existence_loss.detach(),
        "vector_class_loss": class_loss.detach(),
        "vector_center_loss": center_loss.detach(),
        "vector_size_loss": size_loss.detach(),
        "vector_render_loss": render_loss.detach(),
    }


__all__ = [
    "furniture_vector_target_loss",
    "furniture_vector_targets",
]
