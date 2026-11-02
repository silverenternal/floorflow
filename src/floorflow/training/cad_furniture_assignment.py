"""Room assignment targets and losses for furniture queries."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_mask_losses import pairwise_mask_cost


def _room_index_for_mask(
    mask: np.ndarray,
    room_labels: np.ndarray,
    room_ids: tuple[int, ...],
) -> int:
    ring = binary_dilation(mask, iterations=3) & ~mask
    values = room_labels[ring]
    minlength = max(room_ids) + 1 if room_ids else 1
    counts = np.bincount(
        values.reshape(-1),
        minlength=minlength,
    )
    if room_ids:
        room_counts = counts[np.asarray(room_ids, dtype=np.int64)]
        if int(room_counts.max()) > 0:
            return int(room_counts.argmax())
    nearest = distance_transform_edt(
        ~mask,
        return_distances=False,
        return_indices=True,
    )
    nearest_values = room_labels[tuple(nearest)]
    counts = np.bincount(
        nearest_values.reshape(-1),
        minlength=max(room_ids) + 1 if room_ids else 1,
    )
    room_counts = counts[np.asarray(room_ids, dtype=np.int64)]
    return int(room_counts.argmax()) if int(room_counts.max()) else -1


def room_assignment_targets(
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    target_labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Return room-index targets for targets and background for padding."""

    background = len(taxonomy.room_ids)
    output = torch.full(
        target_classes.shape,
        background,
        dtype=torch.long,
        device=target_classes.device,
    )
    masks = target_masks.detach().cpu().numpy()
    labels = target_labels.detach().cpu().numpy()
    valid = target_valid.detach().cpu().numpy()
    for batch in range(masks.shape[0]):
        for query in range(masks.shape[1]):
            if not valid[batch, query]:
                continue
            room_index = _room_index_for_mask(
                masks[batch, query] > 0.5,
                labels[batch],
                taxonomy.room_ids,
            )
            if room_index >= 0:
                output[batch, query] = room_index
    return output


def furniture_room_assignment_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    target_labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise matched query rooms and unmatched background queries."""

    assignment_logits = predictions.get(
        "room_assignment_logits",
        predictions.get("furniture_room_assignment_logits"),
    )
    if assignment_logits is None:
        raise KeyError("missing room assignment logits")
    background = len(taxonomy.room_ids)
    if assignment_logits.shape[-1] != background + 1:
        raise ValueError("assignment logits do not match room taxonomy")
    targets = room_assignment_targets(
        target_classes,
        target_masks,
        target_valid,
        target_labels,
        taxonomy,
    )
    matched_losses = []
    background_losses = []
    correct = 0
    count = 0
    class_logits = predictions.get(
        "furniture_query_class_logits",
        predictions.get("class_logits"),
    )
    mask_logits = predictions.get(
        "furniture_query_mask_logits",
        predictions.get("mask_logits"),
    )
    if class_logits is None or mask_logits is None:
        raise KeyError("missing furniture query class or mask logits")
    if mask_logits.shape[-2:] != target_masks.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    class_probability = class_logits.softmax(dim=-1)
    mask_probability = mask_logits.sigmoid()
    for index in range(target_classes.shape[0]):
        target_count = int(target_valid[index].sum())
        query_count = assignment_logits.shape[1]
        matched = torch.zeros(
            query_count,
            dtype=torch.bool,
            device=assignment_logits.device,
        )
        if target_count:
            class_cost = -class_probability[index][
                :,
                target_classes[index, :target_count],
            ]
            mask_cost = pairwise_mask_cost(
                mask_probability[index],
                target_masks[index, :target_count],
            )
            cost = torch.nan_to_num(
                class_cost + mask_cost,
                nan=1.0e6,
                posinf=1.0e6,
                neginf=-1.0e6,
            )
            rows, columns = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            row_tensor = torch.as_tensor(
                rows,
                device=assignment_logits.device,
            )
            matched[row_tensor] = True
            matched_losses.append(
                F.cross_entropy(
                    assignment_logits[index, row_tensor],
                    targets[index, torch.as_tensor(columns, device=targets.device)],
                )
            )
            correct += int(
                (
                    assignment_logits[index, row_tensor].argmax(dim=-1)
                    == targets[
                        index,
                        torch.as_tensor(columns, device=targets.device),
                    ]
                ).sum()
            )
            count += len(rows)
        unmatched = ~matched
        if bool(unmatched.any()):
            background_target = torch.full(
                (int(unmatched.sum()),),
                background,
                dtype=torch.long,
                device=assignment_logits.device,
            )
            background_losses.append(
                F.cross_entropy(
                    assignment_logits[index, unmatched],
                    background_target,
                )
            )
            correct += int(
                (
                    assignment_logits[index, unmatched].argmax(dim=-1)
                    == background
                ).sum()
            )
            count += int(unmatched.sum())
    matched_loss = (
        torch.stack(matched_losses).mean()
        if matched_losses
        else assignment_logits.sum() * 0.0
    )
    background_loss = (
        torch.stack(background_losses).mean()
        if background_losses
        else assignment_logits.sum() * 0.0
    )
    total = matched_loss + 0.5 * background_loss
    return total, {
        "furniture_room_assignment_loss": total.detach(),
        "furniture_room_assignment_matched_loss": matched_loss.detach(),
        "furniture_room_assignment_background_loss": background_loss.detach(),
        "furniture_room_assignment_accuracy": (
            total.new_tensor(correct / max(1, count))
        ),
    }


__all__ = [
    "furniture_room_assignment_loss",
    "room_assignment_targets",
]
