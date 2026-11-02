"""Deterministic semantic-success checks for predicted typed edits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class RoomControlRequest:
    source_class_id: int
    target_class_id: int
    source_logit_delta: float
    target_logit_delta: float


@dataclass(frozen=True)
class OperationSuccess:
    success: bool
    reason: str
    changed_pixels: int
    source_pixels_before: int
    source_removed_fraction: float
    target_share_of_changed: float
    target_pixels_after: int


def control_region_mask(
    edit_region: tuple[int, int, int, int],
    control_size: tuple[int, int],
) -> np.ndarray:
    """Return the clamped spatial mask used by predicted instruction projection."""

    width, height = (int(value) for value in control_size)
    x, y, box_width, box_height = (int(value) for value in edit_region)
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left + 1, min(width, x + box_width))
    bottom = max(top + 1, min(height, y + box_height))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def room_control_request(
    source_controls: dict[str, torch.Tensor],
    edited_controls: dict[str, torch.Tensor],
    edit_region: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> RoomControlRequest:
    """Infer the requested source and target room classes from control deltas."""

    source_logits = source_controls["room_logits"].float()
    edited_logits = edited_controls["room_logits"].float()
    if source_logits.shape != edited_logits.shape:
        raise ValueError("source and edited room logits must share a shape")
    if source_logits.ndim != 3:
        raise ValueError("room logits must be [C,H,W]")
    mask = torch.from_numpy(
        control_region_mask(
            edit_region,
            (int(source_logits.shape[-1]), int(source_logits.shape[-2])),
        )
    )
    if not bool(mask.any()):
        raise ValueError("edit region is empty")
    delta = (edited_logits - source_logits)[:, mask].mean(dim=1)
    source_index = int(delta.argmin())
    target_index = int(delta.argmax())
    if source_index == target_index:
        raise ValueError("control request does not identify distinct room classes")
    return RoomControlRequest(
        source_class_id=int(taxonomy.room_ids[source_index]),
        target_class_id=int(taxonomy.room_ids[target_index]),
        source_logit_delta=float(delta[source_index]),
        target_logit_delta=float(delta[target_index]),
    )


def operation_success(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    operation: str,
    *,
    request: RoomControlRequest | None = None,
    source_removed_fraction_min: float = 0.5,
    target_share_of_changed_min: float = 0.5,
    minimum_target_pixels: int = 4,
) -> OperationSuccess:
    """Evaluate whether one executed operation realized its local intent."""

    before = np.asarray(before_labels)
    after = np.asarray(after_labels)
    mask = np.asarray(region, dtype=bool)
    if before.shape != after.shape or before.shape != mask.shape:
        raise ValueError("before, after, and region must share a shape")
    changed = mask & (before != after)
    changed_pixels = int(changed.sum())

    if operation in {"room_type", "room_add", "room_remove"}:
        if request is None:
            raise ValueError("room operations require a control request")
        source_class_id = request.source_class_id
        if operation == "room_remove" and not bool(
            (mask & (before == source_class_id)).any()
        ):
            room_values = before[
                mask & np.isin(before, np.asarray(taxonomy.room_ids))
            ]
            if room_values.size:
                source_class_id = int(np.bincount(room_values).argmax())
        source_pixels = mask & (before == source_class_id)
        source_pixels_before = int(source_pixels.sum())
        source_removed_fraction = (
            float((after[source_pixels] != source_class_id).mean())
            if source_pixels_before
            else 0.0
        )
        target_mask = mask & (after == request.target_class_id)
        if operation == "room_remove":
            target_mask = (
                mask
                & (after != 0)
                & (after != source_class_id)
                & np.isin(after, np.asarray(taxonomy.room_ids))
            )
        target_pixels_after = int(target_mask.sum())
        target_share = (
            float(
                (changed & (after == request.target_class_id)).sum()
                / changed_pixels
            )
            if changed_pixels
            else 0.0
        )
        if operation == "room_remove":
            success = (
                changed_pixels > 0
                and source_removed_fraction >= source_removed_fraction_min
                and target_pixels_after > 0
            )
            reason = "source_class_removed" if success else "source_class_retained"
        else:
            success = (
                changed_pixels > 0
                and target_share >= target_share_of_changed_min
                and target_pixels_after >= minimum_target_pixels
            )
            reason = "requested_class_realized" if success else "requested_class_missing"
        return OperationSuccess(
            success=bool(success),
            reason=reason,
            changed_pixels=changed_pixels,
            source_pixels_before=source_pixels_before,
            source_removed_fraction=source_removed_fraction,
            target_share_of_changed=target_share,
            target_pixels_after=target_pixels_after,
        )

    if operation == "opening_delete":
        family = np.asarray(taxonomy.opening_ids)
    elif operation == "furniture_delete":
        family = np.asarray(taxonomy.furniture_ids)
    else:
        raise ValueError(f"unsupported operation: {operation}")

    before_pixels = int((mask & np.isin(before, family)).sum())
    after_pixels = int((mask & np.isin(after, family)).sum())
    success = before_pixels > 0 and after_pixels == 0
    return OperationSuccess(
        success=bool(success),
        reason="family_deleted" if success else "family_retained_or_absent",
        changed_pixels=changed_pixels,
        source_pixels_before=before_pixels,
        source_removed_fraction=(
            float(1.0 - after_pixels / before_pixels)
            if before_pixels
            else 0.0
        ),
        target_share_of_changed=0.0,
        target_pixels_after=after_pixels,
    )


__all__ = [
    "OperationSuccess",
    "RoomControlRequest",
    "control_region_mask",
    "operation_success",
    "room_control_request",
]
