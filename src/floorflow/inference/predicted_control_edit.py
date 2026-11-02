"""Project predicted-control instructions onto a frozen raster plan."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy import ndimage

from floorflow.data.taxonomy import FloorplanTaxonomy


def room_labels_from_controls(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    size: int = 256,
) -> torch.Tensor:
    logits = controls["room_logits"].float()[None]
    resized = F.interpolate(
        logits,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    )[0]
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=logits.device,
        dtype=torch.long,
    )
    class_indices = resized.argmax(dim=0)
    labels = room_ids[class_indices]
    present = controls["program"].bool().to(logits.device)[class_indices]
    present &= resized.amax(dim=0) > 0.0
    return torch.where(present, labels, torch.zeros_like(labels))


def project_predicted_instruction(
    before_labels: torch.Tensor,
    edited_controls: dict[str, torch.Tensor],
    edit_region: tuple[int, int, int, int],
    control_size: tuple[int, int],
    taxonomy: FloorplanTaxonomy,
    *,
    operation: str = "control_delta",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace labels only inside the declared region and verify exact freezing."""

    height, width = before_labels.shape[-2:]
    edited_labels = room_labels_from_controls(
        edited_controls,
        taxonomy,
        size=max(height, width),
    )
    if tuple(edited_labels.shape) != (height, width):
        edited_labels = F.interpolate(
            edited_labels[None, None].float(),
            size=(height, width),
            mode="nearest",
        )[0, 0].long()
    mask = torch.zeros((height, width), dtype=torch.bool)
    x, y, box_width, box_height = edit_region
    control_width, control_height = control_size
    left = max(0, round(x * width / max(control_width, 1)))
    top = max(0, round(y * height / max(control_height, 1)))
    right = min(
        width,
        round((x + box_width) * width / max(control_width, 1)) + 1,
    )
    bottom = min(
        height,
        round((y + box_height) * height / max(control_height, 1)) + 1,
    )
    mask[top:bottom, left:right] = True
    if operation in {"opening_delete", "furniture_delete"}:
        family_ids = (
            taxonomy.opening_ids
            if operation == "opening_delete"
            else taxonomy.furniture_ids
        )
        family = torch.tensor(
            family_ids,
            device=before_labels.device,
            dtype=before_labels.dtype,
        )
        changed = mask & torch.isin(before_labels, family)
        after = before_labels.clone()
        after[changed] = 0
    elif operation == "room_remove":
        dilated = F.max_pool2d(
            mask.float()[None, None],
            5,
            stride=1,
            padding=2,
        )[0, 0] > 0.5
        ring_values = before_labels[dilated & ~mask]
        nonzero = ring_values[ring_values != 0]
        values = nonzero if nonzero.numel() else ring_values
        target = (
            torch.mode(values).values
            if values.numel()
            else torch.zeros((), dtype=before_labels.dtype)
        )
        changed = mask
        after = torch.where(changed, target, before_labels)
    elif operation == "room_add":
        changed = (
            mask
            & (edited_labels != before_labels)
            & (edited_labels != 0)
        )
        if not bool(changed.any()):
            return before_labels, mask
        values = edited_labels[changed].long()
        target = torch.bincount(values).argmax().to(before_labels.dtype)
        changed = mask & (edited_labels == target) & (before_labels != target)
        changed = _largest_connected_component(changed)
        after = before_labels.clone()
        after[changed] = target
    else:
        changed = (
            mask
            & (edited_labels != before_labels)
            & (edited_labels != 0)
        )
        if not bool(changed.any()):
            return before_labels, mask
        after = before_labels.clone()
        after[changed] = edited_labels[changed]
    outside_changes = int((after != before_labels)[~mask].sum())
    if outside_changes:
        raise RuntimeError("predicted instruction leaked outside the declared region")
    return after, mask


def _largest_connected_component(mask: torch.Tensor) -> torch.Tensor:
    """Keep the largest four-connected component of one edit target."""

    array = mask.detach().cpu().numpy()
    components, count = ndimage.label(array)
    if count <= 1:
        return mask
    areas = [
        int((components == index).sum())
        for index in range(1, int(count) + 1)
    ]
    selected = int(max(range(1, int(count) + 1), key=lambda index: areas[index - 1]))
    return torch.from_numpy(components == selected).to(mask.device)


__all__ = ["project_predicted_instruction", "room_labels_from_controls"]
