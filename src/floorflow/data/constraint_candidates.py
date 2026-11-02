"""Deterministic legal and near-violation candidates for constraint learning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from floorflow.data.taxonomy import FloorplanTaxonomy

VIOLATION_FAMILIES = (
    "room_overlap",
    "wall_gap",
    "opening_detached",
    "room_outside_footprint",
    "door_connectivity",
    "room_label_mismatch",
    "outside_leakage",
)
HARD_VIOLATION_FAMILIES = VIOLATION_FAMILIES


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    group_id: str
    role: str
    violation_family: str | None
    severity: int
    transform: str
    seed: int
    operation: str


@dataclass(frozen=True)
class MaterializedCandidate:
    controls: dict[str, torch.Tensor]
    bbox: tuple[int, int, int, int]
    changed_pixels: int
    violation_labels: dict[str, bool]
    transform: str


def _bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    rows, columns = torch.nonzero(mask, as_tuple=True)
    if len(rows) == 0:
        raise ValueError("candidate mask is empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def _union_bbox(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x0 = min(left[0], right[0])
    y0 = min(left[1], right[1])
    x1 = max(left[0] + left[2], right[0] + right[2])
    y1 = max(left[1] + left[3], right[1] + right[3])
    return x0, y0, x1 - x0, y1 - y0


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_height, source_width = source_size
    target_height, target_width = target_size
    x, y, width, height = bbox
    left = round(x * target_width / max(source_width, 1))
    top = round(y * target_height / max(source_height, 1))
    right = round((x + width) * target_width / max(source_width, 1))
    bottom = round((y + height) * target_height / max(source_height, 1))
    left = min(max(left, 0), target_width - 1)
    top = min(max(top, 0), target_height - 1)
    right = min(max(right, left + 1), target_width)
    bottom = min(max(bottom, top + 1), target_height)
    return left, top, right - left, bottom - top


def _central_patch(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    fraction: float = 0.35,
) -> torch.Tensor:
    height, width = size
    x, y, box_width, box_height = bbox
    patch_width = max(1, round(box_width * fraction))
    patch_height = max(1, round(box_height * fraction))
    center_x = min(max(x + box_width // 2, 0), width - 1)
    center_y = min(max(y + box_height // 2, 0), height - 1)
    left = max(center_x - patch_width // 2, 0)
    top = max(center_y - patch_height // 2, 0)
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[
        top : min(top + patch_height, height),
        left : min(left + patch_width, width),
    ] = True
    return mask


def _resize_mask(
    mask: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if tuple(mask.shape) == size:
        return mask
    return F.interpolate(
        mask.float()[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5


def _copy_with_channel(
    controls: dict[str, torch.Tensor],
    channel: str,
    tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output = dict(controls)
    output[channel] = tensor
    return output


def _active_room(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    room_count: int,
) -> int:
    probability = controls["room_logits"].sigmoid()
    patch = _central_patch(tuple(probability.shape[-2:]), bbox)
    mass = (probability[:, patch] > 0.5).float().sum(dim=1)
    return int(mass.argmax()) if float(mass.max()) > 0 else 0


def _legal_perturbation(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    logits = controls["room_logits"].clone()
    mask = _central_patch(tuple(logits.shape[-2:]), bbox)
    room = _active_room(controls, bbox, len(taxonomy.room_ids))
    logits[room] = torch.where(mask, logits[room] + 0.2, logits[room])
    return _copy_with_channel(controls, "room_logits", logits), bbox


def _room_overlap(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    logits = controls["room_logits"].clone()
    mask = _central_patch(tuple(logits.shape[-2:]), bbox, fraction=0.22)
    source = _active_room(controls, bbox, len(taxonomy.room_ids))
    program = controls["program"].bool()
    alternatives = [
        index
        for index in torch.nonzero(program, as_tuple=False).flatten().tolist()
        if index != source
    ]
    other = alternatives[0] if alternatives else (source + 1) % len(taxonomy.room_ids)
    logits[other] = torch.where(mask, torch.full_like(logits[other], 20.0), logits[other])
    return _copy_with_channel(controls, "room_logits", logits), bbox


def _wall_gap(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    structure = controls["structure"].clone()
    room_count = len(taxonomy.room_ids)
    mask = _resize_mask(
        _central_patch(tuple(controls["room_logits"].shape[-2:]), bbox, fraction=0.12),
        tuple(structure.shape[-2:]),
    )
    structure[room_count] = torch.where(
        mask,
        torch.zeros_like(structure[room_count]),
        structure[room_count],
    )
    return _copy_with_channel(controls, "structure", structure), bbox


def _opening_detached(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    structure = controls["structure"].clone()
    room_count = len(taxonomy.room_ids)
    wall = structure[room_count] > 0.5
    valid = torch.ones_like(wall)
    if footprint is not None:
        valid = _resize_mask(
            torch.from_numpy(np.asarray(footprint, dtype=bool)),
            tuple(wall.shape),
        )
    distance = ndimage.distance_transform_edt(~wall.cpu().numpy())
    score = torch.from_numpy(distance)
    score = torch.where(valid, score, torch.zeros_like(score))
    flat = int(score.argmax())
    row, column = divmod(flat, score.shape[1])
    mask = torch.zeros_like(wall)
    mask[
        max(row - 1, 0) : min(row + 2, mask.shape[0]),
        max(column - 2, 0) : min(column + 3, mask.shape[1]),
    ] = True
    structure[room_count + 1] = torch.where(
        mask,
        torch.ones_like(structure[room_count + 1]),
        structure[room_count + 1],
    )
    return _copy_with_channel(controls, "structure", structure), _bbox(mask)


def _room_outside_footprint(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    logits = controls["room_logits"].clone()
    if footprint is None:
        height, width = logits.shape[-2:]
        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[0:2, 0:2] = True
    else:
        footprint_tensor = _resize_mask(
            torch.from_numpy(np.asarray(footprint, dtype=bool)),
            tuple(logits.shape[-2:]),
        )
        outside_distance = ndimage.distance_transform_edt(
            ~footprint_tensor.cpu().numpy()
        )
        score = torch.from_numpy(outside_distance)
        score[footprint_tensor] = 0.0
        flat = int(score.argmax())
        row, column = divmod(flat, score.shape[1])
        mask = torch.zeros_like(footprint_tensor)
        mask[
            max(row - 1, 0) : min(row + 2, mask.shape[0]),
            max(column - 1, 0) : min(column + 2, mask.shape[1]),
        ] = True
        mask &= ~footprint_tensor
    active = _active_room(controls, bbox, len(taxonomy.room_ids))
    logits[active] = torch.where(mask, torch.full_like(logits[active], 20.0), logits[active])
    return _copy_with_channel(controls, "room_logits", logits), _bbox(mask)


def _door_connectivity(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    structure = controls["structure"].clone()
    room_count = len(taxonomy.room_ids)
    wall = structure[room_count] > 0.5
    valid = torch.ones_like(wall)
    if footprint is not None:
        valid = _resize_mask(
            torch.from_numpy(np.asarray(footprint, dtype=bool)),
            tuple(wall.shape),
        )
    interior = ndimage.binary_erosion(valid.cpu().numpy(), iterations=4)
    candidates = wall & torch.from_numpy(interior)
    if not candidates.any():
        return _opening_detached(controls, bbox, taxonomy, footprint)
    rows, columns = torch.nonzero(candidates, as_tuple=True)
    row = int(rows[(len(rows) - 1) // 2])
    column = int(columns[(len(columns) - 1) // 2])
    mask = torch.zeros_like(wall)
    mask[max(row - 1, 0) : row + 2, max(column - 2, 0) : column + 3] = True
    structure[room_count + 1] = torch.where(
        mask,
        torch.ones_like(structure[room_count + 1]),
        structure[room_count + 1],
    )
    return _copy_with_channel(controls, "structure", structure), _bbox(mask)


def _room_label_mismatch(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    logits = controls["room_logits"].clone()
    program = controls["program"].bool()
    absent = (~program).nonzero(as_tuple=False).flatten()
    target = int(absent[0]) if len(absent) else 0
    mask = _central_patch(tuple(logits.shape[-2:]), bbox, fraction=0.18)
    logits[target] = torch.where(mask, torch.full_like(logits[target], 20.0), logits[target])
    return _copy_with_channel(controls, "room_logits", logits), bbox


def _outside_leakage(
    controls: dict[str, torch.Tensor],
    bbox: tuple[int, int, int, int],
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    logits = controls["room_logits"].clone()
    height, width = logits.shape[-2:]
    x, y, box_width, box_height = bbox
    mask = torch.zeros((height, width), dtype=torch.bool)
    far_x = width - max(2, min(width, x + box_width + 3))
    far_y = height - max(2, min(height, y + box_height + 3))
    mask[far_y : min(far_y + 2, height), far_x : min(far_x + 2, width)] = True
    if not bool(mask.any()):
        mask[-2:, -2:] = True
    active = _active_room(controls, bbox, len(taxonomy.room_ids))
    logits[active] = torch.where(mask, logits[active] + 0.75, logits[active])
    return _copy_with_channel(controls, "room_logits", logits), bbox


def build_candidate_specs(record: Any) -> tuple[CandidateSpec, ...]:
    """Create one positive, one legal alternative, and seven typed negatives."""

    group_id = f"{record.split}:{record.name}:{record.operation}"
    specs = [
        CandidateSpec(
            candidate_id=f"{group_id}:positive",
            group_id=group_id,
            role="positive",
            violation_family=None,
            severity=0,
            transform="identity",
            seed=record.seed,
            operation=record.operation,
        ),
        CandidateSpec(
            candidate_id=f"{group_id}:legal_alternative",
            group_id=group_id,
            role="legal_alternative",
            violation_family=None,
            severity=0,
            transform="argmax_preserving_room_reweight",
            seed=record.seed + 7919,
            operation=record.operation,
        ),
    ]
    for index, family in enumerate(VIOLATION_FAMILIES):
        specs.append(
            CandidateSpec(
                candidate_id=f"{group_id}:{family}",
                group_id=group_id,
                role="near_violation",
                violation_family=family,
                severity=1 + (record.seed + index) % 3,
                transform=family,
                seed=record.seed + 104729 * (index + 1),
                operation=record.operation,
            )
        )
    return tuple(specs)


def materialize_candidate(
    before: dict[str, torch.Tensor],
    positive_after: dict[str, torch.Tensor],
    record: Any,
    spec: CandidateSpec,
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
) -> MaterializedCandidate:
    """Apply a frozen candidate recipe without reading any split beyond train/val."""

    record_size = tuple(record.control_size)
    room_size = tuple(positive_after["room_logits"].shape[-2:])
    structure_size = tuple(positive_after["structure"].shape[-2:])
    room_bbox = _scale_bbox(tuple(record.bbox), record_size, room_size)
    structure_bbox = _scale_bbox(tuple(record.bbox), record_size, structure_size)
    if spec.transform == "identity":
        controls = copy.deepcopy(positive_after)
        bbox = tuple(record.bbox)
    elif spec.transform == "argmax_preserving_room_reweight":
        controls, _ = _legal_perturbation(positive_after, room_bbox, taxonomy)
        bbox = tuple(record.bbox)
    elif spec.transform == "room_overlap":
        controls, _ = _room_overlap(positive_after, room_bbox, taxonomy)
        bbox = tuple(record.bbox)
    elif spec.transform == "wall_gap":
        controls, local_bbox = _wall_gap(positive_after, structure_bbox, taxonomy)
        bbox = _union_bbox(
            tuple(record.bbox),
            _scale_bbox(local_bbox, structure_size, record_size),
        )
    elif spec.transform == "opening_detached":
        controls, local_bbox = _opening_detached(
            positive_after,
            structure_bbox,
            taxonomy,
            footprint,
        )
        bbox = _union_bbox(
            tuple(record.bbox),
            _scale_bbox(local_bbox, structure_size, record_size),
        )
    elif spec.transform == "room_outside_footprint":
        controls, local_bbox = _room_outside_footprint(
            positive_after,
            room_bbox,
            taxonomy,
            footprint,
        )
        bbox = _union_bbox(
            tuple(record.bbox),
            _scale_bbox(local_bbox, room_size, record_size),
        )
    elif spec.transform == "door_connectivity":
        controls, local_bbox = _door_connectivity(
            positive_after,
            structure_bbox,
            taxonomy,
            footprint,
        )
        bbox = _union_bbox(
            tuple(record.bbox),
            _scale_bbox(local_bbox, structure_size, record_size),
        )
    elif spec.transform == "room_label_mismatch":
        controls, _ = _room_label_mismatch(positive_after, room_bbox, taxonomy)
        bbox = tuple(record.bbox)
    elif spec.transform == "outside_leakage":
        controls, _ = _outside_leakage(positive_after, room_bbox, taxonomy)
        bbox = tuple(record.bbox)
    else:
        raise ValueError(f"unknown candidate transform: {spec.transform}")
    labels = {family: family == spec.violation_family for family in VIOLATION_FAMILIES}
    changed = sum(
        int((controls[key] != positive_after[key]).sum())
        for key in ("room_logits", "structure", "furniture", "support")
    )
    return MaterializedCandidate(
        controls=controls,
        bbox=bbox,
        changed_pixels=max(changed, 1),
        violation_labels=labels,
        transform=spec.transform,
    )


def constraint_diagnostics(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    record: Any,
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
) -> torch.Tensor:
    """Return continuous, label-free severity features in family order."""

    size = (128, 128)
    room_count = len(taxonomy.room_ids)
    after_room = F.interpolate(
        after["room_logits"].sigmoid()[None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]
    active = after_room > 0.5
    active_count = active.sum(dim=0)
    overlap = (active_count > 1).float().sum() / (active_count > 0).float().sum().clamp_min(1.0)
    if footprint is None:
        valid = torch.ones(size, dtype=torch.bool)
    else:
        valid = _resize_mask(
            torch.from_numpy(np.asarray(footprint, dtype=bool)),
            size,
        )
    outside = ~valid
    room_outside = (active & outside).float().sum() / active.float().sum().clamp_min(1.0)
    program = after["program"].bool()
    mismatch = (active & ~program[:, None, None]).float().sum() / active.float().sum().clamp_min(1.0)

    before_structure = F.interpolate(
        before["structure"].float()[None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]
    after_structure = F.interpolate(
        after["structure"].float()[None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]
    before_wall = before_structure[room_count] > 0.5
    wall = after_structure[room_count] > 0.5
    wall_gap = (
        (before_wall.float().sum() - wall.float().sum()).clamp_min(0.0)
        / before_wall.float().sum().clamp_min(1.0)
    )
    opening = after_structure[room_count + 1] > 0.5
    near_wall = F.max_pool2d(
        wall.float()[None, None],
        7,
        stride=1,
        padding=3,
    )[0, 0] > 0.5
    opening_detached = (
        (opening & ~near_wall).float().sum()
        / opening.float().sum().clamp_min(1.0)
    )
    near_rooms = F.max_pool2d(
        active.float(),
        7,
        stride=1,
        padding=3,
    ) > 0.5
    connected_rooms = near_rooms.sum(dim=0)
    outside_near = F.max_pool2d(
        outside.float()[None, None],
        7,
        stride=1,
        padding=3,
    )[0, 0] > 0.5
    interior_opening = opening & ~outside_near
    door_connectivity = (
        (interior_opening & (connected_rooms < 2)).float().sum()
        / interior_opening.float().sum().clamp_min(1.0)
    )

    x, y, box_width, box_height = record.bbox
    record_width, record_height = record.control_size
    outside_changes = 0
    total_changes = 0
    for key in ("room_logits", "structure", "furniture", "support"):
        delta = (after[key].float() - before[key].float()).abs()
        while delta.ndim > 2:
            delta = delta.amax(dim=0)
        height, width = delta.shape
        left = round(x * width / max(record_width, 1))
        top = round(y * height / max(record_height, 1))
        right = round((x + box_width) * width / max(record_width, 1))
        bottom = round((y + box_height) * height / max(record_height, 1))
        changed = delta > 0
        inside = torch.zeros_like(changed)
        inside[
            max(top, 0) : min(bottom, height),
            max(left, 0) : min(right, width),
        ] = True
        total_changes += int(changed.sum())
        outside_changes += int((changed & ~inside).sum())
    leakage = outside_changes / max(total_changes, 1)
    return torch.stack(
        [
            overlap,
            wall_gap,
            opening_detached,
            room_outside,
            door_connectivity,
            mismatch,
            torch.tensor(leakage),
        ]
    ).float()


def is_hard_legal(
    hard_flags: dict[str, bool],
    violation_labels: dict[str, bool],
) -> bool:
    return all(hard_flags.values()) and not any(
        violation_labels.get(family, False) for family in HARD_VIOLATION_FAMILIES
    )


__all__ = [
    "HARD_VIOLATION_FAMILIES",
    "VIOLATION_FAMILIES",
    "CandidateSpec",
    "MaterializedCandidate",
    "build_candidate_specs",
    "constraint_diagnostics",
    "is_hard_legal",
    "materialize_candidate",
]
