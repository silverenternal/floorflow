"""Deterministic local edits on serialized coarse floorplan controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class LocalEdit:
    controls: dict[str, torch.Tensor]
    operation: str
    control_key: str
    bbox: tuple[int, int, int, int]
    control_size: tuple[int, int]
    changed_pixels: int


def _component(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
) -> np.ndarray | None:
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    areas = np.bincount(labels.ravel())
    candidates = [
        index
        for index in range(1, count + 1)
        if minimum <= int(areas[index]) <= maximum
    ]
    if not candidates:
        return None
    selected = int(rng.choice(candidates))
    return labels == selected


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("edit mask is empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def edit_room_type(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int,
) -> LocalEdit | None:
    """Replace one room-class connected component with another present class."""

    logits = controls["room_logits"].clone()
    class_map = logits.argmax(dim=0).cpu().numpy()
    rng = np.random.default_rng(seed)
    candidates = []
    for class_index in range(len(taxonomy.room_ids)):
        component = _component(
            class_map == class_index,
            rng,
            minimum=40,
            maximum=max(41, class_map.size // 3),
        )
        if component is not None:
            candidates.append((class_index, component))
    if len(candidates) < 2:
        return None
    source_index, source_component = candidates[int(rng.integers(len(candidates)))]
    alternatives = [
        item for item in candidates if item[0] != source_index
    ]
    target_index, _ = alternatives[int(rng.integers(len(alternatives)))]
    edit_mask = torch.from_numpy(source_component)
    logits[source_index][edit_mask] -= 12.0
    logits[target_index][edit_mask] += 12.0
    output = dict(controls)
    output["room_logits"] = logits
    return LocalEdit(
        controls=output,
        operation="room_type_change",
        control_key="room_logits",
        bbox=_bbox(source_component),
        control_size=tuple(logits.shape[-2:]),
        changed_pixels=int(source_component.sum()),
    )


def _room_components(
    class_map: np.ndarray,
    room_count: int,
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
) -> list[tuple[int, np.ndarray]]:
    components = []
    for class_index in range(room_count):
        labels, count = ndimage.label(class_map == class_index)
        if count == 0:
            continue
        areas = np.bincount(labels.ravel())
        for component_index in range(1, count + 1):
            if minimum <= int(areas[component_index]) <= maximum:
                components.append(
                    (class_index, labels == component_index)
                )
    rng.shuffle(components)
    return components


def edit_room_add(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int,
) -> LocalEdit | None:
    """Split one room component and assign half to an absent room type."""

    logits = controls["room_logits"].clone()
    class_map = logits.argmax(dim=0).cpu().numpy()
    program = controls["program"].clone().bool()
    absent = np.flatnonzero(~program.cpu().numpy())
    if absent.size == 0:
        return None
    rng = np.random.default_rng(seed)
    candidates = _room_components(
        class_map,
        len(taxonomy.room_ids),
        rng,
        minimum=200,
        maximum=max(201, class_map.size // 4),
    )
    if not candidates:
        return None
    source_index, component = candidates[0]
    rows, columns = np.nonzero(component)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    split_mask = np.zeros_like(component)
    if right - left >= bottom - top:
        midpoint = (left + right) // 2
        split_mask[top:bottom, midpoint:right] = True
    else:
        midpoint = (top + bottom) // 2
        split_mask[midpoint:bottom, left:right] = True
    split_mask &= component
    if int(split_mask.sum()) < 40:
        return None
    target_index = int(absent[int(rng.integers(len(absent)))])
    edit_mask = torch.from_numpy(split_mask)
    logits[source_index][edit_mask] -= 12.0
    logits[target_index][edit_mask] += 12.0
    program[target_index] = True
    output = dict(controls)
    output["room_logits"] = logits
    output["program"] = program
    return LocalEdit(
        controls=output,
        operation="room_add",
        control_key="room_logits",
        bbox=_bbox(split_mask),
        control_size=tuple(logits.shape[-2:]),
        changed_pixels=int(split_mask.sum()),
    )


def edit_room_remove(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int,
) -> LocalEdit | None:
    """Merge one room component into its strongest adjacent room."""

    logits = controls["room_logits"].clone()
    class_map = logits.argmax(dim=0).cpu().numpy()
    program = controls["program"].clone().bool()
    rng = np.random.default_rng(seed)
    candidates = _room_components(
        class_map,
        len(taxonomy.room_ids),
        rng,
        minimum=40,
        maximum=max(41, class_map.size // 4),
    )
    for source_index, component in candidates:
        neighbors = ndimage.binary_dilation(component, iterations=4) & ~component
        neighbor_labels = class_map[neighbors]
        neighbor_labels = neighbor_labels[
            (neighbor_labels >= 0)
            & (neighbor_labels < len(taxonomy.room_ids))
            & (neighbor_labels != source_index)
        ]
        if neighbor_labels.size == 0:
            continue
        counts = np.bincount(neighbor_labels)
        target_index = int(counts.argmax())
        edit_mask = torch.from_numpy(component)
        logits[source_index][edit_mask] -= 12.0
        logits[target_index][edit_mask] += 12.0
        if not ((class_map == source_index) & ~component).any():
            program[source_index] = False
        output = dict(controls)
        output["room_logits"] = logits
        output["program"] = program
        return LocalEdit(
            controls=output,
            operation="room_remove",
            control_key="room_logits",
            bbox=_bbox(component),
            control_size=tuple(logits.shape[-2:]),
            changed_pixels=int(component.sum()),
        )
    return None


def edit_opening_delete(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int,
) -> LocalEdit | None:
    """Delete one connected opening component from the structure control."""

    structure = controls["structure"].clone()
    room_count = len(taxonomy.room_ids)
    gate = structure[room_count + 1].cpu().numpy() >= 0.5
    rng = np.random.default_rng(seed)
    component = _component(
        gate,
        rng,
        minimum=2,
        maximum=max(3, gate.size // 20),
    )
    if component is None:
        return None
    edit_mask = torch.from_numpy(component)
    structure[room_count + 1][edit_mask] = 0.0
    structure[
        room_count + 2 : room_count + 2 + len(taxonomy.opening_ids)
    ][:, edit_mask] = 0.0
    output = dict(controls)
    output["structure"] = structure
    return LocalEdit(
        controls=output,
        operation="opening_delete",
        control_key="structure",
        bbox=_bbox(component),
        control_size=tuple(structure.shape[-2:]),
        changed_pixels=int(component.sum()),
    )


def edit_furniture_delete(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int,
) -> LocalEdit | None:
    """Delete one connected furniture component from the furniture control."""

    furniture = controls["furniture"].clone()
    room_count = len(taxonomy.room_ids)
    class_logits = furniture[room_count:-1]
    present = (class_logits.max(dim=0).values > 0.5).cpu().numpy()
    rng = np.random.default_rng(seed)
    component = _component(
        present,
        rng,
        minimum=3,
        maximum=max(4, present.size // 30),
    )
    if component is None:
        return None
    edit_mask = torch.from_numpy(component)
    class_logits[:, edit_mask] = 0.0
    furniture[room_count:-1] = class_logits
    output = dict(controls)
    output["furniture"] = furniture
    return LocalEdit(
        controls=output,
        operation="furniture_delete",
        control_key="furniture",
        bbox=_bbox(component),
        control_size=tuple(furniture.shape[-2:]),
        changed_pixels=int(component.sum()),
    )


LOCAL_EDIT_OPERATIONS = {
    "room_add": edit_room_add,
    "room_remove": edit_room_remove,
    "room_type": edit_room_type,
    "opening_delete": edit_opening_delete,
    "furniture_delete": edit_furniture_delete,
}


__all__ = [
    "LOCAL_EDIT_OPERATIONS",
    "LocalEdit",
    "edit_furniture_delete",
    "edit_opening_delete",
    "edit_room_add",
    "edit_room_remove",
    "edit_room_type",
]
