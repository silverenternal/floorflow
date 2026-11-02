"""Typed non-uniform routing for local floorplan edits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FAMILIES = ("room", "wall", "opening", "furniture")
POLICIES = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)


@dataclass(frozen=True)
class TypedEditGraph:
    """Semantic masks induced by one declared edit."""

    anchor: np.ndarray
    affected_rooms: np.ndarray
    adjacent_rooms: np.ndarray
    adjacent_walls: np.ndarray
    global_outline: np.ndarray
    weak_connections: np.ndarray
    changed_families: tuple[str, ...]


@dataclass(frozen=True)
class TypedEditComposition:
    """Composited labels for one graph-routing policy."""

    visible: np.ndarray
    room_labels: np.ndarray
    topology: np.ndarray
    routing_mask: np.ndarray
    outside_region_changes: int


def _binary(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


def _family(labels: np.ndarray, value: int, ids: Sequence[int]) -> np.ndarray:
    return labels == int(value) if len(ids) == 0 else np.isin(labels, ids)


def family_masks(
    labels: np.ndarray,
    *,
    room_ids: Sequence[int],
    wall_id: int,
    opening_ids: Sequence[int],
    furniture_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Map class labels to the four semantic families."""

    return {
        "room": _family(labels, 0, tuple(room_ids)),
        "wall": labels == int(wall_id),
        "opening": _family(labels, 0, tuple(opening_ids)),
        "furniture": _family(labels, 0, tuple(furniture_ids)),
    }


def build_typed_edit_graph(
    before_labels: np.ndarray,
    edit_region: np.ndarray,
    changed_mask: np.ndarray,
    *,
    after_labels: np.ndarray | None = None,
    room_ids: Sequence[int],
    wall_id: int,
    opening_ids: Sequence[int],
    furniture_ids: Sequence[int],
    changed_families: Sequence[str] | None = None,
) -> TypedEditGraph:
    """Build anchor-to-room-to-wall routing masks for one edit."""

    labels = np.asarray(before_labels)
    after = None if after_labels is None else np.asarray(after_labels)
    region = _binary(edit_region, "edit_region")
    changed = _binary(changed_mask, "changed_mask") & region
    if labels.shape != region.shape or labels.shape != changed.shape:
        raise ValueError("labels, edit_region, and changed_mask must share a shape")
    if after is not None and after.shape != labels.shape:
        raise ValueError("before and after labels must share a shape")
    families = family_masks(
        labels,
        room_ids=room_ids,
        wall_id=wall_id,
        opening_ids=opening_ids,
        furniture_ids=furniture_ids,
    )
    room_components, _count = ndimage.label(families["room"])
    selected = np.unique(room_components[changed & families["room"]])
    selected = selected[selected > 0]
    if after is not None:
        after_families = family_masks(
            after,
            room_ids=room_ids,
            wall_id=wall_id,
            opening_ids=opening_ids,
            furniture_ids=furniture_ids,
        )
        after_room_components, _after_count = ndimage.label(
            after_families["room"]
        )
        after_selected = np.unique(
            after_room_components[changed & after_families["room"]]
        )
        after_selected = after_selected[after_selected > 0]
    else:
        after_families = None
        after_room_components = None
        after_selected = np.zeros(0, dtype=np.int64)
    if selected.size:
        affected = np.isin(room_components, selected)
    elif after_room_components is not None and after_selected.size:
        affected = np.isin(after_room_components, after_selected)
    else:
        affected = ndimage.binary_dilation(
            changed & families["room"],
            iterations=2,
        )
    adjacent_rooms = (
        ndimage.binary_dilation(affected, iterations=2)
        & families["room"]
        & ~affected
    )
    before_walls = families["wall"]
    wall_family = (
        before_walls
        if after_families is None
        else before_walls | after_families["wall"]
    )
    adjacent_walls = (
        ndimage.binary_dilation(affected | changed, iterations=2)
        & wall_family
    )
    eroded = ndimage.binary_erosion(region, iterations=2)
    global_outline = region & ~eroded
    weak = region & ~(
        changed
        | affected
        | adjacent_rooms
        | adjacent_walls
        | global_outline
    )
    if changed_families is None:
        family_sets = [families]
        if after_families is not None:
            family_sets.append(after_families)
        resolved_families = tuple(
            name
            for name in FAMILIES
            if any(bool((changed & source[name]).any()) for source in family_sets)
        )
    else:
        resolved_families = tuple(
            name for name in changed_families if name in FAMILIES
        )
    return TypedEditGraph(
        anchor=changed,
        affected_rooms=affected,
        adjacent_rooms=adjacent_rooms,
        adjacent_walls=adjacent_walls,
        global_outline=global_outline,
        weak_connections=weak,
        changed_families=resolved_families,
    )


def routing_mask(
    graph: TypedEditGraph,
    edit_region: np.ndarray,
    footprint: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    """Return the pixels whose rendered post-edit values are accepted."""

    region = _binary(edit_region, "edit_region")
    support = _binary(footprint, "footprint")
    if policy == "typed_edit_graph":
        allowed = graph.anchor.copy()
        if "room" in graph.changed_families:
            allowed |= graph.affected_rooms | graph.adjacent_rooms
        if "wall" in graph.changed_families or "opening" in graph.changed_families:
            allowed |= graph.adjacent_walls
        return allowed & region & support
    if policy == "dense_cross_attention":
        return ndimage.binary_dilation(region, iterations=2) & support
    if policy == "no_edit_gating":
        return support.copy()
    if policy == "no_structural_graph":
        return region & support
    raise ValueError(f"unknown routing policy: {policy}")


def compose_typed_edit(
    before: np.ndarray,
    after: np.ndarray,
    before_rooms: np.ndarray,
    after_rooms: np.ndarray,
    before_topology: np.ndarray,
    after_topology: np.ndarray,
    graph: TypedEditGraph,
    edit_region: np.ndarray,
    footprint: np.ndarray,
    *,
    policy: str,
) -> TypedEditComposition:
    """Apply one routing policy to visible, room, and topology outputs."""

    arrays = (
        before,
        after,
        before_rooms,
        after_rooms,
        before_topology,
        after_topology,
    )
    if any(np.asarray(array).shape != np.asarray(before).shape for array in arrays):
        raise ValueError("all rendered arrays must share a shape")
    accepted = routing_mask(
        graph,
        edit_region,
        footprint,
        policy=policy,
    )
    outside_region = _binary(footprint, "footprint") & ~_binary(
        edit_region,
        "edit_region",
    )
    visible = np.where(accepted, after, before).astype(before.dtype, copy=False)
    return TypedEditComposition(
        visible=visible,
        room_labels=np.where(accepted, after_rooms, before_rooms).astype(
            before_rooms.dtype,
            copy=False,
        ),
        topology=np.where(accepted, after_topology, before_topology).astype(
            before_topology.dtype,
            copy=False,
        ),
        routing_mask=accepted,
        outside_region_changes=int(
            np.count_nonzero(
                visible[outside_region] != before[outside_region]
            )
        ),
    )


__all__ = [
    "FAMILIES",
    "POLICIES",
    "TypedEditComposition",
    "TypedEditGraph",
    "build_typed_edit_graph",
    "compose_typed_edit",
    "family_masks",
    "routing_mask",
]
