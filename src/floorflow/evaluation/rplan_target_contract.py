"""Audit one tensorized RPLAN sample against model target contracts."""

from __future__ import annotations

import torch


def _is_prefix(mask: torch.Tensor) -> bool:
    valid = mask.bool().flatten()
    count = int(valid.sum())
    return bool(valid[:count].all())


def audit_rplan_target_item(item: dict[str, object]) -> dict[str, object]:
    """Return deterministic violations for one collated-compatible item."""

    violations: list[str] = []
    room_exists = item["room_exists"].bool().flatten()
    room_count = int(item["room_count"])
    source_room_count = int(item.get("source_room_count", room_count))
    if room_count != int(room_exists.sum()):
        violations.append("room_count_mismatch")
    if room_count > room_exists.numel():
        violations.append("room_count_out_of_range")
    if source_room_count < room_count:
        violations.append("source_room_count_smaller")
    if not _is_prefix(room_exists):
        violations.append("room_valid_nonprefix")

    adjacency = item["adjacency"].bool()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        violations.append("adjacency_not_square")
    else:
        if not torch.equal(adjacency, adjacency.T):
            violations.append("adjacency_asymmetric")
        if torch.diagonal(adjacency).any():
            violations.append("adjacency_self_edge")
        slots = min(adjacency.shape[0], room_exists.numel())
        inactive = ~room_exists[:slots]
        if adjacency[:slots, :slots][inactive].any() or adjacency[:slots, :slots][:, inactive].any():
            violations.append("adjacency_inactive_room")

    point_valid = item["room_point_valid"].bool()
    for room in torch.nonzero(room_exists, as_tuple=False).flatten().tolist():
        if int(point_valid[room].sum()) < 3:
            violations.append("room_has_too_few_valid_points")
        elif not _is_prefix(point_valid[room]):
            violations.append("room_point_valid_nonprefix")
        if not torch.isfinite(item["room_polygons"][room][point_valid[room]]).all():
            violations.append("room_polygon_nonfinite")

    boundary_valid = item["boundary_valid"].bool().flatten()
    if int(boundary_valid.sum()) < 3:
        violations.append("boundary_too_few_points")
    elif not _is_prefix(boundary_valid):
        violations.append("boundary_valid_nonprefix")

    wall_valid = item["wall_valid"].bool().flatten()
    opening_valid = item["opening_valid"].bool().flatten()
    opening_host = item["opening_host"].long().flatten()
    if opening_valid.shape != opening_host.shape:
        violations.append("opening_host_shape_mismatch")
    else:
        hosts = opening_host[opening_valid]
        if hosts.numel() and (
            (hosts < 0).any()
            or (hosts >= wall_valid.numel()).any()
            or not wall_valid[hosts.clamp(0, wall_valid.numel() - 1)].all()
        ):
            violations.append("opening_invalid_host")
        if (~opening_valid & (opening_host >= 0)).any():
            violations.append("inactive_opening_has_host")

    junction_valid = item["junction_valid"].bool().flatten()
    junction_degree = item["junction_degree"].long().flatten()
    if junction_valid.shape != junction_degree.shape:
        violations.append("junction_shape_mismatch")
    elif (junction_degree[junction_valid] < 0).any():
        violations.append("junction_negative_degree")

    unique = tuple(sorted(set(violations)))
    return {
        "passed": not unique,
        "violations": unique,
        "room_count": room_count,
        "source_room_count": source_room_count,
        "valid_room_points": [
            int(point_valid[room].sum())
            for room in torch.nonzero(room_exists, as_tuple=False).flatten().tolist()
        ],
        "wall_count": int(wall_valid.sum()),
        "opening_count": int(opening_valid.sum()),
        "junction_count": int(junction_valid.sum()),
    }


__all__ = ["audit_rplan_target_item"]
