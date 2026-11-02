"""Differentiable geometry operators shared by the hard-constraint projector."""

from __future__ import annotations

import torch

from floorflow.chip_flow.model import decode_boxes_batch


def boxes_to_edges(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left, bottom, right and top box edges."""

    return (
        boxes[..., 0],
        boxes[..., 1],
        boxes[..., 0] + boxes[..., 2],
        boxes[..., 1] + boxes[..., 3],
    )


def resolve_projector_config(
    model: torch.nn.Module,
    sparse_pairs: bool | None,
    projection_strategy: str | None,
    grid_size: int | None,
) -> tuple[bool, str, int]:
    """Resolve explicit runtime overrides against checkpoint model defaults."""

    return (
        bool(getattr(model, "projector_sparse_pairs", True))
        if sparse_pairs is None
        else bool(sparse_pairs),
        str(getattr(model, "projector_strategy", "legacy"))
        if projection_strategy is None
        else str(projection_strategy),
        int(getattr(model, "projector_grid_size", 16))
        if grid_size is None
        else int(grid_size),
    )


def candidate_overlap_pairs(
    boxes: torch.Tensor,
    *,
    grid_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unique upper-triangle box pairs sharing an occupancy-grid cell."""

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    _batch, count, _ = boxes.shape
    left, bottom, right, top = boxes_to_edges(boxes)
    chart_left = left.amin(dim=1)
    chart_bottom = bottom.amin(dim=1)
    chart_right = right.amax(dim=1)
    chart_top = top.amax(dim=1)
    cell_width = ((chart_right - chart_left) / grid_size).clamp_min(1.0e-6)
    cell_height = ((chart_top - chart_bottom) / grid_size).clamp_min(1.0e-6)
    low_x = ((left - chart_left[:, None]) / cell_width[:, None]).floor().long()
    high_x = ((right - chart_left[:, None]) / cell_width[:, None]).floor().long()
    low_y = ((bottom - chart_bottom[:, None]) / cell_height[:, None]).floor().long()
    high_y = ((top - chart_bottom[:, None]) / cell_height[:, None]).floor().long()
    candidate = (
        (low_x[:, :, None] <= high_x[:, None, :])
        & (low_x[:, None, :] <= high_x[:, :, None])
        & (low_y[:, :, None] <= high_y[:, None, :])
        & (low_y[:, None, :] <= high_y[:, :, None])
        & torch.triu(
            torch.ones(count, count, dtype=torch.bool, device=boxes.device),
            diagonal=1,
        )[None]
    )
    batch_index, first, second = candidate.nonzero(as_tuple=True)
    return batch_index, first, second


def _pair_overlap(
    boxes: torch.Tensor,
    batch_index: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = boxes[batch_index]
    first_boxes = selected[torch.arange(selected.shape[0]), first]
    second_boxes = selected[torch.arange(selected.shape[0]), second]
    first_left, first_bottom, first_right, first_top = boxes_to_edges(first_boxes)
    second_left, second_bottom, second_right, second_top = boxes_to_edges(
        second_boxes
    )
    overlap_x = (
        torch.minimum(first_right, second_right)
        - torch.maximum(first_left, second_left)
    ).clamp_min(0.0)
    overlap_y = (
        torch.minimum(first_top, second_top)
        - torch.maximum(first_bottom, second_bottom)
    ).clamp_min(0.0)
    active = (overlap_x > 0.0) & (overlap_y > 0.0)
    return overlap_x, overlap_y, active


def overlap_pair_count(boxes: torch.Tensor, *, grid_size: int = 16) -> int:
    """Count exact overlapping upper-triangle pairs after grid filtering."""

    batch_index, first, second = candidate_overlap_pairs(
        boxes,
        grid_size=grid_size,
    )
    if first.numel() == 0:
        return 0
    _, _, active = _pair_overlap(boxes, batch_index, first, second)
    return int(active.sum())


def separate_overlaps(
    boxes: torch.Tensor,
    movable: torch.Tensor,
    constraints: torch.Tensor,
    *,
    boundary_axes_locked: bool = False,
    lock_boundary_corners: bool = True,
    sparse_pairs: bool = True,
    grid_size: int = 16,
) -> torch.Tensor:
    """Separate overlapping boxes with an out-of-place differentiable update."""

    batch, count, _ = boxes.shape
    if sparse_pairs:
        batch_index, first, second = candidate_overlap_pairs(
            boxes,
            grid_size=grid_size,
        )
        if first.numel() == 0:
            return boxes
        overlap_x, overlap_y, active = _pair_overlap(
            boxes,
            batch_index,
            first,
            second,
        )
    else:
        first, second = torch.triu_indices(
            count,
            count,
            offset=1,
            device=boxes.device,
        )
        batch_index = torch.arange(batch, device=boxes.device)[:, None]
        batch_index = batch_index.expand(-1, first.numel()).reshape(-1)
        first = first[None].expand(batch, -1).reshape(-1)
        second = second[None].expand(batch, -1).reshape(-1)
        left, bottom, right, top = boxes_to_edges(boxes)
        overlap_x = (
            torch.minimum(right[:, :, None], right[:, None, :])
            - torch.maximum(left[:, :, None], left[:, None, :])
        ).clamp_min(0.0)[
            torch.arange(batch, device=boxes.device)[:, None],
            first.reshape(batch, -1),
            second.reshape(batch, -1),
        ].reshape(-1)
        overlap_y = (
            torch.minimum(top[:, :, None], top[:, None, :])
            - torch.maximum(bottom[:, :, None], bottom[:, None, :])
        ).clamp_min(0.0)[
            torch.arange(batch, device=boxes.device)[:, None],
            first.reshape(batch, -1),
            second.reshape(batch, -1),
        ].reshape(-1)
        active = (overlap_x > 0.0) & (overlap_y > 0.0)

    if not bool(active.any()):
        return boxes
    centers = boxes[..., :2] + boxes[..., 2:] / 2.0
    move = movable.float()
    if boundary_axes_locked:
        codes = constraints[..., 4].long()
        move_x = move * (codes & 3 == 0).float()
        move_y = move * (codes & 12 == 0).float()
        if not lock_boundary_corners:
            corners = (codes & 3 != 0) & (codes & 12 != 0)
            move_x = torch.where(corners, move, move_x)
            move_y = torch.where(corners, move, move_y)
    else:
        move_x, move_y = move, move
    axis_x = overlap_x <= overlap_y
    penetration = torch.where(axis_x, overlap_x, overlap_y) + 1.0e-4
    first_move = torch.where(
        axis_x,
        move_x[batch_index, first],
        move_y[batch_index, first],
    )
    second_move = torch.where(
        axis_x,
        move_x[batch_index, second],
        move_y[batch_index, second],
    )
    move_total = first_move + second_move
    valid = active & (move_total > 0.0)
    delta_x = centers[batch_index, first, 0] - centers[batch_index, second, 0]
    delta_y = centers[batch_index, first, 1] - centers[batch_index, second, 1]
    direction_x = torch.where(delta_x.eq(0), -torch.ones_like(delta_x), delta_x.sign())
    direction_y = torch.where(delta_y.eq(0), -torch.ones_like(delta_y), delta_y.sign())
    direction = torch.where(axis_x, direction_x, direction_y)
    first_shift = torch.where(
        valid,
        direction * penetration * first_move / move_total.clamp_min(1.0),
        torch.zeros_like(direction),
    )
    second_shift = torch.where(
        valid,
        -direction * penetration * second_move / move_total.clamp_min(1.0),
        torch.zeros_like(direction),
    )
    flat_dx = boxes.new_zeros(batch * count)
    flat_dy = boxes.new_zeros(batch * count)
    first_flat = batch_index * count + first
    second_flat = batch_index * count + second
    use_x = axis_x & valid
    use_y = (~axis_x) & valid
    flat_dx.index_add_(0, first_flat[use_x], first_shift[use_x])
    flat_dx.index_add_(0, second_flat[use_x], second_shift[use_x])
    flat_dy.index_add_(0, first_flat[use_y], first_shift[use_y])
    flat_dy.index_add_(0, second_flat[use_y], second_shift[use_y])
    output_x = boxes[..., 0] + flat_dx.reshape(batch, count)
    output_y = boxes[..., 1] + flat_dy.reshape(batch, count)
    return torch.stack([output_x, output_y, boxes[..., 2], boxes[..., 3]], dim=-1)


def project_boundary(
    latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Snap boundary-coded blocks with only preplaced positions held fixed."""

    boxes = decode_boxes_batch(latent, areas)
    left, bottom, right, top = boxes_to_edges(boxes)
    chart_left = left.amin(dim=1)
    chart_bottom = bottom.amin(dim=1)
    chart_right = right.amax(dim=1)
    chart_top = top.amax(dim=1)
    width = boxes[..., 2]
    height = boxes[..., 3]
    codes = constraints[..., 4].long()
    movable = ~constraints[..., 1].ne(0)
    centers_x = boxes[..., 0] + width / 2.0
    centers_y = boxes[..., 1] + height / 2.0
    centers_x = torch.where(
        (codes & 1).ne(0) & movable,
        chart_left[:, None] + width / 2.0,
        centers_x,
    )
    centers_x = torch.where(
        (codes & 2).ne(0) & movable,
        chart_right[:, None] - width / 2.0,
        centers_x,
    )
    centers_y = torch.where(
        (codes & 4).ne(0) & movable,
        chart_top[:, None] - height / 2.0,
        centers_y,
    )
    centers_y = torch.where(
        (codes & 8).ne(0) & movable,
        chart_bottom[:, None] + height / 2.0,
        centers_y,
    )
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    return torch.stack(
        [centers_x / scale[:, None], centers_y / scale[:, None], latent[..., 2]],
        dim=-1,
    )


def boundary_violation_count(
    boxes: torch.Tensor,
    constraints: torch.Tensor,
) -> int:
    left, bottom, right, top = boxes_to_edges(boxes)
    chart_left = left.amin(dim=1)
    chart_bottom = bottom.amin(dim=1)
    chart_right = right.amax(dim=1)
    chart_top = top.amax(dim=1)
    codes = constraints[..., 4].long()
    deltas = torch.stack(
        (
            left - chart_left[:, None],
            chart_right[:, None] - right,
            chart_top[:, None] - top,
            bottom - chart_bottom[:, None],
        ),
        dim=1,
    )
    bits = boxes.new_tensor((1, 2, 4, 8)).long()
    invalid = (codes[:, None, :] & bits[None, :, None]).ne(0) & deltas.abs().gt(
        1.0e-6
    )
    return int(invalid.sum())


def mib_shape_violation_count(
    boxes: torch.Tensor,
    constraints: torch.Tensor,
) -> int:
    mib = constraints[..., 2].long()
    valid = mib > 0
    if not bool(valid.any()):
        return 0
    groups, inverse = torch.unique(mib[valid], return_inverse=True)
    dense = torch.zeros_like(mib)
    dense[valid] = inverse + 1
    shapes = boxes[..., 2:].round(decimals=4)
    same_shape = shapes[:, :, None].eq(shapes[:, None, :]).all(dim=-1)
    same_group = dense[:, :, None].eq(dense[:, None, :])
    positions = torch.arange(mib.shape[1], device=mib.device)
    earlier = positions[None, :, None].gt(positions[None, None, :])
    representative = valid & ~(same_shape & same_group & earlier).any(dim=-1)
    counts = torch.zeros(
        (mib.shape[0], len(groups) + 1),
        dtype=torch.long,
        device=mib.device,
    ).scatter_add(1, dense, representative.long())
    return int((counts[:, 1:] - 1).clamp_min(0).sum())


def joint_conflict_energy(
    latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    grid_size: int = 16,
) -> torch.Tensor:
    """Per-sample soft energy for overlap, boundary and MIB conflicts."""

    boxes = decode_boxes_batch(latent, areas)
    batch, _count, _ = boxes.shape
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    batch_index, first, second = candidate_overlap_pairs(
        boxes,
        grid_size=grid_size,
    )
    overlap_area = latent.new_zeros(batch)
    if first.numel():
        overlap_x, overlap_y, active = _pair_overlap(
            boxes,
            batch_index,
            first,
            second,
        )
        overlap_area = overlap_area.index_add(
            0,
            batch_index,
            overlap_x * overlap_y * active.to(latent.dtype),
        )
    left, bottom, right, top = boxes_to_edges(boxes)
    chart_left = left.amin(dim=1)
    chart_bottom = bottom.amin(dim=1)
    chart_right = right.amax(dim=1)
    chart_top = top.amax(dim=1)
    codes = constraints[..., 4].long()
    boundary = (
        (codes & 1).ne(0) * (left - chart_left[:, None]).abs()
        + (codes & 2).ne(0) * (chart_right[:, None] - right).abs()
        + (codes & 4).ne(0) * (chart_top[:, None] - top).abs()
        + (codes & 8).ne(0) * (bottom - chart_bottom[:, None]).abs()
    ).sum(dim=1)
    rho = latent[..., 2]
    mib = constraints[..., 2].long()
    mib_energy = torch.zeros(batch, device=latent.device, dtype=latent.dtype)
    valid_mib = mib > 0
    groups = torch.unique(mib[valid_mib])
    if groups.numel():
        member = mib[:, :, None].eq(groups[None, None, :]) & valid_mib[:, :, None]
        member_f = member.to(latent.dtype)
        count_member = member_f.sum(dim=1).clamp_min(1.0)
        mean = (rho[:, :, None] * member_f).sum(dim=1) / count_member
        mib_energy = (
            (rho[:, :, None] - mean[:, None, :]).square() * member_f
        ).sum(dim=(1, 2))
    return (
        overlap_area / scale.square()
        + boundary / scale
        + mib_energy
    )


__all__ = [
    "boundary_violation_count",
    "candidate_overlap_pairs",
    "joint_conflict_energy",
    "mib_shape_violation_count",
    "overlap_pair_count",
    "project_boundary",
    "resolve_projector_config",
    "separate_overlaps",
]
