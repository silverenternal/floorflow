"""Vectorized flow losses and training physics for same-size chip batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.objectives.token_optimal_transport import paired_token_mse


def batch_physics_terms(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_b2b: torch.Tensor,
    n_p2b: torch.Tensor,
    n_pins: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate the five differentiable training terms for ``[B, N, ...]``.

    Grouping is optimized separately: its exact minimum-spanning-tree value is
    not differentiable in the reference implementation. It remains in the
    evaluation report and is not silently replaced by a surrogate.
    """

    x = boxes[..., 0]
    y = boxes[..., 1]
    width = boxes[..., 2]
    height = boxes[..., 3]
    x1 = x + width
    y1 = y + height
    left = torch.maximum(x[:, :, None], x[:, None, :])
    right = torch.minimum(x1[:, :, None], x1[:, None, :])
    top = torch.maximum(y[:, :, None], y[:, None, :])
    bottom = torch.minimum(y1[:, :, None], y1[:, None, :])
    intersection = (right - left).clamp_min(0.0) * (
        bottom - top
    ).clamp_min(0.0)
    valid = ~torch.eye(
        boxes.shape[1],
        dtype=torch.bool,
        device=boxes.device,
    )
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    overlap = intersection[:, valid].sum(dim=1) * 0.5 / scale.square()

    log_width = width.clamp_min(1.0e-6).log()
    log_height = height.clamp_min(1.0e-6).log()
    mib = torch.zeros_like(scale)
    mib_ids = constraints[..., 2]
    for group in torch.unique(mib_ids[mib_ids > 0]):
        members = mib_ids == group
        count = members.sum(dim=1)
        denominator = count.clamp_min(1)
        mean_width = (
            log_width.masked_fill(~members, 0.0).sum(dim=1)
            / denominator
        )
        mean_height = (
            log_height.masked_fill(~members, 0.0).sum(dim=1)
            / denominator
        )
        deviation = (
            (log_width - mean_width[:, None]).abs()
            + (log_height - mean_height[:, None]).abs()
        )
        mib = mib + torch.where(count > 1, (deviation * members).sum(1), 0.0)

    x_min = x.min(dim=1).values
    y_min = y.min(dim=1).values
    x_max = x1.max(dim=1).values
    y_max = y1.max(dim=1).values
    codes = constraints[..., 4].to(dtype=torch.long)
    boundary = (
        (codes & 1).to(boxes.dtype) * (x - x_min[:, None]).abs()
        + (codes & 2).to(boxes.dtype) * (x_max[:, None] - x - width).abs()
        + (codes & 4).to(boxes.dtype) * (y_max[:, None] - y - height).abs()
        + (codes & 8).to(boxes.dtype) * (y - y_min[:, None]).abs()
    ).sum(dim=1) / scale

    centers = boxes[..., :2] + boxes[..., 2:] * 0.5
    edge_rows = torch.arange(b2b.shape[1], device=b2b.device)[None, :]
    edge_mask = edge_rows < n_b2b[:, None]
    left_id = b2b[..., 0].long().clamp(0, boxes.shape[1] - 1)
    right_id = b2b[..., 1].long().clamp(0, boxes.shape[1] - 1)
    gather_left = left_id[..., None].expand(-1, -1, 2)
    gather_right = right_id[..., None].expand(-1, -1, 2)
    block_distance = (
        centers.gather(1, gather_left) - centers.gather(1, gather_right)
    ).abs().sum(dim=2)
    block_weights = b2b[..., 2] * edge_mask
    block_total = (block_distance * block_weights).sum(dim=1)
    block_weight_sum = block_weights.sum(dim=1)

    pin_rows = torch.arange(p2b.shape[1], device=p2b.device)[None, :]
    pin_edge_mask = pin_rows < n_p2b[:, None]
    pin_id = p2b[..., 0].long().clamp(0, pins.shape[1] - 1)
    block_id = p2b[..., 1].long().clamp(0, boxes.shape[1] - 1)
    pin_gather = pin_id[..., None].expand(-1, -1, 2)
    block_gather = block_id[..., None].expand(-1, -1, 2)
    pin_distance = (
        centers.gather(1, block_gather) - pins.gather(1, pin_gather)
    ).abs().sum(dim=2)
    pin_weights = p2b[..., 2] * pin_edge_mask
    pin_total = (pin_distance * pin_weights).sum(dim=1)
    pin_weight_sum = pin_weights.sum(dim=1)
    wirelength = (block_total + pin_total) / (
        scale * (block_weight_sum + pin_weight_sum).clamp_min(1.0)
    )

    layout_width = x.max(dim=1).values - x.min(dim=1).values
    layout_height = y1.max(dim=1).values - y.min(dim=1).values
    area = layout_width * layout_height / scale.square()
    terms = {
        "overlap": overlap,
        "mib": mib,
        "boundary": boundary,
        "wl": wirelength,
        "area": area,
        "b2b": block_total / (
            scale * block_weight_sum.clamp_min(1.0)
        ),
        "p2b": pin_total / (
            scale * pin_weight_sum.clamp_min(1.0)
        ),
    }
    weights = {
        "overlap": 0.01,
        "mib": 0.01,
        "boundary": 0.01,
        "wl": 0.01,
        "area": 0.01,
    }
    terms["total"] = sum(weight * terms[name] for name, weight in weights.items())
    return terms


def _bilinear_splat(
    points: torch.Tensor,
    values: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """Differentiably splat per-point values into a square grid."""

    scaled = points.clamp(0.0, 1.0) * (resolution - 1)
    x0 = scaled[..., 0].floor().long()
    y0 = scaled[..., 1].floor().long()
    x1 = (x0 + 1).clamp_max(resolution - 1)
    y1 = (y0 + 1).clamp_max(resolution - 1)
    fx = scaled[..., 0] - x0
    fy = scaled[..., 1] - y0
    indices = torch.stack(
        (
            y0 * resolution + x0,
            y0 * resolution + x1,
            y1 * resolution + x0,
            y1 * resolution + x1,
        ),
        dim=-1,
    ).reshape(points.shape[0], -1)
    weights = torch.stack(
        (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ),
        dim=-1,
    ).reshape(points.shape[0], -1)
    flat = values[..., None].expand(-1, -1, 4).reshape(points.shape[0], -1)
    return torch.zeros(
        (points.shape[0], resolution * resolution),
        dtype=points.dtype,
        device=points.device,
    ).scatter_add(1, indices, flat * weights)


def _route_path_obstruction(
    occupancy: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    resolution: int,
    samples: int = 5,
) -> torch.Tensor:
    """Sample both L-shaped routes through an occupancy field."""

    batch, edges, _ = starts.shape
    if edges == 0:
        return starts.new_zeros(batch)
    field = occupancy.reshape(batch, 1, resolution, resolution)
    field = field / field.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    fraction = torch.linspace(
        0.0,
        1.0,
        int(samples),
        device=starts.device,
        dtype=starts.dtype,
    )[None, None, :]
    horizontal = (
        starts[..., :1] * (1.0 - fraction)
        + ends[..., :1] * fraction
    )
    fixed_start_y = starts[..., 1:2].expand_as(horizontal)
    fixed_end_y = ends[..., 1:2].expand_as(horizontal)
    vertical = (
        starts[..., 1:2] * (1.0 - fraction)
        + ends[..., 1:2] * fraction
    )
    fixed_end_x = ends[..., :1].expand_as(vertical)
    fixed_start_x = starts[..., :1].expand_as(vertical)
    first = torch.cat(
        [
            torch.stack([horizontal, fixed_start_y], dim=-1),
            torch.stack([fixed_end_x, vertical], dim=-1),
        ],
        dim=2,
    )
    second = torch.cat(
        [
            torch.stack([fixed_start_x, vertical], dim=-1),
            torch.stack([horizontal, fixed_end_y], dim=-1),
        ],
        dim=2,
    )

    def sampled_mean(path: torch.Tensor) -> torch.Tensor:
        repeated = field[:, None].expand(
            -1,
            edges,
            -1,
            -1,
            -1,
        ).reshape(batch * edges, 1, resolution, resolution)
        grid = path.reshape(batch * edges, path.shape[2], 1, 2) * 2.0 - 1.0
        values = F.grid_sample(
            repeated,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return values.reshape(batch, edges, path.shape[2]).mean(dim=-1)

    obstruction = torch.minimum(sampled_mean(first), sampled_mean(second))
    demand = weights * (ends - starts).abs().sum(dim=-1)
    selected = valid & demand.gt(0.0)
    return (obstruction * demand * selected).sum(dim=1) / demand.masked_fill(
        ~selected,
        0.0,
    ).sum(dim=1).clamp_min(1.0e-6)


def batch_routing_proxy_terms(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_b2b: torch.Tensor,
    n_p2b: torch.Tensor,
    n_pins: torch.Tensor,
    *,
    resolution: int = 8,
) -> dict[str, torch.Tensor]:
    """Cheap differentiable proxies for routed congestion pressure.

    The proxy combines macro-occupancy hotspots, close-macro spacing risk, and
    net-demand hotspots. Every term is dimensionless and invariant to translation
    and uniform coordinate rescaling, so values from different PDKs and die sizes
    can be compared. Route and pin peaks are expressed as a fraction of the valid
    demand mass; occupancy is expressed as a fraction of normalized macro area.
    """

    if resolution < 2:
        raise ValueError("routing proxy resolution must be at least 2")
    centers = boxes[..., :2] + boxes[..., 2:] * 0.5
    corners = torch.cat([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], dim=-1)
    lower = corners[..., :2].amin(dim=1)
    upper = corners[..., 2:].amax(dim=1)
    span = (upper - lower).amax(dim=1).clamp_min(1.0e-6)
    normalized_centers = (centers - lower[:, None]) / span[:, None, None]
    normalized_areas = areas / span.square()[:, None]
    occupancy = _bilinear_splat(
        normalized_centers,
        normalized_areas,
        resolution,
    )
    top_k = min(4, occupancy.shape[1])
    occupancy_peak = occupancy.topk(top_k, dim=1).values.mean(dim=1)
    occupancy_mass = normalized_areas.sum(dim=1).clamp_min(1.0e-6)
    occupancy_hotspot = occupancy_peak / occupancy_mass

    delta = normalized_centers[:, :, None] - normalized_centers[:, None, :]
    distance = torch.sqrt(delta.square().sum(dim=-1) + 1.0e-8)
    pair_mask = ~torch.eye(
        boxes.shape[1],
        dtype=torch.bool,
        device=boxes.device,
    )[None]
    area_pair = (
        normalized_areas[:, :, None] * normalized_areas[:, None, :]
    ).clamp_min(0.0).sqrt()
    spacing_risk = (
        torch.exp(-distance / 0.05) * area_pair * pair_mask
    ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)

    block_rows = torch.arange(b2b.shape[1], device=b2b.device)[None, :]
    block_mask = block_rows < n_b2b[:, None]
    left = b2b[..., 0].long().clamp(0, boxes.shape[1] - 1)
    right = b2b[..., 1].long().clamp(0, boxes.shape[1] - 1)
    left_centers = centers.gather(
        1,
        left[..., None].expand(-1, -1, 2),
    )
    right_centers = centers.gather(
        1,
        right[..., None].expand(-1, -1, 2),
    )
    block_weights = b2b[..., 2] * block_mask
    b2b_demand = (
        (left_centers - right_centers).abs().sum(dim=-1)
        / span[:, None]
        * block_weights
    )
    b2b_midpoints = (
        0.5 * (left_centers + right_centers) - lower[:, None]
    ) / span[:, None, None]
    b2b_grid = _bilinear_splat(b2b_midpoints, b2b_demand, resolution)
    b2b_demand_mass = b2b_demand.sum(dim=1).clamp_min(1.0e-6)
    b2b_route_peak = (
        b2b_grid.topk(top_k, dim=1).values.mean(dim=1)
        / b2b_demand_mass
    )

    pin_rows = torch.arange(p2b.shape[1], device=p2b.device)[None, :]
    pin_mask = pin_rows < n_p2b[:, None]
    pin_id = p2b[..., 0].long().clamp(0, pins.shape[1] - 1)
    block_id = p2b[..., 1].long().clamp(0, boxes.shape[1] - 1)
    pin_xy = pins.gather(1, pin_id[..., None].expand(-1, -1, 2))
    block_xy = centers.gather(
        1,
        block_id[..., None].expand(-1, -1, 2),
    )
    pin_weights = p2b[..., 2] * pin_mask
    p2b_demand = (
        (pin_xy - block_xy).abs().sum(dim=-1)
        / span[:, None]
        * pin_weights
    )
    p2b_midpoints = (
        0.5 * (pin_xy + block_xy) - lower[:, None]
    ) / span[:, None, None]
    p2b_grid = _bilinear_splat(
        p2b_midpoints,
        p2b_demand,
        resolution,
    )
    p2b_demand_mass = p2b_demand.sum(dim=1).clamp_min(1.0e-6)
    p2b_route_peak = (
        p2b_grid.topk(top_k, dim=1).values.mean(dim=1)
        / p2b_demand_mass
    )
    b2b_present = block_mask.to(boxes.dtype).sum(dim=1).gt(0)
    p2b_present = pin_mask.to(boxes.dtype).sum(dim=1).gt(0)
    present_count = (
        b2b_present.to(boxes.dtype) + p2b_present.to(boxes.dtype)
    ).clamp_min(1.0)
    route_peak = (
        b2b_route_peak * b2b_present.to(boxes.dtype)
        + p2b_route_peak * p2b_present.to(boxes.dtype)
    ) / present_count
    pin_valid = torch.arange(pins.shape[1], device=pins.device)[None, :] < n_pins[:, None]
    normalized_pins = (pins - lower[:, None]) / span[:, None, None]
    pin_grid = _bilinear_splat(
        normalized_pins,
        pin_valid.to(dtype=pins.dtype),
        resolution,
    )
    pin_peak = (
        pin_grid.topk(top_k, dim=1).values.mean(dim=1)
        / n_pins.clamp_min(1).to(dtype=pins.dtype)
    )
    block_starts = (left_centers - lower[:, None]) / span[:, None, None]
    block_ends = (right_centers - lower[:, None]) / span[:, None, None]
    pin_starts = (pin_xy - lower[:, None]) / span[:, None, None]
    pin_ends = (block_xy - lower[:, None]) / span[:, None, None]
    b2b_obstruction = _route_path_obstruction(
        occupancy,
        block_starts,
        block_ends,
        block_weights,
        block_mask,
        resolution=resolution,
    )
    p2b_obstruction = _route_path_obstruction(
        occupancy,
        pin_starts,
        pin_ends,
        pin_weights,
        pin_mask,
        resolution=resolution,
    )
    route_obstruction = (
        b2b_obstruction * b2b_present.to(boxes.dtype)
        + p2b_obstruction * p2b_present.to(boxes.dtype)
    ) / present_count
    return {
        "occupancy_peak": occupancy_hotspot,
        "spacing_risk": spacing_risk,
        "route_demand_peak": route_peak,
        "b2b_route_demand_peak": b2b_route_peak,
        "p2b_route_demand_peak": p2b_route_peak,
        "pin_concentration_peak": pin_peak,
        "route_obstruction": route_obstruction.clamp(0.0, 1.0),
        "b2b_route_obstruction": b2b_obstruction.clamp(0.0, 1.0),
        "p2b_route_obstruction": p2b_obstruction.clamp(0.0, 1.0),
    }


ROUTING_PROXY_REFERENCE = {
    "pin_concentration_peak": 0.07,
    "route_demand_peak": 0.02,
    "occupancy_peak": 0.06,
    "spacing_risk": 1.0e-4,
    "route_obstruction": 0.1,
}


def batch_routing_proxy_loss(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_b2b: torch.Tensor,
    n_p2b: torch.Tensor,
    n_pins: torch.Tensor,
    *,
    resolution: int = 8,
) -> torch.Tensor:
    """Return a scale-normalized routing proxy suitable for an objective.

    The raw terms have different physical units and magnitudes. Summing them
    directly lets pin concentration dominate the optimizer, so normalize each
    term against a fixed reference before combining them.
    """

    terms = batch_routing_proxy_terms(
        boxes,
        areas,
        b2b,
        p2b,
        pins,
        n_b2b,
        n_p2b,
        n_pins,
        resolution=resolution,
    )
    normalized = {
        name: terms[name] / ROUTING_PROXY_REFERENCE[name]
        for name in ROUTING_PROXY_REFERENCE
    }
    return (
        0.5 * normalized["pin_concentration_peak"]
        + 0.25 * normalized["route_demand_peak"]
        + 0.25 * normalized["occupancy_peak"]
        + 0.05 * normalized["spacing_risk"]
        + 0.1 * normalized["route_obstruction"]
    )


def batch_grouping_proxy(
    boxes: torch.Tensor,
    constraints: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable soft-MST gap proxy for positive cluster ids."""

    gap = differentiable_pair_gap(boxes)
    cluster = constraints[..., 3]
    same = (cluster[:, :, None] == cluster[:, None, :]) & (
        cluster[:, :, None] > 0
    )
    same = same & ~torch.eye(
        boxes.shape[1],
        dtype=torch.bool,
        device=boxes.device,
    )[None]
    masked = torch.where(
        same,
        gap,
        torch.full_like(gap, 1.0e6),
    )
    valid = same.any(dim=2)
    soft_min = (
        -float(temperature)
        * torch.logsumexp(-masked / float(temperature), dim=2)
        + float(temperature)
        * same.sum(dim=2).clamp_min(1).to(gap.dtype).log()
    )
    values = torch.where(valid, soft_min, torch.zeros_like(soft_min))
    return values.sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def differentiable_pair_gap(boxes: torch.Tensor) -> torch.Tensor:
    """Return Trinity's smallest-axis rectangle gap for every block pair."""

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B,N,4]")
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = x0 + boxes[..., 2]
    y1 = y0 + boxes[..., 3]
    gap_x = (
        x0[:, None, :] - x1[:, :, None]
    ).clamp_min(0.0).maximum(
        (x0[:, :, None] - x1[:, None, :]).clamp_min(0.0)
    )
    gap_y = (
        y0[:, None, :] - y1[:, :, None]
    ).clamp_min(0.0).maximum(
        (y0[:, :, None] - y1[:, None, :]).clamp_min(0.0)
    )
    x_overlap = (
        x1[:, :, None].minimum(x1[:, None, :])
        - x0[:, :, None].maximum(x0[:, None, :])
    ).gt(0.0)
    y_overlap = (
        y1[:, :, None].minimum(y1[:, None, :])
        - y0[:, :, None].maximum(y0[:, None, :])
    ).gt(0.0)
    return torch.where(
        y_overlap,
        gap_x,
        torch.where(x_overlap, gap_y, gap_x.minimum(gap_y)),
    )


def flow_loss(
    velocity: torch.Tensor,
    interpolant: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    huber_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow MSE and the one-step clean-layout estimate."""

    clean_velocity = target - source
    error = velocity - clean_velocity
    loss = (
        F.smooth_l1_loss(error, torch.zeros_like(error), beta=huber_beta)
        if huber_beta > 0.0
        else error.pow(2).mean()
    )
    estimate = interpolant + (1.0 - time[:, None, None]) * velocity
    return loss, estimate


def matched_endpoint_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Optimal set matching on the predicted clean endpoint."""

    return paired_token_mse(estimate, target)


__all__ = [
    "ROUTING_PROXY_REFERENCE",
    "batch_grouping_proxy",
    "batch_physics_terms",
    "batch_routing_proxy_loss",
    "batch_routing_proxy_terms",
    "differentiable_pair_gap",
    "flow_loss",
    "matched_endpoint_loss",
]
