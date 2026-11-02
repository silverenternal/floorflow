"""Differentiable spatial occupancy feedback for chip block trajectories."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _pin_chart(
    pins: torch.Tensor | None,
    pin_lengths: torch.Tensor | None,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a shared target-free lower/span chart from valid input pins."""

    fallback_lower = torch.zeros((batch, 2), dtype=dtype, device=device)
    fallback_span = torch.ones((batch, 1), dtype=dtype, device=device)
    valid_batch = torch.zeros(batch, dtype=torch.bool, device=fallback_lower.device)
    if pins is None or pins.numel() == 0:
        return fallback_lower, fallback_span, valid_batch
    pin_count = pins.shape[0] // batch
    if pin_count * batch != pins.shape[0]:
        raise ValueError("packed pins must be divisible by the batch size")
    lengths = (
        torch.full(
            (batch,),
            pin_count,
            dtype=torch.long,
            device=pins.device,
        )
        if pin_lengths is None
        else pin_lengths.to(pins.device).long()
    )
    rows = torch.arange(pin_count, device=pins.device)[None]
    valid_pin = rows < lengths[:, None]
    grid = pins.reshape(batch, pin_count, 2)
    lower = grid.masked_fill(~valid_pin[..., None], float("inf")).amin(dim=1)
    upper = grid.masked_fill(~valid_pin[..., None], float("-inf")).amax(dim=1)
    valid_batch = valid_pin.any(dim=1) & (
        upper - lower
    ).amax(dim=1).gt(1.0e-6)
    span = (upper - lower).amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    lower = torch.where(
        valid_batch[:, None],
        lower,
        fallback_lower,
    )
    span = torch.where(
        valid_batch[:, None],
        span,
        fallback_span,
    )
    return lower.to(dtype), span.to(dtype), valid_batch


def block_geometry_in_chart(
    block_state: torch.Tensor,
    areas: torch.Tensor,
    *,
    lower: torch.Tensor | None = None,
    span: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map block centers and extents into the shared inference chart."""

    if block_state.ndim != 3 or block_state.shape[-1] != 3:
        raise ValueError("block_state must be [B,N,3]")
    if areas.shape != block_state.shape[:2]:
        raise ValueError("areas must match block_state")
    batch = block_state.shape[0]
    flow_scale = areas.detach().float().sum(dim=1).sqrt().clamp_min(
        1.0e-6
    )
    physical_center = block_state[..., :2].float() * flow_scale[:, None, None]
    physical_width = (
        areas.detach().float().clamp_min(1.0e-6)
        * block_state[..., 2].float().exp()
    ).sqrt()
    physical_height = (
        areas.detach().float().clamp_min(1.0e-6)
        / physical_width.clamp_min(1.0e-6)
    )
    physical_extent = torch.stack(
        [physical_width, physical_height],
        dim=-1,
    )
    physical_lower = (
        physical_center - 0.5 * physical_extent
    ).amin(dim=1)
    physical_upper = (
        physical_center + 0.5 * physical_extent
    ).amax(dim=1)
    if lower is None:
        lower = physical_lower
    if span is None:
        span = (physical_upper - physical_lower).amax(
            dim=1,
            keepdim=True,
        ).clamp_min(flow_scale[:, None] * 1.0e-3)
    if lower.shape != (batch, 2) or span.shape != (batch, 1):
        raise ValueError("chart lower/span must be [B,2]/[B,1]")
    chart_lower = torch.minimum(lower, physical_lower)
    chart_upper = torch.maximum(
        lower + span,
        physical_upper,
    )
    chart_span = (chart_upper - chart_lower).amax(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    center = (physical_center - chart_lower[:, None]) / chart_span[:, None]
    extent = physical_extent / chart_span[..., None]
    return (
        center.clamp(0.0, 1.0),
        extent.clamp_min(1.0e-6),
        chart_lower,
        chart_span,
    )


def _path_obstruction(
    occupancy: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    grid: int,
    samples: int = 5,
) -> torch.Tensor:
    """Sample both L-shaped routes through a normalized occupancy field."""

    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[-1] != 2:
        raise ValueError("route endpoints must be [edges,2]")
    if batch_index.shape != (starts.shape[0],):
        raise ValueError("route batch indices must match edges")
    if starts.shape[0] == 0:
        return starts.new_zeros(0)
    fraction = torch.linspace(
        0.0,
        1.0,
        int(samples),
        device=starts.device,
        dtype=starts.dtype,
    )[None]
    horizontal = starts[:, :1] * (1.0 - fraction) + ends[:, :1] * fraction
    vertical = starts[:, 1:2] * (1.0 - fraction) + ends[:, 1:2] * fraction
    first = torch.cat(
        [
            torch.stack(
                [horizontal, starts[:, 1:2].expand_as(horizontal)],
                dim=-1,
            ),
            torch.stack(
                [ends[:, :1].expand_as(vertical), vertical],
                dim=-1,
            ),
        ],
        dim=1,
    )
    second = torch.cat(
        [
            torch.stack(
                [starts[:, :1].expand_as(vertical), vertical],
                dim=-1,
            ),
            torch.stack(
                [horizontal, ends[:, 1:2].expand_as(horizontal)],
                dim=-1,
            ),
        ],
        dim=1,
    )
    field = occupancy[:, 1].reshape(-1, 1, grid, grid)
    field = field / field.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
    selected = field[batch_index]

    def path_mean(path: torch.Tensor) -> torch.Tensor:
        grid_points = path.reshape(-1, path.shape[1], 1, 2) * 2.0 - 1.0
        values = F.grid_sample(
            selected,
            grid_points,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return values.reshape(starts.shape[0], path.shape[1]).mean(dim=-1)

    return torch.minimum(path_mean(first), path_mean(second)).clamp(0.0, 1.0)


def route_obstruction_by_block(
    occupancy: torch.Tensor,
    unit_centers: torch.Tensor,
    cache: dict[str, torch.Tensor],
    *,
    lower: torch.Tensor,
    span: torch.Tensor,
    grid: int,
) -> torch.Tensor:
    """Aggregate routed-edge obstruction and demand onto sparse block endpoints."""

    batch, count, _ = unit_centers.shape
    flat_centers = unit_centers.reshape(batch * count, 2)
    obstruction_sum = flat_centers.new_zeros(batch * count)
    demand_sum = flat_centers.new_zeros(batch * count)

    def accumulate(
        block_rows: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        weights: torch.Tensor,
        edge_batches: torch.Tensor,
    ) -> None:
        nonlocal obstruction_sum, demand_sum
        demand = weights.to(starts.dtype) * (ends - starts).abs().sum(dim=-1)
        valid = demand.gt(0.0)
        if not bool(valid.any()):
            return
        obstruction = _path_obstruction(
            occupancy,
            starts[valid].clamp(0.0, 1.0),
            ends[valid].clamp(0.0, 1.0),
            edge_batches[valid],
            grid=grid,
        )
        selected_demand = demand[valid]
        weighted = obstruction * selected_demand
        rows = block_rows[valid]
        obstruction_sum.index_add_(0, rows, weighted)
        demand_sum.index_add_(0, rows, selected_demand)

    source_rows = cache["b2b_source"]
    target_rows = cache["b2b_target"]
    if source_rows.numel():
        source_batch = source_rows // count
        source_local = source_rows % count
        target_local = target_rows % count
        starts = unit_centers[source_batch, source_local]
        ends = unit_centers[source_batch, target_local]
        weights = cache["b2b_weight"]
        accumulate(source_rows, starts, ends, weights, source_batch)
        accumulate(target_rows, ends, starts, weights, source_batch)

    pin_rows = cache["p2b_pin"]
    if pin_rows.numel() and cache["pins"].numel():
        pin_per_batch = cache["pins"].shape[0] // batch
        pin_batch = (pin_rows // pin_per_batch).clamp_max(batch - 1)
        pin_xy = (
            cache["pins"][pin_rows].to(unit_centers.dtype)
            - lower[pin_batch]
        ) / span[pin_batch]
        block_rows = cache["p2b_block"]
        block_xy = flat_centers[block_rows]
        accumulate(
            block_rows,
            pin_xy,
            block_xy,
            cache["p2b_weight"],
            pin_batch,
        )

    obstruction = obstruction_sum / demand_sum.clamp_min(1.0e-6)
    obstruction = obstruction.reshape(batch, count)
    edge_degree = demand_sum.reshape(batch, count)
    edge_degree = edge_degree / edge_degree.amax(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return torch.stack([obstruction, edge_degree.log1p()], dim=-1)


class SpatialOccupancyCoupler(nn.Module):
    """Rasterize current block occupancy and return per-block state residuals."""

    def __init__(
        self,
        hidden: int,
        *,
        grid_size: int = 16,
        channels: int = 8,
        bottleneck: int | None = None,
        route_aware: bool = False,
    ) -> None:
        super().__init__()
        if hidden <= 0 or grid_size < 2 or channels <= 0:
            raise ValueError("invalid spatial-occupancy dimensions")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.grid_size = int(grid_size)
        self.channels = int(channels)
        self.route_aware = bool(route_aware)
        self.input_channels = 5 if self.route_aware else 3
        self.bottleneck = width
        self.field = nn.Sequential(
            nn.Conv2d(self.input_channels, channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.SiLU(),
        )
        self.field_encoder = nn.Linear(channels + 3, width)
        self.reduce = nn.Sequential(
            nn.Linear(hidden + width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.reduce[-1].weight, std=0.02)
        nn.init.zeros_(self.reduce[-1].bias)
        self.obstruction_encoder = (
            nn.Linear(2, width) if self.route_aware else None
        )
        if self.obstruction_encoder is not None:
            nn.init.zeros_(self.obstruction_encoder.weight)
            nn.init.zeros_(self.obstruction_encoder.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        areas: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if areas.shape != (batch, count):
            raise ValueError("areas must have shape (batch, n)")

        grid = self.grid_size
        lower = None
        span = None
        if self.route_aware and cache is not None:
            lower, span, _valid_pin_chart = _pin_chart(
                cache.get("pins"),
                cache.get("pin_lengths"),
                batch=batch,
                dtype=block_state.dtype,
                device=block_state.device,
            )
        unit_centers, extent, chart_lower, chart_span = (
            block_geometry_in_chart(
            block_state,
            areas,
            lower=lower,
            span=span,
            )
        )
        grid_centers = 2.0 * unit_centers - 1.0
        area_weight = areas.float() / areas.float().sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        offsets = unit_centers.new_tensor(
            [[0.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]]
        )
        extent = [
            unit_centers,
            unit_centers + offsets[1] * extent,
            unit_centers + offsets[2] * extent,
            unit_centers + offsets[3] * extent,
            unit_centers + offsets[4] * extent,
        ]
        points = torch.stack(extent, dim=2).reshape(batch, count * 5, 2)
        values = torch.stack(
            [
                areas.new_full(areas.shape, 0.2),
                area_weight * 0.2,
                block_state[..., 2].float().abs() * 0.2,
            ],
            dim=-1,
        ).repeat_interleave(5, dim=1)
        occupancy = block_tokens.new_zeros(
            batch,
            self.input_channels,
            grid * grid,
        )
        occupancy[:, :3] = self._splat(
            points,
            values.to(block_tokens.dtype),
            grid=grid,
            channels=3,
        )
        if self.route_aware:
            if cache is None:
                raise ValueError("route-aware occupancy requires edge cache")
            route = self._route_fields(
                unit_centers,
                areas,
                cache,
                lower=chart_lower,
                span=chart_span,
                grid=grid,
                dtype=occupancy.dtype,
            )
            occupancy[:, 3:5] = route
        field = self.field(
            occupancy.reshape(batch, self.input_channels, grid, grid)
        )
        sampled = F.grid_sample(
            field,
            grid_centers[:, :, None],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        context = self.field_encoder(
            torch.cat([sampled, block_state.float()], dim=-1)
        )
        obstruction_features = None
        if self.route_aware:
            obstruction_features = route_obstruction_by_block(
                occupancy,
                unit_centers,
                cache,
                lower=chart_lower,
                span=chart_span,
                grid=grid,
            )
            context = context + self.obstruction_encoder(
                obstruction_features.to(context.dtype)
            )
        residual = self.reduce(
            torch.cat(
                [block_tokens, context.to(block_tokens.dtype)],
                dim=-1,
            )
        )
        residual = self.gate * residual
        diagnostics = {
            "spatial_occupancy_residual_norm": residual.detach().norm(dim=-1),
            "spatial_occupancy_grid_fill": occupancy[:, :2].sum(dim=-1).gt(0).float().mean(),
            "spatial_extent_mass": occupancy[:, 1].sum(dim=-1).detach().mean(),
        }
        if self.route_aware:
            diagnostics["spatial_net_fill"] = occupancy[:, 3].gt(0).float().mean()
            diagnostics["spatial_pin_fill"] = occupancy[:, 4].gt(0).float().mean()
            diagnostics["spatial_route_obstruction"] = (
                obstruction_features[..., 0].detach().mean()
            )
            diagnostics["spatial_route_degree"] = (
                obstruction_features[..., 1].detach().mean()
            )
        return residual, diagnostics

    @staticmethod
    def _splat(
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        grid: int,
        channels: int,
    ) -> torch.Tensor:
        output = values.new_zeros(values.shape[0], channels, grid * grid)
        if points.numel() == 0:
            return output
        scaled = points.clamp(0.0, 1.0) * (grid - 1)
        lower = scaled.floor().long().clamp(0, grid - 1)
        fraction = scaled - lower.float()
        batch_index = torch.arange(
            points.shape[0],
            device=points.device,
        )[:, None].expand(-1, points.shape[1])
        for dx, dy, weight in (
            (0, 0, (1 - fraction[..., 0]) * (1 - fraction[..., 1])),
            (1, 0, fraction[..., 0] * (1 - fraction[..., 1])),
            (0, 1, (1 - fraction[..., 0]) * fraction[..., 1]),
            (1, 1, fraction[..., 0] * fraction[..., 1]),
        ):
            x = (lower[..., 0] + dx).clamp(0, grid - 1)
            y = (lower[..., 1] + dy).clamp(0, grid - 1)
            flat_index = (
                batch_index * grid * grid + y * grid + x
            ).reshape(-1)
            flat_output = output.reshape(-1)
            for channel in range(channels):
                flat_output.scatter_add_(
                    0,
                    flat_index,
                    values[..., channel].reshape(-1) * weight.reshape(-1),
                )
        return output

    def _route_fields(
        self,
        unit_centers: torch.Tensor,
        areas: torch.Tensor,
        cache: dict[str, torch.Tensor],
        *,
        lower: torch.Tensor,
        span: torch.Tensor,
        grid: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, count = areas.shape
        net = areas.new_zeros(batch, grid * grid)
        source = cache["b2b_source"]
        target = cache["b2b_target"]
        if source.numel():
            source_batch = source // count
            source_local = source % count
            target_local = target % count
            safe_batch = source_batch.clamp_max(batch - 1)
            source_xy = unit_centers[safe_batch, source_local]
            target_xy = unit_centers[safe_batch, target_local]
            midpoint = 0.5 * (source_xy + target_xy)
            demand = (
                (target_xy - source_xy).abs().sum(dim=-1)
                * cache["b2b_weight"].to(dtype)
            )
            net = self._splat_flat(
                source_batch,
                midpoint,
                demand,
                batch=batch,
                grid=grid,
                dtype=dtype,
            )
        pin = areas.new_zeros(batch, grid * grid)
        pin_index = cache["p2b_pin"]
        block_index = cache["p2b_block"]
        pins = cache["pins"]
        if pin_index.numel() and pins.numel():
            pins_per_batch = pins.shape[0] // batch
            pin_batch = pin_index // pins_per_batch
            pin_batch = pin_batch.clamp_max(batch - 1)
            pin_xy = (
                (pins[pin_index].to(dtype) - lower[pin_batch])
                / span[pin_batch]
            ).clamp(0.0, 1.0)
            block_xy = unit_centers.view(-1, 2)[block_index]
            midpoint = 0.5 * (pin_xy + block_xy)
            demand = (
                (pin_xy - block_xy).abs().sum(dim=-1)
                * cache["p2b_weight"].to(dtype)
            )
            pin = self._splat_flat(
                pin_batch,
                midpoint,
                demand,
                batch=batch,
                grid=grid,
                dtype=dtype,
            )
        route = torch.stack([net, pin], dim=1)
        mass = route.amax(dim=-1, keepdim=True)
        return route / mass.clamp_min(1.0e-6)

    @staticmethod
    def _splat_flat(
        batch_index: torch.Tensor,
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        batch: int,
        grid: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        output = values.new_zeros(batch, grid * grid, dtype=dtype)
        if points.numel() == 0:
            return output
        scaled = points.clamp(0.0, 1.0) * (grid - 1)
        lower = scaled.floor().long().clamp(0, grid - 1)
        fraction = scaled - lower.float()
        for dx, dy, weight in (
            (0, 0, (1 - fraction[:, 0]) * (1 - fraction[:, 1])),
            (1, 0, fraction[:, 0] * (1 - fraction[:, 1])),
            (0, 1, (1 - fraction[:, 0]) * fraction[:, 1]),
            (1, 1, fraction[:, 0] * fraction[:, 1]),
        ):
            x = (lower[:, 0] + dx).clamp(0, grid - 1)
            y = (lower[:, 1] + dy).clamp(0, grid - 1)
            flat_index = batch_index * grid * grid + y * grid + x
            output.reshape(-1).scatter_add_(
                0,
                flat_index,
                values * weight,
            )
        return output


__all__ = [
    "SpatialOccupancyCoupler",
    "block_geometry_in_chart",
    "route_obstruction_by_block",
]
