"""Second-stage structural refinement conditioned on a coarse CAD layout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.structure_mita_router import MultiScaleStructureRoute


def coarse_structure_control(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    size: int = 128,
) -> torch.Tensor:
    """Build room, wall, and opening channels at low control resolution."""

    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    rooms = torch.stack(
        [(labels == class_id).float() for class_id in room_ids],
        dim=1,
    )
    wall = (labels == taxonomy.class_id("wall")).float()[:, None]
    opening_ids = torch.tensor(taxonomy.opening_ids, device=labels.device)
    opening = torch.isin(labels, opening_ids).float()[:, None]
    opening_types = torch.stack(
        [(labels == class_id).float() for class_id in opening_ids],
        dim=1,
    )
    footprint_channel = footprint.float()
    if footprint_channel.ndim == 3:
        footprint_channel = footprint_channel[:, None]
    control = torch.cat(
        [rooms, wall, opening, opening_types, footprint_channel],
        dim=1,
    )
    return F.adaptive_max_pool2d(control, (int(size), int(size)))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(1, out_channels // 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CoarseStructureRefiner(nn.Module):
    """Upsample coarse CAD structure while predicting independent channels."""

    def __init__(
        self,
        input_channels: int,
        num_opening_classes: int,
        base_channels: int = 32,
        *,
        masked_input: bool = False,
        zero_masked_input: bool = True,
        extra_input_channels: int = 0,
        direct_control_output: bool = False,
        host_conditioned_openings: bool = False,
        host_conditioning_v2: bool = False,
        predict_junctions: bool = False,
        predict_room_boundary: bool = False,
        predict_centerline: bool = False,
        global_route_attention: bool = False,
        route_grid: int = 16,
        route_router_count: int = 16,
        route_top_k: int = 8,
        route_heads: int = 4,
    ) -> None:
        super().__init__()
        if host_conditioning_v2 and not (
            host_conditioned_openings
            and predict_junctions
            and predict_room_boundary
        ):
            raise ValueError(
                "host_conditioning_v2 requires openings, junctions, "
                "and room boundary heads"
            )
        base = int(base_channels)
        self.masked_input = bool(masked_input)
        self.zero_masked_input = bool(zero_masked_input)
        self.direct_control_output = bool(direct_control_output)
        self.host_conditioned_openings = bool(host_conditioned_openings)
        self.host_conditioning_v2 = bool(host_conditioning_v2)
        self.predict_junctions = bool(predict_junctions)
        self.predict_room_boundary = bool(predict_room_boundary)
        self.predict_centerline = bool(predict_centerline)
        mask_channels = 1 if self.masked_input else 0
        self.enc128 = _ResidualBlock(
            input_channels + int(extra_input_channels) + mask_channels,
            base,
        )
        self.enc64 = _ResidualBlock(base, base * 2)
        self.enc32 = _ResidualBlock(base * 2, base * 4)
        self.dec64 = _ResidualBlock(base * 6, base * 2)
        self.dec128 = _ResidualBlock(base * 3, base)
        self.dec256 = _ResidualBlock(base, base)
        self.global_route = (
            MultiScaleStructureRoute(
                base * 4,
                base * 2,
                base,
                route_grid=route_grid,
                router_count=route_router_count,
                top_k=route_top_k,
                heads=route_heads,
            )
            if global_route_attention
            else None
        )
        self.wall = nn.Conv2d(base, 1, 1)
        self.opening = nn.Conv2d(base, 1, 1)
        self.opening_type = nn.Conv2d(base, int(num_opening_classes), 1)
        self.opening_host_scale = (
            nn.Parameter(
                torch.tensor(0.01 if self.host_conditioning_v2 else 0.0)
            )
            if self.host_conditioned_openings
            else None
        )
        self.opening_type_host_scale = (
            nn.Parameter(
                torch.tensor(0.01 if self.host_conditioning_v2 else 0.0)
            )
            if self.host_conditioned_openings
            else None
        )
        self.opening_host_fusion = (
            nn.Conv2d(3, 1, 1) if self.host_conditioning_v2 else None
        )
        self.opening_type_host_fusion = (
            nn.Conv2d(3, int(num_opening_classes), 1)
            if self.host_conditioning_v2
            else None
        )
        self.junction = nn.Conv2d(base, 1, 1) if predict_junctions else None
        self.junction_degree = (
            nn.Conv2d(base, 5, 1) if predict_junctions else None
        )
        self.room_boundary = (
            nn.Conv2d(base, 1, 1) if predict_room_boundary else None
        )
        self.centerline = (
            nn.Conv2d(base, 1, 1) if predict_centerline else None
        )
        self.centerline_wall_scale = (
            nn.Parameter(torch.full((1,), 0.1))
            if predict_centerline
            else None
        )
        for head in (
            self.junction,
            self.junction_degree,
            self.room_boundary,
            self.centerline,
        ):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        for head in (
            self.opening_host_fusion,
            self.opening_type_host_fusion,
        ):
            if head is not None:
                nn.init.normal_(head.weight, std=0.01)
                nn.init.zeros_(head.bias)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        control: torch.Tensor,
        output_size: tuple[int, int] | None = None,
        validity_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.masked_input:
            if validity_mask is None:
                raise ValueError("validity_mask is required for masked_input")
            if validity_mask.ndim == 3:
                validity_mask = validity_mask[:, None]
            validity_mask = F.interpolate(
                validity_mask.float(),
                size=control.shape[-2:],
                mode="nearest",
            )
            if self.zero_masked_input:
                control = control * validity_mask
            control = torch.cat([control, validity_mask], dim=1)
        skip128 = self.enc128(control)
        skip64 = self.enc64(F.avg_pool2d(skip128, 2))
        low = self.enc32(F.avg_pool2d(skip64, 2))
        route_diagnostics = None
        if self.global_route is not None:
            low, route_diagnostics = self.global_route(
                low,
                skip64,
                skip128,
            )
        hidden64 = self.dec64(
            torch.cat([self._up(low, skip64), skip64], dim=1)
        )
        hidden128 = self.dec128(
            torch.cat([self._up(hidden64, skip128), skip128], dim=1)
        )
        if self.direct_control_output:
            hidden_out = hidden128
        else:
            hidden_out = self.dec256(
                F.interpolate(
                    hidden128,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if output_size is None:
            output_size = (
                hidden128.shape[-2:]
                if self.direct_control_output
                else (control.shape[-2] * 2, control.shape[-1] * 2)
            )
        if hidden_out.shape[-2:] != tuple(output_size):
            hidden_out = F.interpolate(
                hidden_out,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        wall_logits = self.wall(hidden_out)
        centerline_logits = (
            self.centerline(hidden_out) if self.centerline is not None else None
        )
        if centerline_logits is not None:
            wall_logits = (
                wall_logits
                + torch.tanh(self.centerline_wall_scale).to(wall_logits.dtype)
                * centerline_logits
            )
        junction_logits = (
            self.junction(hidden_out) + wall_logits
            if self.junction is not None
            else None
        )
        junction_degree_logits = (
            self.junction_degree(hidden_out)
            if self.junction_degree is not None
            else None
        )
        room_boundary_logits = (
            self.room_boundary(hidden_out)
            if self.room_boundary is not None
            else None
        )
        opening_logits = self.opening(hidden_out)
        opening_type_logits = self.opening_type(hidden_out)
        if self.host_conditioned_openings:
            wall_probability = wall_logits.sigmoid()
            host_scale = torch.exp(
                self.opening_host_scale.clamp(max=2.0)
            ) - 1.0
            type_scale = torch.exp(
                self.opening_type_host_scale.clamp(max=2.0)
            ) - 1.0
            if (
                self.host_conditioning_v2
                and self.opening_host_fusion is not None
                and self.opening_type_host_fusion is not None
                and junction_logits is not None
                and room_boundary_logits is not None
            ):
                host_context = torch.cat(
                    [
                        wall_probability,
                        junction_logits.sigmoid(),
                        room_boundary_logits.sigmoid(),
                    ],
                    dim=1,
                )
                opening_logits = (
                    opening_logits
                    + host_scale * self.opening_host_fusion(host_context)
                )
                opening_type_logits = (
                    opening_type_logits
                    + type_scale
                    * self.opening_type_host_fusion(host_context)
                )
            else:
                opening_logits = opening_logits + host_scale * wall_logits
                opening_type_logits = (
                    opening_type_logits
                    + type_scale * wall_probability
                )
        output = {
            "wall_logits": wall_logits,
            "opening_logits": opening_logits,
            "opening_type_logits": opening_type_logits,
        }
        if centerline_logits is not None:
            output["centerline_logits"] = centerline_logits
        if junction_logits is not None:
            output["junction_logits"] = junction_logits
            output["junction_degree_logits"] = junction_degree_logits
        if room_boundary_logits is not None:
            output["room_boundary_logits"] = room_boundary_logits
        if route_diagnostics is not None:
            output["route_diagnostics"] = route_diagnostics
        return output


__all__ = ["CoarseStructureRefiner", "coarse_structure_control"]
