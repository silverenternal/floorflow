"""Factory for legacy and dynamic CAD semantic stage heads."""

from __future__ import annotations

from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_stage_refiner import DynamicStageRefiner
from floorflow.heads.room_segmenter import ConvNormAct

STAGE_NAMES = ("rooms", "walls", "openings", "furniture")


def build_cad_stage_heads(
    taxonomy: FloorplanTaxonomy,
    field_channels: int,
    stage_hidden: int,
    *,
    dynamic: bool,
    route_attention: bool = False,
    route_grid: int = 8,
    route_router_count: int = 12,
    route_top_k: int = 6,
    route_heads: int = 4,
) -> tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]:
    """Build stage heads without leaking construction details into the model."""

    if dynamic:
        specifications = {
            "rooms": (len(taxonomy.room_ids), 0),
            "walls": (1, len(taxonomy.room_ids)),
            "openings": (len(taxonomy.opening_ids), 4),
            "furniture": (len(taxonomy.furniture_ids), len(taxonomy.room_ids)),
        }
        refiners = nn.ModuleDict(
            {
                name: DynamicStageRefiner(
                    field_channels,
                    prior_channels,
                    field_channels,
                    hidden_channels=stage_hidden,
                    conditioning_channels=conditioning_channels,
                    route_attention=(
                        route_attention
                        and name in {"walls", "openings", "furniture"}
                    ),
                    route_grid=route_grid,
                    route_router_count=route_router_count,
                    route_top_k=route_top_k,
                    route_heads=route_heads,
                )
                for name, (prior_channels, conditioning_channels) in (
                    specifications.items()
                )
            }
        )
        return refiners, nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

    trunks = nn.ModuleDict(
        {
            name: ConvNormAct(field_channels, stage_hidden)
            for name in STAGE_NAMES
        }
    )
    films = nn.ModuleDict(
        {
            name: nn.Linear(field_channels, stage_hidden * 2)
            for name in STAGE_NAMES
        }
    )
    output_channels = {
        "rooms": len(taxonomy.room_ids),
        "walls": 1,
        "openings": len(taxonomy.opening_ids),
        "furniture": len(taxonomy.furniture_ids),
    }
    heads = nn.ModuleDict(
        {
            name: nn.Conv2d(stage_hidden, channels, 1)
            for name, channels in output_channels.items()
        }
    )
    for module in films.values():
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
    for module in heads.values():
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
    return nn.ModuleDict(), trunks, films, heads


__all__ = ["STAGE_NAMES", "build_cad_stage_heads"]
