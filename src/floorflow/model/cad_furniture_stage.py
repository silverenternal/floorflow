"""Program-conditioned vector furniture stage for the CAD action graph."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.dynamic_object_instance import (
    DynamicFurnitureInstanceHead,
)
from floorflow.heads.room_conditioned_furniture_semantics import (
    RoomConditionedFurnitureSemantics,
)


class CadFurnitureStage(nn.Module):
    """Place furniture queries after room and program evidence are available."""

    def __init__(
        self,
        in_channels: int,
        num_furniture_classes: int,
        *,
        program_channels: int,
        stage_token_dim: int,
        hidden_dim: int = 96,
        max_queries: int = 24,
        min_queries: int = 4,
        num_rooms: int = 1,
        enable_room_assignment: bool = False,
        enable_semantic_router: bool = False,
        semantic_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_furniture_classes = int(num_furniture_classes)
        self.head = DynamicFurnitureInstanceHead(
            in_channels,
            num_furniture_classes,
            hidden_dim=hidden_dim,
            max_queries=max_queries,
            min_queries=min_queries,
        )
        self.program = nn.Sequential(
            nn.Linear(program_channels, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels),
        )
        self.stage_film = nn.Linear(stage_token_dim, in_channels * 2)
        self.room_projection = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.enable_room_assignment = bool(enable_room_assignment)
        self.room_assignment = (
            nn.Linear(hidden_dim, int(num_rooms) + 1)
            if self.enable_room_assignment
            else None
        )
        self.semantic_router = (
            RoomConditionedFurnitureSemantics(
                hidden_dim,
                int(num_rooms),
                int(num_furniture_classes),
                hidden_dim=semantic_hidden,
            )
            if enable_semantic_router
            else None
        )
        nn.init.normal_(
            self.stage_film.weight,
            mean=0.0,
            std=1.0e-3,
        )
        nn.init.zeros_(self.stage_film.bias)

    def forward(
        self,
        features: torch.Tensor,
        controls: dict[str, torch.Tensor],
        footprint: torch.Tensor,
        *,
        stage_token: torch.Tensor,
        room_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        program = self.program(controls["program"].float())
        scale, bias = self.stage_film(stage_token.float()).chunk(2, dim=-1)
        conditioned = (
            features.float()
            + program[..., None, None]
            + features
            * scale[..., None, None].tanh()
            + bias[..., None, None].tanh()
        )
        room_probabilities = F.interpolate(
            (
                controls["room_logits"]
                if room_logits is None
                else room_logits
            ).float().softmax(dim=1),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mass = room_probabilities.flatten(2).sum(dim=-1).clamp_min(1.0e-4)
        room_descriptors = torch.einsum(
            "brhw,bchw->brc",
            room_probabilities,
            conditioned,
        ) / mass[..., None]
        room_tokens = self.room_projection(room_descriptors)
        room_probability = room_probabilities.amax(dim=1, keepdim=True)
        class_density = controls.get("furniture")
        if class_density is not None:
            class_density = class_density[:, : self.num_furniture_classes]
        output = self.head(
            conditioned,
            room_probability,
            footprint,
            class_density=class_density,
            room_tokens=room_tokens,
        )
        if self.semantic_router is not None:
            room_presence = room_probabilities.amax(dim=(2, 3))
            semantic = self.semantic_router(
                output["query_features"],
                room_presence,
            )
            foreground = (
                output["class_logits"][..., :-1]
                + semantic["class_residual"]
            )
            output["class_logits"] = torch.cat(
                [foreground, output["class_logits"][..., -1:]],
                dim=-1,
            )
            output["furniture_semantic_gate"] = semantic[
                "semantic_gate"
            ]
            output["furniture_room_semantic_context"] = semantic[
                "room_context"
            ]
        output["raw_mask_logits"] = output["mask_logits"]
        if self.room_assignment is not None:
            assignment_logits = self.room_assignment(output["query_features"])
            assignment_probability = assignment_logits.softmax(dim=-1)
            foreground_probability = assignment_probability[..., :-1]
            background_probability = assignment_probability[..., -1:]
            assigned_support = torch.einsum(
                "bqr,brhw->bqhw",
                foreground_probability,
                room_probabilities,
            )
            support = assigned_support * (
                1.0 - background_probability[..., None]
            )
            output["mask_logits"] = (
                output["mask_logits"]
                + support.clamp_min(1.0e-4).log()
            )
            output["room_assignment_logits"] = assignment_logits
        output["furniture_query_room_probability"] = room_probability
        return output


__all__ = ["CadFurnitureStage"]
