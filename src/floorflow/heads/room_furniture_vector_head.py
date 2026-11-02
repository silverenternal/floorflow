"""Room-slot furniture primitives with direct vector outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.geometry.diff_ras import diff_soft_polygon_masks


class RoomFurnitureVectorHead(nn.Module):
    """Predict furniture rectangles from room-conditioned query slots."""

    def __init__(
        self,
        feature_channels: int,
        num_rooms: int,
        num_furniture_classes: int,
        *,
        hidden_dim: int = 96,
        max_per_room: int = 6,
        heads: int = 4,
        layers: int = 1,
        max_size_fraction: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_rooms = int(num_rooms)
        self.max_per_room = int(max_per_room)
        self.max_size_fraction = float(max_size_fraction)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, 1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.room_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.box_projection = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(1, 1, self.max_per_room, hidden_dim) * 0.02
        )
        self.decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    hidden_dim,
                    heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.exists = nn.Linear(hidden_dim, 1)
        self.class_logits = nn.Linear(hidden_dim, num_furniture_classes)
        self.center = nn.Linear(hidden_dim, 2)
        self.size = nn.Linear(hidden_dim, 2)

    def _room_context(
        self,
        features: torch.Tensor,
        room_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.feature_projection(features)
        probabilities = F.interpolate(
            room_logits.float().softmax(dim=1),
            size=hidden.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mass = probabilities.flatten(2).sum(dim=-1).clamp_min(1.0e-5)
        descriptors = torch.einsum(
            "brhw,bchw->brc",
            probabilities,
            hidden,
        ) / mass[..., None]
        y = torch.linspace(
            0.0,
            1.0,
            hidden.shape[-2],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        x = torch.linspace(
            0.0,
            1.0,
            hidden.shape[-1],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        mean_x = (
            probabilities * grid_x[None, None]
        ).flatten(2).sum(dim=-1) / mass
        mean_y = (
            probabilities * grid_y[None, None]
        ).flatten(2).sum(dim=-1) / mass
        variance_x = (
            probabilities * (grid_x[None, None] - mean_x[..., None, None]).square()
        ).flatten(2).sum(dim=-1) / mass
        variance_y = (
            probabilities * (grid_y[None, None] - mean_y[..., None, None]).square()
        ).flatten(2).sum(dim=-1) / mass
        half_width = variance_x.clamp_min(1.0e-5).sqrt() * 1.5
        half_height = variance_y.clamp_min(1.0e-5).sqrt() * 1.5
        boxes = torch.stack(
            [
                (mean_x - half_width).clamp(0.0, 1.0),
                (mean_y - half_height).clamp(0.0, 1.0),
                (mean_x + half_width).clamp(0.0, 1.0),
                (mean_y + half_height).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        box_tokens = self.box_projection(
            torch.cat([boxes, mass[..., None]], dim=-1)
        )
        room_tokens = self.room_projection(descriptors) + box_tokens
        memory = hidden.flatten(2).transpose(1, 2)
        return room_tokens, boxes, mass, memory

    def forward(
        self,
        features: torch.Tensor,
        room_logits: torch.Tensor,
        *,
        render_grid: int | None = None,
    ) -> dict[str, torch.Tensor]:
        room_tokens, boxes, mass, memory = self._room_context(
            features,
            room_logits,
        )
        queries = (
            room_tokens[:, :, None]
            + self.slot_embedding
        ).flatten(1, 2)
        hidden = queries
        for layer in self.decoder:
            hidden = layer(hidden, memory)
        batch = features.shape[0]
        hidden = hidden.reshape(
            batch,
            self.num_rooms,
            self.max_per_room,
            -1,
        )
        hidden = self.output_norm(hidden)
        center_relative = self.center(hidden).sigmoid()
        box_size = (boxes[:, :, 2:] - boxes[:, :, :2]).clamp_min(1.0e-4)
        centers = (
            boxes[:, :, None, :2]
            + center_relative * box_size[:, :, None]
        )
        half_size = (
            self.size(hidden).sigmoid()
            * box_size[:, :, None]
            * self.max_size_fraction
            * 0.5
        )
        half_x = torch.minimum(
            half_size[..., 0],
            torch.minimum(
                centers[..., 0] - boxes[:, :, None, 0],
                boxes[:, :, None, 2] - centers[..., 0],
            ).clamp_min(0.0),
        )
        half_y = torch.minimum(
            half_size[..., 1],
            torch.minimum(
                centers[..., 1] - boxes[:, :, None, 1],
                boxes[:, :, None, 3] - centers[..., 1],
            ).clamp_min(0.0),
        )
        x0 = (centers[..., 0] - half_x).clamp(0.0, 1.0)
        x1 = (centers[..., 0] + half_x).clamp(0.0, 1.0)
        y0 = (centers[..., 1] - half_y).clamp(0.0, 1.0)
        y1 = (centers[..., 1] + half_y).clamp(0.0, 1.0)
        polygons = torch.stack(
            [
                torch.stack([x0, y0], dim=-1),
                torch.stack([x1, y0], dim=-1),
                torch.stack([x1, y1], dim=-1),
                torch.stack([x0, y1], dim=-1),
            ],
            dim=-2,
        )
        output = {
            "furniture_exists_logits": self.exists(hidden).squeeze(-1),
            "furniture_class_logits": self.class_logits(hidden),
            "furniture_center": centers,
            "furniture_size": half_size * 2.0,
            "furniture_polygons": polygons,
            "room_boxes": boxes,
            "room_mass": mass,
        }
        if render_grid is not None:
            flat_polygons = polygons.flatten(1, 2)
            masks = diff_soft_polygon_masks(
                flat_polygons,
                int(render_grid),
            )
            output["furniture_render"] = masks.reshape(
                batch,
                self.num_rooms,
                self.max_per_room,
                int(render_grid),
                int(render_grid),
            )
        return output


__all__ = ["RoomFurnitureVectorHead"]
