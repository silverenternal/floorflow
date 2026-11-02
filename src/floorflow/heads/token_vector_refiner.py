"""Token-state conditioned residual refinement for vector CAD outputs."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.representation.tensor_checks import assert_true_async


class TokenVectorRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        max_walls: int,
        max_openings: int,
        polygon_points: int,
        heads: int = 4,
        residual_scale: float = 0.02,
        min_confidence: float = 0.05,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.max_walls = int(max_walls)
        self.max_openings = int(max_openings)
        self.polygon_points = int(polygon_points)
        self.residual_scale = float(residual_scale)
        self.min_confidence = float(min_confidence)
        if self.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.room_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.room_norm = nn.LayerNorm(hidden_dim)
        self.room_delta = nn.Linear(
            hidden_dim,
            self.polygon_points * 2,
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.wall_delta = nn.Linear(
            hidden_dim,
            self.max_walls * 2 * 2,
        )
        self.opening_center_delta = nn.Linear(
            hidden_dim,
            self.max_openings * 2,
        )
        self.opening_width_delta = nn.Linear(
            hidden_dim,
            self.max_openings,
        )
        self.channel_gate = nn.Linear(hidden_dim, 3)
        self.room_entity_gate = nn.Linear(hidden_dim, 1)
        self.wall_entity_gate = nn.Linear(4, 1)
        self.opening_entity_gate = nn.Linear(3, 1)
        self.confidence_temperature = nn.Parameter(torch.ones(()))

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        token_hidden: torch.Tensor,
        token_mask: torch.Tensor,
        token_confidence: torch.Tensor | None = None,
        stage_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if token_hidden.ndim != 3 or token_mask.shape != token_hidden.shape[:2]:
            raise ValueError("token hidden states and mask must align")
        assert_true_async(
            token_mask.any(dim=1).all(),
            "every sample needs at least one token state",
        )
        room_features = predictions["room_features"]
        room_context = self.room_attention(
            room_features,
            token_hidden,
            token_hidden,
            key_padding_mask=~token_mask,
            need_weights=False,
        )[0]
        room_delta = self.room_delta(
            self.room_norm(room_features + room_context)
        ).reshape(*room_features.shape[:2], self.polygon_points, 2)
        weights = token_mask[..., None].to(token_hidden.dtype)
        global_context = (
            token_hidden * weights
        ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        global_context = self.global_norm(global_context)
        confidence = (
            token_confidence
            if token_confidence is not None
            else token_hidden.new_ones(token_hidden.shape[0])
        )
        confidence = confidence.clamp(0.0, 1.0) ** (
            1.0 / self.confidence_temperature.clamp_min(1.0e-3)
        )
        channel_logits = self.channel_gate(global_context)
        if stage_confidence is not None:
            channel_logits = channel_logits + stage_confidence[
                :, torch.tensor([0, 3, 5], device=stage_confidence.device)
            ]
        gates = channel_logits.sigmoid() * confidence[:, None]
        fallback = confidence < self.min_confidence
        gates = gates.masked_fill(fallback[:, None], 0.0)
        wall_delta = self.wall_delta(global_context).reshape(
            token_hidden.shape[0],
            self.max_walls,
            2,
            2,
        )
        opening_center_delta = self.opening_center_delta(global_context).reshape(
            token_hidden.shape[0],
            self.max_openings,
            2,
        )
        opening_width_delta = self.opening_width_delta(global_context)
        room_gates = (
            self.room_entity_gate(room_context).squeeze(-1).sigmoid()
            * gates[:, 0, None]
        )
        wall_gates = (
            self.wall_entity_gate(
                predictions["wall_endpoints"].flatten(2)
            ).squeeze(-1).sigmoid()
            * gates[:, 1, None]
        )
        opening_context = torch.cat(
            [
                predictions["opening_centers"],
                predictions["opening_widths"].unsqueeze(-1),
            ],
            dim=-1,
        )
        opening_gates = (
            self.opening_entity_gate(opening_context).squeeze(-1).sigmoid()
            * gates[:, 2, None]
        )
        scale = self.residual_scale
        room_update = scale * room_gates[..., None, None] * torch.tanh(room_delta)
        wall_update = scale * wall_gates[..., None, None] * torch.tanh(wall_delta)
        center_update = (
            scale * opening_gates[..., None] * torch.tanh(opening_center_delta)
        )
        width_update = (
            scale * opening_gates * torch.tanh(opening_width_delta)
        )
        return {
            **predictions,
            "room_polygons": predictions["room_polygons"] + room_update,
            "wall_endpoints": predictions["wall_endpoints"] + wall_update,
            "opening_centers": predictions["opening_centers"] + center_update,
            "opening_widths": predictions["opening_widths"] + width_update,
            "token_refinement_applied": token_hidden.new_tensor(1.0),
            "token_refiner_confidence": confidence,
            "token_refiner_gate_room": gates[:, 0],
            "token_refiner_gate_wall": gates[:, 1],
            "token_refiner_gate_opening": gates[:, 2],
            "token_refiner_entity_gate_room": room_gates.mean(dim=1),
            "token_refiner_entity_gate_wall": wall_gates.mean(dim=1),
            "token_refiner_entity_gate_opening": opening_gates.mean(dim=1),
            "token_refiner_fallback": fallback.float(),
            "token_refiner_room_residual_norm": room_update.flatten(1).norm(dim=1),
            "token_refiner_wall_residual_norm": wall_update.flatten(1).norm(dim=1),
            "token_refiner_opening_residual_norm": (
                center_update.flatten(1).norm(dim=1)
                + width_update.norm(dim=1)
            ),
        }


__all__ = ["TokenVectorRefiner"]
