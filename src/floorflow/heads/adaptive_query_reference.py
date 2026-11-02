"""Image-conditioned reference sampling for room-query decoders."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def prototype_reference_points(
    reference_logits: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Place query anchors at saliency centroids of their initial grid cells."""

    if reference_logits.ndim != 4:
        raise ValueError("reference logits must be [B,C,H,W]")
    saliency = reference_logits.float().sigmoid().amax(dim=1, keepdim=True)
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    height, width = saliency.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=saliency.device)
    x = torch.linspace(0.0, 1.0, width, device=saliency.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    anchors = []
    for index in range(count):
        row, column = divmod(index, columns)
        top = row * height // rows
        bottom = max(top + 1, (row + 1) * height // rows)
        left = column * width // columns
        right = max(left + 1, (column + 1) * width // columns)
        weights = saliency[:, 0, top:bottom, left:right]
        mass = weights.sum(dim=(1, 2)).clamp_min(1.0e-6)
        anchors.append(
            torch.stack(
                [
                    (weights * xx[top:bottom, left:right]).sum(dim=(1, 2))
                    / mass,
                    (weights * yy[top:bottom, left:right]).sum(dim=(1, 2))
                    / mass,
                ],
                dim=-1,
            )
        )
    return torch.stack(anchors, dim=1)


class AdaptiveQueryReferenceRefiner(nn.Module):
    """Sample image features at learned, sample-dependent room references."""

    def __init__(
        self,
        query_dim: int,
        feature_channels: int,
        *,
        hidden_dim: int = 64,
        reference_scale: float = 0.25,
        points_per_query: int = 4,
        live_reference_residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(
            int(feature_channels),
            int(query_dim),
            1,
        )
        self.reference_head = nn.Sequential(
            nn.Linear(int(query_dim) * 2, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * int(points_per_query)),
        )
        self.context = nn.Sequential(
            nn.Linear(int(query_dim) * 2, int(query_dim)),
            nn.SiLU(),
            nn.Linear(int(query_dim), int(query_dim)),
        )
        self.reference_scale = float(reference_scale)
        self.points_per_query = int(points_per_query)
        self.live_reference_residual_scale = float(
            live_reference_residual_scale
        )
        self.last_reference_points: torch.Tensor | None = None
        nn.init.normal_(
            self.reference_head[-1].weight,
            mean=0.0,
            std=1.0e-3,
        )
        nn.init.zeros_(self.reference_head[-1].bias)
        nn.init.normal_(self.context[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.context[-1].bias)

    def initial_reference_points(
        self,
        queries: torch.Tensor,
        reference_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        count = max(1, queries.shape[1])
        columns = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / columns))
        grid = torch.arange(count, device=queries.device, dtype=queries.dtype)
        x = (grid % columns + 0.5) / float(columns)
        y = (grid // columns + 0.5) / float(rows)
        uniform = torch.stack([x, y], dim=-1)[None].expand(
            queries.shape[0],
            -1,
            -1,
        )
        if reference_logits is None:
            return uniform
        anchors = prototype_reference_points(reference_logits, count)
        return 0.5 * (uniform + anchors.to(queries.dtype))

    def forward(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        reference_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if queries.ndim != 3 or features.ndim != 4:
            raise ValueError("queries must be [B,Q,D] and features [B,C,H,W]")
        projected = self.feature_projection(features)
        global_context = projected.mean(dim=(2, 3))[:, None].expand(
            -1,
            queries.shape[1],
            -1,
        )
        joint = torch.cat([queries, global_context], dim=-1)
        uniform_reference = self.initial_reference_points(queries)
        base_reference = self.initial_reference_points(
            queries,
            reference_logits,
        )
        offsets = torch.tanh(self.reference_head(joint)).reshape(
            queries.shape[0],
            queries.shape[1],
            self.points_per_query,
            2,
        )
        reference_samples = (
            base_reference[:, :, None]
            + self.reference_scale * offsets
        ).clamp(0.0, 1.0)
        reference = reference_samples[:, :, 0]
        grid = reference_samples * 2.0 - 1.0
        sampled = F.grid_sample(
            projected,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).permute(0, 2, 3, 1).mean(dim=2)
        update = self.context(
            torch.cat([queries, sampled], dim=-1)
        )
        if self.live_reference_residual_scale > 0.0:
            update = (
                update
                + self.live_reference_residual_scale * sampled
            )
        self.last_reference_points = reference.detach()
        return update, {
            "reference_points": reference,
            "reference_samples": reference_samples,
            "reference_displacement": (offsets * self.reference_scale).abs().mean(),
            "sampled_feature_norm": sampled.norm(dim=-1).mean(),
            "reference_prior_shift": (
                base_reference - uniform_reference
            ).norm(dim=-1).mean(),
        }


__all__ = ["AdaptiveQueryReferenceRefiner", "prototype_reference_points"]
