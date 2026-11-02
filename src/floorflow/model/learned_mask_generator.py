"""Contour-conditioned learned mask generator for sparse CAD controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.model.dynamic_budget import (
    allocate_group_token_budgets,
    family_minimum_budgets,
    mean_group_priority,
)


@dataclass(frozen=True)
class GeneratedMask:
    spatial: torch.Tensor
    channel: torch.Tensor
    probability: torch.Tensor
    hard: torch.Tensor
    coverage: torch.Tensor


class LearnedMaskGenerator(nn.Module):
    """Predict where and which coarse-control channels should be completed."""

    def __init__(
        self,
        control_channels: int,
        hidden_channels: int = 48,
        spatial_resolution: int = 64,
    ) -> None:
        super().__init__()
        self.control_channels = int(control_channels)
        self.spatial_resolution = int(spatial_resolution)
        hidden = int(hidden_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(self.control_channels + 1, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.spatial_head = nn.Conv2d(hidden, 1, 1)
        self.channel_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.control_channels),
        )
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.channel_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        control: torch.Tensor,
        footprint: torch.Tensor,
        *,
        target_ratio: float = 0.25,
        known_mask: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        channel_groups: tuple[torch.Tensor, ...] | None = None,
        minimum_visible: tuple[float, ...] | None = None,
    ) -> GeneratedMask:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        if visibility is not None:
            if visibility.ndim == 3:
                visibility = visibility[:, None]
            if visibility.shape[-2:] != footprint.shape[-2:]:
                raise ValueError("visibility must share the footprint spatial shape")
            if visibility.shape[1] not in {1, control.shape[1]}:
                raise ValueError("visibility must have one or all control channels")
            visible = visibility.bool().expand_as(control)
            if bool((control[~visible] != 0).any()):
                raise ValueError("learned mask generator received hidden controls")
        if known_mask is not None:
            if known_mask.ndim == 3:
                known_mask = known_mask[:, None]
            if known_mask.shape[-2:] != footprint.shape[-2:]:
                raise ValueError("known_mask spatial shape does not match footprint")
            known_mask = known_mask.bool().expand_as(control)
            known_mask = known_mask & footprint.bool().expand_as(control)
        inputs = torch.cat([control.float(), footprint.float()], dim=1)
        inputs = F.adaptive_avg_pool2d(
            inputs,
            (self.spatial_resolution, self.spatial_resolution),
        )
        hidden = self.trunk(inputs)
        spatial_logits = self.spatial_head(hidden)
        average = hidden.mean(dim=(2, 3))
        maximum = hidden.amax(dim=(2, 3))
        channel_logits = self.channel_head(torch.cat([average, maximum], dim=-1))
        spatial = torch.sigmoid(self.spatial_scale * spatial_logits)
        channel = torch.sigmoid(self.channel_scale * channel_logits)
        probability = spatial * channel[:, :, None, None]
        probability = F.interpolate(
            probability,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) * footprint.float()
        flat = probability.flatten(1)
        region = footprint.expand_as(probability).flatten(1).bool()
        spatial_count = probability.shape[-2] * probability.shape[-1]
        hard = torch.zeros_like(probability)
        if channel_groups is not None:
            if minimum_visible is None:
                minimum_visible = (0.15, 0.35, 0.50, 0.50)
            if len(minimum_visible) != len(channel_groups):
                raise ValueError("minimum_visible must match channel_groups")
            groups = tuple(
                group.to(device=probability.device, dtype=torch.long)
                for group in channel_groups
            )
            group_visible = torch.tensor(
                minimum_visible,
                device=probability.device,
                dtype=torch.float32,
            )[None].expand(probability.shape[0], -1)
        for index in range(probability.shape[0]):
            valid = region[index]
            known = (
                known_mask[index].flatten()
                if known_mask is not None
                else torch.zeros_like(valid)
            )
            candidates = valid & ~known
            target_count = max(
                1,
                int(round(valid.sum().item() * float(target_ratio))),
            )
            known_count = int(known.sum().item())
            remaining = max(
                0,
                min(int(candidates.sum().item()), target_count - known_count),
            )
            if remaining <= 0:
                hard[index] = known.reshape_as(hard[index])
                continue
            if channel_groups is None:
                scores = flat[index].masked_fill(~candidates, float("-inf"))
                selected = scores.topk(remaining).indices
            else:
                priorities, capacities = mean_group_priority(
                    probability[index : index + 1],
                    footprint[index : index + 1].float(),
                    groups,
                )
                group_known = []
                spatial_offsets = torch.arange(
                    spatial_count,
                    device=probability.device,
                )
                for group in groups:
                    group_indices = (
                        group[:, None] * spatial_count
                        + spatial_offsets[None]
                    ).flatten()
                    group_known.append(int(known[group_indices].sum()))
                group_known = torch.tensor(
                    group_known,
                    device=probability.device,
                    dtype=torch.long,
                )[None]
                minimum = family_minimum_budgets(
                    capacities,
                    group_visible[index : index + 1],
                    target_ratio,
                )
                available = torch.maximum(
                    capacities - group_known,
                    torch.zeros_like(capacities),
                )
                minimum = (minimum - group_known).clamp_min(0)
                minimum = torch.minimum(minimum, available)
                minimum_sum = minimum.sum()
                if int(minimum_sum) > remaining:
                    minimum = (
                        minimum.float() * (float(remaining) / float(minimum_sum))
                    ).floor().long()
                budgets = allocate_group_token_budgets(
                    priorities,
                    available,
                    torch.tensor(
                        [remaining],
                        device=probability.device,
                        dtype=torch.long,
                    ),
                    minimum_counts=minimum,
                    maximum_counts=available,
                )[0]
                selected_parts = []
                for group, budget in zip(groups, budgets):
                    group_indices = (
                        group[:, None] * spatial_count + spatial_offsets[None]
                    ).flatten()
                    group_candidates = region[index][group_indices] & ~known[
                        group_indices
                    ]
                    if int(budget) <= 0:
                        continue
                    group_scores = flat[index][group_indices].masked_fill(
                        ~group_candidates,
                        float("-inf"),
                    )
                    selected_parts.append(
                        group_indices[
                            group_scores.topk(int(budget)).indices
                        ]
                    )
                selected = (
                    torch.cat(selected_parts)
                    if selected_parts
                    else torch.empty(0, device=probability.device, dtype=torch.long)
                )
            proposal = torch.zeros_like(valid)
            proposal[selected] = True
            hard[index] = (known | proposal).reshape_as(hard[index])
        hard = hard * footprint.float()
        coverage = hard.sum(dim=(1, 2, 3)) / (
            footprint.sum(dim=(1, 2, 3)) * self.control_channels
        ).clamp_min(1.0)
        return GeneratedMask(
            spatial=spatial,
            channel=channel,
            probability=probability,
            hard=hard.bool(),
            coverage=coverage,
        )


__all__ = ["GeneratedMask", "LearnedMaskGenerator"]
