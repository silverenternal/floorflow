"""Differentiable topology auxiliary heads and SVG-derived targets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


class TopologyAuxHeads(nn.Module):
    def __init__(self, channels: int, num_rooms: int) -> None:
        super().__init__()
        self.num_rooms = int(num_rooms)
        hidden = max(32, channels // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(hidden, 1, 1)
                for name in (
                    "wall",
                    "wall_direction_x",
                    "wall_direction_y",
                    "opening_contact",
                    "containment",
                    "room_boundary",
                )
            }
        )
        self.adjacency_head = nn.Linear(hidden, self.num_rooms * self.num_rooms)
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        output = {name: head(hidden).squeeze(1) for name, head in self.heads.items()}
        pooled = hidden.mean(dim=(-2, -1))
        output["adjacency"] = self.adjacency_head(pooled).reshape(-1, self.num_rooms, self.num_rooms)
        return output


def _ids(taxonomy: FloorplanTaxonomy, predicate) -> torch.Tensor:
    return torch.tensor(
        [index for index, name in enumerate(taxonomy.class_names) if predicate(name)],
        dtype=torch.long,
    )


def derive_topology_targets(
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, torch.Tensor]:
    device = target.device
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    wall_ids = _ids(taxonomy, lambda name: name == "wall").to(device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device, dtype=torch.long)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device, dtype=torch.long)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device, dtype=torch.long)
    wall = torch.isin(target, wall_ids).float()
    rooms = torch.isin(target, room_ids).float()
    furniture = torch.isin(target, furniture_ids).float()
    openings = torch.isin(target, opening_ids).float()
    wall_near = F.max_pool2d(wall[:, None], 5, stride=1, padding=2)[:, 0]
    room_near = F.max_pool2d(rooms[:, None], 7, stride=1, padding=3)[:, 0]
    room_eroded = -F.max_pool2d(-rooms[:, None], 3, stride=1, padding=1)[:, 0]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)
    wall_direction_x = F.conv2d(wall[:, None], sobel_x, padding=1)[:, 0].clamp(-1.0, 1.0) * 0.5 + 0.5
    wall_direction_y = F.conv2d(wall[:, None], sobel_y, padding=1)[:, 0].clamp(-1.0, 1.0) * 0.5 + 0.5
    room_class_ids = list(taxonomy.room_ids)
    adjacency = torch.zeros(
        (target.shape[0], len(room_class_ids), len(room_class_ids)),
        device=device,
    )
    for left_index, left_id in enumerate(room_class_ids):
        left = target == left_id
        for right_index in range(left_index, len(room_class_ids)):
            right = target == room_class_ids[right_index]
            if left_index == right_index:
                adjacency[:, left_index, right_index] = 1.0
                continue
            touching = (left[:, :, :-1] & right[:, :, 1:]).any(dim=(1, 2))
            touching |= (left[:, :-1, :] & right[:, 1:, :]).any(dim=(1, 2))
            adjacency[:, left_index, right_index] = touching.float()
            adjacency[:, right_index, left_index] = touching.float()
    return {
        "wall": wall,
        "wall_direction_x": wall_direction_x,
        "wall_direction_y": wall_direction_y,
        "opening_contact": openings * wall_near,
        "containment": furniture * room_near,
        "room_boundary": (rooms - room_eroded).clamp(0.0, 1.0) * footprint,
        "adjacency": adjacency,
    }


def topology_aux_loss(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = derive_topology_targets(target, footprint, taxonomy)
    footprint = footprint[:, None] if footprint.ndim == 3 else footprint
    losses = {}
    for name, prediction in predictions.items():
        losses[name] = F.binary_cross_entropy_with_logits(
            prediction.reshape(targets[name].shape),
            targets[name],
            reduction="mean",
        )
    total = sum(losses.values()) / max(1, len(losses))
    components = {f"{name}_loss": value.detach() for name, value in losses.items()}
    components["topology_loss"] = total.detach()
    return total, components
