"""Typed graph encoder for legal and near-violation floorplan states."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.graph_edit_encoder import GraphEditEncoder

FAMILY_NAMES = ("wet", "circulation", "living", "service", "other")


def room_family_ids(taxonomy: FloorplanTaxonomy) -> torch.Tensor:
    """Map room channels to a small semantic family vocabulary."""

    family_by_name = {
        "Bath": 0,
        "Sauna": 0,
        "Hall": 1,
        "Entry": 1,
        "DraughtLobby": 1,
        "LivingRoom": 2,
        "Den": 2,
        "Dining": 2,
        "Kitchen": 3,
        "Utility": 3,
        "Storage": 3,
        "Garage": 3,
        "CarPort": 3,
    }
    ids = []
    for class_id in taxonomy.room_ids:
        name = taxonomy.class_names[class_id].removeprefix("room:")
        ids.append(family_by_name.get(name, 4))
    return torch.tensor(ids, dtype=torch.long)


def d4_transform_graph(
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    transform: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a D4 symmetry to normalized centroid features."""

    output = node_features.clone()
    x = node_features[..., 4]
    y = node_features[..., 5]
    index = int(transform) % 8
    if index == 1:
        output[..., 4], output[..., 5] = y, 1.0 - x
    elif index == 2:
        output[..., 4], output[..., 5] = 1.0 - x, 1.0 - y
    elif index == 3:
        output[..., 4], output[..., 5] = 1.0 - y, x
    elif index == 4:
        output[..., 4], output[..., 5] = 1.0 - x, y
    elif index == 5:
        output[..., 4], output[..., 5] = y, x
    elif index == 6:
        output[..., 4], output[..., 5] = x, 1.0 - y
    elif index == 7:
        output[..., 4], output[..., 5] = 1.0 - y, 1.0 - x
    return output, adjacency


def permute_graph_nodes(
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a node permutation consistently to features and edges."""

    return (
        node_features[..., permutation, :],
        adjacency[..., permutation, :][..., :, permutation],
    )


class ConstraintStateEncoder(nn.Module):
    """Encode graph state, predict violations, and expose a normality energy."""

    def __init__(
        self,
        node_features: int = 8,
        hidden_dim: int = 384,
        embedding_dim: int = 192,
        operations: int = 5,
        families: int = len(FAMILY_NAMES),
        violation_families: int = 7,
        layers: int = 6,
        global_features: int = 15,
        action_features: int = 15,
        hard_mask_features: int = 8,
    ) -> None:
        super().__init__()
        self.graph = GraphEditEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            operations=operations,
            families=families,
            layers=layers,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        context_size = global_features + action_features + hard_mask_features
        self.context_projection = nn.Sequential(
            nn.Linear(context_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_scale = nn.Parameter(torch.zeros(1))
        self.context_violation_head = nn.Sequential(
            nn.Linear(context_size, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, violation_families),
        )
        self.context_legal_head = nn.Sequential(
            nn.Linear(context_size, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.violation_head = nn.Linear(hidden_dim, violation_families)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.legal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        family_ids: torch.Tensor,
        operation_ids: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
        action_features: torch.Tensor | None = None,
        hard_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.graph(
            node_features.float(),
            adjacency.float(),
            family_ids,
            operation_ids,
            node_mask=node_mask,
        )
        batch = node_features.shape[0]
        if global_features is None:
            global_features = node_features.new_zeros(batch, 15)
        if action_features is None:
            action_features = node_features.new_zeros(batch, 15)
        if hard_mask is None:
            hard_mask = node_features.new_zeros(batch, 8)
        context = torch.cat(
            [
                global_features.float(),
                action_features.float(),
                hard_mask.float(),
            ],
            dim=-1,
        )
        global_token = (
            encoded["global_token"]
            + self.context_scale.to(encoded["global_token"].dtype)
            * self.context_projection(context)
        )
        embedding = nn.functional.normalize(
            self.projection(global_token),
            dim=-1,
        )
        return {
            "embedding": embedding,
            "violation_logits": (
                self.violation_head(global_token)
                + 0.5 * self.context_violation_head(context)
            ),
            "energy": self.energy_head(global_token).squeeze(-1),
            "legal_logits": (
                self.legal_head(global_token).squeeze(-1)
                + 0.5 * self.context_legal_head(context).squeeze(-1)
            ),
            "global_token": global_token,
            "node_tokens": encoded["node_tokens"],
        }


__all__ = [
    "FAMILY_NAMES",
    "ConstraintStateEncoder",
    "d4_transform_graph",
    "permute_graph_nodes",
    "room_family_ids",
]
