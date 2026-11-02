from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.joint_loss import joint_distribution_cfm_loss
from floorflow.objectives.loss_audit import (
    ObjectiveTerm,
    gradient_contribution_rows,
)


class _Output:
    in_channels = 8


class _Base(nn.Module):
    target_channels = 7

    def __init__(self) -> None:
        super().__init__()
        self.output = _Output()
        self.proj = nn.Conv2d(7, 8, 1)

    def _encode_features(self, value, time, footprint, floor_ids=None):
        del time, floor_ids
        return self.proj(value) + footprint.float()

    def _velocity_from_features(self, features, time=None):
        del time
        return features[:, :7]

    def forward_with_aux(self, value, time, footprint, floor_ids=None):
        del time, floor_ids
        velocity = self.proj(value)[:, :7]
        return velocity, {"room_partition_logits": value[:, :1].expand(-1, 2, -1, -1)}


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_joint_distribution_objective_backpropagates() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    model = DistributionCFMModel(
        _Base(),
        hierarchy,
        hidden_channels=8,
        pixel_grid=4,
        element_grid=4,
        queries_per_group={
            "wall": 1,
            "opening": 1,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
    )
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 2:8, 2:8] = taxonomy.room_ids[0]
    target[:, 9:14, 9:14] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 1, 16, 16)
    contour = model.contour_initial_state(footprint)
    loss, components = joint_distribution_cfm_loss(
        model,
        target,
        contour["cfm"]["state"],
        footprint,
        taxonomy,
        hierarchy,
        queries_per_group={
            "wall": 1,
            "opening": 1,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
        class_weights=torch.ones(taxonomy.num_classes),
        structural_weight=0.1,
        topology_closure_weight=0.1,
        contour_output=contour,
    )
    assert "joint_total_loss" in components
    assert components["joint_endpoint_loss"] > 0
    assert torch.isfinite(components["joint_endpoint_loss"])
    assert "joint_structural_loss" in components
    assert "joint_topology_closure_loss" in components
    assert "joint_furniture_vector_loss" in components
    assert torch.isfinite(components["joint_furniture_vector_loss"])
    assert torch.isfinite(components["joint_structural_loss"])
    rows = gradient_contribution_rows(
        [
            ObjectiveTerm("pixel", components["_pixel_term"]),
            ObjectiveTerm(
                "structural",
                components["_structural_term"],
                weight=0.1,
            ),
        ],
        model.parameters(),
        primary="pixel",
    )
    structural = next(row for row in rows if row["term"] == "structural")
    assert structural["gradient_norm"] > 0.0
    loss.backward()
    assert model.hub.pixel.family_queries.grad is not None
    assert model.cfm_bridge.element_blend.grad is not None
