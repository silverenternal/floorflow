from __future__ import annotations

import torch

from floorflow.objectives.design_reward import DesignRewardHead
from floorflow.heads.furniture_action import FurnitureActionHead, apply_furniture_action
from floorflow.heads.registry import default_floorplan_heads
from floorflow.model.layout_router import LayoutRouter
from floorflow.model.layout_stream_mixer import LayoutStreamMixer, stream_gate_diagnostics
from floorflow.model.modality_encoders import FloorplanModalityEncoders
from floorflow.inference.repair_registry import (
    RepairAcceptanceController,
    RepairRegistry,
    RepairStrategy,
    build_floorplan_repair_registry,
)
from floorflow.model.room_graph_transport import RoomGraphTensorRing
from floorflow.heads.spatial_velocity_head import SpatialVelocityHead
from floorflow.inference.structural_attention_repair import StructuralAttentionRepair
from collections import Counter
from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.room_set import RoomSetHead, labels_to_room_targets, room_set_loss, room_set_metrics


def test_modality_and_stream_modules() -> None:
    encoders = FloorplanModalityEncoders(16, 4, 6, 5, 7)
    tokens = encoders(torch.randn(2, 3, 4), torch.randn(2, 4, 6), torch.randn(2, 2, 5), torch.randn(2, 5, 7))
    assert all(value.shape[-1] == 16 for value in tokens.values())
    transport = RoomGraphTensorRing(16, rank=4, order=2)
    assert transport(tokens["room"]).shape == tokens["room"].shape
    mixer = LayoutStreamMixer(16, ("room", "wall", "furniture"))
    mixed, gates = mixer(tokens["room"], tokens)
    assert mixed.shape == tokens["room"].shape
    assert torch.allclose(gates.sum(dim=-1), torch.ones_like(gates.sum(dim=-1)), atol=1.0e-5)
    five_stream = LayoutStreamMixer(
        16,
        ("room", "wall", "opening", "furniture", "geometry"),
    )
    five_tokens = {
        **tokens,
        "geometry": torch.randn(2, 4, 16),
    }
    mixed_five, gates_five = five_stream(tokens["room"], five_tokens)
    diagnostics = stream_gate_diagnostics(gates_five)
    assert mixed_five.shape == tokens["room"].shape
    assert 0.0 <= float(diagnostics["gate_entropy"].detach()) <= 1.0
    assert 0.0 <= float(diagnostics["gate_load_balance"].detach()) <= 1.0
    without_furniture, _ = five_stream(
        tokens["room"],
        {key: value for key, value in five_tokens.items() if key != "furniture"},
    )
    assert torch.isfinite(without_furniture).all()


def test_heads_router_reward_and_repair() -> None:
    head = SpatialVelocityHead(16, ("room", "wall"), output_dim=2)
    velocity, gates = head(torch.randn(2, 8, 16), {"room": torch.randn(2, 3, 16), "wall": torch.randn(2, 4, 16)})
    assert velocity.shape == (2, 8, 2)
    assert gates
    router = LayoutRouter(16, ("velocity", "repair"))
    mixed, routes, diagnostics = router(
        torch.randn(2, 16),
        {"velocity": torch.randn(2, 8, 2), "repair": torch.randn(2, 8, 2)},
        torch.rand(2, 4),
        torch.rand(2),
    )
    assert mixed.shape == (2, 8, 2)
    assert set(routes) == {"velocity", "repair"}
    assert diagnostics["route_entropy"].ndim == 0
    reward = DesignRewardHead(16)
    assert reward(torch.randn(2, 16), torch.randn(2, 8)).shape == (2, 4)
    repair = StructuralAttentionRepair(8)
    delta, gate = repair(torch.randn(2, 8, 16, 16), torch.rand(2, 16, 16), torch.rand(2, 16, 16), torch.rand(2, 16, 16))
    assert delta.shape == (2, 2, 16, 16)
    assert gate.shape == (2, 1, 16, 16)


def test_furniture_and_registries() -> None:
    action_head = FurnitureActionHead(16)
    logits, transform = action_head(torch.randn(2, 3, 16), torch.randn(2, 3, 16))
    assert logits.shape == (2, 3, 5)
    assert transform.shape == (2, 3, 3)
    moved = apply_furniture_action(torch.randn(2, 2), 1, torch.randn(2, 3))
    assert moved.shape == (2, 2)
    registry = RepairRegistry()
    registry.register(
        RepairStrategy("identity", "test", lambda labels: labels.clone(), lambda left, right: True)
    )
    assert registry.candidates(torch.zeros(2, 2))[0][2]
    controller = RepairAcceptanceController(2)
    controller.fit([[1.0, 0.0], [0.0, 1.0]], [True, False])
    assert default_floorplan_heads().weights()["semantic_loss"] == 1.0
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    floorplan_registry = build_floorplan_repair_registry(torch.ones(8, 8), taxonomy)
    assert floorplan_registry.candidates(torch.zeros(8, 8, dtype=torch.long))


def test_room_set_head_and_matching() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 3}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, 2:12, 2:12] = taxonomy.class_id("room:Bedroom")
    labels[:, 16:28, 16:28] = taxonomy.class_id("room:Kitchen")
    classes, boxes, mask = labels_to_room_targets(labels, taxonomy)
    assert int(mask.sum()) == 2
    head = RoomSetHead(16, taxonomy.num_classes)
    predictions = head(torch.randn(1, 16, 8, 8))
    loss = room_set_loss(predictions, classes, boxes, mask)
    assert torch.isfinite(loss)
    assert set(room_set_metrics(predictions, classes, boxes, mask)) == {
        "matched_class_accuracy",
        "matched_box_iou",
        "predicted_instances_per_sample",
    }
