from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.boundary_geometry import (
    BoundaryGeometryEncoder,
    contour_geometry_tokens,
)
from floorflow.heads.room_region_refiner import RoomRegionRefiner
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.model.hierarchical_pipeline import (
    CoarseRoomRegionHead,
    HierarchicalRoomPipeline,
    RoomQueryGraphRefiner,
    SemanticRoomGraphReasoner,
)
from floorflow.objectives.token_optimal_transport import (
    room_instance_token_targets,
    room_token_transport_loss,
    sinkhorn_plan,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4, "Bath": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_coarse_region_head_is_zero_initialized_without_embedding_term() -> None:
    head = CoarseRoomRegionHead(12, 3, hidden_dim=24)
    with torch.no_grad():
        head.class_embedding.zero_()
    output = head(torch.randn(2, 12, 32, 32), resolution=8)
    assert output.shape == (2, 3, 8, 8)
    assert torch.allclose(output, torch.zeros_like(output))


def test_boundary_geometry_tokens_encode_ordered_corners() -> None:
    footprint = torch.zeros(2, 32, 32)
    footprint[:, 4:28, 6:26] = 1.0
    points, valid = contour_geometry_tokens(footprint, max_points=16)
    assert points.shape == (2, 16, 2)
    assert valid.shape == (2, 16)
    assert torch.all((points >= 0.0) & (points <= 1.0))
    assert valid.sum(dim=1).min() >= 4
    encoded = BoundaryGeometryEncoder(hidden_dim=24, max_points=16)(footprint)
    assert encoded["tokens"].shape == (2, 16, 24)
    assert torch.isfinite(encoded["tokens"]).all()


def test_room_graph_has_normalized_adjacency_and_backpropagates() -> None:
    features = torch.randn(2, 12, 16, 16, requires_grad=True)
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    tokens = torch.randn(2, 3, 24, requires_grad=True)
    graph = SemanticRoomGraphReasoner(12, 3, hidden_dim=24)
    with torch.no_grad():
        graph.output_scale.fill_(1.0)
    output = graph(features, logits, tokens)
    assert output["graph_tokens"].shape == (2, 3, 24)
    assert output["adjacency"].shape == (2, 3, 3)
    assert torch.allclose(
        output["adjacency"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1.0e-5,
    )
    output["semantic_bias"].mean().backward()
    assert tokens.grad is not None


def test_room_query_graph_refiner_uses_mask_adjacency_and_backpropagates() -> None:
    queries = torch.randn(2, 4, 24, requires_grad=True)
    mask_logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    footprint = torch.zeros(2, 1, 8, 8)
    footprint[:, :, 1:7, 1:7] = 1.0
    refiner = RoomQueryGraphRefiner(hidden_dim=24)
    output = refiner(queries, mask_logits, footprint)
    assert output["queries"].shape == queries.shape
    assert output["adjacency"].shape == (2, 4, 4)
    assert torch.allclose(
        output["adjacency"].sum(dim=-1),
        torch.ones(2, 4),
        atol=1.0e-5,
    )
    output["queries"].mean().backward()
    assert queries.grad is not None
    assert refiner.message[0].weight.grad is not None


def test_hierarchical_pipeline_shapes_and_transport_plan() -> None:
    taxonomy = _taxonomy()
    pipeline = HierarchicalRoomPipeline(
        12,
        len(taxonomy.room_ids),
        hidden_dim=24,
        tokens=4,
        coarse_resolution=8,
    )
    footprint = torch.zeros(2, 32, 32)
    footprint[:, 3:29, 4:28] = 1.0
    output = pipeline(torch.randn(2, 12, 32, 32), footprint=footprint)
    assert output["program"]["presence_logits"].shape == (
        2,
        len(taxonomy.room_ids),
    )
    assert output["coarse_room_logits"].shape == (
        2,
        len(taxonomy.room_ids),
        8,
        8,
    )
    assert torch.equal(output["coarse_room_logits"], output["refined_room_logits"])
    assert output["room_region_attention"].shape == (
        2,
        len(taxonomy.room_ids),
        8,
        8,
    )
    assert output["boundary_geometry"]["tokens"].shape == (2, 64, 24)
    cost = torch.tensor(
        [[[0.1, 0.9, 0.8], [0.8, 0.2, 0.9], [0.9, 0.8, 0.3]]]
    )
    plan = sinkhorn_plan(cost, epsilon=0.05, iterations=40)
    assert torch.allclose(plan.sum(dim=1), torch.full((1, 3), 1.0 / 3.0), atol=1.0e-4)
    assert torch.allclose(plan.sum(dim=2), torch.full((1, 3), 1.0 / 3.0), atol=1.0e-4)
    assert plan[0].argmax(dim=1).tolist() == [0, 1, 2]


def test_token_transport_loss_and_valid_mask() -> None:
    predicted = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [9.0, 9.0]]])
    classes = torch.tensor([[1, 2, 3]])
    valid = torch.tensor([[True, True, False]])
    loss, diagnostics = room_token_transport_loss(
        predicted,
        target,
        predicted_classes=classes,
        target_classes=classes,
        target_valid=valid,
        epsilon=0.05,
        iterations=30,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(diagnostics["room_transport_target_recall"]) <= 1.0


def test_geometry_only_transport_matches_with_unknown_room_classes() -> None:
    predicted = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    loss, diagnostics = room_token_transport_loss(
        predicted,
        target,
        target_valid=torch.ones(1, 2, dtype=torch.bool),
        epsilon=0.05,
        iterations=30,
    )
    assert torch.isfinite(loss)
    assert float(diagnostics["room_transport_target_recall"]) > 0.9


def test_room_instance_targets_are_normalized_and_sorted() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[0, 1:6, 1:6] = taxonomy.room_ids[0]
    labels[0, 7:11, 8:13] = taxonomy.room_ids[1]
    labels[1, 2:14, 2:14] = taxonomy.room_ids[2]
    classes, tokens, valid = room_instance_token_targets(
        labels,
        taxonomy,
        max_rooms=4,
    )
    assert classes.shape == tokens.shape[:2] == valid.shape == (2, 4)
    assert valid[0].sum() == 2
    assert valid[1].sum() == 1
    assert tokens[0, 0, 0] > tokens[0, 1, 0]
    assert torch.all((tokens[:, :, :4] >= 0.0) & (tokens[:, :, :4] <= 1.0))


def test_hierarchical_pipeline_integrates_with_flow_model() -> None:
    model = FloorplanFlowModel(
        target_channels=4,
        base_channels=8,
        levels=2,
        semantic_channel_attention=False,
        slot_mixing=False,
        dynamic_output_head=False,
        topology_aux=False,
        floor_conditioning=False,
        semantic_endpoint_head=False,
        room_partition_head=True,
        num_room_classes=3,
        hierarchical_rooms=True,
        hierarchical_hidden=16,
        hierarchical_tokens=2,
        hierarchical_coarse_resolution=8,
    )
    _, aux = model.forward_with_aux(
        torch.zeros(2, 4, 32, 32),
        torch.zeros(2),
        torch.ones(2, 32, 32),
        layout_prior_logits=torch.randn(2, 3, 32, 32),
    )
    assert aux["room_partition_logits"].shape == (2, 3, 32, 32)
    assert aux["hierarchical_rooms"]["coarse_room_logits"].shape == (2, 3, 8, 8)
    assert aux["hierarchical_rooms"]["refined_room_logits"].shape == (2, 3, 8, 8)
    assert aux["room_program"]["presence_logits"].shape == (2, 3)


def test_hierarchical_branch_is_identity_at_initialization() -> None:
    kwargs = {
        "target_channels": 4,
        "base_channels": 8,
        "levels": 2,
        "semantic_channel_attention": False,
        "slot_mixing": False,
        "dynamic_output_head": False,
        "topology_aux": False,
        "floor_conditioning": False,
        "semantic_endpoint_head": False,
        "room_partition_head": True,
        "num_room_classes": 3,
    }
    torch.manual_seed(11)
    baseline = FloorplanFlowModel(**kwargs)
    torch.manual_seed(11)
    hierarchical = FloorplanFlowModel(
        **kwargs,
        hierarchical_rooms=True,
        hierarchical_hidden=16,
        hierarchical_tokens=2,
        hierarchical_coarse_resolution=8,
    )
    inputs = (
        torch.zeros(2, 4, 32, 32),
        torch.zeros(2),
        torch.ones(2, 32, 32),
    )
    _, baseline_aux = baseline.forward_with_aux(*inputs)
    _, hierarchical_aux = hierarchical.forward_with_aux(*inputs)
    assert torch.allclose(
        baseline_aux["room_partition_logits"],
        hierarchical_aux["room_partition_logits"],
    )


def test_region_refiner_is_identity_initialized() -> None:
    refiner = RoomRegionRefiner(12, 3, hidden_dim=24)
    features = torch.randn(2, 12, 16, 16)
    coarse = torch.randn(2, 3, 8, 8)
    tokens = torch.randn(2, 3, 24)
    output = refiner(features, coarse, tokens)
    assert torch.equal(output["refined_room_logits"], coarse)
    output["refined_room_probabilities"].mean().backward()
    assert refiner.residual_scale.grad is not None
