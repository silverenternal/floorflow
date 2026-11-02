from __future__ import annotations

import torch

from floorflow.flow.coarse_to_face import CoarseToFaceFlowHead
from floorflow.heads.geometry_heads import GeometryHeatmapHeads
from floorflow.heads.opening_interval_head import WallIntervalHead
from floorflow.heads.topology_moe import SpatialExpertMoE
from floorflow.model.family_stream_mixer import SemanticFamilyStreamMixer
from floorflow.model.builder import ModelMacro
from floorflow.model.specs import ArchSpec


def test_family_stream_mixer_shapes_and_gate_diagnostics() -> None:
    mixer = SemanticFamilyStreamMixer(16)
    hidden = torch.randn(2, 16, 8, 8)
    condition = torch.randn(2, 16, 8, 8)
    topology = torch.randn(2, 5, 16)
    output, diagnostics = mixer(hidden, condition, topology)
    assert output.shape == hidden.shape
    assert diagnostics["stream_gates"].shape == (2, 4)
    assert torch.isfinite(diagnostics["gate_entropy"])
    assert 0.0 < float(diagnostics["gate_load_balance"].detach()) <= 1.0


def test_spatial_expert_moe_routes_to_top_k_experts() -> None:
    moe = SpatialExpertMoE(12, 2, experts=4, top_k=2)
    hidden = torch.randn(2, 12, 7, 9, requires_grad=True)
    output, diagnostics = moe(hidden)
    assert output.shape == (2, 2, 7, 9)
    assert diagnostics["expert_usage"].shape == (2, 4)
    assert torch.allclose(
        diagnostics["expert_usage"].sum(dim=-1),
        torch.ones(2),
        atol=1.0e-5,
    )
    output.mean().backward()
    assert hidden.grad is not None


def test_geometry_heads_accept_expert_residuals() -> None:
    heads = GeometryHeatmapHeads(8, hidden_dim=16, experts=4, top_k=2)
    output = heads(torch.randn(1, 8, 16, 16))
    assert output["wall_centerline_logits"].shape == (1, 1, 16, 16)
    assert output["opening_center_logits"].shape == (1, 1, 16, 16)
    assert output["moe_diagnostics"]["expert_usage"].shape == (1, 4)


def test_coarse_to_face_semantic_tokens_receive_gradient() -> None:
    head = CoarseToFaceFlowHead(
        8,
        4,
        3,
        hidden_dim=16,
        max_faces=4,
        token_dim=12,
    )
    with torch.no_grad():
        head.semantic_gate.fill_(1.0)
        head.output_scale.fill_(1.0)
        head.output.weight.normal_()
    x_t = torch.randn(1, 4, 16, 16)
    time = torch.rand(1)
    features = torch.randn(1, 8, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    coarse = torch.randn(1, 3, 16, 16)
    tokens = torch.randn(1, 5, 12, requires_grad=True)
    output = head(x_t, time, features, footprint, coarse, semantic_tokens=tokens)
    assert output.shape == x_t.shape
    output.mean().backward()
    assert tokens.grad is not None
    assert tokens.grad.abs().sum() > 0


def test_p2_modules_compose_in_the_backbone() -> None:
    spec = ArchSpec(
        name="p2-tiny",
        target_channels=4,
        base_channels=16,
        levels=2,
        resolution=(32, 32),
        backbone_kwargs={
            "semantic_channel_attention": False,
            "slot_mixing": False,
            "floor_conditioning": False,
            "topology_aux": False,
            "semantic_endpoint_head": False,
            "fourier_conditioning": False,
            "family_streams": True,
            "family_mita_attention": True,
            "room_partition_head": True,
            "num_room_classes": 3,
            "room_program_head": True,
            "hierarchical_rooms": True,
            "hierarchical_hidden": 16,
            "hierarchical_tokens": 4,
            "hierarchical_coarse_resolution": 16,
            "coarse_to_face_cfm": True,
            "coarse_face_hidden": 16,
            "coarse_face_max_faces": 4,
            "geometry_heads": True,
            "geometry_hidden": 16,
            "geometry_experts": 4,
            "geometry_top_k": 2,
        },
    )
    model = ModelMacro(spec).build()
    velocity, auxiliary = model.forward_with_aux(
        torch.randn(1, 4, 32, 32),
        torch.rand(1),
        torch.ones(1, 1, 32, 32),
    )
    assert velocity.shape == (1, 4, 32, 32)
    assert auxiliary["geometry"]["moe_diagnostics"]["expert_usage"].shape == (1, 4)
    assert auxiliary["hierarchical_rooms"]["room_graph_tokens"].shape == (1, 3, 16)
    assert auxiliary["family_stream"]["stream_gates"].shape == (1, 4)


def test_wall_interval_head_keeps_openings_on_host_walls() -> None:
    head = WallIntervalHead(16, num_types=2)
    wall_tokens = torch.randn(1, 2, 16)
    opening_tokens = torch.randn(1, 5, 16)
    endpoints = torch.tensor(
        [[[0.0, 0.0, 10.0, 0.0], [0.0, 5.0, 10.0, 5.0]]]
    )
    output = head(wall_tokens, opening_tokens, endpoints)
    assert output["center_xy"].shape == (1, 2, 2)
    assert torch.allclose(output["center_xy"][..., 1], torch.tensor([[0.0, 5.0]]))
    assert output["type_logits"].shape == (1, 2, 2)

    empty = head(wall_tokens, torch.zeros(1, 0, 16), endpoints)
    assert torch.isfinite(empty["center_xy"]).all()
