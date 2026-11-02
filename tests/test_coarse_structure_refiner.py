from __future__ import annotations

import pytest
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.structure_mita_router import structure_route_metrics


def test_coarse_structure_refiner_shapes_and_control() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room = taxonomy.room_ids[0]
    labels = torch.full((1, 64, 64), room)
    footprint = torch.ones(1, 64, 64)
    control = coarse_structure_control(
        labels,
        footprint,
        taxonomy,
        size=16,
    )
    assert control.shape == (
        1,
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        16,
        16,
    )
    refiner = CoarseStructureRefiner(
        control.shape[1],
        len(taxonomy.opening_ids),
        base_channels=8,
    )
    output = refiner(control)
    assert output["wall_logits"].shape == (1, 1, 32, 32)
    assert output["opening_logits"].shape == (1, 1, 32, 32)
    assert output["opening_type_logits"].shape == (
        1,
        len(taxonomy.opening_ids),
        32,
        32,
    )


def test_typed_structure_refiner_exposes_junctions_and_direct_output() -> None:
    refiner = CoarseStructureRefiner(
        8,
        2,
        base_channels=8,
        extra_input_channels=6,
        direct_control_output=True,
        host_conditioned_openings=True,
        predict_junctions=True,
        predict_room_boundary=True,
    )
    output = refiner(torch.zeros(1, 14, 16, 16))
    assert output["wall_logits"].shape == (1, 1, 16, 16)
    assert output["junction_logits"].shape == (1, 1, 16, 16)
    assert output["junction_degree_logits"].shape == (1, 5, 16, 16)
    assert output["room_boundary_logits"].shape == (1, 1, 16, 16)


def test_typed_host_conditioning_v2_learns_junction_boundary_gate() -> None:
    refiner = CoarseStructureRefiner(
        8,
        2,
        base_channels=8,
        direct_control_output=True,
        host_conditioned_openings=True,
        host_conditioning_v2=True,
        predict_junctions=True,
        predict_room_boundary=True,
    )
    with torch.no_grad():
        refiner.opening_host_scale.fill_(0.5)
        refiner.opening_type_host_scale.fill_(0.5)
    output = refiner(torch.randn(1, 8, 16, 16))
    (
        output["opening_logits"].square().mean()
        + output["opening_type_logits"].square().mean()
    ).backward()
    assert refiner.opening_host_fusion.weight.grad is not None
    assert refiner.opening_type_host_fusion.weight.grad is not None
    with pytest.raises(ValueError, match="host_conditioning_v2"):
        CoarseStructureRefiner(
            8,
            2,
            base_channels=8,
            host_conditioned_openings=True,
            host_conditioning_v2=True,
        )


def test_structure_refiner_global_mita_route_is_trainable() -> None:
    refiner = CoarseStructureRefiner(
        8,
        2,
        base_channels=8,
        direct_control_output=True,
        host_conditioned_openings=True,
        predict_junctions=True,
        global_route_attention=True,
        route_grid=4,
        route_router_count=4,
        route_top_k=2,
        route_heads=4,
    )
    output = refiner(torch.randn(2, 8, 16, 16))
    diagnostics = output["route_diagnostics"]
    assert torch.isfinite(diagnostics["routing_entropy"])
    assert torch.isfinite(diagnostics["route_budget_soft_mean"])
    output["wall_logits"].square().mean().backward()
    assert refiner.global_route is not None
    assert refiner.global_route.route.q_proj.weight.grad is not None
    assert refiner.global_route.budget[0].weight.grad is not None
    metrics = structure_route_metrics(diagnostics)
    assert metrics["structure_route_budget"] > 0.0
    assert metrics["structure_route_entropy"] >= 0.0
