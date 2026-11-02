from __future__ import annotations

import math

import pytest
import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.external_graph_conditioning import (
    NODE_SEMANTIC_DIM,
    apply_external_conditioning,
    external_netlist_hypergraph,
    normalize_orientation,
)
from floorflow.chip_flow.external_portfolio import (
    ExternalLegalizerCandidate,
    select_candidate_with_analytical_response,
)
from floorflow.chip_flow.global_spec import GlobalSpecConditioner, global_spec_vector
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchy_targets import build_hierarchy_targets
from floorflow.chip_flow.legalizer_response import apply_legalizer_displacement
from floorflow.chip_flow.netlist_hypergraph import (
    NetlistHyperedgeConditioner,
    NetlistHypergraphContext,
)
from floorflow.chip_flow.ppa_external import ExternalMacroGraph
from floorflow.chip_flow.source_mixing import source_shelf_probability_at_step
from floorflow.chip_flow.structured_heads import MembershipHierarchyCoupler
from floorflow.chip_flow.training_manifest import validate_resume_config
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


def _chip_batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 0.0]] * 2
    )
    constraints[0, :2, 2] = 1
    constraints[1, :3, 2] = 1
    constraints[0, :2, 3] = 1
    constraints[0, 2:4, 3] = 2
    constraints[1, 1:4, 3] = 1
    boxes = torch.stack(
        [
            torch.stack(
                [
                    torch.arange(count).float(),
                    torch.zeros(count),
                    torch.ones(count),
                    torch.ones(count),
                ],
                dim=-1,
            )
        ]
        * 2
    )
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6]],
                [[0.1, 0.5], [0.7, 0.9]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
        "target_latent": torch.randn(2, count, 3),
    }


def test_global_spec_vector_and_conditioner_are_masked() -> None:
    vector = global_spec_vector(
        core_width=torch.tensor([20.0]),
        core_height=torch.tensor([10.0]),
        die_width=torch.tensor([22.0]),
        die_height=torch.tensor([12.0]),
        macro_area=torch.tensor([100.0]),
    )
    assert vector.shape == (1, 7)
    assert torch.isfinite(vector).all()
    assert 0.0 < float(vector[0, 5]) <= 1.0
    conditioner = GlobalSpecConditioner(hidden=16)
    assert torch.equal(conditioner(vector), torch.zeros(1, 16))
    conditioner.gate.data.fill_(1.0)
    torch.nn.init.normal_(conditioner.network[-1].weight, std=0.02)
    assert torch.equal(
        conditioner(vector, torch.zeros(1, dtype=torch.bool)),
        torch.zeros(1, 16),
    )
    assert torch.allclose(
        conditioner(vector),
        conditioner(vector, torch.ones(1, dtype=torch.bool)),
    )
    assert not torch.equal(conditioner(vector), torch.zeros(1, 16))


def test_typed_encoder_semantic_features_are_opt_in_and_receive_gradients() -> None:
    torch.manual_seed(4)
    batch = _chip_batch()
    base = TypedGraphEncoder(hidden=32, heads=4, graph_layers=1).eval()
    conditioned = TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=1,
        global_spec_conditioning=True,
        node_semantic_dim=3,
        port_semantic_dim=2,
    ).eval()
    conditioned.load_state_dict(base.state_dict(), strict=False)
    base.cross_attention.gate.data.fill_(1.0)
    conditioned.cross_attention.gate.data.fill_(1.0)
    features = torch.randn(2, 5, 19)
    global_spec = torch.randn(2, 7)
    node_features = torch.randn(2, 5, 3)
    port_features = torch.randn(2, 2, 2)
    with torch.no_grad():
        expected = base(
            features,
            batch["constraints"],
            areas=batch["areas"],
            b2b=batch["b2b"],
            p2b=batch["p2b"],
            pins=batch["pins"],
            n_b2b=batch["n_b2b"],
            n_p2b=batch["n_p2b"],
            n_pins=batch["n_pins"],
        ).block_tokens
        actual = conditioned(
            features,
            batch["constraints"],
            areas=batch["areas"],
            b2b=batch["b2b"],
            p2b=batch["p2b"],
            pins=batch["pins"],
            n_b2b=batch["n_b2b"],
            n_p2b=batch["n_p2b"],
            n_pins=batch["n_pins"],
            global_spec=global_spec,
            node_features=node_features,
            port_features=port_features,
        ).block_tokens
    assert torch.allclose(actual, expected, atol=1.0e-6)
    conditioned.node_semantic_encoder.gate.data.fill_(1.0)
    conditioned.port_semantic_encoder.gate.data.fill_(1.0)
    conditioned.global_spec_conditioner.gate.data.fill_(1.0)
    torch.nn.init.normal_(conditioned.node_semantic_encoder.network[-1].weight, std=0.02)
    changed = conditioned(
        features,
        batch["constraints"],
        areas=batch["areas"],
        b2b=batch["b2b"],
        p2b=batch["p2b"],
        pins=batch["pins"],
        n_b2b=batch["n_b2b"],
        n_p2b=batch["n_p2b"],
        n_pins=batch["n_pins"],
        global_spec=global_spec,
        node_features=node_features,
        port_features=port_features,
    ).block_tokens
    assert not torch.allclose(changed, expected)
    changed.square().mean().backward()
    assert conditioned.node_semantic_encoder.network[-1].weight.grad is not None


def test_opt_in_conditioning_fails_when_required_inputs_are_missing() -> None:
    batch = _chip_batch()
    features = torch.randn(2, 5, 19)
    common = {
        "areas": batch["areas"],
        "b2b": batch["b2b"],
        "p2b": batch["p2b"],
        "pins": batch["pins"],
        "n_b2b": batch["n_b2b"],
        "n_p2b": batch["n_p2b"],
        "n_pins": batch["n_pins"],
    }
    global_model = TypedGraphEncoder(
        hidden=32,
        heads=4,
        global_spec_conditioning=True,
    )
    node_model = TypedGraphEncoder(
        hidden=32,
        heads=4,
        node_semantic_dim=3,
    )
    port_model = TypedGraphEncoder(
        hidden=32,
        heads=4,
        port_semantic_dim=2,
    )
    with pytest.raises(ValueError, match="global_spec"):
        global_model(features, batch["constraints"], **common)
    with pytest.raises(ValueError, match="node_features"):
        node_model(features, batch["constraints"], **common)
    with pytest.raises(ValueError, match="port_features"):
        port_model(features, batch["constraints"], **common)


def test_dynamic_hyperedge_conditioner_depends_on_current_state() -> None:
    torch.manual_seed(6)
    model = NetlistHyperedgeConditioner(16, feature_dim=4)
    incidence = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    context = NetlistHypergraphContext(
        features=torch.randn(2, 3, 4),
        block_incidence=incidence,
        mask=torch.ones(2, 3, dtype=torch.bool),
    )
    tokens = torch.randn(2, 4, 16)
    left = model(torch.randn(2, 4, 3), tokens, context)
    right = model(torch.randn(2, 4, 3), tokens, context)
    assert left.shape == (2, 4, 3)
    assert torch.equal(left, torch.zeros_like(left))
    model.gate.data.fill_(1.0)
    torch.nn.init.normal_(model.network[-1].weight, std=0.02)
    left = model(torch.zeros(2, 4, 3), tokens, context)
    right = model(torch.ones(2, 4, 3), tokens, context)
    assert not torch.allclose(left, right)
    with_ports = NetlistHypergraphContext(
        features=context.features,
        block_incidence=incidence,
        mask=context.mask,
        port_incidence=torch.ones(2, 3, 1, dtype=torch.bool),
    )
    assert not torch.allclose(
        model(torch.zeros(2, 4, 3), tokens, with_ports),
        left,
    )
    port_state = torch.randn(2, 1, 2)
    port_tokens = torch.randn(2, 1, 16)
    port_mask = torch.ones(2, 1, dtype=torch.bool)
    with_port_geometry = model(
        torch.zeros(2, 4, 3),
        tokens,
        with_ports,
        port_state=port_state,
        port_tokens=port_tokens,
        port_mask=port_mask,
    )
    assert not torch.allclose(
        with_port_geometry,
        model(torch.zeros(2, 4, 3), tokens, with_ports),
    )
    empty = NetlistHypergraphContext(
        features=torch.randn(2, 3, 4),
        block_incidence=torch.zeros_like(incidence),
        mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.equal(
        model(torch.randn(2, 4, 3), tokens, empty),
        torch.zeros(2, 4, 3),
    )
    left.square().mean().backward()
    assert model.network[-1].weight.grad is not None


def _external_candidate(
    name: str,
    wirelength: float | None,
    *,
    eligible: bool = True,
):
    return ExternalLegalizerCandidate(
        name=name,
        boxes=torch.zeros(1, 4).numpy(),
        route="scale_pack",
        v_rel=0.1,
        boundary=0.0,
        grouping=0.0,
        displacement_mean_um=1.0,
        rung=2,
        solves=4,
        geometry={
            "checker_feasible": eligible,
            "inside_core": eligible,
            "overlap_pair_count": 0 if eligible else 1,
            "fixed_shape_max_relative_error": 0.0,
        },
        analytical_wirelength=wirelength,
    )


def test_analytical_selection_requires_complete_eligible_scores() -> None:
    baseline = _external_candidate("baseline", 100.0)
    candidate = _external_candidate("candidate", 90.0)
    alternative = _external_candidate("alternative", 95.0)
    selected = select_candidate_with_analytical_response(
        baseline,
        (candidate, alternative),
    )
    shuffled = select_candidate_with_analytical_response(
        baseline,
        (alternative, candidate),
        analytical_scores={
            "candidate": 90.0,
            "baseline": 100.0,
            "alternative": 95.0,
        },
    )
    assert selected.name == "candidate"
    assert shuffled.name == selected.name
    baseline = _external_candidate("baseline", None)
    try:
        select_candidate_with_analytical_response(baseline, (candidate,))
    except RuntimeError as error:
        assert "missing for eligible candidate baseline" in str(error)
    else:
        raise AssertionError("missing analytical scores must fail closed")


def test_candidate_consensus_is_identity_initialized_and_set_equivariant() -> None:
    torch.manual_seed(8)
    model = CandidateConsensusFuser(hidden=16, candidate_feature_dim=3, context_dim=2)
    states = torch.randn(2, 3, 5, 3)
    features = torch.randn(2, 3, 3)
    context = torch.randn(2, 2)
    fused, _ = model(states, features, context=context)
    assert torch.equal(fused, states[:, 0])
    model.gate.data.fill_(1.0)
    fused, _ = model(states, features, context=context)
    permuted_states = states[:, [0, 2, 1]]
    permuted_features = features[:, [0, 2, 1]]
    permuted, _ = model(permuted_states, permuted_features, context=context)
    assert torch.allclose(fused, permuted, atol=1.0e-6)
    permuted.square().mean().backward()
    assert model.scorer[-1].weight.grad is not None
    invalid = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="reference candidate"):
        model(states, features, invalid, context=context)


def test_external_graph_conditioning_and_source_cap() -> None:
    graph = ExternalMacroGraph(
        dbu_per_micron=1000,
        die=(0.0, 0.0, 20.0, 20.0),
        core=(1.0, 1.0, 19.0, 19.0),
        macros=(
            {
                "name": "u1",
                "width_dbu": 2000,
                "height_dbu": 1000,
                "orientation": "R0",
                "placement_status": "PLACED",
            },
            {
                "name": "u2",
                "width_dbu": 1000,
                "height_dbu": 1000,
                "orientation": "MY",
                "placement_status": "FIXED",
            },
        ),
        ports={"p0": (1.0, 10.0)},
        net_macros={"n0": {"u1", "u2"}},
        net_ports={"n0": {"p0"}},
    )
    batch = {
        "areas": torch.ones(1, 2),
        "pins": torch.zeros(1, 1, 2),
    }
    conditioned = apply_external_conditioning(graph, batch)
    assert conditioned["node_features"].shape == (1, 2, NODE_SEMANTIC_DIM)
    assert conditioned["node_features"][0, 0, 0] == 1.0
    assert conditioned["node_features"][0, 1, 4] == 1.0
    assert conditioned["global_spec"].shape == (1, 7)
    assert 0.0 < float(conditioned["global_spec"][0, 5]) < 0.1
    assert float(conditioned["global_spec"][0, 6]) > 0.0
    hypergraph = external_netlist_hypergraph(graph)
    assert hypergraph["net_block_incidence"].shape == (1, 1, 2)
    assert math.isclose(
        float(conditioned["port_features"][0, 0, 0]),
        1.0,
    )
    assert normalize_orientation("R180") == "S"
    assert normalize_orientation("MY90") == "FE"
    with pytest.raises(ValueError, match="unsupported macro orientation"):
        normalize_orientation("UNKNOWN")
    args = type(
        "Args",
        (),
        {
            "hierarchy_source_shelf_probability": 0.5,
            "hierarchy_source_shelf_probability_final": 0.9,
            "hierarchy_source_shelf_probability_steps": 10,
            "hierarchy_source_max_shelf_probability": 0.6,
            "weight_schedule": "linear",
        },
    )()
    assert source_shelf_probability_at_step(args, 10) == 0.6
    args.hierarchy_source_max_shelf_probability = 1.1
    with pytest.raises(ValueError, match="maximum shelf probability"):
        source_shelf_probability_at_step(args, 0)


def test_legalizer_displacement_respects_trust_radius() -> None:
    prediction = torch.zeros(1, 2, 3)
    displacement = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    updated = apply_legalizer_displacement(
        prediction,
        displacement,
        strength=1.0,
        max_displacement=0.1,
    )
    assert torch.allclose(updated[0, 0, :2].norm(), torch.tensor(0.1), atol=1.0e-6)
    assert torch.equal(updated[0, 1], prediction[0, 1])


def test_membership_and_standalone_legalizer_branches_are_constructible() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        membership_hierarchy_conditioning=True,
        standalone_legalizer_response=True,
        hierarchy_coordinate_parameterization=True,
    )
    assert model.conditioners.membership is not None
    assert model.conditioners.legalizer_head is not None
    assert math.isfinite(
        float(sum(parameter.detach().sum() for parameter in model.parameters()))
    )


def test_membership_hierarchy_residual_only_reaches_member_blocks() -> None:
    torch.manual_seed(14)
    coupler = MembershipHierarchyCoupler(hidden=16)
    coupler.global_gate.data.zero_()
    coupler.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(coupler.structure_reduce[-1].weight, std=0.05)
    membership = torch.zeros(1, 4, 9)
    membership[0, [0, 2], 0] = 1.0
    structural_mask = torch.zeros(1, 9, dtype=torch.bool)
    structural_mask[0, 0] = True
    residual = coupler(
        torch.randn(1, 4, 16),
        torch.randn(1, 4, 3),
        membership,
        structural_mask,
        torch.randn(1, 4),
        torch.randn(1, 9, 3),
        torch.randn(1, 9, 16),
        torch.zeros(1),
    )
    assert residual[0, 0].abs().sum() > 0
    assert residual[0, 2].abs().sum() > 0
    assert torch.equal(residual[0, [1, 3]], torch.zeros(2, 3))


def test_hyperedge_branch_rejects_a_missing_netlist_context() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        netlist_hyperedge_conditioning=True,
    )
    batch = _chip_batch()
    context = model.prepare_context(batch)
    targets = build_hierarchy_targets(batch, context)
    with pytest.raises(ValueError, match="netlist context"):
        model(targets, torch.full((2,), 0.5), context)


def test_new_architecture_branches_round_trip_through_checkpoint_loader() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        global_spec_conditioning=True,
        node_semantic_dim=3,
        port_semantic_dim=2,
        netlist_hyperedge_conditioning=True,
        membership_hierarchy_conditioning=True,
        standalone_legalizer_response=True,
        hierarchy_coordinate_parameterization=True,
    )
    config = {
        "architecture": "typed_hierarchy",
        "hidden": 32,
        "heads": 4,
        "depth": 1,
        "head_dim": 16,
        "typed_layers": 1,
        "global_spec_conditioning": True,
        "node_semantic_dim": 3,
        "port_semantic_dim": 2,
        "netlist_hyperedge_conditioning": True,
        "membership_hierarchy_conditioning": True,
        "standalone_legalizer_response": True,
        "hierarchy_coordinate_parameterization": True,
    }
    checkpoint = {
        "config": config,
        "ema": model.state_dict(),
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.typed.global_spec_conditioner is not None
    assert restored.typed.node_semantic_encoder is not None
    assert restored.typed.port_semantic_encoder is not None
    assert restored.conditioners.hyperedge is not None
    assert restored.conditioners.membership is not None
    assert restored.conditioners.legalizer_head is not None
    assert restored.hierarchy_coordinate_parameterization


def test_resume_validation_rejects_changed_new_architecture_config() -> None:
    saved = {
        "global_spec_conditioning": False,
        "node_semantic_dim": 0,
        "netlist_hyperedge_conditioning": False,
        "hierarchy_source_max_shelf_probability": 0.6,
        "hierarchy_coordinate_parameterization": False,
    }
    current = {
        **saved,
        "netlist_hyperedge_conditioning": True,
    }
    with pytest.raises(ValueError, match="netlist_hyperedge_conditioning"):
        validate_resume_config(saved, current, weights_only=False)
    validate_resume_config(saved, current, weights_only=True)


def test_standalone_membership_coupling_receives_gradient() -> None:
    torch.manual_seed(12)
    batch = _chip_batch()
    base = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
    ).eval()
    coupled = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        membership_hierarchy_conditioning=True,
    ).eval()
    coupled.load_state_dict(base.state_dict(), strict=False)
    base.typed.cross_attention.gate.data.fill_(1.0)
    coupled.typed.cross_attention.gate.data.fill_(1.0)
    base_context = base.prepare_context(batch)
    coupled_context = coupled.prepare_context(batch)
    targets = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        assert torch.equal(
            base(targets, time, base_context).block_state,
            coupled(targets, time, coupled_context).block_state,
        )
    coupling = coupled.conditioners.membership
    assert coupling is not None
    coupling.global_gate.data.fill_(1.0)
    coupling.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(coupling.global_reduce[-1].weight, std=0.02)
    torch.nn.init.normal_(coupling.structure_reduce[-1].weight, std=0.02)
    changed = coupled(targets, time, coupled_context).block_state
    changed.square().mean().backward()
    assert coupling.structure_reduce[-1].weight.grad is not None
