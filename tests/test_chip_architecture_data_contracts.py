from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.external_graph_conditioning import (
    external_netlist_hypergraph,
    external_used_ports,
)
from floorflow.chip_flow.external_portfolio import (
    ExternalLegalizerCandidate,
    select_candidate_with_analytical_response,
)
from floorflow.chip_flow.global_spec import GlobalSpecConditioner, global_spec_vector
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.legalizer_response import apply_legalizer_displacement
from floorflow.chip_flow.netlist_hypergraph import (
    NetlistHyperedgeConditioner,
    NetlistHypergraphContext,
    netlist_context_from_batch,
)
from floorflow.chip_flow.ppa_external import ExternalMacroGraph, model_batch
from floorflow.chip_flow.semantic_conditioning import MaskedSemanticEncoder
from floorflow.chip_flow.source_mixing import source_shelf_probability_at_step


def _graph() -> ExternalMacroGraph:
    return ExternalMacroGraph(
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
            {
                "name": "u3",
                "width_dbu": 1000,
                "height_dbu": 1000,
                "orientation": "MX",
                "placement_status": "PLACED",
            },
            {
                "name": "u4",
                "width_dbu": 2000,
                "height_dbu": 2000,
                "orientation": "R180",
                "placement_status": "PLACED",
            },
        ),
        ports={"p_used": (1.0, 10.0), "p_unused": (10.0, 10.0)},
        net_macros={"n0": {"u1", "u2", "u3", "u4"}},
        net_ports={"n0": {"p_used"}},
    )


def test_external_port_features_match_the_model_pin_order() -> None:
    graph = _graph()
    used = external_used_ports(graph)
    batch = model_batch(graph, torch.device("cpu"))
    hypergraph = external_netlist_hypergraph(graph, used)
    assert used == ["p_used"]
    assert batch["n_pins"].item() == 1
    assert batch["port_features"].shape == (1, 1, 5)
    assert batch["net_port_incidence"].shape[-1] == 1
    context = netlist_context_from_batch(batch)
    assert context is not None
    assert context.port_incidence is not None
    assert hypergraph["net_port_incidence"].shape[-1] == 1


def test_external_semantics_and_hypergraph_run_through_hierarchy_forward() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        global_spec_conditioning=True,
        node_semantic_dim=12,
        port_semantic_dim=5,
        netlist_hyperedge_conditioning=True,
        hierarchy_coordinate_parameterization=True,
    ).eval()
    batch = model_batch(_graph(), torch.device("cpu"))
    context = model.prepare_context(batch)
    targets = build_hierarchy_targets(batch, context)
    prediction = model(targets, torch.full((1,), 0.5), context)
    assert prediction.block_state.shape == (1, 4, 3)
    assert prediction.port_state.shape == (1, 1, 2)
    assert torch.isfinite(prediction.block_state).all()
    assert torch.isfinite(prediction.port_state).all()


def test_netlist_context_rejects_unexpected_port_incidence_width() -> None:
    batch = {
        "areas": torch.ones(1, 2),
        "pins": torch.zeros(1, 1, 2),
        "net_features": torch.zeros(1, 1, 4),
        "net_block_incidence": torch.ones(1, 1, 2, dtype=torch.bool),
        "net_port_incidence": torch.ones(1, 1, 2, dtype=torch.bool),
        "net_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="align with batch"):
        netlist_context_from_batch(batch)


def test_external_graph_without_nets_exports_an_empty_contract() -> None:
    graph = replace(_graph(), net_macros={}, net_ports={})
    hypergraph = external_netlist_hypergraph(graph, [])
    assert hypergraph["net_features"].shape == (1, 0, 4)
    assert hypergraph["net_block_incidence"].shape == (1, 0, 4)
    assert hypergraph["net_port_incidence"].shape == (1, 0, 0)
    batch = model_batch(graph, torch.device("cpu"))
    context = netlist_context_from_batch(batch)
    assert context is not None
    assert context.features.shape == (1, 0, 4)
    assert not context.mask.any()


def test_candidate_consensus_ignores_padded_invalid_values() -> None:
    model = CandidateConsensusFuser(
        hidden=16,
        candidate_feature_dim=3,
        context_dim=2,
    )
    model.gate.data.fill_(1.0)
    states = torch.randn(2, 2, 4, 3)
    states[:, 1] = float("nan")
    features = torch.randn(2, 2, 3)
    features[:, 1] = float("nan")
    mask = torch.tensor([[True, False], [True, False]])
    fused, weights = model(states, features, mask, torch.randn(2, 2))
    assert torch.isfinite(fused).all()
    assert torch.allclose(weights[:, 1], torch.zeros(2))
    with pytest.raises(ValueError, match="valid candidate"):
        model(states, features, torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 2))


def test_masked_conditioners_sanitize_nonfinite_hidden_rows() -> None:
    global_conditioner = GlobalSpecConditioner(hidden=8)
    global_conditioner.gate.data.fill_(1.0)
    global_output = global_conditioner(
        torch.tensor([[float("nan")] * 7]),
        torch.tensor([False]),
    )
    assert torch.equal(global_output, torch.zeros(1, 8))

    semantic = MaskedSemanticEncoder(input_dim=3, hidden=8)
    semantic.gate.data.fill_(1.0)
    semantic_output = semantic(
        torch.full((1, 2, 3), float("nan")),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    assert torch.equal(semantic_output, torch.zeros(1, 2, 8))


def test_hyperedge_conditioner_ignores_nonfinite_invalid_net_rows() -> None:
    torch.manual_seed(23)
    model = NetlistHyperedgeConditioner(16, feature_dim=4)
    model.gate.data.fill_(1.0)
    torch.nn.init.normal_(model.network[-1].weight, std=0.02)
    incidence = torch.tensor([[[True, True], [False, False]]])
    mask = torch.tensor([[True, False]])
    features = torch.randn(1, 2, 4)
    invalid = features.clone()
    invalid[:, 1] = float("nan")
    tokens = torch.randn(1, 2, 16)
    expected = model(
        torch.zeros(1, 2, 3),
        tokens,
        NetlistHypergraphContext(features, incidence, mask),
    )
    actual = model(
        torch.zeros(1, 2, 3),
        tokens,
        NetlistHypergraphContext(invalid, incidence, mask),
    )
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()


def test_source_probability_rejects_invalid_schedule_values() -> None:
    args = type(
        "Args",
        (),
        {
            "hierarchy_source_shelf_probability": -0.1,
            "hierarchy_source_shelf_probability_final": None,
            "hierarchy_source_shelf_probability_steps": 10,
            "hierarchy_source_max_shelf_probability": 1.0,
            "weight_schedule": "linear",
        },
    )()
    with pytest.raises(ValueError, match="finite and in"):
        source_shelf_probability_at_step(args, 0)
    args.hierarchy_source_shelf_probability = 0.2
    with pytest.raises(ValueError, match="non-negative"):
        source_shelf_probability_at_step(args, -1)
    args.weight_schedule = "unknown"
    with pytest.raises(ValueError, match="unknown weight schedule"):
        source_shelf_probability_at_step(args, 0)


def _candidate(
    name: str,
    *,
    congestion: float | None = None,
) -> ExternalLegalizerCandidate:
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
            "checker_feasible": True,
            "inside_core": True,
            "overlap_pair_count": 0,
            "fixed_shape_max_relative_error": 0.0,
        },
        analytical_wirelength=100.0,
        analytical_congestion=congestion,
    )


def test_analytical_reranking_rejects_ambiguous_or_invalid_scores() -> None:
    duplicate = (_candidate("candidate"), _candidate("candidate"))
    with pytest.raises(RuntimeError, match="names must be unique"):
        select_candidate_with_analytical_response(_candidate("baseline"), duplicate)
    invalid = _candidate("candidate", congestion=float("nan"))
    with pytest.raises(RuntimeError, match="congestion is invalid"):
        select_candidate_with_analytical_response(_candidate("baseline"), (invalid,))
    nonfinite = replace(
        _candidate("nonfinite"),
        v_rel=float("nan"),
    )
    assert not nonfinite.eligible
    incomplete = replace(_candidate("incomplete"), geometry={})
    assert not incomplete.eligible
    fallback = select_candidate_with_analytical_response(
        _candidate("baseline"),
        (nonfinite, incomplete),
    )
    assert fallback.name == "baseline"


def test_global_spec_and_legalizer_reject_nonfinite_configuration() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        global_spec_vector(
            core_width=torch.tensor([float("nan")]),
            core_height=torch.tensor([10.0]),
            die_width=torch.tensor([12.0]),
            die_height=torch.tensor([12.0]),
            macro_area=torch.tensor([20.0]),
        )
    with pytest.raises(ValueError, match="trust region"):
        apply_legalizer_displacement(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 2),
            max_displacement=float("nan"),
        )
