from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from floorflow.chip_flow.branch_registry import (
    enabled_branches,
    validate_branch_scope,
)
from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.training_scopes import (
    parse_scope_schedule,
    validate_atom_isolation_schedule,
)
from floorflow.chip_flow.training_validation import (
    validate_atom_metric_objective_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_registry_rejects_external_only_branches_for_internal_protocol() -> None:
    config = json.loads(
        (ROOT / "configs/chip_architecture_mainline_v4.json").read_text()
    )
    branches = config["branches"] | config["objective"]
    enabled = enabled_branches(branches)
    assert "hierarchy_coordinate_parameterization" in enabled
    assert "observed_hierarchy_coupling" in enabled
    assert "global_bbox_consensus" in enabled
    assert "boundary_anchor_conditioning" in enabled
    assert "group_moment_conditioning" in enabled
    assert branches["relation_specific_attention"] is True
    assert branches["dynamic_graph_input"] is True
    assert branches["symmetric_b2b_messages"] is True
    assert branches["spatial_route_field"] is True
    assert branches["structured_heads"] is True
    assert branches["structured_hierarchy_coupling"] is True
    assert branches["structured_factorized_block"] is True
    assert branches["structured_dual_ports"] is True
    assert branches["structured_legalizer_response"] is True
    assert branches["structured_legalizer_apply"] is False
    assert branches["shape_modes"] == 4
    assert branches["shape_mode_center_scale"] == 0.25
    assert branches["shape_mode_apply"] is True
    assert branches["membership_hierarchy_conditioning"] is False
    assert branches["standalone_legalizer_response"] is False
    assert branches["routing_proxy_final_weight"] == 0.01
    assert branches["routing_proxy_delay_steps"] == 300
    assert branches["grouping_proxy_final_weight"] == 0.01
    assert branches["grouping_proxy_delay_steps"] == 150
    assert config["response_sidecar"].endswith(
        "chip_m63_response_screen1k_v1.npz"
    )
    assert branches["legalizer_response_weight"] > 0.0
    assert branches["endpoint_legalizer_target_weight"] > 0.0
    assert config["training_stages"]["require_atom_isolation"] is True
    schedule = parse_scope_schedule(
        config["training_stages"]["certification_schedule"]
    )
    validate_atom_isolation_schedule(schedule)
    validate_atom_metric_objective_schedule(
        SimpleNamespace(**branches),
        schedule,
    )
    assert "netlist_hyperedge_conditioning" not in enabled
    assert "official_cost_final_weight" in enabled
    validate_branch_scope(branches, protocol="internal")

    branches["global_spec_conditioning"] = True
    with pytest.raises(ValueError, match="external-only"):
        validate_branch_scope(branches, protocol="internal")
    branches["global_spec_conditioning"] = False
    branches["netlist_hyperedge_conditioning"] = True
    with pytest.raises(ValueError, match="external-only"):
        validate_branch_scope(branches, protocol="internal")


def test_mainline_corrected_branches_build_together() -> None:
    config = json.loads(
        (ROOT / "configs/chip_architecture_mainline_v4.json").read_text()
    )
    model_config = {
        **config["architecture"],
        "architecture": "typed_hierarchy",
        **config["branches"],
    }

    model = build_chip_model(model_config)

    assert model.observed_hierarchy_coupler is not None
    assert model.global_bbox_consensus is not None
    assert model.boundary_anchor is not None
    assert model.relation_head is not None
    assert model.group_moment_coupler is not None
    assert model.dynamic_relay is not None
    assert model.spatial_occupancy is not None
    assert model.typed.graph_layers[0].symmetric_edges
    assert model.structured_heads is not None
    assert model.structured_heads.hierarchy_coupling is not None
    assert model.structured_heads.factorized_block is not None
    assert model.structured_heads.dual_semantic_ports is not None
    assert model.structured_heads.legalizer_response is not None
    assert model.shape_mode_head is not None
    assert model.conditioners.membership is None
    assert model.conditioners.legalizer_head is None
    assert model.projector
    assert model.projector_strategy == "conflict_aware"
    assert model.projector_sparse_pairs


def test_mainline_objective_fields_have_training_cli_flags() -> None:
    config = json.loads(
        (ROOT / "configs/chip_architecture_mainline_v4.json").read_text()
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "src/floorflow/chip_flow/train_cli.py",
            ROOT / "src/floorflow/chip_flow/train_cli_options.py",
        )
    )
    missing = [
        name
        for name in config["objective"]
        if f"--{name.replace('_', '-')}" not in source
    ]

    assert not missing
