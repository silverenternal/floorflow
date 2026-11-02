"""Canonical registry for opt-in chip-flow architecture branches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BranchSpec:
    key: str
    scope: str
    stage: str
    defaults_on: bool = False


BRANCH_SPECS = (
    BranchSpec("group_moment_conditioning", "internal", "S1", False),
    BranchSpec("hierarchy_coordinate_parameterization", "internal_external", "S1", True),
    BranchSpec("observed_hierarchy_coupling", "internal", "S4", False),
    BranchSpec("global_bbox_consensus", "internal", "S4", False),
    BranchSpec("boundary_anchor_conditioning", "internal", "S2", False),
    BranchSpec("relation_specific_attention", "internal", "S4", False),
    BranchSpec("incidence_normalized_relations", "internal", "S4", False),
    BranchSpec("structural_relation_head", "internal", "S4", False),
    BranchSpec("dynamic_graph_input", "internal", "S4", False),
    BranchSpec("symmetric_b2b_messages", "internal", "S4", False),
    BranchSpec("complexity_conditioning", "internal", "S3", False),
    BranchSpec("degree_expert_conditioning", "internal", "S3", False),
    BranchSpec("complexity_slot_conditioning", "internal", "S3", False),
    BranchSpec("spatial_route_field", "internal_external", "S4", False),
    BranchSpec("netlist_hyperedge_conditioning", "external", "S5", False),
    BranchSpec("port_consensus_conditioning", "internal_external", "S4", True),
    BranchSpec("membership_hierarchy_conditioning", "internal", "S4", True),
    BranchSpec("standalone_legalizer_response", "internal", "S4", True),
    BranchSpec("structured_heads", "internal", "S4", False),
    BranchSpec("structured_legalizer_apply", "internal", "S4", False),
    BranchSpec("global_spec_conditioning", "external", "S5", False),
    BranchSpec("node_semantic_dim", "external", "S5", False),
    BranchSpec("port_semantic_dim", "external", "S5", False),
    BranchSpec("official_cost_final_weight", "internal", "S4B", True),
    BranchSpec("shape_modes", "internal", "S7", False),
)


def enabled_branches(config: Mapping) -> tuple[str, ...]:
    """Return registry branches that are meaningfully enabled by a config."""

    enabled = []
    for spec in BRANCH_SPECS:
        value = config.get(spec.key, spec.defaults_on)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            active = float(value) > 0.0
        else:
            active = bool(value)
        if active:
            enabled.append(spec.key)
    return tuple(enabled)


def validate_branch_scope(
    config: Mapping,
    *,
    protocol: str,
) -> None:
    """Reject external-only branches in internal FloorSet configurations."""

    if protocol not in {"internal", "external"}:
        raise ValueError("protocol must be internal or external")
    for spec in BRANCH_SPECS:
        if spec.scope == "external" and protocol == "internal":
            value = config.get(spec.key, spec.defaults_on)
            active = float(value) > 0.0 if isinstance(value, (int, float)) else bool(value)
            if active:
                raise ValueError(f"{spec.key} is external-only")


__all__ = ["BRANCH_SPECS", "BranchSpec", "enabled_branches", "validate_branch_scope"]
