"""Reusable synthetic fixtures for chip atom health probes."""

from __future__ import annotations

from argparse import Namespace

import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel

ATOM_PREFIXES = {
    "A01": ("typed.block_input.",),
    "A02": ("typed.local_geometry_encoder.",),
    "A03": ("typed.graph_layers.",),
    "A04": ("typed.degree_expert.",),
    "A05": ("typed.global_spec_conditioner.",),
    "A06": ("typed.complexity.",),
    "A07": ("typed.node_semantic_encoder.", "typed.port_semantic_encoder."),
    "A08": ("typed.structural_type",),
    "A09": ("typed.constraint_type", "typed.constraint_to_port."),
    "A10": ("typed.port_input.", "typed.port_type"),
    "A11": ("typed.cross_attention.", "typed.norm."),
    "A12": ("global_head.",),
    "A13": ("structural_head.",),
    "A14": ("port_head.",),
    "A15": ("relation_head.",),
    "A17": ("conditioners.hyperedge.",),
    "A18": ("coordinate_gate.",),
    "A19": ("order_staging.",),
    "A20": ("coarse_to_fine.",),
    "A21": ("port_consensus.",),
    "A22": ("spatial_occupancy.",),
    "A23": ("complexity_slot.",),
    "A24": ("structured_heads.factorized_block.",),
    "A25": ("structured_heads.dual_semantic_ports.",),
    "A26": ("boundary_anchor.",),
    "A27": ("global_bbox_consensus.",),
    "A28": ("projector_residual_head.",),
    "A29": (
        "conditioners.legalizer_head.",
        "structured_heads.legalizer_response.",
    ),
    "A30": ("backbone.hierarchy_bridge.",),
    "A42": ("structured_heads.atom_aggregator.",),
}


def probe_batch(batch: int = 2, count: int = 8, pins: int = 12) -> dict:
    constraints = torch.zeros(batch, count, 5)
    constraints[:, :2, 0] = 1.0
    constraints[:, :3, 2] = 1.0
    constraints[:, :, 3] = torch.arange(count) % 4 + 1
    constraints[:, :, 4] = (torch.arange(count) % 15 + 1).float()
    pairs = count * 3
    source = torch.arange(pairs) % count
    target = (source + 1 + torch.arange(pairs) // count) % count
    net_incidence = torch.zeros(batch, count, count, dtype=torch.bool)
    for index in range(count):
        net_incidence[:, index, index] = True
        net_incidence[:, index, (index + 1) % count] = True
    return {
        "areas": torch.rand(batch, count) + 0.5,
        "boxes": torch.rand(batch, count, 4) * 0.2 + 0.05,
        "constraints": constraints,
        "features": torch.randn(batch, count, 19),
        "source_latent": torch.randn(batch, count, 3) * 0.1,
        "target_latent": torch.randn(batch, count, 3) * 0.1,
        "b2b": torch.stack(
            [source.float(), target.float(), torch.rand(pairs) + 0.1],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "p2b": torch.stack(
            [
                torch.arange(pins).float(),
                (torch.arange(pins) % count).float(),
                torch.rand(pins) + 0.1,
            ],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "pins": torch.rand(batch, pins, 2),
        "n_b2b": torch.full((batch,), pairs),
        "n_p2b": torch.full((batch,), pins),
        "n_pins": torch.full((batch,), pins),
        "global_spec": torch.randn(batch, 7),
        "node_features": torch.randn(batch, count, 4),
        "port_features": torch.randn(batch, pins, 4),
        "net_features": torch.rand(batch, count, 4),
        "net_block_incidence": net_incidence,
        "net_mask": torch.ones(batch, count, dtype=torch.bool),
    }


def probe_args() -> Namespace:
    return Namespace(
        time_std=1.0,
        flow_huber_beta=0.1,
        path_steps=0,
        path_weight=0.2,
        physics_weight=1.0,
        overlap_weight_scale=10.0,
        decode_weight=1.0,
        rho_weight=0.0,
        clean_weight=0.1,
        sparsity_weight=0.01,
        route_balance_weight=0.01,
        projector_residual_iterations=1,
        nfe_consistency_weight=0.1,
        nfe_consistency_steps=1,
        nfe_consistency_target_weight=1.0,
        nfe_consistency_self_weight=0.1,
        hierarchy_global_weight=0.1,
        hierarchy_structural_weight=0.1,
        hierarchy_port_weight=0.1,
        hierarchy_source_mode="random",
        hierarchy_source_noise=0.0,
        hierarchy_source_levels="random",
    )


def full_probe_model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        relation_specific=True,
        structural_relations=True,
        complexity_conditioning=True,
        projector=True,
        projector_iterations=1,
        dynamic_graph_relay=True,
        order_conditioning=True,
        coarse_to_fine_conditioning=True,
        port_consensus_conditioning=True,
        boundary_anchor_conditioning=True,
        spatial_occupancy_conditioning=True,
        complexity_slot_conditioning=True,
        degree_expert_conditioning=True,
        hierarchy_token_bridge=True,
        hierarchy_token_bridge_mode="bounded",
        global_bbox_consensus=True,
        projector_residual_head=True,
        state_conditioned_edge_bias=True,
        structured_heads=True,
        structured_hierarchy_coupling=True,
        structured_factorized_block=True,
        structured_dual_ports=True,
        structured_legalizer_response=True,
        global_spec_conditioning=True,
        node_semantic_dim=4,
        port_semantic_dim=4,
        netlist_hyperedge_conditioning=True,
        standalone_legalizer_response=True,
        hierarchy_coordinate_parameterization=True,
    )


def membership_probe_model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        membership_hierarchy_conditioning=True,
    )


def activate_probe_parameters(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(47)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            scalar_switch = (
                name.endswith(("gate", "coordinate_gate.bias"))
                or name.endswith("scale")
                and parameter.numel() == 1
            )
            if scalar_switch:
                parameter.fill_(0.5)
            elif parameter.numel() > 1 and not parameter.detach().any():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 0.02
                )


__all__ = [
    "ATOM_PREFIXES",
    "activate_probe_parameters",
    "full_probe_model",
    "membership_probe_model",
    "probe_args",
    "probe_batch",
]
