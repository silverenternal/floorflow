"""Runtime health diagnostics for opt-in chip-flow branches."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def _branch_name(parameter_name: str) -> str | None:
    known = (
        ("typed.local_geometry_encoder.", "local_geometry"),
        ("local_geometry_encoder.", "local_geometry"),
        ("typed.global_spec_conditioner.", "global_spec"),
        ("global_spec_conditioner.", "global_spec"),
        ("typed.node_semantic_encoder.", "node_semantic"),
        ("node_semantic_encoder.", "node_semantic"),
        ("typed.port_semantic_encoder.", "port_semantic"),
        ("port_semantic_encoder.", "port_semantic"),
        ("typed.degree_expert.", "degree_expert"),
        ("degree_expert.", "degree_expert"),
        ("typed.complexity.", "complexity"),
        ("complexity.", "complexity"),
        ("conditioners.hyperedge.", "netlist_hyperedge"),
        ("conditioners.membership.", "membership_hierarchy"),
        ("conditioners.legalizer_head.", "legalizer_response"),
        ("dynamic_relay.", "dynamic_relay"),
        ("spatial_occupancy.", "spatial_occupancy"),
        ("port_consensus.", "port_consensus"),
        ("structured_heads.", "structured_heads"),
        ("coordinate_gate.", "coordinate_gate"),
        ("observed_hierarchy_coupler.", "observed_hierarchy"),
        ("shape_mode_head.", "shape_mode"),
    )
    for prefix, branch in known:
        if parameter_name.startswith(prefix):
            return branch
    if parameter_name.endswith("cross_attention.gate"):
        return "typed_cross_attention"
    return None


@torch.no_grad()
def branch_health(
    model: nn.Module,
    *,
    threshold: float = 1.0e-7,
) -> dict[str, dict[str, float | bool | int]]:
    """Report gate magnitude, gradient reachability and active parameter counts."""

    if threshold < 0.0:
        raise ValueError("branch-health threshold must be non-negative")
    rows: dict[str, dict[str, float | bool | int]] = {}
    for name, parameter in model.named_parameters():
        branch = _branch_name(name)
        if branch is None:
            continue
        row = rows.setdefault(
            branch,
            {
                "parameter_count": 0,
                "gate_present": False,
                "gate_abs_max": 0.0,
                "gradient_norm": 0.0,
                "finite": True,
            },
        )
        row["parameter_count"] = int(row["parameter_count"]) + parameter.numel()
        if name.endswith("gate") or name == "coordinate_gate.bias":
            row["gate_present"] = True
            row["gate_abs_max"] = max(
                float(row["gate_abs_max"]),
                float(parameter.detach().abs().max().cpu()),
            )
        gradient_norm = (
            float(parameter.grad.detach().norm().cpu())
            if parameter.grad is not None
            else 0.0
        )
        row["gradient_norm"] = max(
            float(row["gradient_norm"]),
            gradient_norm,
        )
        row["finite"] = bool(row["finite"]) and bool(
            torch.isfinite(parameter.detach()).all()
        )
    for row in rows.values():
        gate_open = (
            not bool(row["gate_present"])
            or float(row["gate_abs_max"]) > threshold
        )
        row["active"] = (
            gate_open
            and float(row.get("gradient_norm", 0.0)) > threshold
            and bool(row["finite"])
        )
    return rows


def dead_branches(
    health: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Return branch names that are configured but have no gate/gradient activity."""

    return tuple(
        name for name, row in health.items() if not bool(row.get("active", False))
    )


__all__ = ["branch_health", "dead_branches"]
