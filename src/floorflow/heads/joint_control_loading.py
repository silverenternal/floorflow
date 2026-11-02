"""Checkpoint loader for legacy and typed joint structure refiners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_repair import JointControlRepair


def load_joint_control_repair(
    checkpoint: str | Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> tuple[JointControlRepair, dict[str, Any]]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config", {})
    model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=int(config.get("room_base_channels", 24)),
        structure_base_channels=int(
            config.get("structure_base_channels", 32)
        ),
        structure_contract=str(
            config.get("structure_contract", "legacy")
        ),
        predict_centerline=bool(config.get("predict_centerline", False)),
        host_conditioning_v2=bool(
            config.get("host_conditioning_v2", False)
        ),
        structure_global_route_attention=bool(
            config.get("structure_global_route_attention", False)
        ),
        structure_route_grid=int(config.get("structure_route_grid", 16)),
        structure_route_router_count=int(
            config.get("structure_route_router_count", 16)
        ),
        structure_route_top_k=int(
            config.get("structure_route_top_k", 8)
        ),
        structure_route_heads=int(
            config.get("structure_route_heads", 4)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), config


__all__ = ["load_joint_control_repair"]
