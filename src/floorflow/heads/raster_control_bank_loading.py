"""Checkpoint loaders for raster room-control models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy


def load_raster_room_model(
    checkpoint: str | Path,
    taxonomy,
    device: torch.device,
) -> nn.Module:
    from floorflow.heads.raster_room_model_build import build_raster_room_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = build_raster_room_model(
        str(config.get("architecture", "resnet18_fpn")),
        len(taxonomy.room_ids),
        decoder_channels=int(config.get("decoder_channels", 96)),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def load_raster_instance_control_model(
    checkpoint: str | Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    *,
    owner_semantic_reassignment: bool | None = None,
    query_count_gate: bool | None = None,
):
    from floorflow.heads.raster_control_bank import RasterInstanceControl

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    owner_reassignment = (
        bool(config.get("owner_semantic_reassignment", False))
        if owner_semantic_reassignment is None
        else bool(owner_semantic_reassignment)
    )
    effective_query_count_gate = bool(
        payload.get(
            "query_count_gate_effective",
            config.get("query_count_gate", False),
        )
    )
    if query_count_gate is not None:
        effective_query_count_gate = bool(query_count_gate)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=int(config.get("decoder_channels", 128)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_queries=int(config.get("queries", 24)),
        layers=int(config.get("layers", 3)),
        heads=int(config.get("heads", 4)),
        full_resolution_masks=bool(config.get("full_resolution_masks", False)),
        full_resolution_channels=int(config.get("full_resolution_channels", 32)),
        query_existence_gate=bool(config.get("query_existence_gate", False)),
        dual_control=bool(
            config.get("dual_control_raster", config.get("dual_control", False))
        ),
        adaptive_query_refinement=bool(
            config.get(
                "adaptive_query_refiner",
                config.get("adaptive_query_refinement", False),
            )
        ),
        partition_consistent=bool(
            config.get("partition_consistent", False)
        ),
        mask_pooled_class_fusion=bool(
            config.get("mask_pooled_class_fusion", False)
        ),
        full_resolution_class_sampler=bool(
            config.get("full_resolution_class_sampler", False)
        ),
        owner_semantic_reassignment=bool(
            owner_reassignment
        ),
        room_boundary_owner_refiner=bool(
            config.get("room_boundary_owner_refiner", False)
        ),
        query_set_head=bool(config.get("query_set_head", False)),
        query_set_hidden=int(config.get("query_set_hidden", 64)),
        query_set_pool=int(config.get("query_set_pool", 48)),
        query_set_layers=int(config.get("query_set_layers", 0)),
        query_set_routing_attention=bool(
            config.get("query_set_routing_attention", False)
        ),
        query_set_route_router_count=int(
            config.get("query_set_route_router_count", 12)
        ),
        query_set_route_top_k=int(
            config.get("query_set_route_top_k", 4)
        ),
        joint_room_slot_state=bool(
            config.get("joint_room_slot_state", False)
        ),
        joint_slot_owner_fusion=bool(
            config.get("joint_slot_owner_fusion", False)
        ),
        query_count_gate=effective_query_count_gate,
        dense_identity_refiner=bool(
            config.get("dense_identity_refiner", False)
        ),
        room_program_head=bool(config.get("room_program_head", False)),
        program_token_conditioning=bool(
            config.get("program_token_conditioning", False)
        ),
        program_class_prior_weight=float(
            config.get("program_class_prior_weight", 0.0)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def load_raster_control_bank(
    checkpoint: str | Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
):
    from floorflow.heads.raster_control_bank import RasterControlBank

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = RasterControlBank(
        taxonomy,
        decoder_channels=int(config.get("decoder_channels", 128)),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


__all__ = [
    "load_raster_control_bank",
    "load_raster_instance_control_model",
    "load_raster_room_model",
]
