"""Model construction for structured conditional training."""

from __future__ import annotations

import argparse

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.training.optimizer import freeze_except_prefixes


def build_condition_model(
    args: argparse.Namespace,
    taxonomy: FloorplanTaxonomy,
) -> FloorplanFlowModel:
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=args.base_channels,
        levels=args.levels,
        input_resolution=(args.image_size, args.image_size),
        condition_dropout=0.0,
        semantic_channel_attention=True,
        family_streams=args.enable_family_streams,
        family_mita_attention=args.enable_family_mita,
        slot_mixing=True,
        num_slots=16,
        dynamic_output_head=False,
        topology_aux=False,
        floor_conditioning=True,
        semantic_endpoint_head=True,
        num_rooms=len(taxonomy.room_ids),
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
        room_partition_semantic_prior=True,
        room_class_ids=tuple(taxonomy.room_ids),
        room_program_head=args.enable_room_program,
        room_group_head=args.enable_room_group_head,
        hierarchical_rooms=args.enable_hierarchical_rooms,
        hierarchical_hidden=args.hierarchical_hidden,
        hierarchical_tokens=args.hierarchical_tokens,
        hierarchical_coarse_resolution=args.hierarchical_coarse_resolution,
        polygon_loop_head=args.enable_polygon_loop,
        polygon_queries=args.polygon_queries,
        polygon_points=args.polygon_points,
        polygon_hidden=args.polygon_hidden,
        room_mask_head=(
            args.enable_room_mask_decoder
            or args.room_mask_weight > 0
            or args.room_mask_partition_weight > 0
        ),
        room_mask_queries=args.room_mask_queries,
        room_mask_resolution=args.room_mask_resolution,
        room_mask_use_partition=True,
        structured_object_head=True,
        semantic_taxonomy=taxonomy,
        geometry_heads=args.enable_geometry_heads,
        geometry_hidden=args.geometry_hidden,
        geometry_experts=args.geometry_experts,
        geometry_top_k=args.geometry_top_k,
        coarse_to_face_cfm=args.enable_coarse_to_face_cfm,
        coarse_face_hidden=args.coarse_face_hidden,
        coarse_face_max_faces=args.coarse_face_max_faces,
        program_conditioning=args.enable_program_conditioning,
        program_hidden=args.program_hidden,
    )
    if args.initialize_from is not None:
        payload = torch.load(
            args.initialize_from,
            map_location="cpu",
            weights_only=False,
        )
        source = (
            payload["ema_model"]
            if args.initialize_ema and payload.get("ema_model") is not None
            else payload["model"]
        )
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in current and current[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(
            f"initialized {len(compatible)}/{len(current)} tensors from {args.initialize_from}",
            flush=True,
        )
    if args.trainable_prefixes:
        freeze_except_prefixes(model, tuple(args.trainable_prefixes))
    for module_name, checkpoint_path in (
        ("room_partition", args.partition_initialize_from),
        ("structured_objects", args.object_initialize_from),
    ):
        if checkpoint_path is None:
            continue
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = payload.get("head", payload.get("model", payload))
        module = getattr(model, module_name)
        current = module.state_dict()
        compatible = {
            key.removeprefix(f"{module_name}."): value
            for key, value in source.items()
            if key.removeprefix(f"{module_name}.") in current
            and current[key.removeprefix(f"{module_name}.")].shape == value.shape
        }
        module.load_state_dict(compatible, strict=False)
        print(
            f"initialized {len(compatible)}/{len(current)} "
            f"{module_name} tensors from {checkpoint_path}",
            flush=True,
        )
    return model


__all__ = ["build_condition_model"]
