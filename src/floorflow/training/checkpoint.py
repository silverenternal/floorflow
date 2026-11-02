"""Checkpoint-to-model construction shared by evaluation scripts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.builder import ModelMacro, load_state_into_model
from floorflow.model.specs import ArchSpec


def preserve_checkpoint_num_styles(path: Path, requested: int) -> int:
    """Inherit checkpoint style capacity unless explicitly requested."""

    if int(requested) > 0:
        return int(requested)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("config", {}).get("num_styles", 0))


def _dynamic_kernels(state: dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    kernels = sorted(
        {
            int(value.shape[-1])
            for key, value in state.items()
            if "dynamic_output_head.branches" in key
            and key.endswith(".weight")
            and value.ndim == 4
        }
    )
    return tuple(kernels) or None


def arch_spec_from_checkpoint(
    checkpoint: dict,
    taxonomy: FloorplanTaxonomy,
    *,
    name: str = "checkpoint-macro",
) -> ArchSpec:
    """Infer a declarative macro spec from a legacy checkpoint."""

    state = checkpoint["model"]
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", config.get("resolution", 256)))
    if isinstance(config.get("resolution"), (list, tuple)):
        resolution = tuple(int(value) for value in config["resolution"])
    else:
        resolution = (image_size, image_size)
    configured_kernels = config.get("output_kernels")
    output_kernels = (
        tuple(int(value) for value in configured_kernels)
        if configured_kernels
        else _dynamic_kernels(state)
    )
    base_channels = int(config.get("base_channels", 48))
    num_room_classes = len(taxonomy.room_ids)
    backbone_kwargs = {
        "condition_dropout": 0.0,
        "semantic_channel_attention": any(
            "semantic_channel_attention." in key for key in state
        ),
        "semantic_relation_bias": any(
            "semantic_channel_attention.channel_prior" in key for key in state
        ),
        "slot_mixing": any("slot_mixer." in key for key in state),
        "num_slots": int(config.get("num_slots", 16)),
        "topology_aux": any("topology_aux." in key for key in state),
        "floor_conditioning": any("floor_embedding.weight" in key for key in state),
        "num_floors": int(config.get("num_floors", 4)),
        "semantic_endpoint_head": any("semantic_endpoint." in key for key in state),
        "num_rooms": len(taxonomy.room_ids),
        "channel_relations": taxonomy.channel_prior(),
        "advanced_streams": any(
            "layout_stream_mixer." in key or "spatial_velocity." in key for key in state
        ),
        "family_streams": any("family_stream_mixer." in key for key in state),
        "family_mita_attention": any(
            "family_stream_mixer.attention." in key and ".q_proj." in key
            for key in state
        ),
        "family_attention_impl": (
            str(config.get("family_attention_impl"))
            if config.get("family_attention_impl") in {"legacy", "official"}
            else "legacy"
        ),
        "fourier_conditioning": int(state["condition_encoder.stem.weight"].shape[1]) >= 10,
        "room_partition_head": any("room_partition." in key for key in state),
        "num_room_classes": num_room_classes,
        "room_partition_semantic_prior": "room_partition.prior_logits.weight" in state,
        "room_class_ids": tuple(taxonomy.room_ids),
        "room_program_head": any("room_program." in key for key in state),
        "room_group_head": any("room_group." in key for key in state),
        "hierarchical_rooms": any("hierarchical_rooms." in key for key in state),
        "hierarchical_hidden": int(config.get("hierarchical_hidden", 96)),
        "hierarchical_tokens": int(config.get("hierarchical_tokens", 8)),
        "hierarchical_coarse_resolution": int(
            config.get("hierarchical_coarse_resolution", 64)
        ),
        "polygon_loop_head": any("polygon_loop." in key for key in state),
        "polygon_queries": int(config.get("polygon_queries", 24)),
        "polygon_points": int(config.get("polygon_points", 24)),
        "polygon_hidden": int(config.get("polygon_hidden", 128)),
        "layoutbridge_refinement": any("dual_stream_injection." in key for key in state),
        "layoutbridge_hidden": int(config.get("layoutbridge_hidden", 64)),
        "room_mask_head": any("room_mask_decoder." in key for key in state),
        "room_mask_queries": int(config.get("room_mask_queries", 32)),
        "room_mask_resolution": int(config.get("room_mask_resolution", 64)),
        "room_mask_use_partition": (
            "room_mask_decoder.mask_features.weight" in state
            and int(state["room_mask_decoder.mask_features.weight"].shape[1])
            > base_channels
        ),
        "structured_object_head": any("structured_objects." in key for key in state),
        "semantic_taxonomy": taxonomy,
        "semantic_anchor_fusion": any("semantic_anchor_fusion." in key for key in state),
        "geometry_heads": any("geometry_heads." in key for key in state),
        "geometry_hidden": int(config.get("geometry_hidden", 64)),
        "geometry_experts": int(config.get("geometry_experts", 1)),
        "geometry_top_k": int(config.get("geometry_top_k", 1)),
        "coarse_to_face_cfm": any("coarse_to_face_cfm." in key for key in state),
        "coarse_face_hidden": int(config.get("coarse_face_hidden", 64)),
        "coarse_face_max_faces": int(config.get("coarse_face_max_faces", 32)),
        "program_conditioning": any("program_conditioner." in key for key in state),
        "program_hidden": int(config.get("program_hidden", 64)),
    }
    return ArchSpec(
        name=name,
        family="floorflow",
        target_channels=int(state["output.weight"].shape[0]),
        base_channels=base_channels,
        levels=int(config.get("levels", 3)),
        resolution=resolution,
        output_kernels=output_kernels,
        dynamic_output=any("dynamic_output_head." in key for key in state),
        backbone_kwargs=backbone_kwargs,
    )


def load_checkpoint_model(
    path: str | Path,
    device: str | torch.device,
    *,
    taxonomy: FloorplanTaxonomy | None = None,
    weight_source: str = "model",
    family_attention_impl: str | None = None,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    taxonomy = taxonomy or FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
    state = checkpoint["ema_model"] if weight_source == "ema" and checkpoint.get("ema_model") else checkpoint["model"]
    config = checkpoint.get("config", {})
    spec = arch_spec_from_checkpoint(checkpoint, taxonomy)
    if family_attention_impl is not None:
        if family_attention_impl not in {"legacy", "official"}:
            raise ValueError("family_attention_impl must be legacy or official")
        spec = replace(
            spec,
            backbone_kwargs={
                **spec.backbone_kwargs,
                "family_attention_impl": family_attention_impl,
            },
        )
    bundle = ModelMacro(spec).build_bundle()
    model = bundle.model.to(device)
    load_report = load_state_into_model(model, state)
    model.macro_spec = spec
    model.macro_resolved = bundle.resolved
    model.macro_load_report = load_report
    model.eval()
    return model, config, taxonomy, state
