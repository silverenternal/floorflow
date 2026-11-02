"""Deterministic setup helpers for raster instance-room training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.heads.raster_control_bank import RasterInstanceControl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_room_encoder(model: RasterInstanceControl, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.room.load_state_dict(payload["model"])


def build_raster_control_model(
    args: Any,
    taxonomy,
    device: torch.device,
) -> RasterInstanceControl:
    return RasterInstanceControl(
        taxonomy,
        decoder_channels=args.decoder_channels,
        hidden_dim=args.hidden_dim,
        num_queries=args.queries,
        layers=args.layers,
        heads=args.heads,
        full_resolution_masks=args.full_resolution_masks,
        full_resolution_channels=args.full_resolution_channels,
        query_existence_gate=args.query_existence_gate,
        dual_control=args.dual_control_raster,
        adaptive_query_refinement=args.adaptive_query_refiner,
        partition_consistent=args.partition_consistent,
        mask_pooled_class_fusion=args.mask_pooled_class_fusion,
        full_resolution_class_sampler=args.full_resolution_class_sampler,
        owner_semantic_reassignment=args.owner_semantic_reassignment,
        room_boundary_owner_refiner=args.room_boundary_owner_refiner,
        query_set_head=args.query_set_head,
        query_set_hidden=args.query_set_hidden,
        query_set_pool=args.query_set_pool,
        query_set_layers=args.query_set_layers,
        query_set_routing_attention=args.query_set_routing_attention,
        query_set_route_router_count=args.query_set_route_router_count,
        query_set_route_top_k=args.query_set_route_top_k,
        joint_room_slot_state=args.joint_room_slot_state,
        joint_slot_owner_fusion=args.joint_slot_owner_fusion,
        query_count_gate=args.query_count_gate,
        dense_identity_refiner=args.dense_identity_refiner,
        room_program_head=args.room_program_head,
        program_token_conditioning=args.program_token_conditioning,
        program_class_prior_weight=args.program_class_prior_weight,
    ).to(device)


def build_room_optimizer(
    model: RasterInstanceControl,
    args: Any,
) -> torch.optim.Optimizer:
    """Build the configured decoder/encoder parameter groups."""

    if not args.train_encoder:
        return torch.optim.AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    encoder_parameter_ids = {
        id(parameter) for parameter in model.room.parameters()
    }
    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
        and parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {
                "params": decoder_parameters,
                "lr": args.learning_rate,
            },
            {
                "params": [
                    parameter
                    for parameter in model.room.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.encoder_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )


def compatible_initial_state(
    state: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    *,
    exclude_room_encoder: bool = False,
) -> dict[str, torch.Tensor]:
    """Select tensors with matching names and shapes for warm starting."""

    return {
        key: value
        for key, value in state.items()
        if key in current
        and current[key].shape == value.shape
        and not (exclude_room_encoder and key.startswith("room."))
    }


def full_resolution_predictions(
    outputs: dict[str, torch.Tensor],
    size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    result = dict(outputs)
    result["mask_logits"] = F.interpolate(
        outputs["mask_logits"],
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    result["aux_mask_logits"] = [
        F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        for value in outputs["aux_mask_logits"]
    ]
    return result


def materialize_float_outputs(value: Any) -> Any:
    """Move mixed-precision model outputs to the fp32 loss boundary."""

    if torch.is_tensor(value):
        return (
            value.float()
            if value.is_floating_point() and value.dtype != torch.float32
            else value
        )
    if isinstance(value, dict):
        return {
            key: materialize_float_outputs(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [materialize_float_outputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_float_outputs(item) for item in value)
    return value


def build_loss_contract(
    args: Any,
    *,
    teacher_present: bool,
    dynamic_weights: bool,
) -> dict[str, Any]:
    joint_room_slot_state = bool(
        getattr(args, "joint_room_slot_state", False)
    )
    query_set_enabled = bool(args.query_set_head or joint_room_slot_state)
    if args.teacher_only:
        active_terms = ["teacher"]
    else:
        active_terms = [
            "room_mask",
            "partition",
            "fused_identity",
            *(
                ["dense"]
                if args.train_encoder or args.dense_identity_refiner
                else []
            ),
            *(
                ["query_set"]
                if query_set_enabled and args.query_set_weight > 0.0
                else []
            ),
            *(
                ["room_program"]
                if args.room_program_head and args.room_program_weight > 0.0
                else []
            ),
            *(["teacher"] if teacher_present else []),
            *(
                ["room_boundary"]
                if args.room_boundary_weight > 0.0
                else []
            ),
        ]
    return {
        "active_terms": sorted(active_terms),
        "dynamic_weights": bool(dynamic_weights),
        "query_separation": {
            "weight": args.query_separation_weight,
            "mode": args.query_separation_mode,
            "class_aware": not args.query_separation_class_agnostic,
        },
        "partition_consistency_weight": args.partition_consistency_weight,
        "partition_confidence_weighting": args.partition_confidence_weighting,
        "query_set_head": args.query_set_head,
        "query_set_routing_attention": bool(
            args.query_set_routing_attention
        ),
        "query_set_route_router_count": int(
            args.query_set_route_router_count
        ),
        "query_set_route_top_k": int(args.query_set_route_top_k),
        "joint_room_slot_state": joint_room_slot_state,
        "joint_slot_owner_fusion": bool(
            getattr(args, "joint_slot_owner_fusion", False)
        ),
        "slot_state_only": bool(getattr(args, "slot_state_only", False)),
        "slot_class_anchor_weight": float(
            getattr(args, "slot_class_anchor_weight", 0.0)
        ),
        "slot_owner_weight": float(
            getattr(args, "slot_owner_weight", 0.0)
        ),
        "query_count_gate": args.query_count_gate,
        "room_program_head": args.room_program_head,
        "program_token_conditioning": bool(
            getattr(args, "program_token_conditioning", False)
        ),
        "program_class_prior_weight": float(
            getattr(args, "program_class_prior_weight", 0.0)
        ),
        "mask_pooled_class_fusion": args.mask_pooled_class_fusion,
        "dense_identity_refiner": args.dense_identity_refiner,
        "adaptive_query_refinement": args.adaptive_query_refiner,
        "room_boundary_weight": args.room_boundary_weight,
    }


def weighted_loss_sum(
    loss_terms: dict[str, torch.Tensor],
    args: Any,
) -> torch.Tensor:
    """Apply the configured scalar weight to each active loss term."""

    weights = {
        "room_mask": 1.0,
        "partition": args.partition_weight,
        "fused_identity": args.fused_identity_weight,
        "dense": args.dense_weight,
        "query_set": args.query_set_weight,
        "room_program": args.room_program_weight,
        "teacher": args.teacher_weight,
        "room_boundary": args.room_boundary_weight,
    }
    return sum(
        float(weights.get(name, 1.0)) * value
        for name, value in loss_terms.items()
    )


def write_partial_report(
    output_dir: Path,
    history: list[dict[str, object]],
    args: Any,
    *,
    status: str = "running",
) -> None:
    payload = {
        "status": status,
        "history": history,
        "config": vars(args),
    }
    (Path(output_dir) / "partial_report.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def training_summary(report: dict) -> dict:
    """Return the concise terminal summary while preserving the full report."""

    return {
        key: report[key]
        for key in (
            "status",
            "scope",
            "train_cases",
            "validation_cases",
            "best_blend",
            "best_selection_score",
            "runtime",
        )
    }


__all__ = [
    "build_loss_contract",
    "build_raster_control_model",
    "build_room_optimizer",
    "compatible_initial_state",
    "full_resolution_predictions",
    "load_room_encoder",
    "materialize_float_outputs",
    "seed_everything",
    "training_summary",
    "weighted_loss_sum",
    "write_partial_report",
]
