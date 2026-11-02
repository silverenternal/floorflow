"""Parameter grouping for legacy and hierarchical floorplan heads."""

from __future__ import annotations

import argparse

import torch

from floorflow.model.backbone import FloorplanFlowModel


HEAD_PREFIXES = (
    "room_partition.",
    "structured_objects.",
    "room_program.",
    "room_group.",
    "hierarchical_rooms.",
    "polygon_loop.",
    "room_mask_decoder.",
    "geometry_heads.",
    "coarse_to_face_cfm.",
    "hierarchical_scale",
    "coarse_room_scale",
    "hierarchical_program_scale",
    "room_program_scale",
    "room_group_scale",
)

HIERARCHY_PREFIXES = (
    "hierarchical_rooms.",
    "polygon_loop.",
    "room_mask_decoder.",
    "hierarchical_scale",
    "coarse_room_scale",
    "hierarchical_program_scale",
    "room_program_scale",
)


def freeze_except_hierarchy_heads(model: FloorplanFlowModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(HIERARCHY_PREFIXES))


def freeze_except_prefixes(
    model: FloorplanFlowModel,
    prefixes: tuple[str, ...],
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))


def make_optimizer(model: FloorplanFlowModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        bucket = head_parameters if name.startswith(HEAD_PREFIXES) else backbone_parameters
        bucket.append(parameter)
    if not backbone_parameters:
        backbone_parameters = [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
    if not head_parameters:
        head_parameters = [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
    fused = bool(args.fused_adamw and torch.cuda.is_available())
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        fused=fused,
    )
