"""CLI and configuration contract for the typed channel trainer."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from floorflow.training.typed_channel_scopes import SCOPES

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v3",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/raster_instance_control_384_promotion256_10ep_v1/best.pt",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument(
        "--centerline-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/p6_geometry_full_v1/best.pt",
    )
    parser.add_argument(
        "--opening-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/opening_interval_decoder_1024_8ep_v1/best.pt",
    )
    parser.add_argument("--room-blend", type=float, default=0.75)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--field-grid", type=int, default=128)
    parser.add_argument("--field-channels", type=int, default=64)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--attention-grid", type=int, default=24)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--use-action-graph", action="store_true", default=True)
    parser.add_argument(
        "--attention-mode",
        choices=("typed", "dense"),
        default="typed",
    )
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--room-iou-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--family-iou-weight", type=float, default=0.5)
    parser.add_argument("--stage-family-weight", type=float, default=1.0)
    parser.add_argument("--wall-precision-weight", type=float, default=0.0)
    parser.add_argument("--opening-precision-weight", type=float, default=0.0)
    parser.add_argument("--opening-host-weight", type=float, default=0.0)
    parser.add_argument(
        "--routing-regularization-weight",
        type=float,
        default=0.01,
    )
    parser.add_argument("--selection-furniture-weight", type=float, default=0.05)
    parser.add_argument("--furniture-query-weight", type=float, default=0.2)
    parser.add_argument(
        "--furniture-query-mask-pos-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--furniture-query-unmatched-mask-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--furniture-query-tversky-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--furniture-containment-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--enable-furniture-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--furniture-query-hidden", type=int, default=128)
    parser.add_argument("--furniture-max-queries", type=int, default=32)
    parser.add_argument("--furniture-min-queries", type=int, default=4)
    parser.add_argument("--furniture-query-threshold", type=float, default=0.1)
    parser.add_argument("--furniture-detach-backbone", action="store_true")
    parser.add_argument(
        "--furniture-room-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--furniture-semantic-router",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--furniture-semantic-hidden", type=int, default=64)
    parser.add_argument(
        "--enable-instance-room-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--room-instance-hidden", type=int, default=128)
    parser.add_argument("--room-instance-queries", type=int, default=32)
    parser.add_argument("--room-instance-layers", type=int, default=2)
    parser.add_argument(
        "--enable-program-instance-router",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-room-adjacency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-room-query-set",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--room-query-set-hidden", type=int, default=96)
    parser.add_argument("--room-query-set-pool", type=int, default=48)
    parser.add_argument("--room-query-set-layers", type=int, default=2)
    parser.add_argument(
        "--room-query-set-routing-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--query-count-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--owner-preserving-room-readout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--room-instance-adaptive-query",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--room-instance-adaptive-query-residual-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--room-instance-mask-aware-adjacency",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--room-instance-relational-adjacency",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--room-instance-class-count-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--relational-adjacency-hidden", type=int, default=128)
    parser.add_argument(
        "--room-instance-count-aware-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--count-aware-second-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--full-resolution-mask-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--typed-opening-host-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--refresh-validation-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--control-evidence-context",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--action-context-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--typed-source-spatial-router",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamic-channel-links",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamic-channel-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source-channel-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamic-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--channel-group-adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cross-attention-lora-rank", type=int, default=8)
    parser.add_argument(
        "--spatial-conv-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--spatial-adapter-bottleneck", type=int, default=16)
    parser.add_argument("--source-hyper-hidden", type=int, default=64)
    parser.add_argument("--train-scope", choices=SCOPES, default="full")
    parser.add_argument(
        "--dynamic-stage-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stage-head-hidden", type=int, default=128)
    parser.add_argument(
        "--dynamic-stage-links",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--action-route-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--action-route-router-count", type=int, default=8)
    parser.add_argument("--action-route-top-k", type=int, default=4)
    parser.add_argument(
        "--refresh-action-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stage-route-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stage-route-grid", type=int, default=8)
    parser.add_argument("--stage-route-router-count", type=int, default=12)
    parser.add_argument("--stage-route-top-k", type=int, default=6)
    parser.add_argument("--stage-route-heads", type=int, default=4)
    parser.add_argument("--instance-room-weight", type=float, default=0.5)
    parser.add_argument(
        "--room-instance-class-balance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--instance-room-partition-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--room-instance-adjacency-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--room-type-adjacency-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--room-type-node-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--dense-type-adjacency-weight", type=float, default=0.0)
    parser.add_argument("--dense-type-node-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-type-edge-overlap-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dense-type-node-overlap-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--room-query-set-loss-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--room-query-class-cardinality-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--room-query-support-assignment-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--room-query-selection-alignment-weight", type=float, default=0.2)
    parser.add_argument("--room-query-presence-rank-weight", type=float, default=0.2)
    parser.add_argument("--room-query-presence-count-weight", type=float, default=0.5)
    parser.add_argument("--room-instance-unmatched-mask-weight", type=float, default=0.2)
    parser.add_argument("--room-instance-mask-pos-weight", type=float, default=2.0)
    parser.add_argument("--room-instance-mask-tversky-weight", type=float, default=0.2)
    parser.add_argument("--room-instance-query-separation-weight", type=float, default=0.0)
    parser.add_argument("--room-instance-query-separation-class-agnostic", action="store_true")
    parser.add_argument("--room-instance-reference-anchor-weight", type=float, default=0.2)
    parser.add_argument("--room-instance-mask-centroid-weight", type=float, default=0.5)
    parser.add_argument(
        "--room-reference-learning-rate-multiplier", type=float, default=10.0
    )
    parser.add_argument(
        "--program-instance-assignment-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--use-teacher-room-proposals",
        action="store_true",
    )
    parser.add_argument(
        "--teacher-proposal-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_teacher_train256_cal64_v1",
    )
    parser.add_argument("--teacher-max-proposals", type=int, default=64)
    parser.add_argument(
        "--furniture-room-assignment-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument("--learned-room-stage", action="store_true")
    parser.add_argument("--dynamic-loss-weights", action="store_true")
    parser.add_argument(
        "--dynamic-balance-reference",
        choices=(
            "auto",
            "ce",
            "room_iou",
            "instance_room",
            "stage_wall",
            "stage_opening",
            "stage_furniture",
        ),
        default="auto",
        help=(
            "Gate-driving objective used to calibrate dynamic loss scales; "
            "auto selects the primary objective for the active train scope."
        ),
    )
    parser.add_argument(
        "--dynamic-balance-update-interval",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dynamic-balance-exponent",
        type=float,
        default=1.0,
    )
    parser.add_argument("--no-normalize-class-weights", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    args = parser.parse_args()
    apply_model_config(args)
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_config(args: argparse.Namespace) -> dict:
    resolved = getattr(args, "resolved_model_config", None)
    if resolved is not None:
        return dict(resolved)
    return {
        "control_channels": (24, 38, 52, 28),
        "program_channels": 24,
        "grid": args.field_grid,
        "field_channels": args.field_channels,
        "head_channels": args.head_channels,
        "heads": 4,
        "layers": args.layers,
        "attention_grid": args.attention_grid,
        "full_resolution_output": True,
        "typed_channel_links": args.attention_mode == "typed",
        "use_action_graph": args.use_action_graph,
        "preserve_room_stage": not args.learned_room_stage,
        "enable_furniture_queries": args.enable_furniture_queries,
        "furniture_query_hidden": args.furniture_query_hidden,
        "furniture_max_queries": args.furniture_max_queries,
        "furniture_min_queries": args.furniture_min_queries,
        "furniture_detach_backbone": args.furniture_detach_backbone,
        "furniture_room_assignment": args.furniture_room_assignment,
        "furniture_semantic_router": args.furniture_semantic_router,
        "furniture_semantic_hidden": args.furniture_semantic_hidden,
        "enable_instance_room_decoder": args.enable_instance_room_decoder,
        "room_instance_hidden": args.room_instance_hidden,
        "room_instance_queries": args.room_instance_queries,
        "room_instance_layers": args.room_instance_layers,
        "enable_program_instance_router": (
            args.enable_program_instance_router
        ),
        "enable_room_adjacency": args.enable_room_adjacency,
        "enable_room_query_set": args.enable_room_query_set,
        "room_query_set_hidden": args.room_query_set_hidden,
        "room_query_set_pool": args.room_query_set_pool,
        "room_query_set_layers": args.room_query_set_layers,
        "room_query_set_routing_attention": (
            args.room_query_set_routing_attention
        ),
        "query_count_gate": args.query_count_gate,
        "owner_preserving_room_readout": (
            args.owner_preserving_room_readout
        ),
        "room_instance_adaptive_query": (
            args.room_instance_adaptive_query
        ),
        "room_instance_adaptive_query_residual_scale": (
            args.room_instance_adaptive_query_residual_scale
        ),
        "room_instance_mask_aware_adjacency": (
            args.room_instance_mask_aware_adjacency
        ),
        "room_instance_relational_adjacency": (
            args.room_instance_relational_adjacency
        ),
        "room_instance_class_count_prior": (
            args.room_instance_class_count_prior
        ),
        "relational_adjacency_hidden": args.relational_adjacency_hidden,
        "room_instance_count_aware_graph": (
            args.room_instance_count_aware_graph
        ),
        "count_aware_second_graph": args.count_aware_second_graph,
        "full_resolution_mask_refinement": (
            args.full_resolution_mask_refinement
        ),
        "typed_opening_host_gate": args.typed_opening_host_gate,
        "refresh_validation_tokens": args.refresh_validation_tokens,
        "control_evidence_context": args.control_evidence_context,
        "action_context_conditioning": args.action_context_conditioning,
        "typed_source_spatial_router": args.typed_source_spatial_router,
        "dynamic_channel_links": args.dynamic_channel_links,
        "dynamic_channel_queries": args.dynamic_channel_queries,
        "source_channel_attention": args.source_channel_attention,
        "dynamic_cross_attention": args.dynamic_cross_attention,
        "channel_group_adapters": args.channel_group_adapters,
        "cross_attention_lora_rank": args.cross_attention_lora_rank,
        "spatial_conv_adapter": args.spatial_conv_adapter,
        "spatial_adapter_bottleneck": args.spatial_adapter_bottleneck,
        "source_hyper_hidden": args.source_hyper_hidden,
        "dynamic_stage_heads": args.dynamic_stage_heads,
        "stage_head_hidden": args.stage_head_hidden,
        "dynamic_stage_links": args.dynamic_stage_links,
        "action_route_attention": args.action_route_attention,
        "action_route_router_count": args.action_route_router_count,
        "action_route_top_k": args.action_route_top_k,
        "refresh_action_tokens": args.refresh_action_tokens,
        "stage_route_attention": args.stage_route_attention,
        "stage_route_grid": args.stage_route_grid,
        "stage_route_router_count": args.stage_route_router_count,
        "stage_route_top_k": args.stage_route_top_k,
        "stage_route_heads": args.stage_route_heads,
    }


def apply_model_config(args: argparse.Namespace) -> None:
    """Load model and loss settings from one reproducible experiment file."""

    if args.model_config is None:
        return
    payload = _load_experiment_config(args.model_config)
    model = dict(payload.get("model", payload))
    if "field_grid" in model:
        model["grid"] = model.pop("field_grid")
    model["control_channels"] = tuple(model.get("control_channels", ()))
    args.resolved_model_config = model
    for name, value in model.items():
        if name == "grid":
            args.field_grid = value
        elif hasattr(args, name):
            setattr(args, name, value)
    loss_policy = payload.get("loss_policy", {})
    weights = loss_policy.get("weights", {})
    for name, value in {
        key: value
        for key, value in loss_policy.items()
        if key != "weights"
    }.items():
        if hasattr(args, name):
            setattr(args, name, value)
    for name, value in weights.items():
        if hasattr(args, name):
            setattr(args, name, value)
    for name, value in payload.items():
        if name in {"model", "loss_policy", "budget"}:
            continue
        if hasattr(args, name):
            path_value = (
                Path(value)
                if name.endswith(("_path", "_root", "_checkpoint"))
                and value is not None
                else value
            )
            setattr(args, name, path_value)


def _load_experiment_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("extends")
    if parent is None:
        return payload
    merged = copy.deepcopy(_load_experiment_config(path.parent / parent))
    for key, value in payload.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


__all__ = ["model_config", "parse_args", "seed_everything"]
