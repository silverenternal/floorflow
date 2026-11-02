"""Command-line contract for raster instance-room head training."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_instance_control_v1",
    )
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on every valid training floor after empty-target filtering.",
    )
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument(
        "--train-sampling",
        choices=("prefix", "seeded", "curriculum"),
        default="seeded",
        help="Use the established prefix subset or a deterministic shuffled subset.",
    )
    parser.add_argument("--train-pool-size", type=int, default=None)
    parser.add_argument("--curriculum-rare-boost", type=float, default=2.0)
    parser.add_argument("--curriculum-strength", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--page-floor-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop each multi-storey source page to the selected floor.",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--triton-kernels", action="store_true")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--full-resolution-masks", action="store_true")
    parser.add_argument("--full-resolution-channels", type=int, default=32)
    parser.add_argument("--query-existence-gate", action="store_true")
    parser.add_argument(
        "--query-set-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--query-set-hidden", type=int, default=192)
    parser.add_argument("--query-set-pool", type=int, default=64)
    parser.add_argument("--query-set-layers", type=int, default=2)
    parser.add_argument(
        "--query-set-routing-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--query-set-route-router-count", type=int, default=12)
    parser.add_argument("--query-set-route-top-k", type=int, default=4)
    parser.add_argument("--joint-room-slot-state", action="store_true")
    parser.add_argument("--joint-slot-owner-fusion", action="store_true")
    parser.add_argument("--slot-state-only", action="store_true")
    parser.add_argument("--slot-class-anchor-weight", type=float, default=0.0)
    parser.add_argument("--slot-owner-weight", type=float, default=0.0)
    parser.add_argument(
        "--query-count-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mask-pooled-class-fusion", action="store_true")
    parser.add_argument("--full-resolution-class-sampler", action="store_true")
    parser.add_argument("--owner-semantic-reassignment", action="store_true")
    parser.add_argument("--room-boundary-owner-refiner", action="store_true")
    parser.add_argument("--room-boundary-weight", type=float, default=0.0)
    parser.add_argument("--room-program-head", action="store_true")
    parser.add_argument("--program-token-conditioning", action="store_true")
    parser.add_argument("--program-class-prior-weight", type=float, default=0.0)
    parser.add_argument("--partition-consistent", action="store_true")
    parser.add_argument("--adaptive-query-refiner", action="store_true")
    parser.add_argument("--control-raster-root", type=Path, default=None)
    parser.add_argument(
        "--dual-control-raster",
        action="store_true",
        help="Concatenate native RGB with the cached predicted control.",
    )
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument(
        "--preserve-room-encoder",
        action="store_true",
        help=(
            "Initialize only non-room modules from --initialize-from so the "
            "explicit --room-checkpoint remains authoritative."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=2.0e-5,
        help="Lower learning rate used when --train-encoder is enabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--partition-weight", type=float, default=0.5)
    parser.add_argument("--partition-confidence-weighting", action="store_true")
    parser.add_argument("--fused-identity-weight", type=float, default=0.5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--dense-identity-refiner", action="store_true")
    parser.add_argument("--unmatched-mask-weight", type=float, default=0.0)
    parser.add_argument("--mask-tversky-weight", type=float, default=0.0)
    parser.add_argument("--existence-loss-weight", type=float, default=0.5)
    parser.add_argument("--query-separation-weight", type=float, default=0.0)
    parser.add_argument(
        "--query-separation-mode",
        choices=("active", "supervised_same_class"),
        default="active",
    )
    parser.add_argument(
        "--query-separation-class-agnostic",
        action="store_true",
    )
    parser.add_argument("--partition-consistency-weight", type=float, default=0.0)
    parser.add_argument("--partition-conflict-only", action="store_true")
    parser.add_argument("--partition-label-smoothing", type=float, default=0.0)
    parser.add_argument("--query-set-weight", type=float, default=0.5)
    parser.add_argument("--query-count-weight", type=float, default=0.5)
    parser.add_argument("--query-class-count-weight", type=float, default=0.1)
    parser.add_argument("--query-gate-start-step", type=int, default=0)
    parser.add_argument("--room-program-weight", type=float, default=0.0)
    parser.add_argument("--room-program-cardinality-weight", type=float, default=0.5)
    parser.add_argument("--selection-count-weight", type=float, default=0.005)
    parser.add_argument(
        "--selection-mode",
        choices=("legacy", "dense_relative_identity"),
        default="dense_relative_identity",
    )
    parser.add_argument("--dynamic-room-loss-weights", action="store_true")
    parser.add_argument("--gradient-balance-interval", type=int, default=50)
    parser.add_argument("--gradient-balance-min-scale", type=float, default=0.10)
    parser.add_argument("--gradient-balance-max-scale", type=float, default=2.0)
    parser.add_argument(
        "--reset-gradient-balancer",
        action="store_true",
        help="Ignore a saved balancer state when warm-starting.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--teacher-manifest", type=Path, default=None)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--teacher-max-proposals", type=int, default=64)
    parser.add_argument("--teacher-class-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Optimize class-agnostic proposal masks only; target masks/classes are evaluation-only.",
    )
    parser.add_argument(
        "--class-only",
        action="store_true",
        help="Freeze proposal geometry and recover only query class parameters.",
    )
    parser.add_argument(
        "--train-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.prefetch_factor < 1:
        raise ValueError("prefetch factor must be positive")
    if args.train_offset < 0 or args.val_offset < 0:
        raise ValueError("dataset offsets must be non-negative")
    if not args.full_train and args.limit_train <= 0:
        raise ValueError("limit train must be positive unless full-train is set")
    if args.teacher_class_weight < 0.0:
        raise ValueError("teacher class weight must be non-negative")
    if (
        args.train_sampling == "curriculum"
        and args.train_pool_size is not None
        and args.train_pool_size < args.limit_train
    ):
        raise ValueError("curriculum pool must cover the requested train cases")
    if args.query_set_head and args.joint_room_slot_state:
        raise ValueError(
            "--query-set-head and --joint-room-slot-state are mutually exclusive"
        )
    if args.slot_state_only and not args.joint_room_slot_state:
        raise ValueError(
            "--slot-state-only requires --joint-room-slot-state"
        )
    if args.slot_state_only and args.class_only:
        raise ValueError(
            "--slot-state-only and --class-only are mutually exclusive"
        )
    if args.joint_slot_owner_fusion and not args.joint_room_slot_state:
        raise ValueError(
            "--joint-slot-owner-fusion requires --joint-room-slot-state"
        )
    if args.slot_class_anchor_weight < 0.0:
        raise ValueError("slot class anchor weight must be non-negative")
    if args.slot_class_anchor_weight > 0.0 and not args.joint_room_slot_state:
        raise ValueError(
            "--slot-class-anchor-weight requires --joint-room-slot-state"
        )
    if args.slot_owner_weight < 0.0:
        raise ValueError("slot owner weight must be non-negative")
    if args.slot_owner_weight > 0.0 and not (
        args.joint_room_slot_state and args.joint_slot_owner_fusion
    ):
        raise ValueError(
            "--slot-owner-weight requires joint slot-owner fusion"
        )
    if args.query_set_weight > 0.0 and not (
        args.query_set_head or args.joint_room_slot_state
    ):
        raise ValueError(
            "--query-set-weight requires --query-set-head or "
            "--joint-room-slot-state"
        )
        if args.query_count_gate and not (
            args.query_set_head or args.joint_room_slot_state
        ):
            raise ValueError(
                "--query-count-gate requires a query-set or joint room-slot head"
            )
    if args.query_gate_start_step < 0:
        raise ValueError("query gate start step must be non-negative")
    if (
        args.query_set_head
        and args.query_set_routing_attention
        and args.query_set_layers <= 0
    ):
        raise ValueError(
            "query-set routing attention requires a query-set head and "
            "at least one query-set layer"
        )
    if args.room_program_weight > 0.0 and not args.room_program_head:
        raise ValueError("--room-program-weight requires --room-program-head")
    if args.program_token_conditioning and not args.room_program_head:
        raise ValueError(
            "--program-token-conditioning requires --room-program-head"
        )
    if args.program_class_prior_weight > 0.0 and not args.room_program_head:
        raise ValueError(
            "--program-class-prior-weight requires --room-program-head"
        )
    if args.partition_consistency_weight > 0.0 and not args.partition_consistent:
        raise ValueError(
            "--partition-consistency-weight requires --partition-consistent"
        )


__all__ = ["parse_args", "validate_args"]
