"""Command-line contract for room-to-structure bridge training."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=ROOT / "outputs/train/fp_joint_control_repair_prefill_seed19_v2/best.pt",
    )
    parser.add_argument(
        "--raster-room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument("--room-query-checkpoint", type=Path, default=None)
    parser.add_argument("--room-mask-threshold", type=float, default=0.6)
    parser.add_argument("--room-mask-strength", type=float, default=8.0)
    parser.add_argument(
        "--room-query-blend",
        type=float,
        default=None,
        help="Use the deployed dense/query probability blend.",
    )
    parser.add_argument(
        "--room-blend-jitter",
        type=float,
        default=0.0,
        help="Uniform training-time jitter around the declared room blend.",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/train/raster_predicted_control_bridge_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/raster_predicted_control_bridge_v1.json",
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument(
        "--structure-contract",
        choices=("legacy", "typed_v2"),
        default="legacy",
    )
    parser.add_argument("--structure-relation-weight", type=float, default=0.0)
    parser.add_argument("--structure-opening-host-weight", type=float, default=0.25)
    parser.add_argument("--structure-room-edge-weight", type=float, default=0.25)
    parser.add_argument("--structure-junction-weight", type=float, default=0.5)
    parser.add_argument("--structure-boundary-weight", type=float, default=0.25)
    parser.add_argument("--predict-centerline", action="store_true")
    parser.add_argument(
        "--host-conditioning-v2",
        action="store_true",
        help="Gate openings by wall, junction, and room-boundary evidence.",
    )
    parser.add_argument("--centerline-weight", type=float, default=0.0)
    parser.add_argument(
        "--structure-global-route-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--structure-route-grid", type=int, default=16)
    parser.add_argument("--structure-route-router-count", type=int, default=16)
    parser.add_argument("--structure-route-top-k", type=int, default=8)
    parser.add_argument("--structure-route-heads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.room_blend_jitter) <= 0.5:
        raise ValueError("room blend jitter must be in [0, 0.5]")
    if args.room_query_blend is not None and not 0.0 <= args.room_query_blend <= 1.0:
        raise ValueError("room query blend must be in [0, 1]")
    if args.room_blend_jitter > 0.0 and args.room_query_blend is None:
        raise ValueError("room blend jitter requires --room-query-blend")
    if args.room_query_blend is not None and args.room_query_checkpoint is None:
        raise ValueError("--room-query-blend requires --room-query-checkpoint")
    if (
        args.structure_relation_weight > 0.0
        and args.structure_contract != "typed_v2"
    ):
        raise ValueError(
            "structure relation loss requires --structure-contract typed_v2"
        )
    if args.centerline_weight < 0.0:
        raise ValueError("centerline weight must be non-negative")
    if args.centerline_weight > 0.0 and not args.predict_centerline:
        raise ValueError(
            "--centerline-weight requires --predict-centerline"
        )
    if (
        args.structure_global_route_attention
        and args.structure_contract != "typed_v2"
    ):
        raise ValueError(
            "structure global routing requires --structure-contract typed_v2"
        )
    if args.host_conditioning_v2 and args.structure_contract != "typed_v2":
        raise ValueError(
            "host conditioning v2 requires --structure-contract typed_v2"
        )


__all__ = ["parse_args", "validate_args"]
