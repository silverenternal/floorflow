"""CLI contract for predicted-control fidelity v3."""

from __future__ import annotations

import argparse
from pathlib import Path

from floorflow.inference.opening_heatmap_decoder import add_opening_heatmap_arguments

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--room-query-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/raster_instance_control_384_promotion256_10ep_v1/best.pt",
    )
    parser.add_argument("--room-identity-gate", type=Path, default=None)
    parser.add_argument("--room-identity-gate-threshold", type=float, default=None)
    parser.add_argument("--room-identity-safety-fallback", action="store_true")
    parser.add_argument("--room-safety-min-wall-continuity", type=float, default=0.98)
    parser.add_argument("--wall-centerline-checkpoint", type=Path, default=None)
    parser.add_argument("--wall-centerline-threshold", type=float, default=0.5)
    parser.add_argument("--wall-centerline-dilation", type=int, default=1)
    parser.add_argument("--wall-centerline-opening-protect-radius", type=int, default=3)
    parser.add_argument(
        "--wall-centerline-mode",
        choices=("add", "consensus", "separate"),
        default="consensus",
    )
    parser.add_argument("--structured-object-checkpoint", type=Path, default=None)
    parser.add_argument("--structured-opening-threshold", type=float, default=0.5)
    parser.add_argument("--interval-opening-readout", action="store_true")
    parser.add_argument("--interval-opening-threshold", type=float, default=0.5)
    add_opening_heatmap_arguments(parser)
    parser.add_argument(
        "--opening-ranker-batch-mode",
        choices=("batch", "per_case"),
        default="per_case",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument(
        "--room-instance-blend",
        type=float,
        default=None,
        help="Use a RasterInstanceControl checkpoint and blend query room masks.",
    )
    parser.add_argument(
        "--room-instance-route",
        choices=("blend", "class_region", "soft"),
        default="blend",
    )
    parser.add_argument("--room-mask-threshold", type=float, default=0.5)
    parser.add_argument("--room-mask-strength", type=float, default=8.0)
    parser.add_argument(
        "--structure-room-source",
        choices=("output", "dense"),
        default="output",
    )
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--reconcile-gap-fraction", type=float, default=0.04)
    parser.add_argument(
        "--wall-skeleton-close-kernel",
        type=int,
        default=0,
        help="Zero uses the adaptive default; otherwise use an odd kernel.",
    )
    parser.add_argument(
        "--wall-skeleton-simplification-fraction",
        type=float,
        default=0.006,
    )
    parser.add_argument(
        "--wall-skeleton-min-segment-length",
        type=float,
        default=2.5,
    )
    parser.add_argument("--ip-wall-reconcile", action="store_true")
    parser.add_argument("--owner-cell-wall-plan", action="store_true")
    parser.add_argument("--owner-cell-edge-support", type=float, default=0.25)
    parser.add_argument("--owner-cell-support-radius", type=int, default=1)
    parser.add_argument("--owner-cell-min-area", type=int, default=32)
    parser.add_argument("--near-duplicate-wall-prune", action="store_true")
    parser.add_argument(
        "--near-duplicate-wall-distance-fraction",
        type=float,
        default=0.006,
    )
    parser.add_argument(
        "--near-duplicate-wall-angle-degrees",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--near-duplicate-wall-overlap",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--ip-wall-milp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ip-wall-tolerance-fraction", type=float, default=0.006)
    parser.add_argument("--ip-wall-merge-gap-fraction", type=float, default=0.02)
    parser.add_argument("--room-merge-min-area", type=int, default=0)
    parser.add_argument("--include-room-decomposition", action="store_true")
    parser.add_argument(
        "--ged-timeout-seconds",
        type=float,
        default=0.5,
        help="Legacy compatibility only; deterministic GED ignores wall-clock time.",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_fidelity_val491_v3.json",
    )
    return parser.parse_args()
