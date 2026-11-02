"""CLI contract for canonical dense/query structure evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from floorflow.inference.opening_heatmap_decoder import (
    add_opening_heatmap_arguments,
)

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--support-gate-checkpoint", type=Path, default=None)
    add_opening_heatmap_arguments(parser)
    parser.add_argument("--topology-safe", action="store_true")
    parser.add_argument("--min-deletable-component-area", type=int, default=32)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--query-weight", type=float, default=None)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--repair-room", action="store_true")
    parser.add_argument(
        "--readout-seam-mode",
        choices=("legacy", "none", "boundary_gated", "wall_gated"),
        default="wall_gated",
    )
    parser.add_argument("--readout-wall-threshold", type=float, default=0.5)
    parser.add_argument("--readout-wall-core-threshold", type=float, default=0.0)
    parser.add_argument("--readout-wall-tail-radius", type=int, default=1)
    parser.add_argument("--readout-opening-threshold", type=float, default=0.5)
    parser.add_argument("--readout-junction-threshold", type=float, default=0.5)
    parser.add_argument("--readout-junction-degree-min", type=int, default=2)
    parser.add_argument("--readout-room-boundary-threshold", type=float, default=0.2)
    parser.add_argument("--readout-opening-support-radius", type=int, default=2)
    parser.add_argument("--readout-opening-min-area", type=int, default=1)
    parser.add_argument(
        "--readout-exterior-mode",
        choices=("fixed", "resolution_scaled"),
        default="fixed",
    )
    parser.add_argument("--owner-cell-plan", action="store_true")
    parser.add_argument(
        "--structure-opening-interval-readout",
        action="store_true",
    )
    parser.add_argument("--canonical-token-readout", action="store_true")
    parser.add_argument("--reconcile-gap-fraction", type=float, default=0.04)
    parser.add_argument("--fpeval-tolerance", type=float, default=0.05)
    parser.add_argument("--ged-timeout-seconds", type=float, default=0.1)
    parser.add_argument(
        "--graph-source",
        choices=("raster", "plan"),
        default="raster",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


__all__ = ["parse_args"]
