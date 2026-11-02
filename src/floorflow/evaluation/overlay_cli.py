"""Command-line contract for deterministic overlay evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_overlay_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dynamic-head-variant",
        choices=("fixed", "mixture", "generated", "query"),
        default=None,
    )
    parser.add_argument("--ensemble-checkpoint", type=Path, default=None)
    parser.add_argument("--ensemble-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--save-labels-dir", type=Path, default=None)
    parser.add_argument("--per-case-output", type=Path, default=None)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument(
        "--retrieval-metric",
        choices=("cosine", "mask_iou", "hybrid"),
        default="cosine",
    )
    parser.add_argument("--program-checkpoint", type=Path, default=None)
    parser.add_argument("--layout-checkpoint", type=Path, default=None)
    parser.add_argument("--program-prior-weight", type=float, default=0.0)
    parser.add_argument("--room-prior-weight", type=float, default=0.0)
    parser.add_argument("--room-segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--room-raster-checkpoint", type=Path, default=None)
    parser.add_argument("--room-raster-blend", type=float, default=0.9)
    parser.add_argument(
        "--room-raster-input-rendering",
        choices=("native", "line", "nonroom"),
        default="native",
    )
    parser.add_argument("--room-raster-support-fallback", action="store_true")
    parser.add_argument("--recursion-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--recursive-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--recursive-graph-blend-scale", type=float, default=1.0)
    parser.add_argument(
        "--room-gate-decouple",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--room-graph-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--room-graph-trust-checkpoint", type=Path, default=None)
    parser.add_argument("--room-graph-trust-threshold", type=float, default=0.35)
    parser.add_argument("--furniture-query-checkpoint", type=Path, default=None)
    parser.add_argument("--furniture-query-threshold", type=float, default=0.45)
    parser.add_argument("--recursive-max-depth", type=int, default=3)
    parser.add_argument("--recursive-min-leaf-pixels", type=int, default=256)
    parser.add_argument("--recursive-context-fraction", type=float, default=0.75)
    parser.add_argument("--recursive-blend", type=float, default=0.1)
    parser.add_argument("--recursive-graph-blend", type=float, default=0.0)
    parser.add_argument("--recursive-graph-steps", type=int, default=2)
    parser.add_argument("--recursive-graph-margin", type=float, default=None)
    parser.add_argument("--room-partition-weight", type=float, default=0.0)
    parser.add_argument(
        "--room-prior-confidence-threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument("--style-separation", type=float, default=None)
    parser.add_argument("--seed-iou-gap", type=float, default=None)
    parser.add_argument("--overlay-room-logit-weight", type=float, default=0.0)
    parser.add_argument("--overlay-room-support-threshold", type=float, default=0.1)
    parser.add_argument("--overlay-wall-threshold", type=float, default=0.3)
    parser.add_argument("--overlay-opening-threshold", type=float, default=0.2)
    parser.add_argument("--overlay-furniture-threshold", type=float, default=0.2)
    parser.add_argument("--overlay-wall-element-weight", type=float, default=0.0)
    parser.add_argument("--overlay-opening-element-weight", type=float, default=0.0)
    parser.add_argument("--overlay-furniture-element-weight", type=float, default=0.0)
    parser.add_argument("--legalize-objects", action="store_true")
    parser.add_argument("--legalization-opening-wall-radius", type=int, default=2)
    parser.add_argument("--legalization-closure-radius", type=int, default=0)
    parser.add_argument("--legalization-dynamic-closure", action="store_true")
    parser.add_argument(
        "--retrieval-weights-report",
        type=Path,
        default=Path("outputs/reports/a1_retrieval_weights_calib191.json"),
    )
    parser.add_argument(
        "--retrieval-baseline",
        type=Path,
        default=Path("outputs/reports/distribution_v6b_fusion_holdout300.json"),
    )
    return parser.parse_args()


__all__ = ["parse_overlay_args"]
