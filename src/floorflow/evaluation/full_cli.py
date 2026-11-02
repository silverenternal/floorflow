"""Command-line contract for full floorplan evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from floorflow.training.runtime_acceleration import PRECISIONS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_full_evaluation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/cache/cubicasa5k",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/eval",
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--no-heun", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.75)
    parser.add_argument("--smooth-kernel", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--objective-mode",
        choices=("velocity", "simplex"),
        default=None,
    )
    parser.add_argument(
        "--weight-source",
        choices=("model", "ema"),
        default="model",
    )
    parser.add_argument("--semantic-decode-weight", type=float, default=0.5)
    parser.add_argument("--repair-rooms", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--structured-flow-steps", type=int, default=0)
    parser.add_argument("--structured-flow-noise-scale", type=float, default=0.75)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--furniture-threshold", type=float, default=0.5)
    parser.add_argument("--wall-threshold", type=float, default=0.5)
    parser.add_argument("--use-geometry-walls", action="store_true")
    parser.add_argument("--geometry-wall-threshold", type=float, default=0.7)
    parser.add_argument("--geometry-wall-dilate", type=int, default=1)
    parser.add_argument("--instance-semantic-vote", action="store_true")
    parser.add_argument(
        "--room-program-input",
        choices=("none", "gt", "gt_counts"),
        default="none",
    )
    parser.add_argument("--program-absent-penalty", type=float, default=8.0)
    parser.add_argument(
        "--coarse-layout-input",
        choices=("none", "gt"),
        default="none",
    )
    parser.add_argument("--coarse-layout-size", type=int, default=64)
    parser.add_argument("--coarse-layout-weight", type=float, default=4.0)
    parser.add_argument(
        "--object-score-scale",
        choices=("logit", "probability"),
        default="probability",
    )
    parser.add_argument("--threshold-calibration", type=Path, default=None)
    parser.add_argument("--wall-source-checkpoint", type=Path, default=None)
    parser.add_argument("--wall-source-threshold", type=float, default=0.7)
    parser.add_argument("--wall-source-dilate", type=int, default=0)
    parser.add_argument("--wall-face-rooms", action="store_true")
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument(
        "--retrieval-metric",
        choices=("cosine", "mask_iou", "hybrid"),
        default="cosine",
    )
    parser.add_argument("--retrieval-opening-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-furniture-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-wall-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-class-weights", type=Path, default=None)
    parser.add_argument("--structural-gate", action="store_true")
    parser.add_argument("--partition-walls-only", action="store_true")
    parser.add_argument(
        "--wall-mode",
        choices=("heads", "partition", "union"),
        default=None,
    )
    parser.add_argument("--allow-incompatible-furniture", action="store_true")
    parser.add_argument("--partition-wall-radius", type=int, default=0)
    parser.add_argument(
        "--object-source",
        choices=("auto", "semantic", "heads"),
        default="auto",
    )
    parser.add_argument(
        "--opening-source",
        choices=("inherit", "heads", "semantic"),
        default="inherit",
    )
    parser.add_argument(
        "--furniture-source",
        choices=("inherit", "heads", "semantic"),
        default="inherit",
    )
    parser.add_argument(
        "--room-source",
        choices=("mask", "partition", "semantic", "hierarchical", "polygon"),
        default="mask",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=PRECISIONS, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--save-examples", type=int, default=16)
    parser.add_argument("--save-rasters", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


__all__ = ["parse_full_evaluation_args"]
