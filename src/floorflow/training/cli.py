"""Command-line contract for structured floorplan training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from floorflow.flow.presets import resolve_flow_preset
from floorflow.training.runtime_acceleration import PRECISIONS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--geometry-cache-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / ".data/cubicasa5k")
    parser.add_argument("--synthetic-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/train/condition_structured")
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--initialize-ema", action="store_true")
    parser.add_argument("--partition-initialize-from", type=Path, default=None)
    parser.add_argument("--object-initialize-from", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--room-boundary-weight", type=float, default=0.15)
    parser.add_argument("--room-tversky-weight", type=float, default=0.40)
    parser.add_argument("--room-group-weight", type=float, default=0.0)
    parser.add_argument("--room-group-head-weight", type=float, default=0.25)
    parser.add_argument("--enable-room-group-head", action="store_true")
    parser.add_argument("--room-program-weight", type=float, default=0.05)
    parser.add_argument("--enable-room-program", action="store_true")
    parser.add_argument("--enable-hierarchical-rooms", action="store_true")
    parser.add_argument("--hierarchical-hidden", type=int, default=96)
    parser.add_argument("--hierarchical-tokens", type=int, default=8)
    parser.add_argument("--hierarchical-coarse-resolution", type=int, default=64)
    parser.add_argument("--room-hierarchy-weight", type=float, default=0.50)
    parser.add_argument("--room-refine-weight", type=float, default=0.25)
    parser.add_argument("--room-transport-weight", type=float, default=0.20)
    parser.add_argument("--condition-flow-weight", type=float, default=0.0)
    parser.add_argument("--condition-flow-noise-scale", type=float, default=0.75)
    parser.add_argument("--condition-flow-matcher", choices=("legacy", "torchcfm", "torchcfm_vp"), default="torchcfm")
    parser.add_argument("--condition-flow-sigma", type=float, default=0.0)
    parser.add_argument("--condition-time-sampling", choices=("uniform", "logit_normal"), default="uniform")
    parser.add_argument("--contour-inward-curriculum", action="store_true")
    parser.add_argument("--contour-band-width", type=float, default=0.22)
    parser.add_argument("--stage-curriculum", action="store_true")
    parser.add_argument(
        "--flow-preset",
        choices=("baseline", "fast_cfm", "fast_vp", "multi_stage_ot"),
        default=None,
    )
    parser.add_argument("--condition-validation-steps", type=int, default=0)
    parser.add_argument("--enable-polygon-loop", action="store_true")
    parser.add_argument("--polygon-queries", type=int, default=24)
    parser.add_argument("--polygon-points", type=int, default=24)
    parser.add_argument("--polygon-hidden", type=int, default=128)
    parser.add_argument("--room-polygon-weight", type=float, default=0.10)
    parser.add_argument("--enable-room-mask-decoder", action="store_true")
    parser.add_argument("--room-mask-queries", type=int, default=32)
    parser.add_argument("--room-mask-resolution", type=int, default=64)
    parser.add_argument("--room-mask-weight", type=float, default=0.0)
    parser.add_argument("--room-mask-partition-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-injection-warmup-epochs", type=int, default=2)
    parser.add_argument("--hierarchy-injection-margin", type=float, default=0.02)
    parser.add_argument("--train-hierarchy-heads-only", action="store_true")
    parser.add_argument("--trainable-prefixes", nargs="+", default=None)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-top-k", type=int, default=2)
    parser.add_argument("--retrieval-eval-top-k", type=int, default=1)
    parser.add_argument("--retrieval-precompute-workers", type=int, default=8)
    parser.add_argument("--retrieval-metric", choices=("cosine", "mask_iou", "hybrid"), default="cosine")
    parser.add_argument("--opening-weight", type=float, default=0.20)
    parser.add_argument("--furniture-weight", type=float, default=0.20)
    parser.add_argument("--wall-weight", type=float, default=0.10)
    parser.add_argument("--enable-geometry-heads", action="store_true")
    parser.add_argument("--geometry-hidden", type=int, default=64)
    parser.add_argument("--geometry-experts", type=int, default=1)
    parser.add_argument("--geometry-top-k", type=int, default=1)
    parser.add_argument("--geometry-weight", type=float, default=0.10)
    parser.add_argument("--enable-family-streams", action="store_true")
    parser.add_argument("--enable-family-mita", action="store_true")
    parser.add_argument("--enable-program-conditioning", action="store_true")
    parser.add_argument("--program-hidden", type=int, default=64)
    parser.add_argument("--structural-consistency-weight", type=float, default=0.10)
    parser.add_argument("--enable-coarse-to-face-cfm", action="store_true")
    parser.add_argument("--coarse-face-hidden", type=int, default=64)
    parser.add_argument("--coarse-face-max-faces", type=int, default=32)
    parser.add_argument("--object-loss-warmup-epochs", type=int, default=1)
    parser.add_argument("--share-backbone-with-objects", action="store_true")
    parser.add_argument("--gradient-audit-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--disable-room-sampler", action="store_true")
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--precision", choices=PRECISIONS, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "partition_room_macro_iou",
            "refined_room_iou",
            "room_mask_room_iou",
            "room_macro_iou",
        ),
        default="partition_room_macro_iou",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.flow_preset is not None:
        preset = resolve_flow_preset(args.flow_preset)
        args.condition_flow_matcher = preset.matcher
        args.condition_time_sampling = preset.time_sampling
        args.condition_flow_sigma = preset.sigma
        args.condition_validation_steps = preset.validation_steps
        args.room_transport_weight = preset.room_transport_weight
    return args
