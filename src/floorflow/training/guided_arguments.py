"""CLI contract for guided MiTA-CFM training."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/guided_room_cfm_v1"),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=2.0e-4)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--partition-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--validation-steps", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--dynamic-mask", action="store_true")
    parser.add_argument("--contour-only-training", action="store_true")
    parser.add_argument("--mask-min", type=float, default=0.15)
    parser.add_argument("--mask-max", type=float, default=0.9)
    parser.add_argument("--validation-mask-ratio", type=float, default=None)
    parser.add_argument("--mask-curriculum", action="store_true")
    parser.add_argument("--zero-control-probability", type=float, default=0.20)
    parser.add_argument("--probe-probability", type=float, default=0.15)
    parser.add_argument("--probe-ratio", type=float, default=0.25)
    parser.add_argument("--sparse-control-attention", action="store_true")
    parser.add_argument(
        "--family-attention-impl",
        choices=("checkpoint", "legacy", "official"),
        default="checkpoint",
    )
    parser.add_argument(
        "--sparse-attention-type",
        choices=("mita", "mha", "adaptive"),
        default="mita",
    )
    parser.add_argument(
        "--mask-policy",
        choices=("spatial", "attribute", "salience", "composite", "adaptive"),
        default="spatial",
    )
    parser.add_argument("--mask-discriminator", action="store_true")
    parser.add_argument("--learned-mask-generator", action="store_true")
    parser.add_argument("--mask-coverage-weight", type=float, default=0.1)
    parser.add_argument("--distribution-head-checkpoint", type=Path, default=None)
    parser.add_argument("--distribution-hidden", type=int, default=96)
    parser.add_argument("--distribution-gate-hidden", type=int, default=64)
    parser.add_argument("--enable-room-mask-head", action="store_true")
    parser.add_argument("--room-mask-queries", type=int, default=32)
    parser.add_argument("--room-mask-hidden", type=int, default=128)
    parser.add_argument("--room-mask-resolution", type=int, default=64)
    parser.add_argument("--room-mask-weight", type=float, default=0.0)
    parser.add_argument("--room-mask-partition-weight", type=float, default=0.0)
    parser.add_argument("--x1-structural-weight", type=float, default=0.1)
    parser.add_argument("--x1-family-bce-weight", type=float, default=0.10)
    parser.add_argument("--x1-family-dice-weight", type=float, default=0.05)
    parser.add_argument("--x1-opening-wall-weight", type=float, default=0.05)
    parser.add_argument("--x1-wall-boundary-weight", type=float, default=0.05)
    parser.add_argument("--route-balance-weight", type=float, default=0.01)
    parser.add_argument("--gradient-balance-interval", type=int, default=50)
    parser.add_argument("--generator-adversarial-weight", type=float, default=0.02)
    return parser.parse_args()


__all__ = ["parse_args"]
