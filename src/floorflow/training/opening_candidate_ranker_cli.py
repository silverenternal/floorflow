"""CLI contract for opening candidate-ranker training."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--cal-limit", type=int, default=96)
    parser.add_argument("--cal-offset", type=int, default=0)
    parser.add_argument("--holdout-limit", type=int, default=192)
    parser.add_argument("--holdout-offset", type=int, default=96)
    parser.add_argument("--candidate-threshold", type=float, default=0.15)
    parser.add_argument(
        "--candidate-feature-version",
        choices=("v2", "v3"),
        default="v2",
    )
    parser.add_argument("--min-peak-distance", type=float, default=4.0)
    parser.add_argument("--opening-host-raster-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-ranker", action="store_true")
    parser.add_argument("--ranker-epochs", type=int, default=200)
    parser.add_argument("--ranker-hidden", type=int, default=64)
    parser.add_argument("--ranker-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/opening_candidate_ranker_v1",
    )
    return parser.parse_args()


__all__ = ["parse_args"]
