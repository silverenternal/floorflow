"""CLI for contour-conditioned distribution-CFM training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def validate_distribution_runtime_args(args: argparse.Namespace) -> str | None:
    """Return a startup error for known unsafe training combinations."""

    if (
        args.compile_model
        and args.gradient_balance_interval > 0
    ):
        return (
            "torch.compile is incompatible with periodic gradient balancing "
            "because balancing needs multiple backward traversals"
        )
    return None


def parse_distribution_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("cubicasa", "skv4", "mixed"),
        default="cubicasa",
    )
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument("--initialize-distribution", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=96)
    parser.add_argument("--base-lr", type=float, default=2.0e-6)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--selection", choices=("raw", "ema"), default="ema")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--pixel-grid", type=int, default=16)
    parser.add_argument("--element-grid", type=int, default=16)
    parser.add_argument("--element-mask-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--num-styles", type=int, default=0)
    parser.add_argument("--style-rank", type=int, default=8)
    parser.add_argument("--style-manifest", type=Path, default=None)
    parser.add_argument("--style-contrast-weight", type=float, default=0.1)
    parser.add_argument("--style-contrast-margin", type=float, default=0.01)
    parser.add_argument("--style-semantic-weight", type=float, default=0.05)
    parser.add_argument("--style-orthogonality-weight", type=float, default=0.01)
    parser.add_argument("--style-dropout", type=float, default=0.10)
    parser.add_argument("--style-lr", type=float, default=1.0e-2)
    parser.add_argument("--parent-reliability-init", type=float, default=0.0)
    parser.add_argument("--hierarchy-support-strength", type=float, default=None)
    parser.add_argument("--element-blend-init", type=float, default=None)
    parser.add_argument("--element-feedback-init", type=float, default=None)
    parser.add_argument(
        "--bridge-mode",
        choices=("blend", "residual"),
        default="residual",
    )
    parser.add_argument(
        "--cfm-structure-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--static-condition-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--typed-family-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--multiscale-structure-injection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamic-head-variant",
        choices=("fixed", "mixture", "generated", "query"),
        default="generated",
    )
    parser.add_argument("--dynamic-element-queries", action="store_true")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--element-weight", type=float, default=0.25)
    parser.add_argument("--coupling-weight", type=float, default=0.1)
    parser.add_argument("--source-weight", type=float, default=0.1)
    parser.add_argument("--family-iou-weight", type=float, default=0.3)
    parser.add_argument("--room-iou-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--structural-weight", type=float, default=0.0)
    parser.add_argument("--topology-closure-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-route-weight", type=float, default=0.01)
    parser.add_argument("--positive-weight-clip", type=float, default=20.0)
    parser.add_argument("--flow-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--flow-change-weight", type=float, default=0.0)
    parser.add_argument("--gradient-balance-interval", type=int, default=50)
    parser.add_argument("--gradient-balance-min-scale", type=float, default=1.0e-3)
    parser.add_argument("--gradient-balance-max-scale", type=float, default=4.0)
    parser.add_argument("--loss-metric-patience", type=int, default=2)
    parser.add_argument("--loss-metric-drop", type=float, default=1.0e-3)
    parser.add_argument("--validation-metric-drop", type=float, default=1.0e-3)
    parser.add_argument("--element-warmup-updates", type=int, default=700)
    parser.add_argument("--element-warmup-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--flow-preset",
        choices=("baseline", "fast_cfm", "fast_vp", "multi_stage_ot"),
        default="fast_cfm",
    )
    parser.add_argument(
        "--flow-matcher",
        choices=("legacy", "torchcfm", "torchcfm_vp"),
        default=None,
    )
    parser.add_argument(
        "--time-sampling",
        choices=("uniform", "logit_normal"),
        default=None,
    )
    parser.add_argument("--flow-sigma", type=float, default=None)
    parser.add_argument("--time-sampling-mean", type=float, default=None)
    parser.add_argument("--time-sampling-std", type=float, default=None)
    parser.add_argument(
        "--minibatch-ot",
        choices=("none", "token"),
        default="token",
    )
    parser.add_argument("--ot-token-grid", type=int, default=8)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--dynamic-source-mask", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    error = validate_distribution_runtime_args(args)
    if error is not None:
        parser.error(error)
    return args


__all__ = ["parse_distribution_args", "validate_distribution_runtime_args"]
