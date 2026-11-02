"""Grouped optional CLI arguments for chip training."""

from __future__ import annotations

import argparse


def add_architecture_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--architecture-protocol",
        choices=("internal", "external"),
        default="internal",
    )
    parser.add_argument(
        "--loss-balance-mode",
        choices=("fixed", "normalized"),
        default="fixed",
    )
    parser.add_argument("--loss-balance-floor", type=float, default=1.0e-6)


def add_channel_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-scope",
        default="all",
        help="Composable scope, e.g. channel:local+downstream or fusion+downstream.",
    )
    parser.add_argument(
        "--training-scope-schedule",
        default=None,
        help="Comma-separated start:scope transitions, first start must be zero.",
    )
    parser.add_argument("--typed-channel-cache-root", type=str, default=None)
    parser.add_argument("--typed-channel-cache-split", default="train")


def add_architecture_training_args(parser: argparse.ArgumentParser) -> None:
    add_architecture_protocol_args(parser)
    add_channel_training_args(parser)


def add_structured_head_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--structured-heads", action="store_true")
    parser.add_argument(
        "--structured-hierarchy-coupling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--structured-factorized-block",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--structured-factorized-shared-trunk",
        action="store_true",
    )
    parser.add_argument(
        "--structured-rho-trust-radius",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--structured-dual-ports",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--structured-legalizer-response",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--structured-legalizer-apply",
        action="store_true",
    )
    parser.add_argument(
        "--structured-legalizer-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--structured-legalizer-max-displacement",
        type=float,
        default=0.1,
    )
    parser.add_argument("--shape-modes", type=int, default=0)
    parser.add_argument("--shape-mode-width", type=int)
    parser.add_argument("--shape-mode-max-delta", type=float, default=0.5)
    parser.add_argument("--shape-mode-center-scale", type=float, default=0.0)
    parser.add_argument(
        "--shape-mode-top-fractions",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--shape-mode-apply",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shape-mode-selection",
        choices=("learned", "pareto"),
        default="learned",
    )
    parser.add_argument("--shape-mode-loss-weight", type=float, default=0.0)
    parser.add_argument("--shape-mode-generator-weight", type=float, default=0.1)
    parser.add_argument("--shape-mode-diversity-weight", type=float, default=0.01)
    parser.add_argument("--shape-mode-usage-weight", type=float, default=0.01)
    parser.add_argument(
        "--shape-mode-selection-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--shape-mode-diversity-margin", type=float, default=0.05)
    parser.add_argument("--shape-mode-usage-temperature", type=float, default=0.1)
    parser.add_argument("--require-atom-isolation", action="store_true")
    parser.add_argument("--stage-lr-restart", action="store_true")
    parser.add_argument("--stage-lr-warmup-steps", type=int, default=25)


__all__ = [
    "add_architecture_protocol_args",
    "add_architecture_training_args",
    "add_channel_training_args",
    "add_structured_head_args",
]
