#!/usr/bin/env python3
"""Audit whether two predicted-control roots satisfy the same source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.evaluation.control_source_parity import audit_control_source_pair

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v3",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_source_parity_v1_v3.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    report = audit_control_source_pair(
        args.reference_root,
        args.candidate_root,
        splits=tuple(args.splits),
        cache_root=args.cache_root,
        bootstrap_iterations=args.bootstrap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
