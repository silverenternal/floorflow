#!/usr/bin/env python3
"""Audit deployed model parameters and dataset coverage from checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parameter_count(checkpoint: Path, key: str | None = "model") -> int | None:
    if not checkpoint.is_file():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    def count(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        return 0

    return count(payload if key is None else payload.get(key, payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        type=Path,
        default=Path("outputs/train/distribution_retrieval_free_merged_style_v2.pt"),
    )
    parser.add_argument(
        "--room-segmenter",
        type=Path,
        default=Path("outputs/train/room_segmenter_crop_consistency_v1/best.pt"),
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("outputs/train/coarse_structure_selfcontrol_full_v1/best.pt"),
    )
    parser.add_argument(
        "--furniture",
        type=Path,
        default=Path("outputs/train/coarse_furniture_selfcontrol_full_v1/best.pt"),
    )
    parser.add_argument(
        "--support",
        type=Path,
        default=Path("outputs/train/coarse_support_selfcontrol_full_v1/best.pt"),
    )
    parser.add_argument(
        "--graph-adapter",
        type=Path,
        default=Path("outputs/train/recursive_graph_adapter_1024_v1/best.pt"),
    )
    parser.add_argument(
        "--furniture-query-head",
        type=Path,
        default=Path("outputs/train/dynamic_furniture_queries_1024_v2/best.pt"),
    )
    parser.add_argument(
        "--graph-trust-gate",
        type=Path,
        default=Path("outputs/train/room_graph_trust_gate_smoke_v1/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/model_inventory_v2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = {
        "distribution_cfm": args.distribution,
        "room_segmenter": args.room_segmenter,
        "structure_refiner": args.structure,
        "furniture_refiner": args.furniture,
        "support_refiner": args.support,
        "recursive_graph_adapter": args.graph_adapter,
        "furniture_query_head": args.furniture_query_head,
        "graph_trust_gate": args.graph_trust_gate,
    }
    counts = {
        name: parameter_count(
            path,
            key=None if name == "distribution_cfm" else "model",
        )
        for name, path in modules.items()
    }
    available = {
        name: value for name, value in counts.items() if value is not None
    }
    report = {
        "modules": {
            name: {
                "checkpoint": str(path),
                "parameters": counts[name],
                "parameters_m": (
                    counts[name] / 1_000_000
                    if counts[name] is not None
                    else None
                ),
            }
            for name, path in modules.items()
        },
        "deployed_total_m": sum(available.values()) / 1_000_000,
        "core_distribution_cfm_m": (
            counts["distribution_cfm"] / 1_000_000
            if counts["distribution_cfm"] is not None
            else None
        ),
        "auxiliary_total_m": (
            sum(available.values()) - available.get("distribution_cfm", 0)
        )
        / 1_000_000,
        "dataset": {
            "train_floorplans": 5126,
            "val_floorplans": 491,
            "test_floorplans": 503,
            "source": "CubiCasa5K",
        },
        "note": (
            "Parameter counts include all tensors in each deployed checkpoint; "
            "the room segmenter is a frozen contour-only prior."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
