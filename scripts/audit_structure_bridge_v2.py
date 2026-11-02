#!/usr/bin/env python3
"""CPU-only audit of the trained structure bridge and its room distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_query_blend50_seed19_v1/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/structure_bridge_audit_v2.json",
    )
    return parser.parse_args()


def _load(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    return torch.load(resolved, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    payload = _load(args.checkpoint)
    if payload is None:
        raise FileNotFoundError(args.checkpoint)
    config = payload.get("config", {})
    dense = _load(args.dense_checkpoint)
    query = _load(args.query_checkpoint)
    structure_params = sum(
        value.numel()
        for key, value in payload["model"].items()
        if key.startswith("structure_refiner.")
    )
    findings = []
    if bool(config.get("structure_only", False)):
        findings.append(
            {
                "severity": "high",
                "code": "room_refiner_bypassed",
                "message": "Structure-only training leaves CoarseLayoutRefiner unused.",
            }
        )
    if config.get("structure_contract", "legacy") == "legacy":
        findings.append(
            {
                "severity": "high",
                "code": "implicit_downsample_resolution",
                "message": (
                    "The legacy refiner creates an internal 256 map and returns "
                    "it at the 128 structure-control resolution."
                ),
            }
        )
    bridge_room = Path(str(config.get("raster_room_checkpoint", "")))
    current_room = args.dense_checkpoint
    if bridge_room.name and bridge_room.resolve() != current_room.resolve():
        findings.append(
            {
                "severity": "high",
                "code": "room_distribution_mismatch",
                "message": (
                    "The structure bridge was trained on a different room "
                    "checkpoint from the current evaluation path."
                ),
            }
        )
    if query is not None:
        findings.append(
            {
                "severity": "info",
                "code": "query_readout_not_in_bridge_training",
                "message": (
                    "The bridge config has no query blend; typed_v2 training can "
                    "now consume and jitter the deployed blend."
                ),
            }
        )
    report = {
        "schema_version": "structure-bridge-audit-v2",
        "checkpoint": str(args.checkpoint),
        "config": {
            "structure_contract": config.get("structure_contract", "legacy"),
            "structure_only": bool(config.get("structure_only", False)),
            "train_cases": int(config.get("limit_train", 0)),
            "epochs": int(config.get("epochs", 0)),
            "room_size": int(config.get("room_size", 224)),
            "structure_size": int(config.get("structure_size", 128)),
            "bridge_room_checkpoint": str(
                config.get("raster_room_checkpoint")
            ),
            "room_query_blend": config.get("room_query_blend"),
        },
        "parameters": {
            "structure_refiner": structure_params,
            "total": sum(value.numel() for value in payload["model"].values()),
        },
        "comparison": {
            "dense_room_checkpoint": str(args.dense_checkpoint),
            "dense_room_metrics": (dense or {}).get("metrics"),
            "query_checkpoint": str(args.query_checkpoint),
            "query_blend": (query or {}).get("blend"),
        },
        "findings": findings,
        "implemented_contract": "configs/predicted_control_structure_typed_v2.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
