#!/usr/bin/env python3
"""CPU-only audit of the effective room learning-head training contract."""

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
        default=ROOT / "outputs/train/room_query_blend50_seed19_v1/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/room_learning_head_audit_v2.json",
    )
    return parser.parse_args()


def _payload(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    return torch.load(resolved, map_location="cpu", weights_only=False)


def _parameter_counts(state: dict[str, torch.Tensor]) -> dict[str, int]:
    groups = {
        "room_encoder": "room.",
        "query_decoder": "mask_decoder.",
        "mask_refiner": "mask_refiner.",
        "mask_pooled_class": "mask_pooled_class_fusion.",
        "query_set": "query_set_head.",
        "room_program": "room_program_head.",
        "dense_identity_refiner": "dense_identity_refiner.",
    }
    return {
        name: sum(
            value.numel()
            for key, value in state.items()
            if key.startswith(prefix)
        )
        for name, prefix in groups.items()
    }


def main() -> None:
    args = parse_args()
    payload = _payload(args.checkpoint)
    if payload is None:
        raise FileNotFoundError(args.checkpoint)
    config = payload.get("config", {})
    state = payload.get("model", {})
    parent = _payload(config.get("initialize_from"))
    dense = _payload(config.get("room_checkpoint"))
    dense_config = (dense or {}).get("config", {})
    parent_config = (parent or {}).get("config", {})
    flags = {
        key: bool(config.get(key, False))
        for key in (
            "adaptive_query_refiner",
            "full_resolution_masks",
            "mask_pooled_class_fusion",
            "query_set_head",
            "query_count_gate",
            "room_program_head",
            "partition_consistent",
            "dense_identity_refiner",
            "train_encoder",
        )
    }
    train_cases = int(config.get("limit_train", 0))
    dense_train_cases = int(dense_config.get("limit_train", 0))
    findings = []
    if dense_train_cases and train_cases < dense_train_cases:
        findings.append(
            {
                "severity": "high",
                "code": "query_head_under_training",
                "message": (
                    "The query head uses fewer training cases than its frozen "
                    "dense encoder."
                ),
            }
        )
    if "loss_contract" not in payload:
        findings.append(
            {
                "severity": "medium",
                "code": "legacy_checkpoint_without_loss_contract",
                "message": "The checkpoint predates explicit loss-contract metadata.",
            }
        )
    if payload.get("blend") is None:
        findings.append(
            {
                "severity": "medium",
                "code": "missing_checkpoint_blend",
                "message": "The evaluator must use an explicit blend override.",
            }
        )
    report = {
        "schema_version": "room-learning-head-audit-v2",
        "checkpoint": str(args.checkpoint),
        "config": {
            "train_cases": train_cases,
            "validation_cases": int(config.get("limit_val", 0)),
            "epochs": int(config.get("epochs", 0)),
            "blend": payload.get("blend"),
            "effective_flags": flags,
        },
        "parent": {
            "path": str(config.get("initialize_from")),
            "train_cases": int(parent_config.get("limit_train", 0)),
            "epochs": int(parent_config.get("epochs", 0)),
        },
        "dense_encoder": {
            "path": str(config.get("room_checkpoint")),
            "train_cases": dense_train_cases,
            "epochs": int(dense_config.get("epochs", 0)),
        },
        "parameter_counts": _parameter_counts(state),
        "loss_contract": payload.get("loss_contract"),
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
