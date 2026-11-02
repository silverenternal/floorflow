#!/usr/bin/env python3
"""Freeze the style-conditioned constrained-design evaluation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cubicasa-checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_style_prompt_r94_v2/best.pt"),
    )
    parser.add_argument(
        "--skv4-checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_skv4_style_v1/best.pt"),
    )
    parser.add_argument(
        "--cubicasa-bank",
        type=Path,
        default=Path("outputs/style_banks/cubicasa_train_v1.json"),
    )
    parser.add_argument(
        "--skv4-bank",
        type=Path,
        default=Path("outputs/style_banks/skv4_train_v2.json"),
    )
    parser.add_argument(
        "--candidate-gate",
        type=Path,
        default=Path("outputs/reports/style_design_gate_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/style_design_manifest_v1.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bank_summary(path: Path) -> dict:
    payload = load(path)
    assignments = payload.get("assignments", {})
    train_assignments = assignments.get("train", {})
    return {
        "path": str(path),
        "sha256": sha256(path),
        "num_styles": int(payload["num_styles"]),
        "train_assignments": len(train_assignments),
        "split_keys": sorted(assignments),
        "style_gates": payload.get("style_gates"),
        "inference_rule": payload.get("inference_rule"),
        "train_only": set(assignments) == {"train"},
    }


def main() -> None:
    args = parse_args()
    gate = load(args.candidate_gate)
    split_sources = {
        "cubicasa_train": Path(".data/cubicasa5k/train.txt"),
        "cubicasa_val": Path(".data/cubicasa5k/val.txt"),
        "cubicasa_test": Path(".data/cubicasa5k/test.txt"),
        "skv4_train": Path("outputs/cache/skv4/train.jsonl"),
        "skv4_val": Path("outputs/cache/skv4/val.jsonl"),
        "skv4_test": Path("outputs/cache/skv4/test.jsonl"),
    }
    banks = {
        "cubicasa": bank_summary(args.cubicasa_bank),
        "skv4": bank_summary(args.skv4_bank),
    }
    checks = {
        "candidate_gate_passed": gate.get("status") == "pass",
        "cubicasa_train_only": banks["cubicasa"]["train_only"],
        "skv4_train_only": banks["skv4"]["train_only"],
        "no_target_control": gate["conditions"][
            "target_derived_coarse_control"
        ]
        is False,
        "no_best_of_n": gate["conditions"]["best_of_n_used"] is False,
    }
    manifest = {
        "status": "frozen" if all(checks.values()) else "blocked",
        "created_utc": datetime.now(UTC).isoformat(),
        "schema_version": "style_design_manifest_v1",
        "objective": "contour-conditioned, style-controllable constrained floorplan generation",
        "checks": checks,
        "seeds": {
            "cubicasa": 19,
            "skv4": 23,
            "sampling": 19,
        },
        "sampling": {
            "steps": 4,
            "samples_per_condition": 1,
            "source_noise_scale": 0.01,
            "solver": "euler",
            "best_of_n_used": False,
        },
        "checkpoints": {
            "cubicasa": {
                "path": str(args.cubicasa_checkpoint),
                "sha256": sha256(args.cubicasa_checkpoint),
            },
            "skv4": {
                "path": str(args.skv4_checkpoint),
                "sha256": sha256(args.skv4_checkpoint),
            },
        },
        "style_banks": banks,
        "splits": {
            name: {
                "path": str(path),
                "cases": line_count(path),
                "sha256": sha256(path),
            }
            for name, path in split_sources.items()
        },
        "candidate_gate": {
            "path": str(args.candidate_gate),
            "sha256": sha256(args.candidate_gate),
            "metrics": gate["metrics"],
            "reference_room_iou_note": gate["reference_room_iou_note"],
        },
        "evaluation_contract": {
            "primary": [
                "legality",
                "coverage",
                "density",
                "MMD",
                "style separation",
                "human preference",
            ],
            "secondary": ["reference room IoU"],
            "forbidden": [
                "target-derived coarse control",
                "test-time target selection",
                "best-of-N",
            ],
        },
        "legality_schema": {
            "fields": [
                "outside_footprint_fraction",
                "room_area_coverage",
                "opening_wall_alignment",
                "furniture_room_containment",
                "conditional_watertight",
                "room_area_distribution_mae",
            ],
            "reference_iou_is_gate": False,
            "reference_iou_field": "diagnostic_reference_room_iou",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
