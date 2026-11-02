#!/usr/bin/env python3
"""Summarize contour-only stage-1 ceiling experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    naive = read_jsonl("outputs/train/b1_mita_cfm_full_v1/train.jsonl")
    program = read_jsonl("outputs/train/contour_program_layout_full_v1/train.jsonl")
    guided = read_jsonl("outputs/train/guided_room_cfm_full_v1/train.jsonl")
    guided_resume = read_jsonl("outputs/train/guided_room_cfm_full_v2/train.jsonl")
    report = {
        "question": "Can a target-control-free contour-to-CAD path reach 0.95?",
        "experiments": {
            "noise_source_miTa_cfm": {
                "best_room_macro_iou": max(
                    row["val_partition_room_macro_iou"] for row in naive
                ),
                "epochs": len(naive),
                "train_samples": 5126,
            },
            "contour_plus_program_layout": {
                "best_room_macro_iou": max(
                    row["room_macro_iou"] for row in program
                ),
                "epochs": len(program),
                "train_samples": 5126,
            },
            "guided_miTa_cfm": {
                "first_run_best_room_macro_iou": max(
                    row["room_macro_iou"] for row in guided
                ),
                "resume_best_room_macro_iou": max(
                    row["room_macro_iou"] for row in guided_resume
                ),
                "epochs": len(guided) + len(guided_resume),
                "train_samples": 5126,
            },
        },
        "val64_step_sweep": {
            "1_step": 0.7368153094668464,
            "2_steps": 0.7245879692567672,
            "4_steps": 0.7104394880436957,
            "8_steps": 0.7015059107195621,
            "12_steps": 0.6981646977989657,
        },
        "source_control_room_iou_val16": 0.808924735723229,
        "guided_plus_refiners_val32_room_iou": 0.2447622457853122,
        "conclusion": (
            "The gap is representational/identifiability, not a missing "
            "optimizer step: multiple contour-only and contour+program models "
            "plateau far below 0.95, while target-derived coarse controls "
            "support the passing layout-conditioned gate."
        ),
        "scope_options": [
            "accept coarse layout control as required input",
            "redefine the claim around layout-conditioned refinement",
            "continue a separate long-term contour-to-layout research track",
        ],
    }
    output = ROOT / "outputs/reports/b1_contour_cfm_blocker.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
