#!/usr/bin/env python3
"""Compare sparse-control MiTA and MHA under the same mask budget."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    mita = read("outputs/reports/c3_sparse_mita_sweep_val64.json")
    mha = read("outputs/reports/c3_sparse_mha_sweep_val64.json")
    ratios = sorted(
        set(mita["control_ratio_sweep"]) & set(mha["control_ratio_sweep"]),
        key=float,
    )
    rows = {
        ratio: {
            "mita_room_iou": mita["control_ratio_sweep"][ratio][
                "room_macro_iou"
            ]["mean"],
            "mha_room_iou": mha["control_ratio_sweep"][ratio][
                "room_macro_iou"
            ]["mean"],
            "mita_instance_f1": mita["control_ratio_sweep"][ratio][
                "instance_room_f1"
            ]["mean"],
            "mha_instance_f1": mha["control_ratio_sweep"][ratio][
                "instance_room_f1"
            ]["mean"],
        }
        for ratio in ratios
    }
    report = {
        "phase": "C3",
        "same_budget": {
            "train_samples": 5126,
            "epochs": 3,
            "initialization": "guided_room_cfm_full_v2/best.pt",
        },
        "by_mask_ratio": rows,
        "mita_not_worse_all_ratios": all(
            row["mita_room_iou"] >= row["mha_room_iou"]
            for row in rows.values()
        ),
        "finding": (
            "MiTA helps full-control refinement but does not establish a "
            "monotonic advantage under heavy masking; sparse-control 0.95 "
            "remains unmet."
        ),
    }
    output = ROOT / "outputs/reports/c3_attention_ablation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
