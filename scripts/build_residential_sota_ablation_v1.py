"""Build matched residential editing ablations from frozen seed reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21, 22, 23)
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
ARMS = {
    "raster2seq_local": "raster2seq_local",
    "no_typed_repair": "hybrid_delta",
    "raw_floorflow": "floorflow",
    "typed_before_floorflow_after": "typed_before_floorflow_after",
}
METRICS = ("delta_iou", "delta_mse", "outside_exact_ratio")


def report_path(seed: int) -> Path:
    return (
        ROOT
        / "outputs/reports"
        / f"residential_edit_hybrid_typedall_seed{seed}_v1.json"
    )


def main() -> None:
    reports = {
        seed: json.loads(report_path(seed).read_text(encoding="utf-8"))
        for seed in SEEDS
    }
    operations = {}
    checks = {}
    for operation in OPERATIONS:
        arms = {}
        for label, source_name in ARMS.items():
            arms[label] = {
                metric: {
                    "mean": float(
                        np.mean(
                            [
                                reports[seed]["operations"][operation]["methods"][
                                    source_name
                                ][metric]
                                for seed in SEEDS
                            ]
                        )
                    ),
                    "min": float(
                        min(
                            reports[seed]["operations"][operation]["methods"][
                                source_name
                            ][metric]
                            for seed in SEEDS
                        )
                    ),
                    "max": float(
                        max(
                            reports[seed]["operations"][operation]["methods"][
                                source_name
                            ][metric]
                            for seed in SEEDS
                        )
                    ),
                }
                for metric in METRICS
            }
        operations[operation] = {
            "arms": arms,
            "typed_gain_vs_no_repair": (
                arms["typed_before_floorflow_after"]["delta_iou"]["mean"]
                - arms["no_typed_repair"]["delta_iou"]["mean"]
            ),
        }
        checks[f"{operation}_typed_iou_above_anchor"] = (
            arms["typed_before_floorflow_after"]["delta_iou"]["min"] > 0.608
        )
        checks[f"{operation}_typed_outside_exact"] = (
            arms["typed_before_floorflow_after"]["outside_exact_ratio"]["min"]
            == 1.0
        )
        checks[f"{operation}_typed_beats_raster2seq"] = (
            arms["typed_before_floorflow_after"]["delta_iou"]["min"]
            > arms["raster2seq_local"]["delta_iou"]["max"]
        )
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa test503 matched editing ablations",
        "seeds": list(SEEDS),
        "operations": operations,
        "checks": checks,
        "interpretation": {
            "typed_before_floorflow_after": (
                "Repair only typed before-edit changes, then use the existing "
                "control-gated FloorFlow after-edit layer."
            ),
            "no_typed_repair": "Local-delta splice without typed before-edit repair.",
            "raster2seq_local": "Matched reconstruction wrapper, not an editing method.",
        },
    }
    output = ROOT / "outputs/reports/residential_sota_ablation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Residential Edit Ablation",
        "",
        "| Operation | Raster2Seq local | No typed repair | Raw FloorFlow | Typed completion | Gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for operation, row in operations.items():
        arms = row["arms"]
        lines.append(
            f"| {operation} | "
            f"{arms['raster2seq_local']['delta_iou']['mean']:.6f} | "
            f"{arms['no_typed_repair']['delta_iou']['mean']:.6f} | "
            f"{arms['raw_floorflow']['delta_iou']['mean']:.6f} | "
            f"{arms['typed_before_floorflow_after']['delta_iou']['mean']:.6f} | "
            f"{row['typed_gain_vs_no_repair']:+.6f} |"
        )
    table = ROOT / "paper/tables/residential_edit_ablation_v1.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
