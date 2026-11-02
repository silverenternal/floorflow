"""Summarize matched FloorFlow v2 architecture screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "class_agnostic_control": "outputs/train/floorflow_v2_control256_v1",
    "v2_compact_full": "outputs/train/floorflow_v2_screen256_v1",
    "compact_no_boundary_graph": "outputs/train/floorflow_v2_ablate_boundary256_v1",
    "compact_no_program_set": "outputs/train/floorflow_v2_ablate_program256_v1",
    "compact_no_point_validity": "outputs/train/floorflow_v2_ablate_point256_v1",
    "resnet18_full_v2": "outputs/train/floorflow_v2_ablate_compact256_v1",
    "resnet18_none_v1": "outputs/train/floorflow_v2_resnet_none256_v1",
    "resnet18_none_v2": "outputs/train/floorflow_v2_resnet_none256_v2",
    "resnet18_boundary_only": "outputs/train/floorflow_v2_resnet_boundary256_v1",
    "resnet18_program_only": "outputs/train/floorflow_v2_resnet_program256_v1",
    "resnet18_point_only": "outputs/train/floorflow_v2_resnet_point256_v1",
    "resnet18_boundary_program": (
        "outputs/train/floorflow_v2_resnet_boundary_program256_v1"
    ),
    "resnet18_boundary_point": (
        "outputs/train/floorflow_v2_resnet_boundary_point256_v1"
    ),
    "floorflow_v3_query_graph": "outputs/train/floorflow_v3_screen256_v1",
}

METRIC_KEYS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "room_count_correct",
    "room_count_mae",
    "room_mask_loss",
    "room_polygon_loss",
    "room_program_set_loss",
    "room_program_loss",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_v2_module_screens_v1.json",
    )
    return parser.parse_args()


def _best_row(report: dict[str, Any]) -> dict[str, Any]:
    return max(report["history"], key=lambda row: row["selection_score"])


def _checkpoint_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("config", {}).get("model", {}))


def _arm_record(name: str, relative: str) -> dict[str, Any] | None:
    directory = ROOT / relative
    report_path = directory / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = _best_row(report)
    metrics = {key: best["val"].get(key) for key in METRIC_KEYS}
    model_config = report.get("effective_model_config")
    if not model_config:
        model_config = _checkpoint_config(directory / "best.pt")
    scores = [row["selection_score"] for row in report["history"]]
    return {
        "name": name,
        "report": str(report_path.relative_to(ROOT)),
        "best_checkpoint": str((directory / "best.pt").relative_to(ROOT)),
        "best_epoch": int(best["epoch"]),
        "parameter_count": int(report["parameter_count"]),
        "architecture": {
            "encoder": model_config.get("encoder_architecture"),
            "boundary_edge_graph": model_config.get("use_boundary_edge_graph"),
            "program_set": model_config.get("enable_program_set"),
            "point_validity": model_config.get("enable_point_validity"),
        },
        "metrics": metrics,
        "selection_score": float(best["selection_score"]),
        "selection_score_range": {
            "min": float(min(scores)),
            "max": float(max(scores)),
        },
    }


def main() -> None:
    args = _parse_args()
    records = []
    missing = []
    for name, relative in ARMS.items():
        record = _arm_record(name, relative)
        if record is None:
            missing.append(relative)
        else:
            records.append(record)
    control = next(
        record for record in records
        if record["name"] == "class_agnostic_control"
    )
    for record in records:
        record["delta_vs_control"] = {
            key: (
                float(record["metrics"][key] - control["metrics"][key])
                if record["metrics"].get(key) is not None
                and control["metrics"].get(key) is not None
                else None
            )
            for key in METRIC_KEYS
        }
    best = max(records, key=lambda record: record["selection_score"])
    payload = {
        "schema": "floorflow-v2-module-screens-v1",
        "test_opened": False,
        "budget": {
            "train_cases": 256,
            "validation_cases": 64,
            "epochs": 8,
            "seed": 2905,
            "device": "cuda:0",
        },
        "control": "class_agnostic_control",
        "best_arm": best["name"],
        "arms": records,
        "missing_arms": missing,
        "claim_boundary": (
            "Internal validation mechanism screen only. Do not present these "
            "numbers as RPLAN or complete-CAD SOTA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_arm": best["name"], "arms": len(records)}))


if __name__ == "__main__":
    main()
