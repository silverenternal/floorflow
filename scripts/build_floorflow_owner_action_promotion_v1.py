"""Summarize the rejected 1024/256 owner-wall action promotion screen."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
OUTPUT = REPORTS / "floorflow_owner_wall_action_promotion1024_v1.json"

PATHS = {
    2905: (
        REPORTS / "floorflow_owner_wall_action1024_seed2905_control_fusion_structure_v1.json",
        REPORTS / "floorflow_owner_wall_action1024_seed2905_action_fusion_structure_v1.json",
    ),
    37: (
        REPORTS / "floorflow_owner_wall_action1024_seed37_control_fusion_structure_v1.json",
        REPORTS / "floorflow_owner_wall_action1024_seed37_action_fusion_structure_v1.json",
    ),
}

KEYS = (
    "room_iou",
    "instance_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "wall_f1",
    "wall_iou",
    "validity",
    "conditional_watertightness",
    "dangling_endpoint_rate",
    "opening_f1",
)


def _metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    runs = []
    for seed, (control_path, action_path) in PATHS.items():
        control = _metrics(control_path)
        action = _metrics(action_path)
        runs.append(
            {
                "seed": seed,
                "control": {key: float(control[key]) for key in KEYS},
                "action": {key: float(action[key]) for key in KEYS},
                "delta": {
                    key: float(action[key] - control[key])
                    for key in KEYS
                },
                "connector_length_ratio": float(
                    action["wall_connector_length_ratio"]
                ),
            }
        )
    payload = {
        "schema": "floorflow-owner-wall-action-promotion-v1",
        "train_cases": 1024,
        "validation_cases": 256,
        "epochs": 8,
        "nfe": 1,
        "runs": runs,
        "decision": {
            "status": "rejected",
            "reason": (
                "Owner-action regresses both Wall F1 and Wall IoU on both "
                "promotion seeds. The 128/64 gain does not generalize."
            ),
            "next": "Switch to FF-PC-06 predicted-control room identity and count.",
        },
        "test_opened": False,
        "claim_boundary": (
            "Internal 1024/256 validation evidence. No test or external SOTA "
            "claim."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
