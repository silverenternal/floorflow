"""Summarize the two-seed owner-edge wall action screen."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
OUTPUT = REPORTS / "floorflow_owner_wall_action_twoseed_128_v1.json"

ARMS = {
    2905: {
        "control": REPORTS / "floorflow_cadaction_control128_fusion025_audit_v1.json",
        "candidate": REPORTS / "floorflow_owner_wall_action128_seed2905_structure_t030_fusion025_v1.json",
        "action_training": ROOT / "outputs/train/floorflow_owner_wall_action128_seed2905_v4/report.json",
    },
    37: {
        "control": REPORTS / "floorflow_cadaction_control128_seed37_fusion025_structure_v1.json",
        "candidate": REPORTS / "floorflow_owner_wall_action128_seed37_fusion025_structure_t030_v1.json",
        "action_training": ROOT / "outputs/train/floorflow_owner_wall_action128_seed37_v1/report.json",
    },
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
    "cad_stage_exit_count",
    "cad_stage_incomplete_count",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs = []
    for seed, paths in ARMS.items():
        control = _read(paths["control"])["metrics"]
        candidate = _read(paths["candidate"])["metrics"]
        training = _read(paths["action_training"])
        runs.append(
            {
                "seed": seed,
                "control": {key: float(control[key]) for key in KEYS},
                "candidate": {key: float(candidate[key]) for key in KEYS},
                "delta": {
                    key: float(candidate[key] - control[key])
                    for key in KEYS
                },
                "connector_length_ratio": float(
                    candidate["wall_connector_length_ratio"]
                ),
                "wall_count": float(candidate["wall_count"]),
                "action_parameter_count": int(training["parameter_count"]),
            }
        )
    summary = {
        key: {
            "control_mean": sum(run["control"][key] for run in runs)
            / len(runs),
            "candidate_mean": sum(run["candidate"][key] for run in runs)
            / len(runs),
            "delta_mean": sum(run["delta"][key] for run in runs)
            / len(runs),
            "improved_seeds": sum(
                (
                    run["delta"][key] < 0.0
                    if key in {"ged", "dangling_endpoint_rate"}
                    else run["delta"][key] > 0.0
                )
                for run in runs
            ),
        }
        for key in KEYS
    }
    payload = {
        "schema": "floorflow-owner-wall-action-screen-v1",
        "seeds": [37, 2905],
        "train_cases": 128,
        "validation_cases": 64,
        "epochs": 8,
        "nfe": 1,
        "action": {
            "module": "CadWallCandidateScorer with encoded score",
            "candidate_source": "merged owner 1-cells",
            "selection": "maximum spanning forest plus confidence threshold",
            "threshold": 0.3,
            "endpoint_residual": "disabled for topology path",
            "exterior": "forced retained",
            "neutral_evidence_prior": 0.5,
        },
        "runs": runs,
        "summary": summary,
        "decision": {
            "status": "pass_two_seed_mechanism",
            "promotion": "eligible_but_effect_is_small",
            "reason": "Wall F1 and Wall IoU improve on both seeds while every room, graph, validity, watertightness, and opening metric is unchanged.",
            "limitation": "Mean gain is about 0.0014 Wall F1 and 0.0005 Wall IoU; connectors remain 4.9-8.3% of wall length.",
        },
        "test_opened": False,
        "claim_boundary": (
            "Internal 128/64 mechanism evidence. No test, external SOTA, or "
            "complete-CAD claim."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
