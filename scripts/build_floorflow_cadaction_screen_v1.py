"""Summarize the finite CAD-action conditioning mechanism screen."""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs/reports"
OUTPUT = REPORT_ROOT / "floorflow_cadaction_screen128_v1.json"

CONTROL_CHECKPOINT = (
    ROOT / "outputs/train/floorflow_cadaction_control128_seed2905_v1/best.pt"
)
WARM_CHECKPOINT = (
    ROOT
    / "outputs/train/floorflow_cadaction_warmstage128_seed2905_v3/best.pt"
)

ARMS = {
    "control": REPORT_ROOT / "floorflow_cadaction_control128_audit_v1.json",
    "control_fusion": (
        REPORT_ROOT / "floorflow_cadaction_control128_fusion025_audit_v1.json"
    ),
    "warmstage": (
        REPORT_ROOT / "floorflow_cadaction_warmstage128_v3_audit_v1.json"
    ),
    "warmstage_fusion": (
        REPORT_ROOT
        / "floorflow_cadaction_warmstage128_v3_fusion025_audit_v1.json"
    ),
}

METRICS = (
    "room_iou",
    "instance_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "conditional_watertightness",
    "dangling_endpoint_rate",
    "wall_f1",
    "wall_iou",
    "opening_f1",
    "cad_stage_exit_count",
    "cad_stage_incomplete_count",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    arms = {
        name: _read(path)["metrics"]
        for name, path in ARMS.items()
    }
    control = torch.load(
        CONTROL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    warm = torch.load(
        WARM_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    changed_non_stage = []
    non_stage_tensors = 0
    for name, value in control["model"].items():
        if "stage_conditioners." in name:
            continue
        non_stage_tensors += 1
        if name not in warm["model"] or not torch.equal(
            value,
            warm["model"][name],
        ):
            changed_non_stage.append(name)
    training = _read(
        ROOT
        / "outputs/train/floorflow_cadaction_warmstage128_seed2905_v3/report.json"
    )
    payload = {
        "schema": "floorflow-finite-cad-action-screen-v1",
        "seed": 2905,
        "train_cases": 128,
        "validation_cases": 64,
        "epochs": 8,
        "nfe": 1,
        "arms": {
            name: {key: float(metrics[key]) for key in METRICS}
            for name, metrics in arms.items()
        },
        "delta": {
            "warmstage_minus_control": {
                key: float(arms["warmstage"][key] - arms["control"][key])
                for key in METRICS
            },
            "warmstage_fusion_minus_control_fusion": {
                key: float(
                    arms["warmstage_fusion"][key]
                    - arms["control_fusion"][key]
                )
                for key in METRICS
            },
            "fusion_gain_control": {
                key: float(
                    arms["control_fusion"][key] - arms["control"][key]
                )
                for key in METRICS
            },
            "fusion_gain_warmstage": {
                key: float(
                    arms["warmstage_fusion"][key] - arms["warmstage"][key]
                )
                for key in METRICS
            },
        },
        "weight_invariant": {
            "non_stage_tensors": non_stage_tensors,
            "changed_non_stage_tensors": len(changed_non_stage),
            "changed_names": changed_non_stage[:20],
            "trainable_parameter_count": int(
                training["trainable_parameter_count"]
            ),
            "total_parameter_count": int(training["parameter_count"]),
        },
        "stage_audit": {
            "stages_per_output": 9,
            "incomplete_stages": 0,
            "token_roundtrip_smoke": "outputs/train/floorflow_cadaction_contract_smoke_seed2905_v3/report.json",
        },
        "decision": {
            "stage_film": "diagnosed_neutral",
            "learned_wall_fusion": "small_positive_but_stage_independent",
            "promotion": False,
            "reason": (
                "Stage-only training leaves every non-stage tensor identical, "
                "but its fusion result remains slightly below the matched "
                "control+fusion arm. Do not sweep FiLM scales further."
            ),
            "next": (
                "Reuse CadWallCandidateScorer as an owner-edge action selector "
                "trained with matched edge targets before adding another head."
            ),
        },
        "test_opened": False,
        "claim_boundary": (
            "Internal 128/64 mechanism screen. No test, SOTA, or complete-CAD "
            "claim."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["delta"], indent=2))


if __name__ == "__main__":
    main()
