"""Summarize CAGN embedding, gate, scratch, and joint-proxy screens."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_cagn_training_screen_v1.json"


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    embedding = _read(
        "outputs/reports/residential_constraint_embedding_v1.json"
    )
    gate = _read(
        "outputs/reports/residential_cagn_validation_v3_seed19.json"
    )
    scratch = _read(
        "outputs/reports/residential_cagn_validation_v3_scratch_seed19.json"
    )
    history = _read(
        "outputs/train/residential_cagn_joint_screen_v1/history.json"
    )
    joint = history[-1]
    soft_keys = [
        key for key in joint if key.startswith("train_soft_")
    ]
    checks = {
        "embedding_passed": embedding["status"] == "passed",
        "full_gate_passed": gate["status"] == "passed",
        "scratch_gate_recorded": scratch["status"] in {"passed", "failed"},
        "joint_soft_terms_finite": all(
            abs(float(joint[key])) < 1.0e12 for key in soft_keys
        ),
        "joint_valid_consistency_exact_zero": (
            float(joint["train_valid_consistency"]) == 0.0
        ),
        "no_nan_or_inf": all(
            abs(float(value)) < 1.0e12
            for key, value in joint.items()
            if isinstance(value, (int, float))
        ),
    }
    report = {
        "schema_version": "residential-cagn-training-screen-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "device": "cuda:0",
        "embedding": {
            "status": embedding["status"],
            "auroc": embedding["evaluation"]["auroc"],
            "ffr": embedding["evaluation"]["false_feasible_rate"],
            "ece": embedding["evaluation"]["ece"],
        },
        "full_gate": {
            "status": gate["status"],
            "ndcg_at_5": gate["learned_gate"]["ndcg_at_5"],
            "selected_violation_rate": (
                gate["learned_gate"]["selected_violation_rate"]
            ),
            "ece": gate["calibration"]["ece"],
        },
        "scratch_gate": {
            "status": scratch["status"],
            "ndcg_at_5": scratch["learned_gate"]["ndcg_at_5"],
            "selected_violation_rate": (
                scratch["learned_gate"]["selected_violation_rate"]
            ),
        },
        "joint_proxy_screen": {
            "epochs": len(history),
            "train_loss": joint["train_loss"],
            "soft_fevp": joint["train_soft_fevp"],
            "valid_consistency": joint["train_valid_consistency"],
            "saved_checkpoint": (
                "outputs/train/residential_cagn_joint_screen_v1/best.pt"
            ),
        },
        "checks": checks,
        "guard_boundary": (
            "The one-epoch proxy screen verifies finite gradients and an "
            "unchanged hard projector. Exact FEVP validity and outside "
            "preservation are evaluated in the frozen CAGN-09 lockbox, not "
            "inferred from soft terms."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
