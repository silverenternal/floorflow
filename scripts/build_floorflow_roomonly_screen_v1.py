"""Aggregate the partition-only objective screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2905, 37)
METRICS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)
LOWER_IS_BETTER = {"count_abs_error"}


def _metrics(path: str) -> dict:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return payload["arms"]["mask_partition"]["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_roomonly_twoseed_v1.json",
    )
    args = parser.parse_args()
    runs = []
    for seed in SEEDS:
        control_path = (
            "outputs/reports/"
            f"floorflow_cad64_control1024_seed{seed}_canonical_v1.json"
        )
        roomonly_path = (
            "outputs/reports/"
            f"floorflow_roomonly1024_seed{seed}_canonical_v1.json"
        )
        control = _metrics(control_path)
        roomonly = _metrics(roomonly_path)
        runs.append(
            {
                "seed": seed,
                "control_report": control_path,
                "roomonly_report": roomonly_path,
                "control": control,
                "roomonly": roomonly,
                "delta": {
                    key: roomonly[key] - control[key] for key in METRICS
                },
            }
        )
    summary = {}
    for key in METRICS:
        deltas = np.asarray(
            [run["delta"][key] for run in runs],
            dtype=np.float64,
        )
        summary[key] = {
            "control_mean": float(
                np.mean([run["control"][key] for run in runs])
            ),
            "roomonly_mean": float(
                np.mean([run["roomonly"][key] for run in runs])
            ),
            "mean_delta": float(deltas.mean()),
            "seed_deltas": [float(value) for value in deltas],
            "improvements": int(
                (
                    deltas < 0.0 if key in LOWER_IS_BETTER else deltas > 0.0
                ).sum()
            ),
        }
    report = {
        "schema": "floorflow-roomonly-objective-twoseed-v1",
        "seeds": list(SEEDS),
        "test_opened": False,
        "runs": runs,
        "summary": summary,
        "decision": (
            "Promote to 4096 only because type accuracy improves on both "
            "seeds and IoU is neutral; count remains a monitored regression."
        ),
        "claim_boundary": (
            "The deployed wall and opening layers are deterministic, so their "
            "training losses are not part of this objective screen."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
