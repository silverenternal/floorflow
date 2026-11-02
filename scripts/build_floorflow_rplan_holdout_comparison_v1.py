"""Build the matched-case FloorFlow versus CE2EPlan validation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)
FLOORFLOW_RUNS = {
    19: (
        "outputs/train/floorflow_mask_only4096_hold512_seed19_v1/report.json",
        "outputs/reports/floorflow_mask4096_seed19_canonical_hold512_v3.json",
    ),
    29: (
        "outputs/train/floorflow_mask_only4096_hold512_seed29_v1/report.json",
        "outputs/reports/floorflow_mask4096_seed29_canonical_hold512_v3.json",
    ),
    37: (
        "outputs/train/floorflow_mask_only4096_hold512_seed37_v1/report.json",
        "outputs/reports/floorflow_mask4096_seed37_canonical_hold512_v3.json",
    ),
    43: (
        "outputs/train/floorflow_mask_only4096_hold512_seed43_v1/report.json",
        "outputs/reports/floorflow_mask4096_seed43_canonical_hold512_v3.json",
    ),
    2905: (
        "outputs/train/floorflow_mask_only4096_hold512_seed2905_v1/report.json",
        "outputs/reports/floorflow_mask4096_seed2905_canonical_hold512_v3.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ce2eplan",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/rplan_ce2eplan_anchor_hold512_seed2905_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/floorflow_mask4096_fiveseed_hold512_v2.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ce2eplan = json.loads(args.ce2eplan.read_text(encoding="utf-8"))
    runs = []
    for seed, (report_name, evaluation_name) in FLOORFLOW_RUNS.items():
        report = json.loads(
            (ROOT / report_name).read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (ROOT / evaluation_name).read_text(encoding="utf-8")
        )
        arm = evaluation["arms"]["mask_partition"]
        runs.append(
            {
                "seed": seed,
                "report": report_name,
                "evaluation": evaluation_name,
                "parameters": int(report["parameter_count"]),
                "training_cases": int(report["train_cases"]),
                "epochs": int(
                    max(
                        row["epoch"]
                        for row in report["history"]
                    )
                    + 1
                ),
                "training_sample_exposures": int(
                    report["train_cases"]
                    * (
                        max(
                            row["epoch"]
                            for row in report["history"]
                        )
                        + 1
                    )
                ),
                "sampling_nfe": 1,
                "successful_cases": int(arm["successful_cases"]),
                "metrics": {
                    key: float(arm["metrics"][key]) for key in METRICS
                },
            }
        )
    mean_floorflow = {
        key: float(np.mean([run["metrics"][key] for run in runs]))
        for key in METRICS
    }
    seed_statistics = {
        key: {
            "mean": mean_floorflow[key],
            "sd": float(np.std([run["metrics"][key] for run in runs], ddof=1)),
            "min": float(min(run["metrics"][key] for run in runs)),
            "max": float(max(run["metrics"][key] for run in runs)),
            "ci95_t": [
                float(value)
                for value in stats.t.interval(
                    0.95,
                    len(runs) - 1,
                    loc=mean_floorflow[key],
                    scale=stats.sem([run["metrics"][key] for run in runs]),
                )
            ],
        }
        for key in METRICS
    }
    anchor = {
        key: float(ce2eplan["metrics"][key]) for key in METRICS
    }
    payload = {
        "schema": "floorflow-ce2eplan-same-validation-cases-v2",
        "validation_cases": int(ce2eplan["validation_cases"]),
        "validation_offset": int(ce2eplan["val_offset"]),
        "test_opened": False,
        "floorflow": {
            "runs": runs,
            "mean_metrics": mean_floorflow,
            "seed_statistics": seed_statistics,
        },
        "ce2eplan": {
            "checkpoint": ce2eplan["checkpoint_name"],
            "parameters": int(ce2eplan["parameters"]),
            "optimizer_steps_declared": int(
                ce2eplan["optimizer_steps_declared_by_checkpoint"]
            ),
            "training_sample_exposures": int(
                ce2eplan["training_sample_exposures_readme"]
            ),
            "sampling_nfe": int(ce2eplan["sampling_steps"]),
            "metrics": anchor,
        },
        "floorflow_minus_ce2eplan": {
            key: mean_floorflow[key] - anchor[key] for key in METRICS
        },
        "claim_boundary": (
            "Five seed means and t intervals on the same 512 validation "
            "cases; CE2EPlan is a released "
            "300k-step/1000-NFE pretrained anchor and this is not a "
            "matched-training or HouseMind comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["floorflow_minus_ce2eplan"]))


if __name__ == "__main__":
    main()
