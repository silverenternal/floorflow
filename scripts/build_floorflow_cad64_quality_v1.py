"""Aggregate five-seed compact-CAD RPLAN quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 29, 37, 43, 2905)
METRICS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_cad64_quality_fiveseed_v1.json",
    )
    args = parser.parse_args()
    anchor = json.loads(
        (
            ROOT
            / "outputs/reports/rplan_ce2eplan_anchor_hold512_seed2905_v1.json"
        ).read_text(encoding="utf-8")
    )
    runs = []
    for seed in SEEDS:
        evaluation_path = (
            "outputs/reports/"
            f"floorflow_cad64_control4096_seed{seed}_canonical_v1.json"
        )
        training_path = (
            "outputs/train/"
            f"floorflow_cad64_control4096_hold512_seed{seed}_v1/report.json"
        )
        evaluation = json.loads(
            (ROOT / evaluation_path).read_text(encoding="utf-8")
        )
        training = json.loads((ROOT / training_path).read_text(encoding="utf-8"))
        arm = evaluation["arms"]["mask_partition"]
        runs.append(
            {
                "seed": seed,
                "evaluation": evaluation_path,
                "training": training_path,
                "parameters": int(training["parameter_count"]),
                "training_sample_exposures": int(
                    training["train_cases"]
                    * (max(row["epoch"] for row in training["history"]) + 1)
                ),
                "sampling_nfe": 1,
                "canonical_success": int(arm["successful_cases"]),
                "token_roundtrip_exact_rate": float(
                    arm["token_roundtrip_exact_rate"]
                ),
                "metrics": arm["metrics"],
            }
        )
    statistics = {}
    for key in METRICS:
        values = np.asarray(
            [float(run["metrics"][key]) for run in runs],
            dtype=np.float64,
        )
        low, high = stats.t.interval(
            0.95,
            len(values) - 1,
            loc=float(values.mean()),
            scale=stats.sem(values),
        )
        statistics[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_t": [float(low), float(high)],
            "ce2eplan_anchor": float(anchor["metrics"][key]),
            "mean_minus_anchor": float(values.mean() - anchor["metrics"][key]),
        }
    report = {
        "schema": "floorflow-cad64-quality-fiveseed-v1",
        "validation_cases_per_seed": 512,
        "validation_offset": 256,
        "test_opened": False,
        "runs": runs,
        "statistics": statistics,
        "claim_boundary": (
            "Five-seed validation evidence for the compact 64-query CAD bank; "
            "CE2EPlan remains a disclosed pretrained external anchor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
