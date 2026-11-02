"""Aggregate five-seed FloorFlow RPLAN structural validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 29, 37, 43, 2905)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/floorflow_mask4096_structure_fiveseed_v2.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for seed in SEEDS:
        relative = (
            "outputs/reports/"
            f"floorflow_mask4096_seed{seed}_structure_hold512_v2.json"
        )
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        runs.append(
            {
                "seed": seed,
                "report": relative,
                "successful_cases": int(payload["successful_cases"]),
                "failure_count": int(payload["failure_count"]),
                "metrics": payload["metrics"],
            }
        )
    metric_names = sorted(
        set.intersection(*(set(run["metrics"]) for run in runs))
    )
    statistics = {}
    for name in metric_names:
        values = np.asarray(
            [float(run["metrics"][name]) for run in runs],
            dtype=np.float64,
        )
        if np.allclose(values, values[0]):
            low = high = float(values[0])
        else:
            low, high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=float(values.mean()),
                scale=stats.sem(values),
            )
        statistics[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_t": [float(low), float(high)],
        }
    report = {
        "schema": "floorflow-rplan-structure-fiveseed-v2",
        "validation_cases_per_seed": 512,
        "validation_offset": 256,
        "test_opened": False,
        "mask_gap_policy": "partition_fill",
        "runs": runs,
        "statistics": statistics,
        "claim_boundary": (
            "RPLAN validation structure only; not a test, leaderboard, "
            "or external SOTA claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: statistics[name]
                for name in (
                    "room_iou",
                    "wall_f1",
                    "opening_f1",
                    "node_f1",
                    "edge_overlap",
                    "ged",
                    "validity",
                    "conditional_watertightness",
                    "dangling_endpoint_rate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
