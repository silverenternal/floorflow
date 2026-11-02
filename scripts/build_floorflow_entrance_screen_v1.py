"""Aggregate the matched learned-entrance-condition architecture screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2905, 37)
QUALITY = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)
STRUCTURE = (
    "room_iou",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "conditional_watertightness",
    "dangling_endpoint_rate",
)
LOWER_IS_BETTER = {
    "count_abs_error",
    "dangling_endpoint_rate",
    "ged",
    "wall_endpoint_error",
}


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _canonical(seed: int, arm: str) -> tuple[str, dict]:
    path = (
        f"outputs/reports/floorflow_{arm}_mask1024_seed{seed}_canonical_v"
        f"{'2' if arm == 'geometry_control' else '1'}.json"
    )
    return path, _read(path)["arms"]["mask_partition"]["metrics"]


def _heads(seed: int, arm: str) -> tuple[str, dict]:
    path = (
        f"outputs/reports/floorflow_{arm}_mask1024_seed{seed}_cad_heads_v"
        f"{'2' if arm == 'geometry_control' else '1'}.json"
    )
    return path, _read(path)["metrics"]


def _structure(seed: int, arm: str) -> tuple[str, dict]:
    path = (
        f"outputs/reports/floorflow_{arm}_mask1024_seed{seed}_structure_v"
        f"{'2' if arm == 'geometry_control' else '1'}.json"
    )
    return path, _read(path)["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/floorflow_entrance_condition_twoseed_v1.json"
        ),
    )
    args = parser.parse_args()
    runs = []
    for seed in SEEDS:
        control_quality_report, control_quality = _canonical(
            seed,
            "geometry_control",
        )
        entrance_quality_report, entrance_quality = _canonical(
            seed,
            "entrance",
        )
        control_heads_report, control_heads = _heads(seed, "geometry_control")
        entrance_heads_report, entrance_heads = _heads(seed, "entrance")
        control_structure_report, control_structure = _structure(
            seed,
            "geometry_control",
        )
        entrance_structure_report, entrance_structure = _structure(
            seed,
            "entrance",
        )
        runs.append(
            {
                "seed": seed,
                "quality": {
                    "control_report": control_quality_report,
                    "entrance_report": entrance_quality_report,
                    "control": control_quality,
                    "entrance": entrance_quality,
                    "delta": {
                        key: entrance_quality[key] - control_quality[key]
                        for key in QUALITY
                    },
                },
                "raw_cad": {
                    "control_report": control_heads_report,
                    "entrance_report": entrance_heads_report,
                    "control": control_heads,
                    "entrance": entrance_heads,
                    "delta": {
                        key: entrance_heads[key] - control_heads[key]
                        for key in control_heads
                    },
                },
                "structure": {
                    "control_report": control_structure_report,
                    "entrance_report": entrance_structure_report,
                    "control": control_structure,
                    "entrance": entrance_structure,
                    "delta": {
                        key: entrance_structure[key] - control_structure[key]
                        for key in STRUCTURE
                    },
                },
            }
        )
    summary = {}
    for section, metrics in (
        ("quality", QUALITY),
        ("structure", STRUCTURE),
        ("raw_cad", ("wall_f1", "wall_endpoint_error")),
    ):
        summary[section] = {}
        for key in metrics:
            deltas = np.asarray(
                [run[section]["delta"][key] for run in runs],
                dtype=np.float64,
            )
            summary[section][key] = {
                "mean_delta": float(deltas.mean()),
                "sd_delta": float(deltas.std(ddof=1)),
                "improvements": int(
                    (
                        deltas < 0.0
                        if key in LOWER_IS_BETTER
                        else deltas > 0.0
                    ).sum()
                ),
                "seed_deltas": [float(value) for value in deltas],
            }
    report = {
        "schema": "floorflow-entrance-condition-twoseed-v1",
        "seeds": list(SEEDS),
        "test_opened": False,
        "runs": runs,
        "summary": summary,
        "decision": (
            "Retain the learned entrance token only as room identity, type, "
            "and count conditioning; raw wall/opening heads are not promoted."
        ),
        "claim_boundary": (
            "Matched train1024/validation256 architecture screen. "
            "Condition-derived canonical opening recall is not a learned-head "
            "result."
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
