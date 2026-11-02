"""Aggregate the compact partition-edge CAD architecture screen."""

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
RAW = ("wall_f1", "wall_endpoint_error")
LOWER_IS_BETTER = {
    "count_abs_error",
    "dangling_endpoint_rate",
    "ged",
    "wall_endpoint_error",
}


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_cad64_twoseed_v1.json",
    )
    args = parser.parse_args()
    runs = []
    for seed in SEEDS:
        prefix = "outputs/reports/floorflow_cad64"
        quality_paths = {
            arm: f"{prefix}_{arm}1024_seed{seed}_canonical_v1.json"
            for arm in ("control", "edge")
        }
        head_paths = {
            arm: f"{prefix}_{arm}1024_seed{seed}_cad_heads_v1.json"
            for arm in ("control", "edge")
        }
        structure_paths = {
            arm: f"{prefix}_{arm}1024_seed{seed}_structure_v1.json"
            for arm in ("control", "edge")
        }
        quality = {
            arm: _read(path)["arms"]["mask_partition"]["metrics"]
            for arm, path in quality_paths.items()
        }
        heads = {
            arm: _read(path)["metrics"] for arm, path in head_paths.items()
        }
        structure = {
            arm: _read(path)["metrics"]
            for arm, path in structure_paths.items()
        }
        runs.append(
            {
                "seed": seed,
                "quality": {
                    "reports": quality_paths,
                    "metrics": quality,
                    "delta": {
                        key: quality["edge"][key] - quality["control"][key]
                        for key in QUALITY
                    },
                },
                "raw_cad": {
                    "reports": head_paths,
                    "metrics": heads,
                    "delta": {
                        key: heads["edge"][key] - heads["control"][key]
                        for key in RAW
                    },
                },
                "structure": {
                    "reports": structure_paths,
                    "metrics": structure,
                    "delta": {
                        key: structure["edge"][key]
                        - structure["control"][key]
                        for key in STRUCTURE
                    },
                },
            }
        )
    summary = {}
    for section, keys in (
        ("quality", QUALITY),
        ("structure", STRUCTURE),
        ("raw_cad", RAW),
    ):
        summary[section] = {}
        for key in keys:
            control = np.asarray(
                [run[section]["metrics"]["control"][key] for run in runs]
            )
            edge = np.asarray(
                [run[section]["metrics"]["edge"][key] for run in runs]
            )
            delta = edge - control
            summary[section][key] = {
                "control_mean": float(control.mean()),
                "edge_mean": float(edge.mean()),
                "mean_delta": float(delta.mean()),
                "seed_deltas": [float(value) for value in delta],
                "improvements": int(
                    (
                        delta < 0.0
                        if key in LOWER_IS_BETTER
                        else delta > 0.0
                    ).sum()
                ),
            }
    report = {
        "schema": "floorflow-cad64-partition-edge-twoseed-v1",
        "seeds": list(SEEDS),
        "training": {
            "train_cases": 1024,
            "validation_cases": 256,
            "epochs": 12,
            "batch_size": 8,
            "wall_queries": 64,
            "opening_queries": 4,
        },
        "test_opened": False,
        "runs": runs,
        "summary": summary,
        "decision": (
            "Promote to 4096 as a graph/geometry candidate. Do not claim "
            "robust raw-wall superiority because seed 37 regresses, and do "
            "not promote learned openings."
        ),
        "claim_boundary": (
            "Matched small-split architecture screen; canonical opening F1 is "
            "condition-derived, not a learned-opening result."
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
