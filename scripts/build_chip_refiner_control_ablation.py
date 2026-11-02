#!/usr/bin/env python3
"""Compare typed hierarchy and graph-token control at the frozen refiner budget."""

from __future__ import annotations

import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_area_gap",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    candidate_path = (
        ROOT
        / "outputs/chip_hierarchy/v44/refiner_official/"
        "candidate_boundary2_detglop_1600_seed19.json"
    )
    control_path = (
        ROOT
        / "outputs/chip_hierarchy/v44/control_official/"
        "control_boundary2_detglop_1600_seed19.json"
    )
    candidate = read(candidate_path)
    control = read(control_path)
    candidate_rows = read(
        candidate_path.with_name(f"{candidate_path.stem}_cases.json")
    )["nfe4_refine1600"]
    control_rows = read(
        control_path.with_name(f"{control_path.stem}_cases.json")
    )["nfe4_refine1600"]
    paired = {
        metric: paired_delta(
            candidate_rows,
            control_rows,
            metric,
            iterations=10000,
            seed=20270000 + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(METRICS)
    }
    report = {
        "status": (
            "candidate_wins_at_frozen_refiner"
            if paired["legalized_hard_cost"]["mean_delta"] < 0.0
            and paired["legalized_v_rel"]["mean_delta"] < 0.0
            else "control_wins_or_ties_at_frozen_refiner"
        ),
        "scope": "Seed 19, Official-100, NFE4 + boundary=2 + 1600-step deterministic GLOP1 refiner",
        "candidate": {
            "path": str(candidate_path.relative_to(ROOT)),
            "cost": float(candidate["results"]["nfe4_refine1600"]["legalized_hard_cost"]),
            "v_rel": float(candidate["results"]["nfe4_refine1600"]["legalized_v_rel"]),
        },
        "control": {
            "path": str(control_path.relative_to(ROOT)),
            "cost": float(control["results"]["nfe4_refine1600"]["legalized_hard_cost"]),
            "v_rel": float(control["results"]["nfe4_refine1600"]["legalized_v_rel"]),
        },
        "paired": paired,
        "boundary": (
            "The frozen refiner is a strong inference-time equalizer. A control win "
            "means typed hierarchy cannot be credited for the fixed-anchor gain."
        ),
    }
    output = ROOT / "outputs/reports/chip_refiner_control_ablation_seed19_v44.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": report["candidate"],
                "control": report["control"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
