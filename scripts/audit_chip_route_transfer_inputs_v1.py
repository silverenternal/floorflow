#!/usr/bin/env python3
"""Audit available labels for a cross-stage chip route-transfer experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = (
    "globalplace__route__wirelength__estimated",
    "globalplace__gpl__routability__congestion",
    "globalroute__timing__setup__ws",
    "globalroute__timing__setup__tns",
    "detailedroute__route__wirelength",
    "detailedroute__route__drc_errors",
    "finish__timing__setup__ws",
    "finish__power__total",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orfs-root",
        type=Path,
        default=ROOT / "outputs/reports/orfs",
    )
    parser.add_argument(
        "--def-root",
        type=Path,
        default=ROOT / "outputs/chip_ppa/orfs/results",
    )
    parser.add_argument(
        "--external-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_e1_external_holdout_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_route_transfer_inputs_v1.json",
    )
    args = parser.parse_args()
    metrics_paths = sorted(args.orfs_root.glob("**/metrics_2_1_2.json"))
    coverage = Counter()
    complete = []
    designs = set()
    for path in metrics_paths:
        values = load(path)
        present = [
            key
            for key in REQUIRED_METRICS
            if values.get(key) not in (None, "N/A", "ERR")
        ]
        coverage[len(present)] += 1
        relative = path.relative_to(args.orfs_root).parts
        if len(relative) >= 3:
            designs.add("/".join(relative[:2]))
        if len(present) == len(REQUIRED_METRICS):
            complete.append(str(path.relative_to(ROOT)))
    def_files = sorted(args.def_root.glob("**/6_final.def"))
    external = load(args.external_report)
    report = {
        "version": "chip-route-transfer-inputs-v1",
        "status": (
            "calibration_ready_confirmation_missing"
            if complete and external["designs"] < 15
            else "ready"
        ),
        "floor_set_training_graph": {
            "available": "pairwise b2b and pin-to-block incidence",
            "true_hypergraph_training_labels": False,
            "optional_external_hypergraph_api": [
                "src/floorflow/chip_flow/netlist_hypergraph.py",
                "src/floorflow/chip_flow/external_graph_conditioning.py",
                "src/floorflow/chip_flow/def_graph.py",
            ],
        },
        "orfs": {
            "metrics_files": len(metrics_paths),
            "complete_metric_files": len(complete),
            "coverage_histogram": dict(sorted(coverage.items())),
            "unique_metric_designs": len(designs),
            "final_def_files": len(def_files),
            "required_metrics": list(REQUIRED_METRICS),
            "complete_examples": complete[:20],
        },
        "external_cohort": {
            "designs": external["designs"],
            "completed_pairs": external["completed_pairs"],
            "drc_zero_both_arms": external["checks"]["drc_zero_both_arms"],
            "ppa_promotable": external["ppa_promotable"],
            "routed_wirelength_relative": external["summary"][
                "primary_relative_wirelength_mean"
            ],
            "congestion_relative": external["summary"][
                "congestion_relative_mean"
            ],
        },
        "reuse": {
            "label_parser": "src/floorflow/chip_flow/analytical_feedback.py",
            "graph_parser": "src/floorflow/chip_flow/def_graph.py",
            "external_screen": "src/floorflow/chip_flow/external_portfolio.py",
            "metric_contract": "src/floorflow/chip_flow/ppa_external.py",
            "inventory": "outputs/reports/chip_external_inventory_v45_expanded_audit.json",
        },
        "decision": {
            "calibration_supported": bool(complete),
            "confirmation_supported": external["designs"] >= 15,
            "do_not_pool_seeds_as_designs": True,
            "do_not_train_a_route_head_without_true_hypergraph_or_cross_stage_labels": True,
        },
        "next_cpu_action": (
            "Build a provenance manifest keyed by unique design, proposal, seed, "
            "placement TCL hash and route recipe; retain seeds as repeated "
            "measurements and reserve a new 15-20 design cohort."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
