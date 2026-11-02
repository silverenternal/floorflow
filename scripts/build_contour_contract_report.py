#!/usr/bin/env python3
"""Contrast the requested contour-only target with measurable data ceilings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ceiling",
        type=Path,
        default=Path("outputs/reports/room_iou_conditional_ceiling.json"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path(
            "outputs/reports/distribution_adapter_v1_publication_gate.json"
        ),
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("outputs/reports/b5_cvpr_readiness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/contour_only_contract.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    ceiling = load(args.ceiling)
    gate = load(args.gate)
    readiness = load(args.readiness)
    strict_fraction = float(
        ceiling["strict_0_95_identifiable_fraction"]
    )
    contour_metrics = {
        key: value
        for key, value in gate["metrics"].items()
        if key.startswith(("cubicasa", "skv4"))
    }
    report = {
        "status": "contract_conflict",
        "requested_contract": {
            "condition": "contour_only",
            "single_sample": True,
            "target_derived_coarse_control": False,
            "best_of_n": False,
            "all_published_metrics_at_least": 0.95,
        },
        "observed_contour_support": {
            "train_cases": ceiling["train_cases"],
            "query_cases": ceiling["query_cases"],
            "top1_room_iou": ceiling["summary"]["top1_iou"],
            "neighbour_mean_room_iou": ceiling["summary"][
                "neighbour_mean_iou"
            ],
            "neighbour_max_room_iou_mean": ceiling["summary"][
                "neighbour_max_iou"
            ],
            "fraction_with_neighbour_at_least_0_95": strict_fraction,
        },
        "current_contour_only_metrics": contour_metrics,
        "layout_conditioned_upper_bound": readiness[
            "layout_conditioned_results"
        ],
        "findings": [
            "Only 0.596% of test cases have a training neighbour at room IoU >= 0.95.",
            "Current contour-only test room IoU is about 0.126, close to the observed neighbour ceiling, not 0.95.",
            "Layout-conditioned metrics above 0.95 require target-derived coarse room/structure controls, which the main contract disables.",
            "Additional optimization budget, parameter count, or best-of-N cannot resolve missing input information.",
        ],
        "supported_conclusion": (
            "The current contour-only, single-sample, target-control-free contract "
            "has no empirical support for a 0.95 exact-reconstruction claim."
        ),
        "allowed_next_contracts": [
            "Add an independently predicted or user-provided sparse structural prior as a separately reported condition.",
            "Evaluate a distributional multi-hypothesis claim without selecting by target.",
            "Retain exact reconstruction as an upper-bound experiment with coarse control and do not report it as contour-only.",
        ],
        "selected_contract": "none_without_user_decision",
        "sources": {
            "ceiling": str(args.ceiling),
            "gate": str(args.gate),
            "readiness": str(args.readiness),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
