#!/usr/bin/env python3
"""Summarize the frozen normalized-routing 5k screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float | None:
    return candidate / control - 1.0 if control != 0.0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_routing_norm_control_seed23_5k_group2_dev.json",
    )
    parser.add_argument(
        "--route",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_routing_norm_route_seed23_5k_group2_dev.json",
    )
    parser.add_argument(
        "--proxy",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_routing_norm_seed23_5k_proxy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_routing_norm_screen.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_report = load(args.control)
    route_report = load(args.route)
    proxy_report = load(args.proxy)
    control = control_report["results"]["nfe4_refine1600"]
    route = route_report["results"]["nfe4_refine1600"]
    fields = (
        "legalized_hard_cost",
        "legalized_v_rel",
        "legalized_hpwl_gap",
        "legalized_area_gap",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_grouping",
        "legalized_mib",
        "legalized_boundary",
        "legalized_rho_mae",
        "legalized_feasible_fraction",
    )
    relative_metrics = {
        field: relative(float(route[field]), float(control[field]))
        for field in fields
    }
    proxy_relative = proxy_report["relative_to_control"]["route"]
    cost_gate = relative_metrics["legalized_hard_cost"] < 0.0
    v_rel_gate = relative_metrics["legalized_v_rel"] < 0.0
    proxy_gate = proxy_relative["normalized_composite"] < 0.0
    feasibility_gate = (
        float(route["legalized_feasible_fraction"]) == 1.0
        and float(control["legalized_feasible_fraction"]) == 1.0
    )
    gates = {
        "cost": cost_gate,
        "v_rel": v_rel_gate,
        "normalized_proxy": proxy_gate,
        "feasibility": feasibility_gate,
    }
    promoted = all(gates.values())
    report = {
        "version": "2026-10-29-chip-v45-routing-norm-screen-v1",
        "status": "promote_to_20k" if promoted else "rejected_5k",
        "scope": (
            "Frozen 32-case legalized dev screen, training/data seed 23, "
            "single inference seed 19. Official-100 was not inspected."
        ),
        "sources": {
            "control": str(args.control.relative_to(ROOT)),
            "route": str(args.route.relative_to(ROOT)),
            "proxy": str(args.proxy.relative_to(ROOT)),
        },
        "control": {field: control[field] for field in fields},
        "route": {field: route[field] for field in fields},
        "relative_to_control": relative_metrics,
        "proxy_relative_to_control": proxy_relative,
        "gates": gates,
        "decision": (
            "The route arm reduces every proxy component but fails the legalized "
            "cost gate and provides only a negligible V_rel change. Do not spend "
            "20k compute on this arm."
        ),
        "retained_negative": (
            "A lower routing proxy remains insufficient to imply lower legalized "
            "cost or routed wirelength. Route-demand proxy improvements must be "
            "validated end to end on untouched designs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
