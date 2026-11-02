#!/usr/bin/env python3
"""Compare matched mechanism-screen reports on the frozen dev subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from floorflow.chip_flow.mechanism_screening import (
    claim_boundary,
    infer_stage,
    screen_status,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--setting", default="nfe4_refine400")
    parser.add_argument(
        "--stage",
        choices=("quick", "confirmation", "full"),
        default=None,
        help="Explicit screening stage; inferred from the setting when omitted.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float | None:
    return candidate / control - 1.0 if control != 0.0 else None


def main() -> None:
    args = parse_args()
    control_path = (
        args.control if args.control.is_absolute() else ROOT / args.control
    ).resolve()
    mechanism_path = (
        args.mechanism
        if args.mechanism.is_absolute()
        else ROOT / args.mechanism
    ).resolve()
    control = load(control_path)["results"][args.setting]
    mechanism = load(mechanism_path)["results"][args.setting]
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
    comparison = {
        field: {
            "control": control[field],
            "mechanism": mechanism[field],
            "relative": relative(float(mechanism[field]), float(control[field])),
        }
        for field in fields
    }
    checks = {
        "cost_improves": comparison["legalized_hard_cost"]["relative"] is not None
        and comparison["legalized_hard_cost"]["relative"] < 0.0,
        "v_rel_improves": comparison["legalized_v_rel"]["relative"] is not None
        and comparison["legalized_v_rel"]["relative"] < 0.0,
        "feasible_fraction_one": (
            mechanism["legalized_feasible_fraction"] == 1.0
            and control["legalized_feasible_fraction"] == 1.0
        ),
    }
    stage = args.stage or infer_stage(args.setting)
    report = {
        "version": "chip-m5-screen-v1",
        "mechanism_id": args.mechanism_id,
        "status": screen_status(stage, all(checks.values())),
        "stage": stage,
        "scope": f"32-case frozen legalized dev; {args.setting}; stage={stage}.",
        "sources": {
            "control": str(control_path.relative_to(ROOT)),
            "mechanism": str(mechanism_path.relative_to(ROOT)),
        },
        "comparison": comparison,
        "checks": checks,
        "claim_boundary": claim_boundary(stage),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
