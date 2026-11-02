#!/usr/bin/env python3
"""Promote the frozen room readout through the page-floor wall consumer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "room_macro_iou": False,
    "instance_f1": False,
    "type_accuracy": False,
    "count_abs_error": True,
    "wall_precision": False,
    "wall_recall": False,
    "wall_f1": False,
    "wall_continuity_rate": False,
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison(path: Path) -> dict:
    rows = _read(path)["case_rows"]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["arm"]), []).append(row)
    control = grouped["dense_boundary"]
    candidate = grouped["query_boundary"]
    paired = {
        key: paired_delta(
            candidate,
            control,
            key,
            iterations=5000,
            seed=17,
            lower_is_better=lower_is_better,
        )
        for key, lower_is_better in KEYS.items()
    }
    summary = _read(path)["summary"]
    safety = {
        key: summary["query_boundary"][key]["mean"]
        for key in (
            "validity",
            "functional_watertight",
            "outside_footprint_fraction",
        )
    }
    return {"paired": paired, "safety": safety}


def _split_passed(comparison: dict) -> bool:
    paired = comparison["paired"]
    safety = comparison["safety"]
    return bool(
        paired["room_macro_iou"]["ci_low"] > 0.0
        and paired["instance_f1"]["ci_low"] > 0.0
        and paired["count_abs_error"]["ci_high"] < 0.0
        and paired["wall_precision"]["ci_low"] > 0.0
        and paired["wall_f1"]["ci_low"] > 0.0
        and paired["wall_continuity_rate"]["ci_low"] > -0.01
        and safety["validity"] == 1.0
        and safety["functional_watertight"] >= 0.99
        and safety["outside_footprint_fraction"] == 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cal",
        type=Path,
        default=ROOT / "outputs/reports/room_wall_pagefloor_fusion_cal64_v2.json",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_wall_pagefloor_fusion_holdout64_v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_wall_pagefloor_promotion_v1.json"
        ),
    )
    args = parser.parse_args()
    calibration = _comparison(args.cal)
    holdout = _comparison(args.holdout)
    passed = _split_passed(calibration) and _split_passed(holdout)
    report = {
        "schema_version": "room-wall-pagefloor-promotion-v1",
        "status": "passed" if passed else "rejected",
        "protocol": "pagefloor_native_cubicasa_256",
        "candidate": "query_boundary",
        "control": "dense_boundary",
        "cal64": calibration,
        "holdout64": holdout,
        "decision": (
            "promote_room_to_wall_consumer"
            if passed
            else "reject_room_to_wall_consumer"
        ),
        "negated_arm": {
            "name": "query_wall_gated_0.5",
            "reason": (
                "Wall F1 rises by deleting or suppressing room seams, but mean "
                "wall continuity falls below the safety margin."
            ),
        },
        "boundary": (
            "This is a consumed room-to-wall mechanism result on the "
            "page-floor protocol, not a full CAD or external SOTA claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
