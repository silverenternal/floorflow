#!/usr/bin/env python3
"""Compare one candidate/control PlaceReasoner routed-PPA pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "route_complete":
        raise ValueError(f"{path} is not route_complete")
    if not payload.get("metrics"):
        raise ValueError(f"{path} has no routed metrics")
    return payload


def _relative(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return value / reference - 1.0


def main() -> None:
    args = parse_args()
    candidate = _load(args.candidate)
    control = _load(args.control)
    for key in ("task", "repo_commit", "dataset_id", "macro_count"):
        if candidate.get(key) != control.get(key):
            raise ValueError(f"pair mismatch on {key}")
    if candidate["input_sha256"] != control["input_sha256"]:
        raise ValueError("pair input hashes differ")
    if candidate["coordinates_sha256"] == control["coordinates_sha256"]:
        raise ValueError("candidate/control coordinates collapsed")

    candidate_metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    relative = {
        key: _relative(float(candidate_metrics[key]), float(control_metrics[key]))
        for key in ("wirelength", "power", "area")
    }
    absolute = {
        "wns_ns": float(candidate_metrics["wns"]) - float(control_metrics["wns"]),
        "tns_ns": float(candidate_metrics["tns"]) - float(control_metrics["tns"]),
        "drc": int(candidate_metrics["drc"]) - int(control_metrics["drc"]),
    }
    report = {
        "version": "chip-placereasoner-pair-v1",
        "status": "pilot_pass_not_statistical",
        "task": candidate["task"],
        "repo_commit": candidate["repo_commit"],
        "dataset_id": candidate["dataset_id"],
        "coordinate_hashes": {
            "candidate": candidate["coordinates_sha256"],
            "control": control["coordinates_sha256"],
            "distinct": True,
        },
        "input_sha256": candidate["input_sha256"],
        "candidate": candidate_metrics,
        "control": control_metrics,
        "relative": relative,
        "absolute_delta": absolute,
        "checks": {
            "both_drc_clean": int(candidate_metrics["drc"]) == 0
            and int(control_metrics["drc"]) == 0,
            "candidate_drc_nonregression": int(candidate_metrics["drc"])
            <= int(control_metrics["drc"]),
            "candidate_wirelength_nonregression": relative["wirelength"] <= 0.0,
            "candidate_power_nonregression": relative["power"] <= 0.0,
            "candidate_area_nonregression": relative["area"] <= 0.0,
        },
        "claim_boundary": (
            "One design/aspect pilot only. It validates routing and metric "
            "collection, not PPA superiority or external generalisation."
        ),
        "sources": {
            "candidate": str(args.candidate),
            "control": str(args.control),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
