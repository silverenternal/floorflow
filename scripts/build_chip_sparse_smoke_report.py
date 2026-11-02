#!/usr/bin/env python3
"""Validate and freeze sparse/slot-transport wiring smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-root",
        type=Path,
        default=ROOT / "outputs/smoke/chip_slot_transport_v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_sparse_v3_smoke_v1.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    train_path = args.smoke_root / "slot_transport_smoke.json"
    checkpoint = (
        args.smoke_root
        / "slot_transport_smoke"
        / "checkpoints"
        / "final.pt"
    )
    official_path = args.smoke_root / "official_smoke.json"
    train = read_json(train_path)
    official = read_json(official_path)
    source_paths = [
        ROOT / "src/floorflow/chip_flow/sparse_slot_flow.py",
        ROOT / "src/floorflow/chip_flow/sparse_hierarchy.py",
        ROOT / "src/floorflow/chip_flow/sparse_flow_model.py",
        ROOT / "src/floorflow/chip_flow/flow_adapter.py",
        ROOT / "scripts/train_chip_slot.py",
        ROOT / "scripts/evaluate_chip_official.py",
        ROOT / "tests/test_chip_slot_transport.py",
    ]
    line_counts = {
        str(path.relative_to(ROOT)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in source_paths
    }
    checks = {
        "training_complete": train.get("status") == "complete",
        "final_checkpoint_exists": checkpoint.is_file(),
        "official_cases_present": official.get("cases") == 2,
        "official_legalized_feasible": (
            official.get("legalized_feasible_fraction") == 1.0
        ),
        "routing_budget_dynamic_across_cases": (
            official.get("top_k_min") != official.get("top_k_max")
        ),
        "router_load_reported": "router_load_max" in official,
        "files_below_600_lines": max(line_counts.values()) < 600,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "two-step CPU wiring smoke; no quality claim",
        "checks": checks,
        "smoke_root": str(args.smoke_root),
        "training_report": str(train_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_parameters": train.get("model_parameters"),
        "train_steps": train.get("config", {}).get("steps"),
        "official": {
            key: official.get(key)
            for key in (
                "cases",
                "sampling_steps",
                "legalized_hard_cost",
                "legalized_feasible_fraction",
                "active_slot_count_min",
                "active_slot_count_max",
                "top_k_min",
                "top_k_max",
                "router_load_max",
            )
        },
        "line_counts": line_counts,
        "frozen_artifacts_untouched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
