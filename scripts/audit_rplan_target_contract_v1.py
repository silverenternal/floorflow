#!/usr/bin/env python3
"""Audit real RPLAN tensor targets without a model or accelerator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    rplan_taxonomy,
)
from floorflow.evaluation.rplan_target_contract import (
    audit_rplan_target_item,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            ROOT
            / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/rplan_target_contract_val_v1.json",
    )
    args = parser.parse_args()
    dataset = RPlanBoundaryDataset(
        args.dataset,
        rplan_taxonomy(),
        offset=args.offset,
        limit=args.limit,
        limits=RPlanTensorLimits(image_size=args.image_size),
    )
    violations: Counter[str] = Counter()
    room_counts: Counter[int] = Counter()
    failed = []
    for index in range(len(dataset)):
        result = audit_rplan_target_item(dataset[index])
        room_counts[int(result["room_count"])] += 1
        if not result["passed"]:
            failed.append(
                {
                    "index": index,
                    "case": dataset.rows[index].get("name"),
                    **result,
                }
            )
            violations.update(result["violations"])
    report = {
        "schema_version": "rplan-target-contract-audit-v1",
        "dataset": str(args.dataset),
        "offset": int(args.offset),
        "cases": len(dataset),
        "passed_cases": len(dataset) - len(failed),
        "failed_cases": len(failed),
        "violations": dict(sorted(violations.items())),
        "room_count_distribution": {
            str(key): value for key, value in sorted(room_counts.items())
        },
        "failures": failed,
        "decision": "pass" if not failed else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "passed_cases": report["passed_cases"],
                "failed_cases": report["failed_cases"],
                "violations": report["violations"],
                "room_count_distribution": report["room_count_distribution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
