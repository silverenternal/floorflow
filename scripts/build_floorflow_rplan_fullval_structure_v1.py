"""Aggregate full-validation FloorFlow structural metric shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = (
    "outputs/reports/floorflow_mask4096_seed37_fullval_structure_shard0_v2.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_structure_shard1_v2.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_structure_shard2_v2.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_structure_shard3_v2.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/floorflow_mask4096_seed37_fullval_structure_v1.json"
        ),
    )
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        default=[ROOT / relative for relative in SHARDS],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = []
    for path in args.shards:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = (
            str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path)
        )
        shards.append(
            {
                "report": relative,
                "validation_offset": int(payload["validation_offset"]),
                "requested_cases": int(payload["requested_cases"]),
                "successful_cases": int(payload["successful_cases"]),
                "failure_count": int(payload["failure_count"]),
                "metrics": payload["metrics"],
            }
        )
    successful = sum(item["successful_cases"] for item in shards)
    failures = sum(item["failure_count"] for item in shards)
    keys = sorted(
        set.intersection(*(set(item["metrics"]) for item in shards))
    )
    aggregate = {
        key: sum(
            float(item["metrics"][key]) * item["successful_cases"]
            for item in shards
        )
        / max(1, successful)
        for key in keys
    }
    report = {
        "schema": "floorflow-rplan-full-validation-structure-v1",
        "checkpoint": (
            "outputs/train/floorflow_mask_only4096_hold512_seed37_v1/best.pt"
        ),
        "mask_gap_policy": "partition_fill",
        "requested_cases": sum(item["requested_cases"] for item in shards),
        "successful_cases": successful,
        "failure_count": failures,
        "metrics": aggregate,
        "shards": shards,
        "test_opened": False,
        "claim_boundary": (
            "RPLAN validation structure only; not a test or external SOTA claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "successful_cases": successful,
                "failure_count": failures,
                "metrics": aggregate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
