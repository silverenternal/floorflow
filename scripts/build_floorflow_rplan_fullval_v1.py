"""Aggregate disjoint FloorFlow RPLAN validation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = (
    "outputs/reports/floorflow_mask4096_seed37_fullval_shard0_v4.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_shard1_v4.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_shard2_v4.json",
    "outputs/reports/floorflow_mask4096_seed37_fullval_shard3_v4.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_mask4096_seed37_fullval_v3.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = []
    for relative in SHARDS:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload["arms"]["mask_partition"]
        shards.append(
            {
                "report": relative,
                "validation_offset": int(payload["validation_offset"]),
                "requested_cases": int(payload["requested_cases"]),
                "successful_cases": int(arm["successful_cases"]),
                "failure_count": int(arm["failure_count"]),
                "metrics": arm["metrics"],
                "token_roundtrip_exact_rate": float(
                    arm["token_roundtrip_exact_rate"]
                ),
            }
        )
    successful = sum(item["successful_cases"] for item in shards)
    failures = sum(item["failure_count"] for item in shards)
    metric_keys = sorted(shards[0]["metrics"])
    aggregate = {
        key: sum(
            item["metrics"][key] * item["successful_cases"]
            for item in shards
        )
        / max(1, successful)
        for key in metric_keys
    }
    report = {
        "schema": "floorflow-rplan-full-validation-v3",
        "checkpoint": (
            "outputs/train/floorflow_mask_only4096_hold512_seed37_v1/best.pt"
        ),
        "requested_cases": sum(item["requested_cases"] for item in shards),
        "successful_cases": successful,
        "failure_count": failures,
        "token_roundtrip_exact_rate": sum(
            item["token_roundtrip_exact_rate"] * item["successful_cases"]
            for item in shards
        )
        / max(1, successful),
        "metrics": aggregate,
        "shards": shards,
        "test_opened": False,
        "claim_boundary": (
            "RPLAN validation only; no test or external leaderboard claim."
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
