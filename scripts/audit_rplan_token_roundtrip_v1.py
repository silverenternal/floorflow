#!/usr/bin/env python3
"""Audit canonical token roundtrips on full RPLAN validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    rplan_taxonomy,
)
from floorflow.data.rplan_room_limits import limit_canonical_plan_rooms
from floorflow.evaluation.rplan_token_roundtrip import (
    audit_rplan_token_roundtrip,
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
    parser.add_argument("--limit", type=int, default=12018)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--quantization-scale", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/rplan_token_roundtrip_val_v1.json",
    )
    args = parser.parse_args()
    taxonomy = rplan_taxonomy()
    limits = RPlanTensorLimits()
    dataset = RPlanBoundaryDataset(
        args.dataset,
        taxonomy,
        offset=args.offset,
        limit=args.limit,
        limits=limits,
    )
    failures = []
    exact_fingerprints = 0
    maximum_error = 0.0
    maximum_tokens = 0
    for index in range(len(dataset)):
        plan = limit_canonical_plan_rooms(
            dataset._plan(index),
            limits.max_rooms,
        )
        try:
            result = audit_rplan_token_roundtrip(
                plan,
                taxonomy,
                max_length=args.max_length,
                quantization_scale=args.quantization_scale,
            )
        except (TypeError, ValueError) as error:
            failures.append(
                {
                    "index": index,
                    "case": dataset.rows[index].get("name"),
                    "error": str(error),
                }
            )
            continue
        exact_fingerprints += int(result["canonical_fingerprint_exact"])
        maximum_error = max(
            maximum_error,
            float(result["max_quantization_error"]),
        )
        maximum_tokens = max(maximum_tokens, int(result["token_count"]))
        if not result["passed"]:
            failures.append(
                {
                    "index": index,
                    "case": dataset.rows[index].get("name"),
                    **result,
                }
            )
    report = {
        "schema_version": "rplan-token-roundtrip-audit-v1",
        "dataset": str(args.dataset),
        "cases": len(dataset),
        "quantization_scale": int(args.quantization_scale),
        "half_step": 0.5 / float(max(1, args.quantization_scale)),
        "maximum_quantization_error": maximum_error,
        "maximum_token_count": maximum_tokens,
        "canonical_fingerprint_exact_cases": exact_fingerprints,
        "passed_cases": len(dataset) - len(failures),
        "failed_cases": len(failures),
        "failures": failures,
        "decision": "pass" if not failures else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "half_step",
                    "maximum_quantization_error",
                    "maximum_token_count",
                    "canonical_fingerprint_exact_cases",
                    "passed_cases",
                    "failed_cases",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
