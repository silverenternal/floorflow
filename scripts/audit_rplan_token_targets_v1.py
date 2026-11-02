#!/usr/bin/env python3
"""Audit full RPLAN token targets against a frozen vocabulary."""

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
from floorflow.data.rplan_room_limits import limit_canonical_plan_rooms
from floorflow.evaluation.rplan_token_contract import (
    audit_rplan_token_contract,
)
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary

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
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=(
            ROOT
            / "outputs/token_vocabulary/rplan_train4096_v2/vocabulary.json"
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12018)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/rplan_token_contract_val_v1.json",
    )
    args = parser.parse_args()
    taxonomy = rplan_taxonomy()
    vocabulary = FloorplanTokenVocabulary.read(args.vocabulary)
    limits = RPlanTensorLimits()
    dataset = RPlanBoundaryDataset(
        args.dataset,
        taxonomy,
        offset=args.offset,
        limit=args.limit,
        limits=limits,
    )
    lengths = Counter()
    unknown = Counter()
    failed = []
    oov_total = 0
    token_total = 0
    for index in range(len(dataset)):
        plan = limit_canonical_plan_rooms(
            dataset._plan(index),
            limits.max_rooms,
        )
        result = audit_rplan_token_contract(
            plan,
            taxonomy,
            vocabulary,
            max_length=args.max_length,
        )
        lengths[int(result["token_length"])] += 1
        unknown.update(result["unknown_tokens"])
        oov_total += int(result["oov_count"])
        token_total += int(result["token_length"])
        if not result["passed"]:
            failed.append(
                {
                    "index": index,
                    "case": dataset.rows[index].get("name"),
                    **result,
                }
            )
    ordered_lengths = sorted(lengths)
    report = {
        "schema_version": "rplan-token-contract-audit-v1",
        "dataset": str(args.dataset),
        "vocabulary": str(args.vocabulary),
        "vocabulary_sha256": vocabulary.sha256(),
        "offset": int(args.offset),
        "cases": len(dataset),
        "max_length": int(args.max_length),
        "maximum_observed_length": max(ordered_lengths, default=0),
        "tokens": token_total,
        "oov_tokens": oov_total,
        "oov_rate": oov_total / max(1, token_total),
        "unknown_token_counts": dict(unknown.most_common()),
        "passed_cases": len(dataset) - len(failed),
        "failed_cases": len(failed),
        "failures": failed,
        "decision": "pass" if not failed else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "max_length",
                    "maximum_observed_length",
                    "tokens",
                    "oov_tokens",
                    "oov_rate",
                    "passed_cases",
                    "failed_cases",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
