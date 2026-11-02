#!/usr/bin/env python3
"""Merge deterministic candidate rows with seed-reset baseline diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-source",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v1.json",
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_component_baseline_reset_v1.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v2.json",
    )
    return parser.parse_args()


def rows(path: Path, arm: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "complete":
        raise ValueError(f"{path} is incomplete")
    return [row for row in payload["rows"] if row["arm"] == arm]


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    merged = rows(args.candidate_source, "candidate") + rows(
        args.baseline_source,
        "baseline",
    )
    for row in merged:
        index = int(row["index"])
        count = int(store.arrays["n"][index])
        constraints = store.arrays["constraints"][index, :count]
        row["blocks"] = count
        row["n_b2b"] = int(store.arrays["n_b2b"][index])
        row["n_p2b"] = int(store.arrays["n_p2b"][index])
        row["n_pins"] = int(store.arrays["n_pins"][index])
        row["boundary_blocks"] = int(np.count_nonzero(constraints[:, 4]))
        row["cluster_groups"] = int(
            np.unique(constraints[constraints[:, 3] > 0, 3]).size
        )
        row["mib_groups"] = int(
            np.unique(constraints[constraints[:, 2] > 0, 2]).size
        )
    report = {
        "status": "complete" if len(merged) == 3000 else "incomplete",
        "scope": "corrected matched official NFE and score-component diagnostics",
        "candidate_source": str(args.candidate_source),
        "baseline_source": str(args.baseline_source),
        "cache": str(args.cache),
        "expected_rows": 3000,
        "rows": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}))


if __name__ == "__main__":
    main()
