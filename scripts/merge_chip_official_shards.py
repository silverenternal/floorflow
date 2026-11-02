#!/usr/bin/env python3
"""Merge exact Official-100 candidate or Trinity shard evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import (
    total_blockcount,
    total_exp,
    total_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("candidate", "trinity"), required=True)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--case-key", default="nfe4_refine3200")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def candidate_rows(shards: list[dict], case_key: str) -> list[dict]:
    rows = []
    for shard in shards:
        case_path = Path(shard["_path"]).with_name(
            f"{Path(shard['_path']).stem}_cases.json"
        )
        rows.extend(load(case_path)[case_key])
    return sorted(rows, key=lambda row: int(row["index"]))


def mean_if_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else None


def merge_candidate(shards: list[dict], case_key: str) -> tuple[dict, list[dict]]:
    rows = candidate_rows(shards, case_key)
    if len(rows) != 100:
        raise ValueError(f"candidate shards must cover 100 cases, got {len(rows)}")
    costs = [float(row["legalized_hard_cost"]) for row in rows]
    counts = [int(row["blocks"]) for row in rows]
    cases = len(rows)
    result = {
        "cases": cases,
        "legalized_hard_cost": total_mean(costs, counts),
        "legalized_hard_cost_exp": total_exp(costs, counts),
        "legalized_hard_cost_blockcount": total_blockcount(costs, counts),
        "legalized_feasible_fraction": float(
            np.mean([bool(row["legalized_feasible"]) for row in rows])
        ),
    }
    for key in (
        "legalized_v_rel",
        "legalized_hpwl_gap",
        "legalized_area_gap",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_grouping",
        "legalized_mib",
        "legalized_boundary",
        "legalized_rho_mae",
    ):
        value = mean_if_present(rows, key)
        if value is not None:
            result[key] = value
    for key in ("raw_overlap", "raw_group", "raw_mib", "raw_boundary", "raw_wl", "raw_area"):
        result[key] = sum(
            float(shard["results"][case_key][key]) * int(shard["results"][case_key]["cases"])
            for shard in shards
        ) / cases
    for key in (
        "latent_update_l1_mean_per_case",
        "latent_update_l2_mean_per_case",
    ):
        result[key] = sum(
            float(shard["results"][case_key][key])
            * int(shard["results"][case_key]["cases"])
            for shard in shards
        ) / cases
    coordinates = sum(
        int(shard["results"][case_key]["cases"])
        for shard in shards
    )
    result["latent_update_rms_per_coordinate"] = float(
        np.sqrt(
            sum(
                float(shard["results"][case_key]["latent_update_rms_per_coordinate"])
                ** 2
                * int(shard["results"][case_key]["cases"])
                for shard in shards
            )
            / coordinates
        )
    )
    result["stage_seconds"] = {
        key: sum(
            float(shard["results"][case_key]["stage_seconds"][key])
            for shard in shards
        )
        for key in shards[0]["results"][case_key]["stage_seconds"]
    }
    result["wall_clock_seconds"] = sum(
        float(shard["results"][case_key]["wall_clock_seconds"])
        for shard in shards
    )
    report = dict(shards[0])
    report["status"] = "merged_exact_shards"
    report["selected_cases"] = cases
    report["results"] = {case_key: result}
    report["shards"] = [shard["_path"] for shard in shards]
    return report, rows


def merge_trinity(shards: list[dict]) -> dict:
    rows = sorted(
        [row for shard in shards for row in shard["rows"]],
        key=lambda row: int(row["index"]),
    )
    if len(rows) != 100:
        raise ValueError(f"Trinity shards must cover 100 cases, got {len(rows)}")
    costs = [float(row["cost"]) for row in rows]
    counts = [int(row["blocks"]) for row in rows]
    report = dict(shards[0])
    report.update(
        {
            "cases": len(rows),
            "rows": rows,
            "mean_cost": float(np.mean(costs)),
            "cost_exp_weighted": total_exp(costs, counts),
            "cost_blockcount_weighted": total_blockcount(costs, counts),
            "mean_v_rel": float(np.mean([float(row["v_rel"]) for row in rows])),
            "mean_hpwl_gap": float(np.mean([float(row["hpwl_gap"]) for row in rows])),
            "mean_area_gap": float(np.mean([float(row["area_gap"]) for row in rows])),
            "mean_seconds": float(np.mean([float(row["seconds"]) for row in rows])),
            "feasible_fraction": float(np.mean([bool(row["feasible"]) for row in rows])),
            "shards": [shard["_path"] for shard in shards],
        }
    )
    return report


def main() -> None:
    args = parse_args()
    shards = []
    for path in args.shard:
        report = load(path)
        report["_path"] = str(path)
        shards.append(report)
    if args.kind == "candidate":
        report, rows = merge_candidate(shards, args.case_key)
    else:
        report = merge_trinity(shards)
        rows = report["rows"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.kind == "candidate":
        cases_path = args.output.with_name(f"{args.output.stem}_cases.json")
        cases_path.write_text(
            json.dumps({args.case_key: rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": str(args.output), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
