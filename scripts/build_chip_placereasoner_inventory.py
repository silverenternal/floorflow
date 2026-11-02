#!/usr/bin/env python3
"""Build a reproducible PlaceReasoner-Bench inventory and provenance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.placereasoner_bench import (
    benchmark_tasks,
    load_benchmark_contract,
    sha256_file,
    validate_benchmark_root,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/chip_placereasoner_bench_v1.json"
DEFAULT_REPO = ROOT / "third_party/PlaceReasoner-Beta"
DEFAULT_BENCH = DEFAULT_REPO / "Benchmark/PlaceReasoner-Bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_benchmark_contract(args.contract)
    tasks = benchmark_tasks(contract)
    benchmark_root = args.benchmark_root.resolve()
    errors = validate_benchmark_root(benchmark_root, tasks)
    rows = []
    for task in tasks:
        design_dir = benchmark_root / task.design
        files = {
            "verilog": design_dir / f"{task.design}.v",
            "sdc": design_dir / "constraint.sdc",
            "floorplan_def": design_dir / task.floorplan_def,
        }
        rows.append(
            {
                "task_id": task.task_id,
                "design": task.design,
                "source_family": task.source_family,
                "top": task.top,
                "aspect_id": task.aspect_id,
                "floorplan_def": str(task.floorplan_def),
                "macro_count": task.macro_count,
                "files": {
                    name: {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path) if path.is_file() else None,
                    }
                    for name, path in files.items()
                },
            }
        )
    benchmark_readme = benchmark_root / "README.md"
    report = {
        "version": "chip-placereasoner-inventory-v1",
        "status": "pass" if not errors else "fail_missing_inputs",
        "repo_url": contract["repo_url"],
        "repo_commit": contract["repo_commit"],
        "repo_license": contract["repo_license"],
        "dataset_id": contract["dataset_id"],
        "dataset_license": contract["dataset_license"],
        "benchmark_readme_sha256": (
            sha256_file(benchmark_readme) if benchmark_readme.is_file() else None
        ),
        "unique_designs": contract["unique_designs"],
        "placement_tasks": contract["placement_tasks"],
        "freshness_rule": contract["freshness_rule"],
        "errors": errors,
        "designs": contract["designs"],
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "unique_designs": report["unique_designs"],
                "placement_tasks": report["placement_tasks"],
                "errors": len(errors),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
