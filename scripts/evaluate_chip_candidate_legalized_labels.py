#!/usr/bin/env python3
"""Produce legalized labels for every saved candidate in a frozen cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.candidate_consensus_evaluation import (
    evaluate_legalized_candidates,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.evaluate_chip_official import score_case

METRIC_NAMES = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
    "legalizer_rung",
    "legalizer_solves",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--projector-iterations", type=int, default=2)
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--legalizer-policy-json", type=Path)
    parser.add_argument("--promotion-min-cases", type=int, default=500)
    return parser.parse_args()


def _evaluate_path(task: tuple[str, int, int, dict | None]) -> dict:
    path, projector_iterations, legalizer_time_limit_ms, options = task
    case = load_candidate_consensus_case(Path(path))
    return evaluate_legalized_candidates(
        case,
        score_case,
        projector_iterations=projector_iterations,
        legalizer_time_limit_ms=legalizer_time_limit_ms,
        legalizer_options=options,
    )


def _metric_means(rows: list[dict]) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for metric in METRIC_NAMES:
        values = [
            float(candidate[metric])
            for row in rows
            for candidate in row["rows"].values()
            if candidate.get(metric) is not None
        ]
        result[metric] = {
            "mean": sum(values) / len(values) if values else None,
            "defined_candidate_rows": len(values),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    paths = sorted(args.cohort_dir.glob("*.npz"))
    if args.limit is not None:
        paths = paths[: max(0, args.limit)]
    if not paths:
        raise ValueError("candidate cohort contains no shards")
    legalizer_options = (
        json.loads(args.legalizer_policy_json.read_text(encoding="utf-8"))
        if args.legalizer_policy_json is not None
        else None
    )
    if legalizer_options is not None and not isinstance(legalizer_options, dict):
        raise TypeError("legalizer policy JSON must contain an object")
    legalizer_time_limit_ms = args.legalizer_time_limit_ms
    if legalizer_options is not None:
        declared_time_limit = legalizer_options.pop("time_limit_ms", None)
        if declared_time_limit is not None:
            legalizer_time_limit_ms = int(declared_time_limit)
    if legalizer_time_limit_ms <= 0:
        raise ValueError("legalizer time limit must be positive")
    started = time.monotonic()
    tasks = [
        (
            str(path),
            args.projector_iterations,
            legalizer_time_limit_ms,
            legalizer_options,
        )
        for path in paths
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, row in enumerate(executor.map(_evaluate_path, tasks), start=1):
            rows.append(row)
            if completed == len(tasks) or completed % 50 == 0:
                elapsed = time.monotonic() - started
                print(f"evaluated {completed}/{len(tasks)} in {elapsed:.1f}s", flush=True)
    rows.sort(key=lambda row: int(row["case_index"]))
    report = {
        "version": "chip-legalized-candidate-labels-v1",
        "status": "complete" if len(rows) >= args.promotion_min_cases else "screen_only",
        "cohort_dir": str(args.cohort_dir),
        "cases": len(rows),
        "promotion_min_cases": args.promotion_min_cases,
        "contract": {
            "scope": "real saved candidates only; no synthetic consensus",
            "projector_iterations": args.projector_iterations,
            "legalizer_time_limit_ms": legalizer_time_limit_ms,
            "legalizer_options": legalizer_options,
            "workers": args.workers,
            "solver_threads_per_worker": 1,
        },
        "elapsed_seconds": time.monotonic() - started,
        "metric_means": _metric_means(rows),
        "rows": rows,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": report["cases"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
