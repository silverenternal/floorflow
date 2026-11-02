#!/usr/bin/env python3
"""Collect target-free local repair actions on a frozen development split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.types import Placement

from floorflow.chip_flow.candidate_consensus_evaluation import (
    _project_boxes,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_structural_selector import load_structural_selector
from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible
from floorflow.chip_flow.legalized_group_actions import (
    FEATURE_NAMES,
    enumerate_legalized_group_actions,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.evaluate_chip_legalized_group_repair import _instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-move-fraction", type=float, default=0.25)
    parser.add_argument("--max-net-increase", type=float, default=0.02)
    parser.add_argument("--max-actions-per-cluster", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(case_rows: list[dict], elapsed_seconds: float) -> dict:
    counts = np.asarray(
        [len(case["actions"]) for case in case_rows],
        dtype=np.int64,
    )
    features = [
        action["features"]
        for case in case_rows
        for action in case["actions"]
    ]
    feature_matrix = (
        np.asarray(features, dtype=np.float64)
        if features
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    )
    return {
        "cases": len(case_rows),
        "cases_with_actions": int(np.count_nonzero(counts)),
        "total_actions": int(counts.sum()),
        "mean_actions_per_case": float(counts.mean()) if len(counts) else 0.0,
        "median_actions_per_case": float(np.median(counts)) if len(counts) else 0.0,
        "max_actions_per_case": int(counts.max()) if len(counts) else 0,
        "action_count_histogram": {
            str(value): int(np.count_nonzero(counts == value))
            for value in sorted(set(counts.tolist()))
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": (
            feature_matrix.mean(axis=0).tolist()
            if len(feature_matrix)
            else []
        ),
        "feature_std": (
            feature_matrix.std(axis=0).tolist()
            if len(feature_matrix)
            else []
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    allowed = {int(value) for value in split[f"{args.split}_cases"]}
    selector = load_structural_selector(args.selector)
    started = time.perf_counter()
    case_rows = []
    selected_counts = [0] * 8
    for path in sorted(args.cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in allowed:
            continue
        if args.limit > 0 and len(case_rows) >= args.limit:
            break
        case = load_candidate_consensus_case(path)
        mask = case.batch.candidate_mask[0]
        features = case.batch.candidate_features[0][mask].numpy()
        choice = selector.select(features)
        selected_counts[choice] += 1
        projected = _project_boxes(
            case,
            case.batch.candidate_states[0][choice],
            iterations=2,
        )
        instance = _instance(case.geometry)
        legalized = scale_pack_legalize_reproducible(
            Placement(xywh=projected, instance=instance),
            time_limit_ms=2000,
        )
        actions = enumerate_legalized_group_actions(
            legalized.xywh,
            case.geometry["constraints"],
            legalized.instance,
            max_move_fraction=args.max_move_fraction,
            max_net_increase=args.max_net_increase,
            max_actions_per_cluster=args.max_actions_per_cluster,
        )
        case_rows.append(
            {
                "case_index": index,
                "selected_candidate": choice,
                "blocks": len(legalized.xywh),
                "legalized_xywh": np.asarray(
                    legalized.xywh,
                    dtype=np.float64,
                ).tolist(),
                "actions": [
                    action.as_dict(include_boxes=False)
                    for action in actions
                ],
            }
        )
    if not case_rows:
        raise ValueError("action collection selected no cases")
    elapsed = time.perf_counter() - started
    report = {
        "version": "chip-legalized-group-actions-v1",
        "status": "development_dataset",
        "cohort_dir": str(args.cohort_dir),
        "selector": str(args.selector),
        "split_report": str(args.split_report),
        "split": args.split,
        "max_move_fraction": args.max_move_fraction,
        "max_net_increase": args.max_net_increase,
        "max_actions_per_cluster": args.max_actions_per_cluster,
        "selection_counts": selected_counts[: int(np.asarray(selected_counts).nonzero()[0].max()) + 1],
        "summary": _summary(case_rows, elapsed),
        "cases": case_rows,
        "boundary": (
            "Development-only action data. Features use legalized geometry, "
            "constraints and graph topology; no target metrics are inputs."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
