#!/usr/bin/env python3
"""Collect labeled multi-block cluster reflow actions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.cost import score_full_fast
from trinity.floorplan.types import Placement

from floorflow.chip_flow.candidate_consensus_evaluation import (
    _project_boxes,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_structural_selector import load_structural_selector
from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible
from floorflow.chip_flow.legalized_cluster_reflow import (
    CLUSTER_REFLOW_FEATURE_NAMES,
    enumerate_cluster_reflow_actions,
)
from floorflow.chip_flow.score_diagnostics import score_diagnostics
from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.evaluate_chip_legalized_group_repair import _instance

METRICS = (
    "hard_cost",
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-move-fraction", type=float, default=0.25)
    parser.add_argument("--max-net-increase", type=float, default=0.02)
    parser.add_argument("--max-actions-per-cluster", type=int, default=16)
    parser.add_argument("--coordinate-resolution-um", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _labels(score, boxes: np.ndarray, geometry: dict) -> dict[str, float]:
    values = score_diagnostics(
        "candidate",
        score,
        boxes,
        geometry["gt_boxes"],
        geometry["b2b"],
        geometry["p2b"],
        geometry["pins"],
    )
    return {
        metric: float(values[f"candidate_{metric}"])
        for metric in METRICS
    }


def _summary(case_rows: list[dict], elapsed_seconds: float) -> dict:
    counts = np.asarray([len(row["actions"]) for row in case_rows], dtype=np.int64)
    features = [
        action["features"]
        for row in case_rows
        for action in row["actions"]
    ]
    matrix = (
        np.asarray(features, dtype=np.float64)
        if features
        else np.empty((0, len(CLUSTER_REFLOW_FEATURE_NAMES)), dtype=np.float64)
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
        "feature_names": list(CLUSTER_REFLOW_FEATURE_NAMES),
        "feature_mean": matrix.mean(axis=0).tolist() if len(matrix) else [],
        "feature_std": matrix.std(axis=0).tolist() if len(matrix) else [],
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
        base_score = score_full_fast(
            Placement(
                xywh=np.asarray(legalized.xywh).copy(),
                instance=copy.deepcopy(legalized.instance),
            )
        )
        base_labels = _labels(base_score, legalized.xywh, case.geometry)
        actions = enumerate_cluster_reflow_actions(
            legalized.xywh,
            case.geometry["constraints"],
            legalized.instance,
            max_move_fraction=args.max_move_fraction,
            max_net_increase=args.max_net_increase,
            max_actions_per_cluster=args.max_actions_per_cluster,
            coordinate_resolution_um=args.coordinate_resolution_um,
        )
        action_rows = []
        for action in actions:
            score = score_full_fast(
                Placement(
                    xywh=action.boxes,
                    instance=copy.deepcopy(legalized.instance),
                )
            )
            labels = _labels(score, action.boxes, case.geometry)
            action_rows.append(
                {
                    **action.as_dict(include_boxes=False),
                    "labels": labels,
                    "label_delta_vs_base": {
                        metric: labels[metric] - base_labels[metric]
                        for metric in METRICS
                    },
                }
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
                "base_labels": base_labels,
                "actions": action_rows,
            }
        )
    if not case_rows:
        raise ValueError("cluster reflow collection selected no cases")
    elapsed = time.perf_counter() - started
    report = {
        "version": "chip-cluster-reflow-actions-v1",
        "status": "development_dataset",
        "cohort_dir": str(args.cohort_dir),
        "selector": str(args.selector),
        "split_report": str(args.split_report),
        "split": args.split,
        "max_move_fraction": args.max_move_fraction,
        "max_net_increase": args.max_net_increase,
        "max_actions_per_cluster": args.max_actions_per_cluster,
        "coordinate_resolution_um": args.coordinate_resolution_um,
        "selection_counts": selected_counts[: int(np.asarray(selected_counts).nonzero()[0].max()) + 1],
        "summary": _summary(case_rows, elapsed),
        "cases": case_rows,
        "boundary": (
            "Action features are target-free. Official metrics are offline "
            "training labels only and are forbidden at inference."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
