#!/usr/bin/env python3
"""Evaluate target-free group repair after the frozen legalizer."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.cost import score_full_fast
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.candidate_consensus_evaluation import (
    _project_boxes,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_structural_selector import load_structural_selector
from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible
from floorflow.chip_flow.legalized_group_repair import repair_legalized_groups
from floorflow.chip_flow.score_diagnostics import score_diagnostics
from floorflow.evaluation.paired_bootstrap import paired_delta

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--max-net-increase", type=float, default=0.02)
    parser.add_argument("--anchor-cases", type=Path)
    parser.add_argument("--anchor-key", default="nfe4_refine1600")
    return parser.parse_args()


def _instance(geometry: dict) -> FloorplanInstance:
    count = len(geometry["areas"])
    targets = np.full((count, 4), -1.0, dtype=np.float32)
    constraints = geometry["constraints"]
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = geometry["gt_boxes"][fixed, 2:]
    targets[preplaced] = geometry["gt_boxes"][preplaced]
    return FloorplanInstance(
        block_count=count,
        area_targets=geometry["areas"],
        constraints=constraints,
        b2b=geometry["b2b"],
        p2b=geometry["p2b"],
        pins_pos=geometry["pins"],
        target_positions=targets,
        gt_positions=geometry["gt_boxes"],
    )


def _metrics(score, boxes: np.ndarray, geometry: dict) -> dict[str, float | int]:
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


def _relative(selected: list[dict], baseline: list[dict]) -> dict[str, float]:
    return {
        metric: float(
            np.mean([row[metric] for row in selected])
            / np.mean([row[metric] for row in baseline])
            - 1.0
        )
        for metric in METRICS
    }


def _wins(selected: list[dict], baseline: list[dict]) -> dict[str, int | float]:
    joint = sum(
        current["hard_cost"] <= reference["hard_cost"]
        and current["v_rel"] <= reference["v_rel"]
        for current, reference in zip(selected, baseline, strict=True)
    )
    structural = sum(
        current["hard_cost"] <= reference["hard_cost"]
        and current["v_rel"] <= reference["v_rel"]
        and current["grouping"] <= reference["grouping"]
        and current["boundary"] <= reference["boundary"]
        and current["p2b_hpwl_gap"] <= reference["p2b_hpwl_gap"]
        for current, reference in zip(selected, baseline, strict=True)
    )
    constraint_safe = sum(
        current["hard_cost"] <= reference["hard_cost"]
        and current["v_rel"] <= reference["v_rel"]
        and current["grouping"] <= reference["grouping"]
        and current["boundary"] <= reference["boundary"]
        and current["p2b_hpwl_gap"] <= reference["p2b_hpwl_gap"]
        and current["area_gap"] <= reference["area_gap"]
        and current["b2b_hpwl_gap"] <= reference["b2b_hpwl_gap"]
        for current, reference in zip(selected, baseline, strict=True)
    )
    return {
        "joint": joint,
        "strict_structural": structural,
        "constraint_safe": constraint_safe,
        "cases": len(selected),
        "joint_rate": joint / len(selected),
        "strict_structural_rate": structural / len(selected),
        "constraint_safe_rate": constraint_safe / len(selected),
    }


def _anchor_rows(
    repaired_rows: list[dict],
    case_indices: list[int],
    path: Path,
    key: str,
) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchor = {int(row["index"]): row for row in payload[key]}
    rows = []
    for case_index, repaired in zip(case_indices, repaired_rows, strict=True):
        if case_index not in anchor:
            raise ValueError(f"anchor is missing case {case_index}")
        rows.append(
            {
                metric: float(anchor[case_index][f"legalized_{metric}"])
                for metric in METRICS
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    allowed = {int(value) for value in split[f"{args.split}_cases"]}
    selector = load_structural_selector(args.selector)
    baseline_rows = []
    repaired_rows = []
    case_indices = []
    rows = []
    for path in sorted(args.cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in allowed:
            continue
        if args.limit > 0 and len(rows) >= args.limit:
            break
        case = load_candidate_consensus_case(path)
        mask = case.batch.candidate_mask[0]
        features = case.batch.candidate_features[0][mask].numpy()
        choice = selector.select(features)
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
        baseline_score = score_full_fast(
            Placement(
                xywh=np.asarray(legalized.xywh).copy(),
                instance=copy.deepcopy(legalized.instance),
            )
        )
        repair = repair_legalized_groups(
            legalized.xywh,
            case.geometry["constraints"],
            legalized.instance,
            max_net_increase=args.max_net_increase,
        )
        repaired_score = score_full_fast(
            Placement(
                xywh=repair.boxes,
                instance=copy.deepcopy(legalized.instance),
            )
        )
        baseline = _metrics(baseline_score, legalized.xywh, case.geometry)
        repaired = _metrics(repaired_score, repair.boxes, case.geometry)
        baseline_rows.append(baseline)
        repaired_rows.append(repaired)
        case_indices.append(index)
        rows.append(
            {
                "case_index": index,
                "selected_candidate": choice,
                "moves": repair.moves,
                "grouping_before": repair.grouping_before,
                "grouping_after": repair.grouping_after,
                "boundary_before": repair.boundary_before,
                "boundary_after": repair.boundary_after,
                "b2b_length_before": repair.b2b_length_before,
                "b2b_length_after": repair.b2b_length_after,
                "p2b_length_before": repair.p2b_length_before,
                "p2b_length_after": repair.p2b_length_after,
                "feasible": repair.feasible,
                "baseline": baseline,
                "repaired": repaired,
            }
        )
    if not rows:
        raise ValueError("group-repair evaluation selected no cases")
    paired = {
        metric: paired_delta(
            repaired_rows,
            baseline_rows,
            metric,
            iterations=args.bootstrap,
            seed=27100 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }
    report = {
        "version": "chip-legalized-group-repair-v1",
        "status": "complete",
        "cases": len(rows),
        "selection_policy": "frozen target-free structural selector",
        "repair_contract": (
            "post-legalizer local rigid component translation; fixed, preplaced "
            "and boundary-coded components are protected; b2b/p2b absolute "
            f"length may increase by at most {args.max_net_increase:.2%}; no target is used"
        ),
        "relative": _relative(repaired_rows, baseline_rows),
        "wins": _wins(repaired_rows, baseline_rows),
        "paired": paired,
        "moves": int(sum(row["moves"] for row in rows)),
        "grouping_before": int(sum(row["grouping_before"] for row in rows)),
        "grouping_after": int(sum(row["grouping_after"] for row in rows)),
        "boundary_before": int(sum(row["boundary_before"] for row in rows)),
        "boundary_after": int(sum(row["boundary_after"] for row in rows)),
        "b2b_length_relative": (
            sum(row["b2b_length_after"] for row in rows)
            / sum(row["b2b_length_before"] for row in rows)
            - 1.0
        ),
        "p2b_length_relative": (
            sum(row["p2b_length_after"] for row in rows)
            / sum(row["p2b_length_before"] for row in rows)
            - 1.0
        ),
        "rows": rows,
    }
    if args.anchor_cases is not None:
        anchor_rows = _anchor_rows(
            repaired_rows,
            case_indices,
            args.anchor_cases,
            args.anchor_key,
        )
        report["versus_anchor"] = {
            "path": str(args.anchor_cases),
            "key": args.anchor_key,
            "relative": _relative(repaired_rows, anchor_rows),
            "wins": _wins(repaired_rows, anchor_rows),
            "paired": {
                metric: paired_delta(
                    repaired_rows,
                    anchor_rows,
                    metric,
                    iterations=args.bootstrap,
                    seed=27200 + offset,
                    lower_is_better=True,
                )
                for offset, metric in enumerate(METRICS)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "moves",
                    "relative",
                    "wins",
                    "grouping_before",
                    "grouping_after",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
