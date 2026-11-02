#!/usr/bin/env python3
"""Evaluate multi-cluster composition of legalized reflow actions."""

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
from trinity.floorplan.types import Placement

from floorflow.chip_flow.candidate_consensus_evaluation import (
    _project_boxes,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_structural_selector import load_structural_selector
from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible
from floorflow.chip_flow.legalized_cluster_reflow import (
    ClusterReflowAction,
    cluster_reflow_action_key,
    enumerate_cluster_reflow_actions,
)
from floorflow.chip_flow.legalized_group_repair import _target_free_checks
from floorflow.chip_flow.score_diagnostics import score_diagnostics
from floorflow.evaluation.paired_bootstrap import paired_delta
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
    parser.add_argument("--anchor-cases", type=Path)
    parser.add_argument("--anchor-key", default="nfe16_refine1600")
    parser.add_argument("--bootstrap", type=int, default=10000)
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
    return {metric: float(values[f"candidate_{metric}"]) for metric in METRICS}


def _safe_portfolio(
    base: np.ndarray,
    actions: list[ClusterReflowAction],
    instance,
    max_net_increase: float,
) -> tuple[np.ndarray, list[ClusterReflowAction]]:
    if not actions:
        return base, []
    _, _, boundary_limit, b2b_base, p2b_base = _target_free_checks(base, instance)
    b2b_limit = b2b_base * (1.0 + max_net_increase)
    p2b_limit = p2b_base * (1.0 + max_net_increase)
    by_cluster: dict[int, ClusterReflowAction] = {}
    for action in actions:
        current = by_cluster.get(action.cluster_id)
        if current is None or cluster_reflow_action_key(
            action.features
        ) < cluster_reflow_action_key(current.features):
            by_cluster[action.cluster_id] = action
    ordered = sorted(
        by_cluster.values(),
        key=lambda action: cluster_reflow_action_key(action.features),
    )
    accepted: list[ClusterReflowAction] = []
    current = base.copy()
    used: set[int] = set()
    for action in ordered:
        indices = {index for index, _dx, _dy in action.moves}
        if used & indices:
            continue
        candidate = current.copy()
        for index, dx, dy in action.moves:
            candidate[index, 0] += dx
            candidate[index, 1] += dy
        feasible, _, boundary, b2b, p2b = _target_free_checks(candidate, instance)
        if (
            feasible
            and boundary <= boundary_limit
            and b2b <= b2b_limit
            and p2b <= p2b_limit
        ):
            current = candidate
            used.update(indices)
            accepted.append(action)
    return current, accepted


def _relative(selected: list[dict], baseline: list[dict]) -> dict[str, float]:
    return {
        metric: float(
            np.mean([row[metric] for row in selected])
            / np.mean([row[metric] for row in baseline])
            - 1.0
        )
        for metric in METRICS
    }


def _wins(selected: list[dict], baseline: list[dict]) -> dict[str, float | int]:
    joint = strict = safe = 0
    for current, reference in zip(selected, baseline, strict=True):
        primary = (
            current["hard_cost"] <= reference["hard_cost"]
            and current["v_rel"] <= reference["v_rel"]
        )
        structural = (
            primary
            and current["grouping"] <= reference["grouping"]
            and current["boundary"] <= reference["boundary"]
            and current["p2b_hpwl_gap"] <= reference["p2b_hpwl_gap"]
        )
        constraint_safe = (
            structural
            and current["area_gap"] <= reference["area_gap"]
            and current["b2b_hpwl_gap"] <= reference["b2b_hpwl_gap"]
        )
        joint += int(primary)
        strict += int(structural)
        safe += int(constraint_safe)
    return {
        "joint": joint,
        "strict_structural": strict,
        "constraint_safe": safe,
        "cases": len(selected),
        "joint_rate": joint / len(selected),
        "strict_structural_rate": strict / len(selected),
        "constraint_safe_rate": safe / len(selected),
    }


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    allowed = {int(value) for value in split[f"{args.split}_cases"]}
    selector = load_structural_selector(args.selector)
    anchor = None
    if args.anchor_cases is not None:
        anchor_payload = json.loads(args.anchor_cases.read_text(encoding="utf-8"))
        anchor = {
            int(row["index"]): {
                metric: float(row[f"legalized_{metric}"])
                for metric in METRICS
            }
            for row in anchor_payload[args.anchor_key]
        }
    base_rows = []
    portfolio_rows = []
    anchor_rows = []
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
        base_score = score_full_fast(
            Placement(
                xywh=np.asarray(legalized.xywh).copy(),
                instance=copy.deepcopy(legalized.instance),
            )
        )
        base = {"case_index": index, **_labels(base_score, legalized.xywh, case.geometry)}
        actions = enumerate_cluster_reflow_actions(
            legalized.xywh,
            case.geometry["constraints"],
            legalized.instance,
            max_move_fraction=args.max_move_fraction,
            max_net_increase=args.max_net_increase,
            max_actions_per_cluster=args.max_actions_per_cluster,
        )
        final_boxes, accepted = _safe_portfolio(
            np.asarray(legalized.xywh),
            actions,
            legalized.instance,
            args.max_net_increase,
        )
        final_score = score_full_fast(
            Placement(
                xywh=final_boxes,
                instance=copy.deepcopy(legalized.instance),
            )
        )
        portfolio = {
            "case_index": index,
            **_labels(final_score, final_boxes, case.geometry),
        }
        base_rows.append(base)
        portfolio_rows.append(portfolio)
        if anchor is not None:
            anchor_rows.append({"case_index": index, **anchor[index]})
        rows.append(
            {
                "case_index": index,
                "selected_candidate": choice,
                "available_actions": len(actions),
                "accepted_clusters": [action.cluster_id for action in accepted],
                "accepted_moves": sum(len(action.moves) for action in accepted),
                "base": base,
                "portfolio": portfolio,
            }
        )
    if not rows:
        raise ValueError("cluster reflow portfolio selected no cases")
    paired = {
        metric: paired_delta(
            portfolio_rows,
            base_rows,
            metric,
            iterations=args.bootstrap,
            seed=32000 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }
    report = {
        "version": "chip-cluster-reflow-portfolio-v1",
        "status": "complete",
        "cases": len(rows),
        "cases_with_actions": sum(row["available_actions"] > 0 for row in rows),
        "cases_with_accepted_actions": sum(bool(row["accepted_clusters"]) for row in rows),
        "base_relative": _relative(portfolio_rows, base_rows),
        "base_wins": _wins(portfolio_rows, base_rows),
        "paired_portfolio_vs_base": paired,
        "rows": rows,
    }
    if anchor_rows:
        report["portfolio_relative_vs_anchor"] = _relative(portfolio_rows, anchor_rows)
        report["portfolio_wins_vs_anchor"] = _wins(portfolio_rows, anchor_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "cases_with_actions": report["cases_with_actions"],
                "cases_with_accepted_actions": report["cases_with_accepted_actions"],
                "base_relative": report["base_relative"],
                "base_wins": report["base_wins"],
                "portfolio_wins_vs_anchor": report.get("portfolio_wins_vs_anchor"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
