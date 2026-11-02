#!/usr/bin/env python3
"""Summarize the frozen cluster-reflow policy against a matched anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.legalized_cluster_reflow import cluster_reflow_action_key
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
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--anchor-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def _relative(rows: list[dict], anchor: dict[int, dict]) -> dict[str, float]:
    return {
        metric: float(
            np.mean([row[metric] for row in rows])
            / np.mean(
                [
                    anchor[int(case["case_index"])][f"legalized_{metric}"]
                    for case in rows
                ]
            )
            - 1.0
        )
        for metric in METRICS
    }


def _wins(rows: list[dict], anchor: dict[int, dict]) -> dict[str, int | float]:
    joint = strict = safe = 0
    for row in rows:
        reference = anchor[int(row["case_index"])]
        primary = (
            row["hard_cost"] <= reference["legalized_hard_cost"]
            and row["v_rel"] <= reference["legalized_v_rel"]
        )
        structural = (
            primary
            and row["grouping"] <= reference["legalized_grouping"]
            and row["boundary"] <= reference["legalized_boundary"]
            and row["p2b_hpwl_gap"] <= reference["legalized_p2b_hpwl_gap"]
        )
        constraint_safe = (
            structural
            and row["area_gap"] <= reference["legalized_area_gap"]
            and row["b2b_hpwl_gap"] <= reference["legalized_b2b_hpwl_gap"]
        )
        joint += int(primary)
        strict += int(structural)
        safe += int(constraint_safe)
    return {
        "joint": joint,
        "strict_structural": strict,
        "constraint_safe": safe,
        "cases": len(rows),
        "joint_rate": joint / len(rows),
        "strict_structural_rate": strict / len(rows),
        "constraint_safe_rate": safe / len(rows),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.actions.read_text(encoding="utf-8"))
    anchor_payload = json.loads(args.anchor_cases.read_text(encoding="utf-8"))
    anchor = {
        int(row["index"]): row
        for row in anchor_payload[args.anchor_key]
    }
    base_rows = []
    selected_rows = []
    selected_actions = []
    for case in payload["cases"]:
        index = int(case["case_index"])
        base = {"case_index": index, **case["base_labels"]}
        selected = base
        chosen = None
        if case["actions"]:
            chosen = min(
                case["actions"],
                key=lambda action: cluster_reflow_action_key(
                    np.asarray(action["features"], dtype=np.float64)
                ),
            )
            selected = {"case_index": index, **chosen["labels"]}
        base_rows.append(base)
        selected_rows.append(selected)
        selected_actions.append(
            None
            if chosen is None
            else {
                "case_index": index,
                "cluster_id": chosen["cluster_id"],
                "moves": chosen["moves"],
            }
        )
    paired = {
        metric: paired_delta(
            selected_rows,
            base_rows,
            metric,
            iterations=args.bootstrap,
            seed=30100 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }
    report = {
        "version": "chip-cluster-reflow-policy-confirmation-v1",
        "status": "complete",
        "cases": len(base_rows),
        "candidate_action_cases": sum(bool(case["actions"]) for case in payload["cases"]),
        "selected_action_cases": sum(action is not None for action in selected_actions),
        "policy": [
            "min_b2b_relative_delta",
            "min_p2b_relative_delta",
            "min_mean_move_l1",
        ],
        "base_relative_vs_anchor": _relative(base_rows, anchor),
        "selected_relative_vs_anchor": _relative(selected_rows, anchor),
        "base_wins": _wins(base_rows, anchor),
        "selected_wins": _wins(selected_rows, anchor),
        "paired_selected_vs_base": paired,
        "selected_actions": selected_actions,
        "claim_boundary": (
            "Target-free action selection; official metrics are labels only. "
            "All candidate views and legalizer calls remain in the declared budget."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "cases",
                    "candidate_action_cases",
                    "selected_action_cases",
                    "base_wins",
                    "selected_wins",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
