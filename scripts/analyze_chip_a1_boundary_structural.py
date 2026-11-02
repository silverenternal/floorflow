#!/usr/bin/env python3
"""Diagnose boundary/grouping interference in matched chip-screen cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "outputs/reports/chip_v45_routing_control_seed23_20k_group2_dev_cases.json"
CANDIDATE = (
    ROOT
    / "outputs/reports/chip_m5_balanced_stream_seed23_20k_refine1600_dev_cases.json"
)
SCREEN = ROOT / "outputs/reports/chip_m5_balanced_stream_full.json"
ABLATION = ROOT / "outputs/reports/chip_typed_hierarchy_ablation_v3.json"
CACHE = ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev"
SETTING = "nfe4_refine1600"

FIELDS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_boundary",
    "legalized_grouping",
    "raw_boundary",
    "legalizer_rung",
    "legalizer_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_a1_boundary_structural_probe.json",
    )
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_metadata(store: PackedCaseStore, indices: list[int]) -> dict[int, dict]:
    metadata = {}
    for index in indices:
        count = int(store.arrays["n"][index])
        constraints = store.arrays["constraints"][index, :count]
        metadata[index] = {
            "block_count": count,
            "boundary_coded_blocks": int((constraints[:, 4] != 0).sum()),
            "graph_degree_per_block": float(
                (
                    int(store.arrays["n_b2b"][index])
                    + int(store.arrays["n_p2b"][index])
                )
                / max(1, count)
            ),
        }
    return metadata


def matched_rows() -> list[dict]:
    control = {
        int(row["index"]): row for row in read(CONTROL)[SETTING]
    }
    candidate = read(CANDIDATE)[SETTING]
    indices = sorted(int(row["index"]) for row in candidate)
    metadata = case_metadata(PackedCaseStore(CACHE), indices)
    rows = []
    for candidate_row in candidate:
        index = int(candidate_row["index"])
        control_row = control[index]
        rows.append(
            {
                "index": index,
                **metadata[index],
                **{
                    field: float(candidate_row[field]) - float(control_row[field])
                    for field in FIELDS
                },
                "candidate_v_rel": float(candidate_row["legalized_v_rel"]),
                "control_v_rel": float(control_row["legalized_v_rel"]),
            }
        )
    return rows


def summarize(rows: list[dict], field: str) -> dict:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {
        "mean_delta": float(values.mean()),
        "mean_contribution_to_all_32": float(values.sum() / 32.0),
        "wins": int((values < 0.0).sum()),
        "ties": int((values == 0.0).sum()),
        "losses": int((values > 0.0).sum()),
        "worst_delta": float(values.max()),
        "best_delta": float(values.min()),
    }


def main() -> None:
    args = parse_args()
    rows = matched_rows()
    screen = read(SCREEN)
    ablation = read(ABLATION)
    thresholds = {
        "block_count": float(
            np.quantile([row["block_count"] for row in rows], 0.75)
        ),
        "boundary_coded_blocks": float(
            np.quantile([row["boundary_coded_blocks"] for row in rows], 0.75)
        ),
        "graph_degree_per_block": float(
            np.quantile([row["graph_degree_per_block"] for row in rows], 0.75)
        ),
    }
    strata = {
        "all": rows,
        "block_count_top_quartile": [
            row for row in rows if row["block_count"] >= thresholds["block_count"]
        ],
        "boundary_code_top_quartile": [
            row
            for row in rows
            if row["boundary_coded_blocks"] >= thresholds["boundary_coded_blocks"]
        ],
        "graph_degree_top_quartile": [
            row
            for row in rows
            if row["graph_degree_per_block"]
            >= thresholds["graph_degree_per_block"]
        ],
    }
    summaries = {
        name: {
            "cases": len(selected),
            "metrics": {
                field: summarize(selected, field) for field in FIELDS
            },
        }
        for name, selected in strata.items()
    }
    worst_v_rel = sorted(
        rows,
        key=lambda row: row["legalized_v_rel"],
        reverse=True,
    )[:5]
    full_control = ablation["full_vs_controls"]["no_structural"]
    hierarchy_boundary_delta = full_control["legalized_boundary"]["relative_delta"]
    top = summaries["block_count_top_quartile"]["metrics"]
    raw_boundary_delta = summaries["all"]["metrics"]["raw_boundary"]["mean_delta"]
    legalized_boundary_delta = summaries["all"]["metrics"][
        "legalized_boundary"
    ]["mean_delta"]
    checks = {
        "full_screen_cost_improves": screen["checks"]["cost_improves"],
        "full_screen_v_rel_fails": not screen["checks"]["v_rel_improves"],
        "full_hierarchy_boundary_worse_than_no_structural": (
            hierarchy_boundary_delta > 0.0
        ),
        "raw_boundary_mean_improves": raw_boundary_delta < 0.0,
        "legalized_boundary_mean_worsens": legalized_boundary_delta > 0.0,
        "broad_high_complexity_failure_not_supported": (
            top["legalized_v_rel"]["mean_delta"] <= 0.0
        ),
    }
    supported = all(checks.values())
    report = {
        "version": "chip-a1-boundary-structural-probe-v1",
        "status": (
            "post_legalization_boundary_tail_interference_supported"
            if supported
            else "inconclusive"
        ),
        "scope": "CPU analysis of frozen matched 32-case legalized reports; no training or checkpoint selection",
        "sources": {
            "control_cases": str(CONTROL.relative_to(ROOT)),
            "candidate_cases": str(CANDIDATE.relative_to(ROOT)),
            "full_screen": str(SCREEN.relative_to(ROOT)),
            "ablation": str(ABLATION.relative_to(ROOT)),
        },
        "thresholds": thresholds,
        "strata": summaries,
        "worst_v_rel_cases": worst_v_rel,
        "matched_ablation": {
            "full_vs_no_structural": full_control,
        },
        "checks": checks,
        "interpretation": (
            "The candidate improves mean raw boundary, but refinement/legalization "
            "reverses the aggregate boundary result and a small set of rung-increasing "
            "tail cases drives the V_rel failure. The block-count top quartile improves "
            "on average, so a broad high-complexity claim is not supported."
        ),
        "next_rule": (
            "Implement one boundary-preserving refiner/projector change that reduces "
            "raw-to-legalized boundary degradation, especially when legalizer rung rises; "
            "then rerun the frozen subset screen."
        ),
        "claim_boundary": "Mechanism diagnostic only; no SOTA, subgroup, or external-PPA claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
