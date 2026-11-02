#!/usr/bin/env python3
"""Audit whether existing OpenROAD feedback predicts routed placement quality.

This is a development-only signal audit. It does not select a candidate, tune an
inference hyperparameter, or turn an ORFS runtime metric into a deployment-time
oracle. The goal is to decide whether a standard-cell/analytical response path is
worth adding to the existing chip model before spending GPU time on training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from floorflow.chip_flow.analytical_feedback import seed_portfolio_diagnostic

ROOT = Path(__file__).resolve().parents[1]

PRE_ROUTE_SIGNALS = (
    "globalplace__route__wirelength__estimated",
    "globalplace__gpl__routability__congestion",
    "detailedplace__dpl__route__wirelength__estimated",
    "detailedplace__design__instance__utilization__stdcell",
)

STATIC_FEATURES = (
    "floorplan__design__instance__count__macros",
    "floorplan__design__instance__count__stdcell",
    "floorplan__design__nets",
    "floorplan__design__io",
    "floorplan__design__core__area",
    "floorplan__design__instance__area__macros",
    "floorplan__design__instance__area__stdcell",
    "floorplan__design__instance__utilization__stdcell",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppa",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_partial.json",
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_external_ppa_v45_partial_endpoint_certification.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_v45.json",
    )
    parser.add_argument(
        "--p2b-rootcause",
        type=Path,
        default=ROOT / "outputs/reports/chip_p2b_routing_rootcause_v44_holdout18.json",
    )
    parser.add_argument(
        "--typed-audit",
        type=Path,
        default=ROOT / "outputs/reports/chip_typed_graph_cpu_v1.json",
    )
    parser.add_argument(
        "--multi-seed-ppa",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    )
    parser.add_argument(
        "--v44-inventory",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_holdout_inventory_v44.json",
    )
    parser.add_argument(
        "--orfs-work-home",
        type=Path,
        default=ROOT / "outputs/chip_ppa/orfs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_signal_audit.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: Any, control: Any) -> float | None:
    if candidate is None or control is None:
        return None
    candidate_value = float(candidate)
    control_value = float(control)
    if control_value == 0.0:
        return None
    return candidate_value / control_value - 1.0


def rank(x: list[float], y: list[float]) -> dict[str, Any]:
    finite = [
        (left, right)
        for left, right in zip(x, y, strict=True)
        if np.isfinite(left) and np.isfinite(right)
    ]
    if len(finite) < 3:
        return {"n": len(finite), "rho": None, "p_value": None}
    left = [row[0] for row in finite]
    right = [row[1] for row in finite]
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {"n": len(finite), "rho": None, "p_value": None}
    result = spearmanr(left, right)
    return {
        "n": len(finite),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def metric_delta(
    candidate: dict[str, Any],
    control: dict[str, Any],
    key: str,
) -> float | None:
    candidate_value = candidate.get(key)
    control_value = control.get(key)
    if candidate_value in (None, "N/A", "ERR") or control_value in (None, "N/A", "ERR"):
        return None
    return float(candidate_value) - float(control_value)


def rank_summary(
    rows: list[dict[str, Any]],
    signal: str,
) -> dict[str, Any]:
    pair_rank = rank(
        [row[signal] for row in rows],
        [row["routed_wirelength_relative"] for row in rows],
    )
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_design[row["case"]].append(row)
        by_family[row["source_family"]].append(row)
    design_rows = [
        {
            "case": case,
            "source_family": selected[0]["source_family"],
            signal: float(np.mean([row[signal] for row in selected])),
            "routed_wirelength_relative": float(
                np.mean([row["routed_wirelength_relative"] for row in selected])
            ),
        }
        for case, selected in sorted(by_design.items())
    ]
    design_rank = rank(
        [row[signal] for row in design_rows],
        [row["routed_wirelength_relative"] for row in design_rows],
    )
    family_leave_one_out = {}
    for omitted in sorted(by_family):
        selected = [row for row in rows if row["source_family"] != omitted]
        family_leave_one_out[omitted] = rank(
            [row[signal] for row in selected],
            [row["routed_wirelength_relative"] for row in selected],
        )
    estimates = [
        result["rho"]
        for result in family_leave_one_out.values()
        if result["rho"] is not None
    ]
    design_leave_one_out = {}
    for omitted in sorted(by_design):
        selected = [row for row in rows if row["case"] != omitted]
        design_leave_one_out[omitted] = rank(
            [row[signal] for row in selected],
            [row["routed_wirelength_relative"] for row in selected],
        )
    design_estimates = [
        result["rho"]
        for result in design_leave_one_out.values()
        if result["rho"] is not None
    ]
    return {
        "pair": pair_rank,
        "design": design_rank,
        "leave_one_design_out": design_leave_one_out,
        "design_sign_consistent": (
            all(value >= 0 for value in design_estimates)
            or all(value <= 0 for value in design_estimates)
        )
        if design_estimates
        else False,
        "leave_one_family_out": family_leave_one_out,
        "family_sign_consistent": (
            all(value >= 0 for value in estimates)
            or all(value <= 0 for value in estimates)
        )
        if estimates
        else False,
    }


def main() -> None:
    args = parse_args()
    endpoints = load(args.endpoints)
    manifest = load(args.manifest)
    p2b = load(args.p2b_rootcause)
    typed = load(args.typed_audit)
    p2b_rows = [
        row for row in p2b["comparisons"] if row["status"] == "completed"
    ]
    p2b_selection = {}
    for signal in ("p2b_hpwl_relative", "b2b_hpwl_relative"):
        realized = [
            (
                row["routed_wirelength_relative"]
                if row[signal] < 0.0
                else 0.0
            )
            for row in p2b_rows
        ]
        p2b_selection[signal] = {
            "correct_sign": sum(
                (row[signal] < 0.0)
                == (row["routed_wirelength_relative"] < 0.0)
                for row in p2b_rows
            ),
            "selected_routed_wirelength_relative_mean": float(
                np.mean(realized)
            ),
            "selected_routed_wirelength_relative_worst": float(max(realized)),
        }
    rows = []
    timing_rows = []
    for pair in endpoints["pairs"]:
        candidate = load(ROOT / pair["candidate_path"])
        control = load(ROOT / pair["control_path"])
        case = pair["case"]
        case_manifest = manifest["cases"][case]
        row = {
            "case": case,
            "source_family": case_manifest["source_family"],
            "globalplace_seconds": candidate.get("globalplace__elapsed_seconds"),
            "routed_wirelength_relative": relative(
                candidate["detailedroute__route__wirelength"],
                control["detailedroute__route__wirelength"],
            ),
        }
        for signal in PRE_ROUTE_SIGNALS:
            row[signal] = relative(candidate.get(signal), control.get(signal))
        for feature in STATIC_FEATURES:
            row[feature] = candidate.get(feature)
        rows.append(row)
        timing_rows.append(
            {
                "case": case,
                "globalplace_ws_delta_ns": metric_delta(
                    candidate,
                    control,
                    "globalplace__timing__setup__ws",
                ),
                "finish_ws_delta_ns": metric_delta(
                    candidate,
                    control,
                    "finish__timing__setup__ws",
                ),
                "globalplace_tns_delta_ns": metric_delta(
                    candidate,
                    control,
                    "globalplace__timing__setup__tns",
                ),
                "finish_tns_delta_ns": metric_delta(
                    candidate,
                    control,
                    "finish__timing__setup__tns",
                ),
            }
        )
    rows = [
        row
        for row in rows
        if row["routed_wirelength_relative"] is not None
    ]
    signal_ranks = {
        signal: rank_summary(rows, signal)
        for signal in PRE_ROUTE_SIGNALS
        if all(row[signal] is not None for row in rows)
    }
    binary_selection = {}
    for signal in signal_ranks:
        predicted = [row[signal] for row in rows]
        realized = [
            row["routed_wirelength_relative"] if row[signal] < 0.0 else 0.0
            for row in rows
        ]
        binary_selection[signal] = {
            "candidate_selected": sum(value < 0.0 for value in predicted),
            "correct_sign": sum(
                (row[signal] < 0.0)
                == (row["routed_wirelength_relative"] < 0.0)
                for row in rows
            ),
            "selected_routed_wirelength_relative_mean": float(
                np.mean(realized)
            ),
            "selected_routed_wirelength_relative_worst": float(max(realized)),
        }
    globalplace_seconds = [
        float(row["globalplace_seconds"])
        for row in rows
        if row["globalplace_seconds"] is not None
    ]
    design_rows = []
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        design_rows.append(
            {
                "case": case,
                "source_family": selected[0]["source_family"],
                **{
                    feature: float(np.mean([row[feature] for row in selected]))
                    for feature in STATIC_FEATURES
                },
                "routed_wirelength_relative": float(
                    np.mean(
                        [row["routed_wirelength_relative"] for row in selected]
                    )
                ),
            }
        )
    channel_contract = typed["channel_contract"]
    timing_ws = [
        row
        for row in timing_rows
        if row["globalplace_ws_delta_ns"] is not None
        and row["finish_ws_delta_ns"] is not None
    ]
    timing_tns = [
        row
        for row in timing_rows
        if row["globalplace_tns_delta_ns"] is not None
        and row["finish_tns_delta_ns"] is not None
    ]
    report = {
        "version": "chip-v46-downstream-signal-audit-v1",
        "status": "descriptive_complete",
        "scope": (
            "Observed v45 development pairs only. This audit cannot serve as a "
            "confirmation result for a newly selected mechanism."
        ),
        "sample": {
            "pairs": len(rows),
            "designs": len(design_rows),
            "families": len({row["source_family"] for row in rows}),
        },
        "current_model_pathway": {
            "available": channel_contract["available"],
            "unavailable": channel_contract["unavailable"],
            "standard_cell_density_field": "absent",
            "analytical_global_placement_feedback": "external_evaluator_only",
            "routing_capacity_grid": "absent",
            "evidence": str(args.typed_audit.relative_to(ROOT)),
        },
        "reference_failures": {
            "source": str(args.p2b_rootcause.relative_to(ROOT)),
            "p2b_to_routed": p2b["rank_correlations"]["p2b_to_routed"],
            "b2b_to_routed": p2b["rank_correlations"]["b2b_to_routed"],
            "binary_selection": p2b_selection,
            "interpretation": (
                "Macro-only connectivity proxies are not routed-wirelength "
                "endpoints or validated deployment-time scores."
            ),
        },
        "pre_route_signal_ranks": signal_ranks,
        "secondary_endpoint_transfer": {
            "globalplace_ws_to_finish_ws": rank(
                [row["globalplace_ws_delta_ns"] for row in timing_ws],
                [row["finish_ws_delta_ns"] for row in timing_ws],
            ),
            "globalplace_tns_to_finish_tns": rank(
                [row["globalplace_tns_delta_ns"] for row in timing_tns],
                [row["finish_tns_delta_ns"] for row in timing_tns],
            ),
            "interpretation": (
                "Global-place WL is not a substitute for post-route timing. "
                "Timing remains a separate certified guard."
            ),
        },
        "binary_selection_diagnostic": binary_selection,
        "seed_portfolio_diagnostic": seed_portfolio_diagnostic(
            args.multi_seed_ppa,
            args.v44_inventory,
            args.orfs_work_home,
        ),
        "physical_feedback_cost": {
            "globalplace_seconds": {
                "n": len(globalplace_seconds),
                "median": float(np.median(globalplace_seconds))
                if globalplace_seconds
                else None,
                "p90": float(np.quantile(globalplace_seconds, 0.9))
                if globalplace_seconds
                else None,
                "max": max(globalplace_seconds) if globalplace_seconds else None,
                "interpretation": (
                    "A runtime global-placement response is deployable only after "
                    "including this latency in the equal-total-budget comparison."
                ),
            }
        },
        "static_features_available_in_existing_metrics": list(STATIC_FEATURES),
        "rows": rows,
        "design_rows": design_rows,
        "decision_rule": {
            "minimum_evidence_for_model_change": (
                "A deployable pre-route signal must improve design-level out-of-fold "
                "routing prediction over p2b-only and retain a consistent sign under "
                "leave-one-family-out evaluation."
            ),
            "current_status": (
                "audit_only; the observed v45 set cannot confirm a mechanism and the "
                "small family-clustered sample is not sufficient for a final claim"
            ),
        },
        "engineering": {
            "new_file_justification": (
                "No existing audit joined certified v45 endpoint pairs to full "
                "OpenROAD stage metrics and leave-one-family-out signal ranks."
            ),
            "reused": [
                "outputs/reports/chip_external_ppa_v45_partial_endpoint_certification.json",
                "outputs/reports/chip_p2b_routing_rootcause_v44_holdout18.json",
                "outputs/reports/chip_typed_graph_cpu_v1.json",
                "OpenROAD METRICS_2.1.2 stage keys",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sample": report["sample"],
                "pre_route_signal_ranks": signal_ranks,
                "binary_selection_diagnostic": binary_selection,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
