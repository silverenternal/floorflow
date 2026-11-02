#!/usr/bin/env python3
"""Connect external structural changes to legalization and routed PPA."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.ppa_external import placement_connectivity
from scripts.run_chip_def_adapter_suite import export_placed_graph

DEFAULT_PPA = ROOT / "outputs/reports/chip_external_ppa_subset_v1.json"
DEFAULT_ADAPTER = ROOT / "outputs/reports/chip_def_adapter"
DEFAULT_CONFIG = ROOT / "configs/chip_external_adapter_v1.json"
DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_p2b_mechanism_trace_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppa", type=Path, default=DEFAULT_PPA)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-graph-export", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control in (None, 0):
        return None
    return float((candidate - control) / control)


def rank_correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "n": len(x),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def design_cluster_interval(
    values: list[float],
    *,
    seed: int = 20261029,
    iterations: int = 10000,
) -> dict[str, Any]:
    """Bootstrap the mean over independent designs, not repeated seed pairs."""

    if not values:
        return {"designs": 0, "mean": None, "ci_low": None, "ci_high": None}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array,
        size=(iterations, len(array)),
        replace=True,
    ).mean(axis=1)
    return {
        "designs": len(values),
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "iterations": iterations,
        "seed": seed,
    }


def ratio(candidate: float | None, control: float | None) -> float | None:
    return relative(candidate, control)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def path_mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def placement_row(root: Path, nickname: str, arm: str, seed: int) -> dict[str, Any]:
    path = root / nickname / f"{arm}_seed{seed}.json"
    return load_json(path) if path.is_file() else {}


def load_case_graphs(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    lock = load_json(args.lock)
    inventory = {
        row["design_key"]: row
        for row in manifest["inventory"]
    }
    graphs = {}
    for design_key, case_config in config["cases"].items():
        case = inventory.get(design_key, {})
        odb = ROOT / str(case_config["input_odb"])
        if not odb.is_file() or case.get("design_key") != design_key:
            continue
        graphs[design_key] = export_placed_graph(
            argparse.Namespace(),
            lock,
            odb,
        )
    return graphs


def case_nicknames(args: argparse.Namespace) -> dict[str, str]:
    manifest = load_json(args.manifest)
    return {
        row["design_key"]: row["design_nickname"]
        for row in manifest["inventory"]
    }


def boxes_connectivity(
    boxes_report: dict[str, Any],
    graph: Any,
) -> dict[str, float]:
    configured = {
        row["name"]: row
        for row in boxes_report.get("macros", [])
    }
    boxes = np.asarray(
        [
            [
                configured[macro["name"]]["x_um"],
                configured[macro["name"]]["y_um"],
                configured[macro["name"]]["width_um"],
                configured[macro["name"]]["height_um"],
            ]
            for macro in graph.macros
        ],
        dtype=np.float64,
    )
    return placement_connectivity(boxes, graph)


def resolved_connectivity(
    placement: dict[str, Any],
    boxes_report: dict[str, Any],
    graph: Any | None,
) -> tuple[dict[str, float], str | None]:
    legalized = placement.get("connectivity", {}).get("legalized_output")
    if legalized and "b2b_hpwl_um" in legalized and "p2b_hpwl_um" in legalized:
        return legalized, "adapter_report"
    if graph is not None and boxes_report:
        try:
            return boxes_connectivity(boxes_report, graph), "reconstructed_boxes"
        except (KeyError, TypeError, ValueError):
            return {}, None
    return {}, None


def run_lookup(report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["case"], row["arm"], int(row["seed"])): row
        for row in report["runs"]
    }


def main() -> None:
    args = parse_args()
    ppa = load_json(args.ppa)
    runs = run_lookup(ppa)
    graphs = {} if args.skip_graph_export else load_case_graphs(args)
    nicknames = case_nicknames(args)
    comparisons = []
    for case in ppa["cases"]:
        nickname = nicknames[case]
        for seed in ppa["seeds"]:
            candidate = runs.get((case, "candidate", int(seed)), {})
            control = runs.get((case, "control", int(seed)), {})
            candidate_place = placement_row(args.adapter_root, nickname, "candidate", seed)
            control_place = placement_row(args.adapter_root, nickname, "control", seed)
            candidate_metrics = candidate.get("metrics_values", {})
            control_metrics = control.get("metrics_values", {})
            boxes_root = ROOT / "outputs/chip_ppa/placements" / nickname
            candidate_boxes = placement_row(
                boxes_root.parent,
                boxes_root.name,
                "candidate",
                seed,
            )
            control_boxes = placement_row(
                boxes_root.parent,
                boxes_root.name,
                "control",
                seed,
            )
            candidate_conn, candidate_conn_source = resolved_connectivity(
                candidate_place,
                candidate_boxes,
                graphs.get(case),
            )
            control_conn, control_conn_source = resolved_connectivity(
                control_place,
                control_boxes,
                graphs.get(case),
            )
            candidate_geom = candidate_place.get("geometry", {})
            control_geom = control_place.get("geometry", {})
            candidate_disp = candidate_place.get("legalization_displacement", {})
            control_disp = control_place.get("legalization_displacement", {})
            candidate_graph = candidate_place.get("graph", {})
            candidate_overlap = candidate_geom.get("model_output", {}).get(
                "overlap_pair_count"
            )
            control_overlap = control_geom.get("model_output", {}).get(
                "overlap_pair_count"
            )
            comparisons.append(
                {
                    "case": case,
                    "seed": int(seed),
                    "status": "completed"
                    if candidate.get("status") == "completed"
                    and control.get("status") == "completed"
                    else "incomplete",
                    "macro_count": candidate_place.get("graph", {}).get("macros"),
                    "legalizer_routes": {
                        "candidate": candidate_place.get("legalizer_route"),
                        "control": control_place.get("legalizer_route"),
                    },
                    "connectivity_source": {
                        "candidate": candidate_conn_source,
                        "control": control_conn_source,
                    },
                    "raw_overlap_pairs": {
                        "candidate": candidate_overlap,
                        "control": control_overlap,
                        "delta": (
                            candidate_overlap - control_overlap
                            if candidate_overlap is not None
                            and control_overlap is not None
                            else None
                        ),
                    },
                    "legalization_displacement_p90_um": {
                        "candidate": candidate_disp.get("p90_um"),
                        "control": control_disp.get("p90_um"),
                        "relative": ratio(
                            candidate_disp.get("p90_um"),
                            control_disp.get("p90_um"),
                        ),
                    },
                    "graph_features": {
                        "macros": candidate_graph.get("macros"),
                        "ports": candidate_graph.get("ports"),
                        "b2b_edges": candidate_graph.get("b2b_edges"),
                        "p2b_edges": candidate_graph.get("p2b_edges"),
                        "b2b_edges_per_macro": safe_div(
                            candidate_graph.get("b2b_edges"),
                            candidate_graph.get("macros"),
                        ),
                        "p2b_edges_per_macro": safe_div(
                            candidate_graph.get("p2b_edges"),
                            candidate_graph.get("macros"),
                        ),
                        "ports_per_macro": safe_div(
                            candidate_graph.get("ports"),
                            candidate_graph.get("macros"),
                        ),
                    },
                    "b2b_hpwl_relative": relative(
                        candidate_conn.get("b2b_hpwl_um"),
                        control_conn.get("b2b_hpwl_um"),
                    ),
                    "p2b_hpwl_relative": relative(
                        candidate_conn.get("p2b_hpwl_um"),
                        control_conn.get("p2b_hpwl_um"),
                    ),
                    "routed_wirelength_relative": relative(
                        candidate_metrics.get("detailedroute__route__wirelength"),
                        control_metrics.get("detailedroute__route__wirelength"),
                    ),
                    "congestion_relative": relative(
                        candidate_metrics.get("globalplace__gpl__routability__congestion"),
                        control_metrics.get("globalplace__gpl__routability__congestion"),
                    ),
                    "timing_ws_delta_ns": (
                        candidate_metrics.get("finish__timing__setup__ws")
                        - control_metrics.get("finish__timing__setup__ws")
                        if candidate_metrics.get("finish__timing__setup__ws") is not None
                        and control_metrics.get("finish__timing__setup__ws") is not None
                        else None
                    ),
                    "power_relative": relative(
                        candidate_metrics.get("finish__power__total"),
                        control_metrics.get("finish__power__total"),
                    ),
                    "stdcell_area_relative": relative(
                        candidate_metrics.get("finish__design__instance__area__stdcell"),
                        control_metrics.get("finish__design__instance__area__stdcell"),
                    ),
                    "drc": {
                        "candidate": candidate_metrics.get(
                            "detailedroute__route__drc_errors"
                        ),
                        "control": control_metrics.get(
                            "detailedroute__route__drc_errors"
                        ),
                    },
                }
            )
            row = comparisons[-1]
            if row["p2b_hpwl_relative"] is not None:
                row["p2b_minus_routed_relative"] = (
                    row["p2b_hpwl_relative"] - row["routed_wirelength_relative"]
                )
            else:
                row["p2b_minus_routed_relative"] = None
    completed = [row for row in comparisons if row["status"] == "completed"]
    route_counts = {}
    for arm in ("candidate", "control"):
        route_counts[arm] = dict(
            Counter(
                row["legalizer_routes"][arm]
                for row in comparisons
                if row["legalizer_routes"][arm] is not None
            )
        )
    routed = [row["routed_wirelength_relative"] for row in completed]
    b2b = [row["b2b_hpwl_relative"] for row in completed]
    p2b = [row["p2b_hpwl_relative"] for row in completed]
    congestion = [row["congestion_relative"] for row in completed]
    route_pairs = [
        (row["b2b_hpwl_relative"], row["routed_wirelength_relative"])
        for row in completed
        if row["b2b_hpwl_relative"] is not None
        and row["routed_wirelength_relative"] is not None
    ]
    port_pairs = [
        (row["p2b_hpwl_relative"], row["routed_wirelength_relative"])
        for row in completed
        if row["p2b_hpwl_relative"] is not None
        and row["routed_wirelength_relative"] is not None
    ]
    overlap_pairs = [
        (
            row["raw_overlap_pairs"]["delta"],
            row["routed_wirelength_relative"],
        )
        for row in completed
        if row["raw_overlap_pairs"]["delta"] is not None
        and row["routed_wirelength_relative"] is not None
    ]
    by_route = {}
    for route_pair in sorted(
        {
            (
                row["legalizer_routes"]["candidate"],
                row["legalizer_routes"]["control"],
            )
            for row in completed
        }
    ):
        selected = [
            row["routed_wirelength_relative"]
            for row in completed
            if (
                row["legalizer_routes"]["candidate"],
                row["legalizer_routes"]["control"],
            )
            == route_pair
            and row["routed_wirelength_relative"] is not None
        ]
        by_route[str(route_pair)] = {
            "pairs": len(selected),
            "routed_wirelength_relative_mean": float(np.mean(selected))
            if selected
            else None,
        }
    correlations = {
        "b2b_to_routed": rank_correlation(
            [row[0] for row in route_pairs],
            [row[1] for row in route_pairs],
        ),
        "p2b_to_routed": rank_correlation(
            [row[0] for row in port_pairs],
            [row[1] for row in port_pairs],
        ),
        "raw_overlap_delta_to_routed": rank_correlation(
            [row[0] for row in overlap_pairs],
            [row[1] for row in overlap_pairs],
        ),
    }
    design_rows = []
    for case in ppa["cases"]:
        rows = [row for row in completed if row["case"] == case]
        if not rows:
            continue

        design_rows.append(
            {
                "case": case,
                "seeds": len(rows),
                "macro_count": path_mean(rows, ("graph_features", "macros")),
                "b2b_edges_per_macro": path_mean(
                    rows,
                    ("graph_features", "b2b_edges_per_macro")
                ),
                "p2b_edges_per_macro": path_mean(
                    rows,
                    ("graph_features", "p2b_edges_per_macro")
                ),
                "ports_per_macro": path_mean(
                    rows,
                    ("graph_features", "ports_per_macro")
                ),
                "displacement_p90_relative": path_mean(
                    rows,
                    ("legalization_displacement_p90_um", "relative")
                ),
                "p2b_hpwl_relative": path_mean(rows, ("p2b_hpwl_relative",)),
                "b2b_hpwl_relative": path_mean(rows, ("b2b_hpwl_relative",)),
                "routed_wirelength_relative": path_mean(
                    rows,
                    ("routed_wirelength_relative",)
                ),
                "congestion_relative": path_mean(rows, ("congestion_relative",)),
                "p2b_minus_routed_relative": path_mean(
                    rows,
                    ("p2b_minus_routed_relative",)
                ),
            }
        )
    design_correlations = {}
    for feature in (
        "macro_count",
        "b2b_edges_per_macro",
        "p2b_edges_per_macro",
        "ports_per_macro",
        "displacement_p90_relative",
        "p2b_hpwl_relative",
        "b2b_hpwl_relative",
    ):
        pairs = [
            (row[feature], row["routed_wirelength_relative"])
            for row in design_rows
            if row[feature] is not None
            and row["routed_wirelength_relative"] is not None
        ]
        design_correlations[feature] = rank_correlation(
            [row[0] for row in pairs],
            [row[1] for row in pairs],
        )
    routed_designs = [
        row["routed_wirelength_relative"]
        for row in design_rows
        if row["routed_wirelength_relative"] is not None
    ]
    feature_groups = {}
    for feature in ("macro_count", "p2b_edges_per_macro", "ports_per_macro"):
        ranked = sorted(
            (
                row
                for row in design_rows
                if row[feature] is not None
                and row["routed_wirelength_relative"] is not None
            ),
            key=lambda row: row[feature],
        )
        if len(ranked) >= 4:
            midpoint = len(ranked) // 2
            feature_groups[feature] = {
                "low": [
                    row["routed_wirelength_relative"]
                    for row in ranked[:midpoint]
                ],
                "high": [
                    row["routed_wirelength_relative"]
                    for row in ranked[midpoint:]
                ],
            }
    report = {
        "version": "2026-10-29-chip-routing-rootcause-v45",
        "status": "descriptive_complete"
        if len(completed) == int(ppa.get("completed_pairs", len(completed)))
        else "incomplete",
        "scope": (
            "Frozen external macro-placement holdout; descriptive, not a formal theorem"
        ),
        "case_count": len(ppa["cases"]),
        "seed_count": len(ppa["seeds"]),
        "completed_pairs": len(completed),
        "legalizer_route_counts": route_counts,
        "legalizer_route_strata": by_route,
        "relative_means": {
            "b2b_macro_hpwl": float(np.mean([value for value in b2b if value is not None]))
            if any(value is not None for value in b2b)
            else None,
            "p2b_macro_hpwl": float(np.mean([value for value in p2b if value is not None]))
            if any(value is not None for value in p2b)
            else None,
            "routed_wirelength": float(np.mean([value for value in routed if value is not None]))
            if any(value is not None for value in routed)
            else None,
            "congestion": float(
                np.mean([value for value in congestion if value is not None])
            )
            if any(value is not None for value in congestion)
            else None,
        },
        "rank_correlations": correlations,
        "design_cluster_routed_wirelength": design_cluster_interval(routed_designs),
        "design_rows": design_rows,
        "design_rank_correlations": design_correlations,
        "design_feature_groups": feature_groups,
        "proxy_selection_constraint": (
            "The eight legacy designs can nominate at most one proxy family. "
            "The proxy must be confirmed on seven or more untouched designs."
        ),
        "comparisons": comparisons,
        "limitations": [
            "Three seeds per case are repeated measurements, not independent designs.",
            "The legacy external set is too small for a stable nonlinear feature-selection claim.",
            "Matched legalizer routes remove the first-order route confound but do not isolate macro geometry from downstream timing repair.",
            "The adapter's pin positions come from a common baseline I/O-placement snapshot.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed_pairs": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
