#!/usr/bin/env python3
"""Calibrate differentiable routing-pressure terms on the legacy PPA holdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.batched import batch_routing_proxy_terms
from floorflow.chip_flow.ppa_external import graph_edges, placement_connectivity
from scripts.run_chip_def_adapter_suite import export_placed_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppa",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_adapter_v44.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_holdout_inventory_v44.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs/chip_orfs_external_lock_v1.json",
    )
    parser.add_argument(
        "--placement-root",
        type=Path,
        default=ROOT / "outputs/chip_ppa/placements_holdout_v44",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_routing_proxy_calibration.json",
    )
    parser.add_argument("--resolution", type=int, default=8)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float | None:
    return float((candidate - control) / control) if control != 0 else None


def rank(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {"n": len(x), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def physical_batch(graph: Any, box_report: dict[str, Any]) -> dict[str, torch.Tensor]:
    boxes_by_name = {row["name"]: row for row in box_report["macros"]}
    boxes = torch.tensor(
        [
            [
                boxes_by_name[macro["name"]]["x_um"],
                boxes_by_name[macro["name"]]["y_um"],
                boxes_by_name[macro["name"]]["width_um"],
                boxes_by_name[macro["name"]]["height_um"],
            ]
            for macro in graph.macros
        ],
        dtype=torch.float32,
    )
    edges = graph_edges(graph)
    b2b = torch.tensor(
        [[left, right, weight] for (left, right), weight in sorted(edges["b2b"].items())],
        dtype=torch.float32,
    ).reshape(-1, 3)
    port_names = [name for name in edges["used_ports"] if name in graph.ports]
    lookup = {name: index for index, name in enumerate(port_names)}
    p2b = torch.tensor(
        [
            [lookup[port], block, weight]
            for (port, block), weight in sorted(edges["p2b"].items())
            if port in lookup
        ],
        dtype=torch.float32,
    ).reshape(-1, 3)
    pins = torch.tensor(
        [graph.ports[name] for name in port_names],
        dtype=torch.float32,
    ).reshape(-1, 2)
    return {
        "boxes": boxes[None],
        "areas": (boxes[:, 2] * boxes[:, 3])[None],
        "b2b": b2b[None],
        "p2b": p2b[None],
        "pins": pins[None],
        "n_b2b": torch.tensor([len(b2b)]),
        "n_p2b": torch.tensor([len(p2b)]),
        "n_pins": torch.tensor([len(pins)]),
    }


def proxy_values(
    graph: Any,
    box_report: dict[str, Any],
    resolution: int,
) -> dict[str, float]:
    batch = physical_batch(graph, box_report)
    terms = batch_routing_proxy_terms(
        batch["boxes"],
        batch["areas"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
        resolution=resolution,
    )
    return {name: float(value[0]) for name, value in terms.items()}


def design_mean_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    result = []
    for case, selected in sorted(grouped.items()):
        design = {"case": case, "seeds": len(selected)}
        for field in fields:
            values = [row[field] for row in selected if row[field] is not None]
            design[field] = float(np.mean(values)) if values else None
        result.append(design)
    return result


def main() -> None:
    args = parse_args()
    ppa = load(args.ppa)
    config = load(args.config)
    inventory = load(args.inventory)
    lock = load(args.lock)
    nicknames = {
        row["design_key"]: row["design_nickname"] for row in inventory["inventory"]
    }
    ppa_rows = {
        (row["case"], int(row["seed"])): row
        for row in ppa["comparisons"]
        if row["status"] == "completed"
    }
    rows = []
    for design_key, case_config in sorted(config["cases"].items()):
        graph = export_placed_graph(
            argparse.Namespace(),
            lock,
            ROOT / case_config["input_odb"],
        )
        nickname = nicknames[design_key]
        for seed in ppa["seeds"]:
            ppa_row = ppa_rows.get((design_key, int(seed)))
            if ppa_row is None:
                continue
            arms = {
                arm: proxy_values(
                    graph,
                    load(args.placement_root / nickname / f"{arm}_seed{seed}.json"),
                    args.resolution,
                )
                for arm in ("candidate", "control")
            }
            connectivity = {}
            for arm in ("candidate", "control"):
                box_report = load(
                    args.placement_root / nickname / f"{arm}_seed{seed}.json"
                )
                batch = physical_batch(graph, box_report)
                connectivity[arm] = placement_connectivity(
                    batch["boxes"][0].numpy(),
                    graph,
                )
            row = {
                "case": design_key,
                "seed": int(seed),
                "routed_wirelength_relative": ppa_row[
                    "detailedroute__route__wirelength"
                ]["relative"],
                "congestion_relative": ppa_row[
                    "globalplace__gpl__routability__congestion"
                ]["relative"],
                "p2b_hpwl_relative": relative(
                    connectivity["candidate"]["p2b_hpwl_um"],
                    connectivity["control"]["p2b_hpwl_um"],
                ),
                "b2b_hpwl_relative": relative(
                    connectivity["candidate"]["b2b_hpwl_um"],
                    connectivity["control"]["b2b_hpwl_um"],
                ),
            }
            for name in arms["candidate"]:
                row[f"{name}_candidate"] = arms["candidate"][name]
                row[f"{name}_control"] = arms["control"][name]
                row[f"{name}_relative"] = relative(
                    arms["candidate"][name],
                    arms["control"][name],
                )
            rows.append(row)
    proxy_fields = [
        name
        for name in rows[0]
        if name.endswith("_relative")
        and name
        not in {
            "routed_wirelength_relative",
            "congestion_relative",
            "p2b_hpwl_relative",
            "b2b_hpwl_relative",
        }
    ]
    pair_correlations = {
        field: rank(
            [row[field] for row in rows],
            [row["routed_wirelength_relative"] for row in rows],
        )
        for field in proxy_fields
        if all(row[field] is not None for row in rows)
    }
    fields = [
        "routed_wirelength_relative",
        "congestion_relative",
        "p2b_hpwl_relative",
        "b2b_hpwl_relative",
        *proxy_fields,
    ]
    design_rows = design_mean_rows(rows, fields)
    design_correlations = {
        field: rank(
            [row[field] for row in design_rows],
            [row["routed_wirelength_relative"] for row in design_rows],
        )
        for field in [
            *proxy_fields,
            "p2b_hpwl_relative",
            "b2b_hpwl_relative",
        ]
        if all(row[field] is not None for row in design_rows)
    }
    leave_one_design_out = {}
    for field in [*proxy_fields, "p2b_hpwl_relative", "b2b_hpwl_relative"]:
        estimates = []
        for omitted in design_rows:
            selected = [row for row in design_rows if row is not omitted]
            result = rank(
                [row[field] for row in selected],
                [row["routed_wirelength_relative"] for row in selected],
            )
            if result["rho"] is not None:
                estimates.append(result["rho"])
        leave_one_design_out[field] = {
            "estimates": estimates,
            "min": min(estimates) if estimates else None,
            "max": max(estimates) if estimates else None,
            "sign_consistent": (
                all(value >= 0 for value in estimates)
                or all(value <= 0 for value in estimates)
            )
            if estimates
            else False,
        }
    report = {
        "version": "chip-v45-routing-proxy-calibration-v2",
        "status": "descriptive_complete",
        "scope": (
            "Dimensionless legacy eight-design calibration only; "
            "not confirmation evidence."
        ),
        "resolution": args.resolution,
        "proxy_fields": proxy_fields,
        "pair_correlations": pair_correlations,
        "design_correlations": design_correlations,
        "leave_one_design_out": leave_one_design_out,
        "rows": rows,
        "design_rows": design_rows,
        "selection_rule": (
            "Nominate one proxy family only if it outperforms p2b-only ranking on "
            "design-level calibration, then test it on untouched designs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "designs": len(design_rows),
                "design_correlations": design_correlations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
