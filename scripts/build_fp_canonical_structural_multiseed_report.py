#!/usr/bin/env python3
"""Aggregate canonical room-graph and existing-output source audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.room_graph_matching import INSTANCE_PRIMARY_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-report", type=Path, action="append", required=True)
    parser.add_argument("--source-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def main() -> None:
    args = parse_args()
    graphs = sorted(
        (_read(path) for path in args.graph_report),
        key=lambda item: int(item["seed"]),
    )
    sources = sorted(
        (_read(path) for path in args.source_report),
        key=lambda item: int(item["seed"]),
    )
    if [item["seed"] for item in graphs] != [item["seed"] for item in sources]:
        raise ValueError("graph and source reports must cover identical seeds")
    metrics = {
        "node_f1": [
            item["summary"][INSTANCE_PRIMARY_PROTOCOL]["node_f1"] for item in graphs
        ],
        "edge_overlap": [
            item["summary"][INSTANCE_PRIMARY_PROTOCOL]["edge_overlap"]
            for item in graphs
        ],
        "ged": [item["ged"]["mean"] for item in graphs],
        "ged_exact_rate": [item["ged"]["exact_rate"] for item in graphs],
        "prediction_source_node_f1": [
            item["summary"]["prediction"]["node_f1"] for item in sources
        ],
        "prediction_source_edge_overlap": [
            item["summary"]["prediction"]["edge_overlap"] for item in sources
        ],
        "prediction_predicted_nodes": [
            item["summary"]["prediction"]["predicted_nodes"] for item in sources
        ],
        "source_target_nodes": [
            item["summary"]["prediction"]["target_nodes"] for item in sources
        ],
    }
    summary = {name: _range(values) for name, values in metrics.items()}
    graph_rows = []
    source_rows = []
    for graph, source in zip(graphs, sources, strict=True):
        graph_rows.append(
            {
                "seed": int(graph["seed"]),
                "node_f1": graph["summary"][INSTANCE_PRIMARY_PROTOCOL]["node_f1"],
                "edge_overlap": graph["summary"][INSTANCE_PRIMARY_PROTOCOL][
                    "edge_overlap"
                ],
                "ged": graph["ged"]["mean"],
                "ged_exact_rate": graph["ged"]["exact_rate"],
            }
        )
        source_rows.append(
            {
                "seed": int(source["seed"]),
                "node_f1": source["summary"]["prediction"]["node_f1"],
                "edge_overlap": source["summary"]["prediction"][
                    "edge_overlap"
                ],
                "predicted_nodes": source["summary"]["prediction"][
                    "predicted_nodes"
                ],
                "target_nodes": source["summary"]["prediction"]["target_nodes"],
            }
        )
    checks = {
        "node_f1_at_least_0_92_all_seeds": summary["node_f1"]["min"] >= 0.92,
        "edge_overlap_at_least_0_85_all_seeds": (
            summary["edge_overlap"]["min"] >= 0.85
        ),
        "ged_below_2_1_all_seeds": summary["ged"]["max"] < 2.1,
        "prediction_source_beats_topology_node_f1_all_seeds": all(
            source["summary"]["prediction"]["node_f1"]
            > source["summary"]["topology_prediction"]["node_f1"]
            for source in sources
        ),
        "prediction_source_beats_topology_edge_all_seeds": all(
            source["summary"]["prediction"]["edge_overlap"]
            > source["summary"]["topology_prediction"]["edge_overlap"]
            for source in sources
        ),
    }
    report = {
        "status": "complete",
        "seeds": [int(item["seed"]) for item in graphs],
        "edited_cases_per_seed": [int(item["cases"]) for item in graphs],
        "canonical_contract": {
            "prediction_source": "prediction",
            "readout": (
                "nearest-room footprint tiling, connected room polygons, "
                "2-pixel polygon-boundary adjacency"
            ),
            "matching": "same type, maximum IoU, accept at IoU >= 0.5",
            "ged": (
                "unit-cost attributed GED; exact within 24 nodes and 15 seconds; "
                "fallback reported explicitly"
            ),
            "deployability": "oracle-control rows; not deployable",
        },
        "summary": summary,
        "graph_rows": graph_rows,
        "source_rows": source_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "Appendix structural result under oracle controls. This does not "
            "establish a room-graph leaderboard SOTA or deployable performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Canonical Structural Metrics",
        "",
        "| Seed | Node F1 | Edge overlap | GED | Exact GED rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in graph_rows:
        lines.append(
            f"| {row['seed']} | {row['node_f1']:.6f} | "
            f"{row['edge_overlap']:.6f} | {row['ged']:.6f} | "
            f"{row['ged_exact_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Mean | Range |",
            "|---|---:|---:|",
        ]
    )
    for name in ("node_f1", "edge_overlap", "ged", "ged_exact_rate"):
        values = summary[name]
        lines.append(
            f"| {name} | {values['mean']:.6f} | "
            f"{values['min']:.6f}-{values['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Oracle-control structural appendix; not a deployable or leaderboard claim.",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
