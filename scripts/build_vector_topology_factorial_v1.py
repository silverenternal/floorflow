#!/usr/bin/env python3
"""Summarize the FEVP-2 topology projection 2x2 factorial."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "no_projection": "outputs/reports/vector_topology_no_projection_v1.json",
    "holes_only": "outputs/reports/vector_topology_holes_only_v1.json",
    "walls_only": "outputs/reports/vector_topology_walls_only_v1.json",
    "holes_and_walls": "outputs/reports/vector_topology_projection_v1.json",
    "vector_polygon_repair": "outputs/reports/vector_polygon_repair_v1.json",
}
OUTPUT = ROOT / "outputs/reports/vector_topology_factorial_v1.json"


def main() -> None:
    arms = {}
    for name, relative in SOURCES.items():
        report = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        summary = report["summary"]
        arms[name] = {
            "source": relative,
            "status": report["status"],
            "cases": summary["cases"],
            "outside_exact_cases": summary.get(
                "outside_exact_cases",
                summary.get("outside_room_exact_cases"),
            ),
            "validity": summary.get(
                "projected_validity",
                summary.get("repaired_validity"),
            ),
            "node_f1": summary.get(
                "projected_node_f1",
                summary.get("repaired_node_f1"),
            ),
            "edge_overlap": summary.get("repaired_edge_overlap"),
            "checks": report["checks"],
        }
    admissible = [
        name
        for name, arm in arms.items()
        if arm["outside_exact_cases"] == arm["cases"]
        and arm["checks"].get(
            "projected_validity_not_lower",
            arm["checks"].get("validity_not_lower", False),
        )
        and arm["checks"].get(
            "node_f1_not_lower",
            True,
        )
    ]
    report = {
        "status": "complete",
        "arms": arms,
        "admissible_arms": admissible,
        "decision": {
            "promote": [],
            "reject": [
                "holes_and_walls",
                "walls_only",
            ],
            "retain_as_safe_control": [
                "no_projection",
                "vector_polygon_repair",
            ],
            "reason": (
                "Raster wall-seam insertion materially degrades Node F1. "
                "Conservative vector-polygon repair preserves locality and "
                "graph metrics but has not yet improved validity in this sample."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "admissible": admissible}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
