"""Tests for paired graph-fidelity and distribution summaries."""

from __future__ import annotations

from floorflow.evaluation.residential_graph_frontier import (
    build_frontier_summary,
)


def _rows():
    rows = []
    for index in range(12):
        operation = "room_add" if index < 6 else "room_remove"
        rows.append(
            {
                "operation": operation,
                "metrics": {
                    "wall_wall_primary": {
                        "node_f1": 0.4,
                        "edge_overlap": 0.3,
                        "ged": 10.0,
                        "target_nodes": 2 + index % 7,
                        "target_edges": 5 + index,
                        "predicted_nodes": 7 + index,
                        "predicted_edges": 12 + index,
                    },
                    "direct_tiled": {
                        "node_f1": 0.7,
                        "edge_overlap": 0.6,
                        "ged": 4.0,
                        "target_nodes": 2 + index % 7,
                        "target_edges": 5 + index,
                        "predicted_nodes": 4 + index % 5,
                        "predicted_edges": 7 + index % 6,
                    },
                },
            }
        )
    return rows


def test_frontier_summary_reports_strata_tails_and_distribution() -> None:
    summary = build_frontier_summary(_rows())
    assert summary["cases"] == 12
    assert set(summary["strata"]["operations"]) == {
        "room_add",
        "room_remove",
    }
    assert summary["overall"]["paired"]["node_f1"]["delta"] > 0.0
    assert summary["overall"]["paired"]["ged"]["delta"] < 0.0
    assert summary["tail"]["candidate_ged_highest"]["cases"] == 2
    assert summary["distribution"]["candidate"]["coverage"] >= 0.0
    assert summary["efficiency"]["extra_neural_nfe"] == 0
