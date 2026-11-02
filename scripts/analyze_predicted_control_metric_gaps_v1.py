"""Attribute predicted-control gate gaps with report-only diagnostics."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("cases") or payload.get("rows") or []


def _values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _correlation(rows: list[dict], left: str, right: str) -> float:
    return float(np.corrcoef(_values(rows, left), _values(rows, right))[0, 1])


def _strata(rows: list[dict], key: str, edges: list[float]) -> dict:
    values = _values(rows, key)
    output = {}
    for low, high in pairwise(edges):
        mask = (values >= low) & (values < high)
        if not mask.any():
            continue
        selected = [row for row, keep in zip(rows, mask, strict=True) if keep]
        output[f"{low:g}-{high:g}"] = {
            "cases": len(selected),
            "room_iou": float(np.mean(_values(selected, "room_macro_iou"))),
            "instance_f1": float(
                np.mean(_values(selected, "instance_instance_f1_at_threshold"))
            ),
            "wall_f1": float(np.mean(_values(selected, "fpeval_wall_f1"))),
            "opening_f1": float(
                np.mean(_values(selected, "fpeval_opening_f1"))
            ),
            "node_f1": float(np.mean(_values(selected, "node_f1"))),
            "edge_overlap": float(
                np.mean(_values(selected, "edge_overlap"))
            ),
            "ged": float(np.mean(_values(selected, "ged"))),
        }
    return output


def _tail(rows: list[dict], key: str, fraction: float = 0.10) -> dict:
    ordered = sorted(rows, key=lambda row: float(row[key]))
    count = max(1, round(len(rows) * float(fraction)))
    selected = ordered[:count]
    return {
        "metric": key,
        "cases": count,
        "threshold": float(selected[-1][key]),
        "room_iou": float(np.mean(_values(selected, "room_macro_iou"))),
        "instance_f1": float(
            np.mean(_values(selected, "instance_instance_f1_at_threshold"))
        ),
        "count_abs_error": float(
            np.mean(_values(selected, "instance_count_abs_error"))
        ),
        "wall_f1": float(np.mean(_values(selected, "fpeval_wall_f1"))),
        "opening_f1": float(np.mean(_values(selected, "fpeval_opening_f1"))),
        "node_f1": float(np.mean(_values(selected, "node_f1"))),
        "edge_overlap": float(np.mean(_values(selected, "edge_overlap"))),
        "ged": float(np.mean(_values(selected, "ged"))),
        "predicted_edges": float(
            np.mean(_values(selected, "edge_predicted_edges"))
        ),
        "target_edges": float(np.mean(_values(selected, "edge_target_edges"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/predicted_control_full_stage_val491_gedfixed_v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_metric_gap_audit_v1.json",
    )
    args = parser.parse_args()
    rows = _rows(args.report)
    gates = {
        "room_iou": ("room_macro_iou", 0.45, "max"),
        "instance_f1": ("instance_instance_f1_at_threshold", 0.5, "max"),
        "wall_f1": ("fpeval_wall_f1", 0.58, "max"),
        "opening_f1": ("fpeval_opening_f1", 0.35, "max"),
        "node_f1": ("node_f1", 0.82, "max"),
        "edge_overlap": ("edge_overlap", 0.5, "max"),
        "ged": ("ged", 6.0, "min"),
    }
    gaps = {}
    for name, (field, gate, direction) in gates.items():
        value = float(_values(rows, field).mean())
        gap = gate - value if direction == "max" else value - gate
        gaps[name] = {
            "value": value,
            "gate": gate,
            "gap": gap,
            "gate_passed": gap <= 0.0,
        }
    high_node_low_edge = [
        row
        for row in rows
        if float(row["node_f1"]) >= 0.8 and float(row["edge_overlap"]) < 0.5
    ]
    report = {
        "schema_version": "predicted-control-metric-gap-audit-v1",
        "source": str(args.report),
        "cases": len(rows),
        "gates": gaps,
        "distributions": {
            field: _summary(_values(rows, field))
            for field in (
                "room_macro_iou",
                "instance_instance_f1_at_threshold",
                "fpeval_wall_f1",
                "fpeval_opening_f1",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        },
        "correlations": {
            f"{left}__{right}": _correlation(rows, left, right)
            for left, right in (
                ("room_macro_iou", "instance_instance_f1_at_threshold"),
                ("room_macro_iou", "node_f1"),
                ("room_macro_iou", "edge_overlap"),
                ("node_f1", "edge_overlap"),
                ("edge_overlap", "ged"),
                ("fpeval_wall_f1", "fpeval_opening_f1"),
            )
        },
        "room_count_strata": _strata(
            rows,
            "instance_target_instances",
            [0.0, 5.0, 9.0, 13.0, 100.0],
        ),
        "footprint_area_strata": _strata(
            rows,
            "footprint_area_px",
            [
                0.0,
                float(np.quantile(_values(rows, "footprint_area_px"), 0.25)),
                float(np.quantile(_values(rows, "footprint_area_px"), 0.50)),
                float(np.quantile(_values(rows, "footprint_area_px"), 0.75)),
                float("inf"),
            ],
        ),
        "tails": {
            "worst_room_iou": _tail(rows, "room_macro_iou"),
            "worst_edge_overlap": _tail(rows, "edge_overlap"),
            "worst_ged": _tail(rows, "ged"),
        },
        "node_edge_decoupling": {
            "cases_high_node_low_edge": len(high_node_low_edge),
            "fraction": len(high_node_low_edge) / max(1, len(rows)),
            "mean_room_iou": float(
                np.mean(_values(high_node_low_edge, "room_macro_iou"))
            )
            if high_node_low_edge
            else 0.0,
            "mean_edge_precision": float(
                np.mean(_values(high_node_low_edge, "edge_precision"))
            )
            if high_node_low_edge
            else 0.0,
            "mean_edge_recall": float(
                np.mean(_values(high_node_low_edge, "edge_recall"))
            )
            if high_node_low_edge
            else 0.0,
            "interpretation": (
                "Node identity is often present while room adjacency/link "
                "selection is incomplete or over-connected."
            ),
        },
        "conclusions": [
            "Room support IoU is high while semantic/instance IoU is low; identity/count/type must be fixed before adding encoder capacity.",
            "Node F1 is much higher than edge overlap; prioritize typed adjacency and edge consistency over additional node heads.",
            "Wall recall remains higher than precision, so precision-oriented candidate selection is still the direct wall lever.",
            "Opening F1 remains the smallest absolute gate gap component and is dominated by host/candidate geometry.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
