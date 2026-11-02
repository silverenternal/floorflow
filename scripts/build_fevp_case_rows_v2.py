#!/usr/bin/env python3
"""Join hardened evidence into schema-validated FEVP v2 case rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_rows_v2.json",
    )
    parser.add_argument(
        "--validity",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_rows_v2.json",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_rows_v2.json",
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_rows_v2.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "schema/fevp_case_row_v2.schema.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_case_rows_v2.json",
    )
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _key(seed: int, operation: str, name: str) -> str:
    return f"{seed}|{operation}|{name}"


def _validate(row: dict, schema: dict) -> list[str]:
    errors = []
    for key in schema["required"]:
        if key not in row:
            errors.append(f"missing {key}")
    for key in (
        "outside_pixel_exact",
        "outside_room_level_exact",
        "outside_room_label_exact",
        "outside_room_polygon_exact",
        "outside_edge_exact",
        "locality_violation_rate",
        "validity",
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "edge_overlap",
    ):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{key} out of range")
    if row.get("schema_version") != "fevp-case-row-v2":
        errors.append("wrong schema_version")
    checks = row.get("validity_checks", {})
    for key in (
        "outer_boundary_closed",
        "room_polygon_closed",
        "no_self_intersection",
        "no_room_overlap",
        "semantic_complete",
    ):
        if key not in checks:
            errors.append(f"missing validity_checks.{key}")
    diagnostics = row.get("watertight_diagnostics", {})
    for key in (
        "hole_count",
        "hole_area_ratio",
        "disconnected_component_count",
        "dangling_endpoint_rate",
        "watertight_issue_codes",
    ):
        if key not in diagnostics:
            errors.append(f"missing watertight_diagnostics.{key}")
    return errors


def main() -> None:
    args = parse_args()
    outside = {
        _key(row["seed"], row["operation"], row["name"]): row
        for row in _read(args.outside)
    }
    validity = {
        _key(row["seed"], row["operation"], row["name"]): row
        for row in _read(args.validity)
    }
    graph = {
        _key(row["seed"], row["operation"], row["case_id"]): row
        for row in _read(args.graph)
    }
    delta = {
        _key(row["seed"], row["operation"], row["case_id"]): row
        for row in _read(args.delta)
    }
    keys = sorted(outside)
    if not all(set(table) == set(keys) for table in (validity, graph, delta)):
        raise ValueError("hardened case tables do not share identical keys")
    rows = []
    for key in keys:
        outer = outside[key]
        valid = validity[key]
        graph_row = graph[key]
        delta_row = delta[key]
        arm = graph_row["arms"]["typed_edit_graph"]
        strict = outer["strict"]
        row = {
            "schema_version": "fevp-case-row-v2",
            "case_id": outer["case_id"],
            "seed": outer["seed"],
            "operation": outer["operation"],
            "source_hash": outer["source_hash"],
            "outside_pixel_exact": strict["outside_pixel_exact"],
            "outside_room_level_exact": strict["outside_room_level_exact"],
            "outside_room_label_exact": strict["outside_room_label_exact"],
            "outside_room_polygon_exact": strict["outside_room_polygon_exact"],
            "outside_edge_exact": strict["outside_edge_exact"],
            "locality_violation_rate": strict["locality_violation_rate"],
            "validity": valid["validity"]["validity"],
            "validity_checks": {
                key_name: valid["validity"][key_name]
                for key_name in (
                    "outer_boundary_closed",
                    "room_polygon_closed",
                    "no_self_intersection",
                    "no_room_overlap",
                    "semantic_complete",
                )
            },
            "target_watertight": valid["target_watertight"],
            "conditional_watertightness": (
                float(valid["after_watertight"])
                if valid["target_watertight"]
                else None
            ),
            "watertight_diagnostics": {
                key_name: valid["watertight"][key_name]
                for key_name in (
                    "hole_count",
                    "hole_area_px",
                    "hole_area_ratio",
                    "disconnected_component_count",
                    "dangling_endpoint_count",
                    "dangling_endpoint_rate",
                    "watertight_issue_codes",
                )
            },
            "node_precision": arm["node_precision"],
            "node_recall": arm["node_recall"],
            "node_f1": arm["node_f1"],
            "edge_precision": arm["edge_precision"],
            "edge_recall": arm["edge_recall"],
            "edge_f1": arm["edge_f1"],
            "edge_overlap": arm["edge_overlap"],
            "ged": arm["ged"],
            "ged_exact": arm["exact_ged"],
            "optional": {
                "delta_iou": delta_row["delta_iou"],
                "delta_mse": delta_row["delta_mse"],
                "ged": arm["ged"],
            },
            "denominators": {
                "outside_valid_pixels": strict["outside_valid_pixels"],
                "outside_changed_pixels": strict["outside_changed_pixels"],
                "unaffected_room_instances": strict["unaffected_room_instances"],
                "changed_room_instances": strict["changed_room_instances"],
            },
        }
        rows.append(row)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    errors = []
    for row in rows:
        row_errors = _validate(row, schema)
        if row_errors:
            errors.append(
                {
                    "case_id": row["case_id"],
                    "errors": row_errors,
                }
            )
    report = {
        "schema_version": "fevp-case-row-collection-v2",
        "status": "complete" if not errors else "failed",
        "cases": len(rows),
        "schema": str(args.schema.relative_to(ROOT)),
        "validation_errors": errors[:20],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(rows),
                "validation_errors": len(errors),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if not errors else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
