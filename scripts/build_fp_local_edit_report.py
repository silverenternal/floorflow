#!/usr/bin/env python3
"""Measure paired local-edit response and unintended global drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("outputs/eval/fp_architecture_ablation_val191_v1/full"),
    )
    parser.add_argument(
        "--edit-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--edit-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_local_edit_response_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_local_edit_audit.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def raster_path(root: Path, name: str) -> Path:
    return root / "rasters" / f"{name.replace('/', '__')}.npz"


def region_mask(
    shape: tuple[int, int],
    normalized: list[float],
) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = normalized
    left = max(0, min(width, round(float(x) * width)))
    top = max(0, min(height, round(float(y) * height)))
    right = max(left + 1, min(width, round((float(x + box_width)) * width)))
    bottom = max(top + 1, min(height, round((float(y + box_height)) * height)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def adjacency_edges(
    labels: np.ndarray,
    room_ids: set[int],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for left, right in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        pairs = np.stack([left.ravel(), right.ravel()], axis=1)
        pairs = pairs[
            (pairs[:, 0] != pairs[:, 1])
            & np.isin(pairs[:, 0], list(room_ids))
            & np.isin(pairs[:, 1], list(room_ids))
        ]
        for first, second in np.unique(
            np.sort(pairs, axis=1),
            axis=0,
        ):
            edges.add((int(first), int(second)))
    return edges


def edge_f1(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    if not left and not right:
        return 1.0
    intersection = len(left & right)
    return 2.0 * intersection / max(1, len(left) + len(right))


def analyze_operation(
    edit_root: Path,
    manifest_path: Path,
    original_root: Path,
    room_ids: set[int],
) -> dict:
    manifest = read_json(manifest_path)
    original_rows = read_json(original_root / "per_case.json")
    edited_rows = read_json(edit_root / "per_case.json")
    rows = []
    for index, record in enumerate(manifest["rows"]):
        if not record["edited"]:
            continue
        name = record["name"]
        original = np.load(
            raster_path(original_root, name),
            allow_pickle=False,
        )["prediction"]
        edited = np.load(
            raster_path(edit_root, name),
            allow_pickle=False,
        )["prediction"]
        changed = original != edited
        region = region_mask(original.shape, record["edit_region_normalized"])
        inside_changed = int((changed & region).sum())
        inside_pixels = int(region.sum())
        outside_changed = int((changed & ~region).sum())
        outside_pixels = int((~region).sum())
        original_metric = original_rows[index]
        edited_metric = edited_rows[index]
        original_outside = original.copy()
        edited_outside = edited.copy()
        original_outside[region] = 0
        edited_outside[region] = 0
        outside_topology_f1 = edge_f1(
            adjacency_edges(original_outside, room_ids),
            adjacency_edges(edited_outside, room_ids),
        )
        conditional = (
            edited_metric["fpeval_watertight"]
            if edited_metric["target_watertight"] > 0.5
            else np.nan
        )
        rows.append(
            {
                "name": name,
                "inside_response_fraction": inside_changed / inside_pixels,
                "outside_drift_fraction": outside_changed / max(1, outside_pixels),
                "localization_precision": inside_changed / max(1, int(changed.sum())),
                "changed_pixels": int(changed.sum()),
                "outside_changed_pixels": outside_changed,
                "fpeval_validity": edited_metric["fpeval_validity"],
                "conditional_watertight": conditional,
                "room_adjacency_f1": edited_metric["room_adjacency_f1"],
                "room_adjacency_delta": (
                    edited_metric["room_adjacency_f1"]
                    - original_metric["room_adjacency_f1"]
                ),
                "outside_topology_f1": outside_topology_f1,
                "door_contact_delta": (
                    edited_metric["door_contact_room_fraction"]
                    - original_metric["door_contact_room_fraction"]
                ),
                "fpeval_edit_cost": edited_metric["fpeval_edit_cost"],
            }
        )
    conditional_values = [
        row["conditional_watertight"]
        for row in rows
        if np.isfinite(row["conditional_watertight"])
    ]
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "inside_response_fraction",
            "outside_drift_fraction",
            "localization_precision",
            "changed_pixels",
            "outside_changed_pixels",
            "fpeval_validity",
            "room_adjacency_delta",
            "door_contact_delta",
            "outside_topology_f1",
            "fpeval_edit_cost",
        )
    }
    summary["conditional_watertight"] = float(
        np.mean(conditional_values)
    )
    response_coverage = float(
        np.mean([row["inside_response_fraction"] > 0.0 for row in rows])
    )
    topology_values = np.asarray(
        [row["outside_topology_f1"] for row in rows],
        dtype=np.float64,
    )
    topology_preservation_coverage = float(
        np.mean(topology_values >= 0.95)
    )
    checks = {
        "validity_at_least_0_99": summary["fpeval_validity"] >= 0.99,
        "mean_outside_drift_at_most_0_05": (
            summary["outside_drift_fraction"] <= 0.05
        ),
        "response_coverage_at_least_0_80": response_coverage >= 0.80,
        "outside_topology_f1_coverage_at_least_0_80": (
            topology_preservation_coverage >= 0.80
        ),
    }
    return {
        "operation": manifest["operation"],
        "cases": len(rows),
        "manifest_sha256": manifest["source_manifest_sha256"],
        "summary": summary,
        "response_coverage": response_coverage,
        "outside_topology": {
            "mean": float(topology_values.mean()),
            "median": float(np.median(topology_values)),
            "p10": float(np.quantile(topology_values, 0.1)),
            "fraction_at_least_0_95": topology_preservation_coverage,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if len(args.edit_root) != len(args.edit_manifest):
        raise ValueError("one manifest is required per edit root")
    taxonomy = FloorplanTaxonomy.read(
        Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json")
    )
    room_ids = set(taxonomy.room_ids)
    operations = [
        analyze_operation(root, manifest, args.original_root, room_ids)
        for root, manifest in zip(
            args.edit_root,
            args.edit_manifest,
            strict=True,
        )
    ]
    report = {
        "status": "complete",
        "scope": "CubiCasa val191 paired pre/post local control edits",
        "best_of_n_used": False,
        "original_root": str(args.original_root),
        "operations": operations,
        "passed_operations": [
            operation["operation"]
            for operation in operations
            if operation["passed"]
        ],
        "decision": (
            "supported"
            if operations and all(operation["passed"] for operation in operations)
            else "partial_or_failed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table_rows = [
        "# FP Local Edit Audit",
        "",
        (
            "| Operation | Cases | Inside response | Outside drift | "
            "Localization | Outside topology F1 | Valid | Watertight | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for operation in operations:
        summary = operation["summary"]
        topology_summary = operation["outside_topology"]
        table_rows.append(
            f"| {operation['operation']} | {operation['cases']} | "
            f"{summary['inside_response_fraction']:.4f} | "
            f"{summary['outside_drift_fraction']:.6f} | "
            f"{summary['localization_precision']:.4f} | "
            f"{topology_summary['median']:.4f} | "
            f"{summary['fpeval_validity']:.4f} | "
            f"{summary['conditional_watertight']:.4f} | "
            f"{'yes' if operation['passed'] else 'no'} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                operation["operation"]: {
                    "passed": operation["passed"],
                    "summary": operation["summary"],
                    "checks": operation["checks"],
                }
                for operation in operations
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
