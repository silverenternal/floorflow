#!/usr/bin/env python3
"""Build the FEVP-2 vector edit benchmark from hardened CubiCasa rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import vector_preservation_metrics

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
_TAXONOMY: FloorplanTaxonomy | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--case-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_case_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_edit_benchmark_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/fevp2_vector_contract_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/vector_edit_benchmark_v1.md",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "schema/vector_edit_row_v1.schema.json",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str) -> None:
    global _TAXONOMY
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _task_rows(seed: int, limit: int | None) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest_path = (
            ROOT
            / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
            / "test/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["rows"]:
            if record["edited"]:
                tasks.append(
                    {
                        "seed": seed,
                        "operation": operation,
                        "record": record,
                    }
                )
    tasks.sort(key=lambda row: (row["operation"], row["record"]["name"]))
    if limit is None:
        return tasks
    quota = max(1, int(limit) // len(OPERATIONS))
    selected = []
    for operation in OPERATIONS:
        selected.extend(
            [
                row
                for row in tasks
                if row["operation"] == operation
            ][:quota]
        )
    return selected[: int(limit)]


def _diagnose(task: dict) -> dict:
    if _TAXONOMY is None:
        raise RuntimeError("worker taxonomy was not initialized")
    seed = int(task["seed"])
    operation = str(task["operation"])
    record = task["record"]
    name = str(record["name"])
    safe_name = _safe(name)
    before_path = (
        ROOT
        / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1"
        / "rasters"
        / f"{safe_name}.npz"
    )
    after_path = (
        ROOT
        / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
        / operation
        / f"{safe_name}.npz"
    )
    before = _load(before_path)
    after = _load(after_path)
    footprint = np.asarray(before["footprint"], dtype=bool)
    region = normalized_region_mask(
        footprint.shape,
        record["edit_region_normalized"],
    ) & footprint
    metrics = vector_preservation_metrics(
        before["target"],
        before["room_labels"],
        after["target"],
        after["room_labels"],
        region,
        _TAXONOMY,
    )
    changed = np.asarray(before["target"]) != np.asarray(after["target"])
    exact_survival = all(
        metrics[key] == 1.0
        for key in (
            "outside_room_level_exact",
            "outside_wall_exact",
            "outside_opening_exact",
            "outside_edge_exact",
        )
    )
    return {
        "schema_version": "vector-edit-row-v1",
        "case_id": f"seed{seed}/{operation}/{name}",
        "name": name,
        "style": name.split("/")[0],
        "operation": operation,
        "source_hash": _digest(before_path),
        "source_paths": {
            "before": str(before_path.relative_to(ROOT)),
            "after": str(after_path.relative_to(ROOT)),
        },
        "edit_region_normalized": record["edit_region_normalized"],
        "edit_region_area": int(region.sum()),
        "footprint_area": int(footprint.sum()),
        "changed_pixels": int(
            np.count_nonzero(changed & footprint)
        ),
        "outside_changed_pixels": int(
            np.count_nonzero(changed & footprint & ~region)
        ),
        **metrics,
        "ground_truth_self_consistent": bool(
            exact_survival and metrics["locality_violation_area"] <= 1.0e-9
        ),
    }


def _summary(rows: list[dict]) -> dict:
    by_operation = {}
    for operation in OPERATIONS:
        selected = [row for row in rows if row["operation"] == operation]
        by_operation[operation] = {
            "cases": len(selected),
            "ground_truth_self_consistent": sum(
                int(row["ground_truth_self_consistent"])
                for row in selected
            ),
            "outside_room_level_exact_mean": float(
                np.mean(
                    [row["outside_room_level_exact"] for row in selected]
                )
            )
            if selected
            else None,
            "outside_wall_exact_mean": float(
                np.mean([row["outside_wall_exact"] for row in selected])
            )
            if selected
            else None,
            "outside_opening_exact_mean": float(
                np.mean([row["outside_opening_exact"] for row in selected])
            )
            if selected
            else None,
            "locality_violation_area_max": max(
                row["locality_violation_area"] for row in selected
            )
            if selected
            else None,
        }
    return {
        "cases": len(rows),
        "ground_truth_self_consistent": sum(
            int(row["ground_truth_self_consistent"]) for row in rows
        ),
        "by_operation": by_operation,
    }


def _protocol() -> dict:
    return {
        "schema_version": "fevp2-vector-contract-v1",
        "protocol": "FloorPlan Editing Verification Protocol",
        "protocol_version": "2.0-vector",
        "canonical_grid": {
            "image_size": 256,
            "label_dtype": "uint8",
        },
        "tolerances": {
            "room_match_min_iou": 0.5,
            "room_polygon_hausdorff_px": 0.5,
            "wall_endpoint_px": 0.5,
            "opening_center_px": 1.0,
            "edge_contact_px": 2.0,
            "locality_violation_area_epsilon": 1.0e-9,
        },
        "metrics": {
            "outside_room_polygon_exact": (
                "Every room whose mask does not intersect the edit region is "
                "matched one-to-one by label and IoU >= 0.5 with boundary "
                "Hausdorff distance <= 0.5 px."
            ),
            "outside_wall_exact": (
                "Wall support outside the edit region, changed pixels, and a "
                "two-pixel dilation is exactly equal; segment support is also "
                "checked with a 0.5-pixel directed tolerance."
            ),
            "outside_opening_exact": (
                "Unaffected openings form a same-type bijection with center "
                "distance <= 1.0 px."
            ),
            "outside_edge_exact": (
                "The room adjacency set among unaffected rooms is identical "
                "under the room mapping."
            ),
            "locality_violation_area": (
                "Symmetric-difference area of matched unaffected rooms and "
                "buffered wall segments, normalized by their total area."
            ),
        },
        "ground_truth_gate": {
            "outside_room_level_exact": 1.0,
            "outside_wall_exact": 1.0,
            "outside_opening_exact": 1.0,
            "outside_edge_exact": 1.0,
            "locality_violation_area": 0.0,
        },
        "attack_suite": [
            "Move one unaffected room polygon vertex by 1 px.",
            "Delete one unaffected wall segment.",
            "Move one unaffected opening center by more than 1 px.",
            "Remove one unaffected room adjacency edge.",
        ],
    }


def main() -> None:
    args = parse_args()
    case_rows = json.loads(args.case_rows.read_text(encoding="utf-8"))["rows"]
    legacy = {
        row["case_id"]: row
        for row in case_rows
        if int(row["seed"]) == int(args.seed)
    }
    tasks = _task_rows(args.seed, args.limit)
    rows = []
    errors = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_initialize,
        initargs=(str(args.taxonomy),),
    ) as executor:
        futures = {
            executor.submit(_diagnose, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except (OSError, RuntimeError, ValueError, ZeroDivisionError) as error:
                errors.append(
                    {
                        "operation": task["operation"],
                        "name": task["record"]["name"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            source = legacy[row["case_id"]]
            row["legacy_raster"] = {
                "validity": source["validity"],
                "conditional_watertightness": source[
                    "conditional_watertightness"
                ],
                "node_f1": source["node_f1"],
                "edge_overlap": source["edge_overlap"],
                "ged": source["ged"],
            }
            rows.append(row)
    rows.sort(key=lambda row: row["case_id"])
    protocol = _protocol()
    summary = _summary(rows)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validation_errors = []
    for row in rows:
        missing = [
            key for key in schema["required"] if key not in row
        ]
        if missing:
            validation_errors.append(
                {"case_id": row["case_id"], "missing": missing}
            )
    status = "complete" if not errors and not args.limit else "smoke"
    report = {
        "schema_version": "vector-edit-benchmark-v1",
        "status": status,
        "seed": int(args.seed),
        "cases": len(rows),
        "errors": errors,
        "protocol": protocol,
        "summary": summary,
        "checks": {
            "no_errors": not errors,
            "all_operations_present": all(
                summary["by_operation"][operation]["cases"] > 0
                for operation in OPERATIONS
            ),
            "ground_truth_vector_self_consistent": (
                summary["ground_truth_self_consistent"] == len(rows)
            ),
            "schema_valid": not validation_errors,
            "source_paths_exist": all(
                (ROOT / path).is_file()
                for row in rows
                for path in row["source_paths"].values()
            ),
        },
        "schema": str(args.schema.relative_to(ROOT)),
        "validation_errors": validation_errors[:20],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FEVP-2 Vector Edit Benchmark v1",
        "",
        f"Cases: {len(rows)}",
        f"Ground-truth self-consistent: {summary['ground_truth_self_consistent']}",
        "",
        "| Operation | Cases | Exact | Room | Wall | Opening | VLA max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, row in summary["by_operation"].items():
        lines.append(
            f"| {operation} | {row['cases']} | "
            f"{row['ground_truth_self_consistent']} | "
            f"{row['outside_room_level_exact_mean']} | "
            f"{row['outside_wall_exact_mean']} | "
            f"{row['outside_opening_exact_mean']} | "
            f"{row['locality_violation_area_max']} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "cases": len(rows),
                "errors": len(errors),
                "checks": report["checks"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
