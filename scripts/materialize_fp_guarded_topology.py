#!/usr/bin/env python3
"""Materialize guarded topology scaffolds for paired local edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.local_delta import compose_local_delta
from floorflow.inference.topology_scaffold import select_guarded_topology

EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/eval"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def region_mask(shape: tuple[int, int], normalized: list[float]) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = normalized
    left = max(0, min(width, round(float(x) * width)))
    top = max(0, min(height, round(float(y) * height)))
    right = max(left + 1, min(width, round((x + box_width) * width)))
    bottom = max(top + 1, min(height, round((y + box_height) * height)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def load_raster(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / "rasters" / f"{name.replace('/', '__')}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    original_root = (
        args.evaluation_root / f"fp_hard_metrics_full_test503_seed{args.seed}_v1"
    )
    rows = []
    for operation, directory in EDITS.items():
        edit_directory = (
            directory
            if args.seed == 19
            else directory.replace("_v1", f"_seed{args.seed}_v1")
        )
        edit_root = args.evaluation_root / edit_directory
        manifest = json.loads(
            (
                args.control_root / directory / "test" / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        per_case = json.loads(
            (edit_root / "per_case.json").read_text(encoding="utf-8")
        )
        for index, record in enumerate(manifest["rows"]):
            if not record["edited"]:
                continue
            name = record["name"]
            before = load_raster(original_root, name)
            after = load_raster(edit_root, name)
            edit_region = region_mask(
                before["target"].shape,
                record["edit_region_normalized"],
            ) & before["footprint"].astype(bool)
            selected, detail = select_guarded_topology(
                before["topology_prediction"],
                after["topology_prediction"],
                edit_region,
                taxonomy,
            )
            composed_visible = compose_local_delta(
                torch.from_numpy(before["prediction"]),
                torch.from_numpy(after["prediction"]),
                torch.from_numpy(edit_region),
            ).numpy()
            composed_rooms = compose_local_delta(
                torch.from_numpy(before["room_labels"]),
                torch.from_numpy(after["room_labels"]),
                torch.from_numpy(edit_region),
            ).numpy()
            target_after = after["target"].copy()
            outside = ~edit_region
            record_report = {
                "operation": operation,
                "name": name,
                "source": detail["source"],
                "local_watertight": bool(
                    detail["local_validity"].get("watertight", False)
                ),
                "selected_watertight": bool(
                    (
                        detail["local_validity"]
                        if detail["source"] == "local_delta"
                        else detail["fallback_validity"]
                    ).get("watertight", False)
                ),
                "visible_exact_outside_ratio": (
                    float(
                        (
                            composed_visible[outside]
                            == before["prediction"][outside]
                        ).mean()
                    )
                    if outside.any()
                    else 1.0
                ),
                "topology_exact_outside_ratio": (
                    float(
                        (
                            selected[outside]
                            == before["topology_prediction"][outside]
                        ).mean()
                    )
                    if outside.any()
                    else 1.0
                ),
                "target_watertight": bool(
                    per_case[index]["target_watertight"] > 0.5
                ),
            }
            rows.append(record_report)
            output_dir = args.output_root / operation
            output_dir.mkdir(parents=True, exist_ok=True)
            output = dict(before)
            output.update(
                {
                    "prediction": composed_visible,
                    "room_labels": composed_rooms,
                    "topology_prediction": selected,
                    "target": target_after,
                }
            )
            np.savez_compressed(
                output_dir / f"{name.replace('/', '__')}.npz",
                **output,
            )
        print(f"materialized {operation}: {len(rows)} cases", flush=True)

    conditional = [
        row["selected_watertight"]
        for row in rows
        if row["target_watertight"]
    ]
    report = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in ("local_delta", "post_edit_fallback", "failed_both")
        },
        "local_acceptance_rate": float(
            np.mean([row["source"] == "local_delta" for row in rows])
        ),
        "conditional_watertight": float(np.mean(conditional)),
        "visible_min_outside_exact_ratio": float(
            min(row["visible_exact_outside_ratio"] for row in rows)
        ),
        "topology_min_outside_exact_ratio": float(
            min(row["topology_exact_outside_ratio"] for row in rows)
        ),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
