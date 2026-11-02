#!/usr/bin/env python3
"""Score Raster2Seq surgical redraw with the FEVP-2 vector protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_plan_preservation_metrics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.local_delta import compose_local_delta

ROOT = Path(__file__).resolve().parents[1]
EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-operation", type=int, default=100)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/test503_matched_v1/predictions",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_edit_baselines_v1.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _validity(
    labels: np.ndarray,
    room_labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    tolerances,
) -> dict:
    plan = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        room_labels=room_labels,
    )
    validity = validity_checks(
        plan,
        footprint,
        taxonomy,
        tolerances=tolerances,
        room_labels=room_labels,
    )
    watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        fpeval_watertight=False,
        tolerances=tolerances,
    )
    return {"validity": validity, "watertight": watertight}


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    tolerances = tolerances_from_contract(load_metric_contract(args.contract))
    operations = {}
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                ROOT
                / "outputs/control_contracts"
                / directory
                / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        records = [row for row in manifest["rows"] if row["edited"]]
        records = records[: int(args.limit_per_operation)]
        rows = []
        for record in records:
            safe_name = str(record["name"]).replace("/", "__")
            before = _load(
                ROOT
                / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
                / f"{safe_name}.npz"
            )
            external = _load(
                args.predictions / operation / f"{safe_name}.npz"
            )
            floorflow = _load(
                ROOT
                / "outputs/eval/fp_control_gated_local_test503_seed19_v1"
                / operation
                / f"{safe_name}.npz"
            )
            changed_cache = _load(
                ROOT
                / "outputs/cache/fp_edit_detection_v1"
                / operation
                / f"{safe_name}.npz"
            )
            footprint = np.asarray(before["footprint"], dtype=bool)
            region = normalized_region_mask(
                footprint.shape,
                record["edit_region_normalized"],
            ) & footprint
            target_after = np.asarray(before["target"]).copy()
            changed = changed_cache["changed"].astype(bool)
            target_after[changed] = changed_cache["after_visible"][changed]
            surgical = compose_local_delta(
                torch.from_numpy(np.asarray(external["prediction_before"])),
                torch.from_numpy(np.asarray(external["prediction_after"])),
                torch.from_numpy(region),
            ).numpy()
            before_plan = extract_vector_plan(
                before["prediction"],
                before["room_labels"],
                taxonomy,
            )
            surgical_plan = extract_vector_plan(
                surgical,
                surgical,
                taxonomy,
            )
            floorflow_plan = extract_vector_plan(
                floorflow["prediction"],
                floorflow["room_labels"],
                taxonomy,
            )
            surgical_preservation = vector_plan_preservation_metrics(
                before_plan,
                surgical_plan,
                region,
                taxonomy,
                changed_mask=np.asarray(before["prediction"]) != surgical,
            )
            floorflow_preservation = vector_plan_preservation_metrics(
                before_plan,
                floorflow_plan,
                region,
                taxonomy,
                changed_mask=(
                    np.asarray(before["prediction"])
                    != np.asarray(floorflow["prediction"])
                ),
            )
            rows.append(
                {
                    "case_id": f"seed19/{operation}/{record['name']}",
                    "surgical": {
                        "preservation": surgical_preservation,
                        "topology": _validity(
                            surgical,
                            surgical,
                            footprint,
                            taxonomy,
                            tolerances,
                        ),
                        "graph": structural_legality_metrics(
                            surgical,
                            target_after,
                            taxonomy,
                        ),
                    },
                    "floorflow": {
                        "preservation": floorflow_preservation,
                        "topology": _validity(
                            floorflow["prediction"],
                            floorflow["room_labels"],
                            footprint,
                            taxonomy,
                            tolerances,
                        ),
                        "graph": structural_legality_metrics(
                            floorflow["prediction"],
                            target_after,
                            taxonomy,
                        ),
                    },
                }
            )
        operations[operation] = {
            "cases": len(rows),
            "surgical": {
                "outside_room_exact": float(
                    np.mean(
                        [
                            row["surgical"]["preservation"][
                                "outside_room_level_exact"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "validity": float(
                    np.mean(
                        [
                            row["surgical"]["topology"]["validity"]["validity"]
                            for row in rows
                        ]
                    )
                ),
                "watertight": float(
                    np.mean(
                        [
                            row["surgical"]["topology"]["watertight"][
                                "watertight_rate"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "node_f1": float(
                    np.mean([row["surgical"]["graph"]["node_f1"] for row in rows])
                ),
                "edge_overlap": float(
                    np.mean(
                        [row["surgical"]["graph"]["edge_overlap"] for row in rows]
                    )
                ),
            },
            "floorflow": {
                "outside_room_exact": float(
                    np.mean(
                        [
                            row["floorflow"]["preservation"][
                                "outside_room_level_exact"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "validity": float(
                    np.mean(
                        [
                            row["floorflow"]["topology"]["validity"]["validity"]
                            for row in rows
                        ]
                    )
                ),
                "watertight": float(
                    np.mean(
                        [
                            row["floorflow"]["topology"]["watertight"][
                                "watertight_rate"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "node_f1": float(
                    np.mean([row["floorflow"]["graph"]["node_f1"] for row in rows])
                ),
                "edge_overlap": float(
                    np.mean(
                        [row["floorflow"]["graph"]["edge_overlap"] for row in rows]
                    )
                ),
            },
            "rows": rows,
        }
    report = {
        "status": "complete",
        "method": "Raster2Seq outside-frozen surgical redraw",
        "scope": f"seed19, first {args.limit_per_operation} edited cases per operation",
        "operations": operations,
        "boundary": (
            "Raster2Seq is a reconstruction model wrapped with an outside-freeze "
            "operation; it is not a native editing-aware method."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                operation: {
                    "surgical": {
                        key: value
                        for key, value in result["surgical"].items()
                    },
                    "floorflow": {
                        key: value
                        for key, value in result["floorflow"].items()
                    },
                }
                for operation, result in operations.items()
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
