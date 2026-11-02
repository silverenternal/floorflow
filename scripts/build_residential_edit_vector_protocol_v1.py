"""Audit typed-splice locality at raster and vector levels on test503."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
)
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_plan_preservation_metrics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.local_delta import compose_typed_delta
from scripts.evaluate_fp_raster2seq_local_redraw import EDITS, region_mask

VECTOR_METRICS = (
    "outside_room_label_exact",
    "outside_room_polygon_exact",
    "outside_wall_exact",
    "outside_opening_exact",
    "outside_edge_exact",
    "outside_room_level_exact",
    "locality_violation_area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-operation", type=int, default=0)
    parser.add_argument(
        "--base-source",
        choices=("raster2seq", "floorflow"),
        default="raster2seq",
    )
    parser.add_argument(
        "--splice-mode",
        choices=("paired", "before_only", "raw"),
        default="paired",
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
        default=ROOT / "outputs/reports/residential_edit_vector_protocol_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    tolerances = tolerances_from_contract(load_metric_contract(args.contract))
    operations = {}
    all_checks = {}
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                ROOT / "outputs/control_contracts" / directory / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        records = [row for row in manifest["rows"] if row["edited"]]
        if args.limit_per_operation > 0:
            records = records[: args.limit_per_operation]
        rows = []
        for record in records:
            safe_name = str(record["name"]).replace("/", "__")
            with np.load(
                ROOT
                / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
                / f"{safe_name}.npz"
            ) as payload:
                footprint = payload["footprint"].astype(bool)
            with np.load(
                ROOT
                / "outputs/external_baselines/raster2seq/test503_matched_v1/predictions"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                raster2seq_before = payload["prediction_before"].astype(np.uint8)
            with np.load(
                ROOT
                / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
                / f"{safe_name}.npz"
            ) as payload:
                floorflow_before = payload["prediction"].astype(np.uint8)
            with np.load(
                ROOT
                / "outputs/eval/fp_control_gated_local_test503_seed19_v1"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                floorflow_after = payload["prediction"].astype(np.uint8)
            before = (
                raster2seq_before
                if args.base_source == "raster2seq"
                else floorflow_before
            )
            region = region_mask(
                before.shape,
                record["edit_region_normalized"],
            ) & footprint
            if args.splice_mode == "raw":
                typed_before = floorflow_before
                typed_after = floorflow_after
            else:
                with np.load(
                    ROOT
                    / "outputs/cache/typed_edit_controls_v1"
                    / operation
                    / f"{safe_name}.npz"
                ) as payload:
                    before_control = payload["before"].astype(np.uint8)
                    after_control = payload["after"].astype(np.uint8)
                typed_before, typed_after = compose_typed_delta(
                    before,
                    before_control,
                    after_control,
                    region,
                )
                if args.splice_mode == "before_only":
                    typed_after = floorflow_after
            before_plan = extract_vector_plan(typed_before, typed_before, taxonomy)
            after_plan = extract_vector_plan(typed_after, typed_after, taxonomy)
            preservation = vector_plan_preservation_metrics(
                before_plan,
                after_plan,
                region,
                taxonomy,
                changed_mask=typed_before != typed_after,
            )
            validity = validity_checks(
                raster_labels_to_wall_plan(
                    typed_after,
                    taxonomy,
                    room_labels=typed_after,
                ),
                footprint,
                taxonomy,
                tolerances=tolerances,
                room_labels=typed_after,
            )
            wall_scope = binary_dilation(region, iterations=2)
            rows.append(
                {
                    "name": str(record["name"]),
                    "outside_pixel_exact": float(
                        np.array_equal(
                            typed_before[~region],
                            typed_after[~region],
                        )
                    ),
                    "outside_wall_band_pixel_exact": float(
                        np.array_equal(
                            typed_before[~wall_scope],
                            typed_after[~wall_scope],
                        )
                    ),
                    **{
                        metric: float(preservation[metric])
                        for metric in VECTOR_METRICS
                    },
                    "validity": float(validity["validity"]),
                }
            )
        summary = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "min": float(np.min([row[metric] for row in rows])),
            }
            for metric in (*VECTOR_METRICS, "outside_pixel_exact", "validity")
        }
        operations[operation] = {
            "cases": len(rows),
            "summary": summary,
            "rows": rows,
        }
        all_checks[f"{operation}_outside_exact_one"] = (
            summary["outside_pixel_exact"]["min"] == 1.0
            and summary["outside_room_level_exact"]["min"] == 1.0
        )
        all_checks[f"{operation}_locality_violation_zero"] = (
            summary["locality_violation_area"]["min"] == 0.0
        )
    report = {
        "status": "passed" if all(all_checks.values()) else "not_passed",
        "scope": "CubiCasa test503 deterministic typed-splice strict-locality audit",
        "operations": operations,
        "checks": all_checks,
        "claim_boundary": (
            "Vector metrics are computed from the declared typed-control splice; "
            "they do not rank reconstruction fidelity across datasets."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **all_checks}, indent=2))


if __name__ == "__main__":
    main()
