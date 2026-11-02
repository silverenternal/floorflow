"""Build complexity strata and failure tails for the frozen residential edit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.data.taxonomy import FloorplanTaxonomy
from scripts.evaluate_fp_raster2seq_local_redraw import EDITS, region_mask

SEED_REPORT = ROOT / "outputs/reports/residential_edit_hybrid_typedall_seed19_v1.json"
TAXONOMY = ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
RASTER_ROOT = ROOT / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"


def room_count(labels: np.ndarray, taxonomy: FloorplanTaxonomy) -> int:
    count = 0
    for class_id in taxonomy.room_ids:
        number, components = cv2.connectedComponents(
            (labels == class_id).astype(np.uint8),
            connectivity=4,
        )
        count += sum(
            np.count_nonzero(components == index) >= 12
            for index in range(1, number)
        )
    return count


def bin_name(value: float, cutoffs: tuple[float, float]) -> str:
    if value <= cutoffs[0]:
        return "low"
    if value <= cutoffs[1]:
        return "medium"
    return "high"


def summarize(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "delta_iou": float(np.mean([row["delta_iou"] for row in rows])),
        "delta_mse": float(np.mean([row["delta_mse"] for row in rows])),
        "outside_exact": float(np.min([row["outside_exact"] for row in rows])),
    }


def main() -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    report = json.loads(SEED_REPORT.read_text(encoding="utf-8"))
    rows = []
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                ROOT / "outputs/control_contracts" / directory / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        metrics = {
            row["name"]: row["typed_before_floorflow_after"]
            for row in report["operations"][operation]["rows"]
        }
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = str(record["name"])
            safe_name = name.replace("/", "__")
            with np.load(RASTER_ROOT / f"{safe_name}.npz") as payload:
                target = payload["target"].astype(np.uint8)
                footprint = payload["footprint"].astype(bool)
            region = region_mask(target.shape, record["edit_region_normalized"])
            region &= footprint
            value = metrics[name]
            rows.append(
                {
                    "operation": operation,
                    "name": name,
                    "delta_iou": value["delta_iou"],
                    "delta_mse": value["delta_mse"],
                    "outside_exact": value["outside_exact_ratio"],
                    "room_count": int(room_count(target, taxonomy)),
                    "edit_area_fraction": float(region.sum() / max(1, footprint.sum())),
                    "wall_pixels_in_region": int(
                        np.count_nonzero(
                            region & (target == taxonomy.class_id("wall"))
                        )
                    ),
                }
            )
    strata = {}
    for field, cutoffs in (
        ("room_count", (4.0, 7.0)),
        ("edit_area_fraction", (0.10, 0.25)),
        ("wall_pixels_in_region", (0.0, 50.0)),
    ):
        strata[field] = {}
        for label in ("low", "medium", "high"):
            selected = [
                row for row in rows if bin_name(float(row[field]), cutoffs) == label
            ]
            strata[field][label] = summarize(selected) if selected else {
                "cases": 0
            }
    strata["operation"] = {
        operation: summarize([row for row in rows if row["operation"] == operation])
        for operation in EDITS
    }
    failures = sorted(rows, key=lambda row: row["delta_iou"])[:25]
    strata_report = {
        "status": "complete",
        "scope": "CubiCasa test503 frozen typed edit complexity strata",
        "seed_report": str(SEED_REPORT.relative_to(ROOT)),
        "strata": strata,
    }
    (ROOT / "outputs/reports/residential_sota_strata_v1.json").write_text(
        json.dumps(strata_report, indent=2) + "\n",
        encoding="utf-8",
    )
    failure_report = {
        "status": "complete",
        "scope": "CubiCasa test503 typed edit failure tail",
        "rows": failures,
        "by_operation": {
            operation: sorted(
                [row for row in rows if row["operation"] == operation],
                key=lambda row: row["delta_iou"],
            )[:10]
            for operation in EDITS
        },
    }
    (ROOT / "outputs/reports/residential_sota_failures_v1.json").write_text(
        json.dumps(failure_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
