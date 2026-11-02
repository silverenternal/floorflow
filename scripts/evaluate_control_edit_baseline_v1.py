"""Evaluate a control-aware weak baseline against FloorFlow test503 editing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import edit_detection_iou_mse
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    rank_biserial_correlation,
)
from floorflow.inference.local_delta import compose_typed_delta
from scripts.evaluate_fp_raster2seq_local_redraw import EDITS, region_mask

METRICS = ("delta_iou", "delta_mse", "outside_exact_ratio")


def paired_rows(rows: list[dict], left: str, right: str) -> dict:
    first = [{"value": row[left]["delta_iou"]} for row in rows]
    second = [{"value": row[right]["delta_iou"]} for row in rows]
    stats = paired_delta(
        first,
        second,
        "value",
        iterations=4000,
        seed=261201,
    )
    differences = np.asarray(
        [
            float(a["value"]) - float(b["value"])
            for a, b in zip(first, second, strict=True)
        ]
    )
    return {
        **stats,
        "rank_biserial": float(rank_biserial_correlation(differences)),
    }


def main() -> None:
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    operations = {}
    checks = {}
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                ROOT / "outputs/control_contracts" / directory / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        rows = []
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = str(record["name"])
            safe_name = name.replace("/", "__")
            with np.load(
                ROOT
                / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
                / f"{safe_name}.npz"
            ) as payload:
                target_before = payload["target"].astype(np.uint8)
                footprint = payload["footprint"].astype(bool)
                floorflow_before = payload["prediction"].astype(np.uint8)
            with np.load(
                ROOT
                / "outputs/external_baselines/raster2seq/test503_matched_v1/predictions"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                r2s_before = payload["prediction_before"].astype(np.uint8)
            with np.load(
                ROOT
                / "outputs/cache/typed_edit_controls_v1"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                control_before = payload["before"].astype(np.uint8)
                control_after = payload["after"].astype(np.uint8)
            with np.load(
                ROOT
                / "outputs/cache/fp_edit_detection_v1"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                changed = payload["changed"].astype(bool)
                after_visible = payload["after_visible"].astype(np.uint8)
            target_after = target_before.copy()
            target_after[changed] = after_visible[changed]
            region = region_mask(
                target_before.shape,
                record["edit_region_normalized"],
            ) & footprint
            wall = (
                (target_before == taxonomy.class_id("wall"))
                | (target_after == taxonomy.class_id("wall"))
            )
            valid = footprint & ~binary_dilation(wall, iterations=1)
            outside = valid & ~region
            control_changed = region & (control_before != control_after)
            with np.load(
                ROOT
                / "outputs/eval/fp_control_gated_local_test503_seed19_v1"
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                after_floorflow = payload["prediction"].astype(np.uint8)
            typed_before, typed_after = compose_typed_delta(
                floorflow_before,
                control_before,
                control_after,
                region,
            )
            candidates = {
                "raster2seq_before": (r2s_before, r2s_before),
                "control_region_replace": (
                    r2s_before,
                    np.where(region, control_after, r2s_before),
                ),
                "control_delta_replace": (
                    r2s_before,
                    np.where(control_changed, control_after, r2s_before),
                ),
                "floorflow_primary": (
                    (typed_before, typed_after)
                    if operation.startswith("room_")
                    else (floorflow_before, after_floorflow)
                ),
                "floorflow_raw": (floorflow_before, after_floorflow),
            }
            values = {}
            for method, (before, after) in candidates.items():
                delta_iou, delta_mse = edit_detection_iou_mse(
                    target_before,
                    target_after,
                    before,
                    after,
                    valid,
                )
                values[method] = {
                    "delta_iou": delta_iou,
                    "delta_mse": delta_mse,
                    "outside_exact_ratio": (
                        float((after[outside] == before[outside]).mean())
                        if outside.any()
                        else 1.0
                    ),
                }
            rows.append({"name": name, **values})
        summary = {
            method: {
                metric: float(
                    np.mean([row[method][metric] for row in rows])
                )
                for metric in METRICS
            }
            for method in candidates
        }
        operations[operation] = {
            "cases": len(rows),
            "summary": summary,
            "paired_floorflow_vs_control_delta": paired_rows(
                rows,
                "floorflow_primary",
                "control_delta_replace",
            ),
            "rows": rows,
        }
        checks[f"{operation}_floorflow_beats_control_delta"] = (
            summary["floorflow_primary"]["delta_iou"]
            > summary["control_delta_replace"]["delta_iou"]
        )
        checks[f"{operation}_floorflow_outside_exact"] = (
            summary["floorflow_primary"]["outside_exact_ratio"] == 1.0
        )
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa test503 editing-aware weak baseline",
        "baseline_boundary": (
            "Both methods consume declared after-edit controls. The control "
            "baseline paints control labels and is intentionally simple."
        ),
        "operations": operations,
        "checks": checks,
    }
    output = ROOT / "outputs/reports/residential_editing_aware_baseline_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
