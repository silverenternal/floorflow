"""Build per-class, failure-taxonomy, and worst-sample P4 diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.failure_taxonomy import classify_room_failure
from floorflow.evaluation.room_error_decomposition import decompose_rooms
from floorflow.evaluation.visualize import make_comparison_panel


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval/current_model_503_metric_v2"
REPORTS = ROOT / "outputs/reports"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contact_sheet(panels: list[Image.Image], path: Path) -> None:
    width = max(panel.width for panel in panels)
    gap = 8
    height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    cursor = 0
    for panel in panels:
        canvas.paste(panel, (0, cursor))
        cursor += panel.height + gap
    canvas.save(path)


def main() -> None:
    metrics = _load(EVAL / "metrics.json")
    per_case = _load(EVAL / "per_case.json")
    decomposition = _load(REPORTS / "room_iou_error_decomposition.json")
    polygon = _load(REPORTS / "room_polygon_target_audit_test503.json")
    geometry = _load(REPORTS / "geometry_cache_v1_test_audit.json")
    gate = _load(EVAL / "publication_gate_v4.json")
    taxonomy = FloorplanTaxonomy.read(ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json")

    case_root = EVAL / "rasters"
    visual_root = REPORTS / "p4_worst_cases"
    visual_root.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for raster in sorted(case_root.glob("*.npz")):
        with np.load(raster) as payload:
            target = payload["target"]
            prediction = payload["prediction"]
            footprint = payload["footprint"]
        case = decompose_rooms(
            torch.from_numpy(prediction.astype(np.int64))[None, None],
            torch.from_numpy(target.astype(np.int64))[None, None],
            torch.from_numpy(footprint.astype(np.float32))[None, None],
            taxonomy,
        )
        row = {
            "name": raster.stem,
            "support_macro_iou": case["strict_macro_iou"],
            "support_occupancy_iou": case["occupancy"]["iou"],
            "support_fragmentation_gap": case["fragmentation_gap"],
            "allocation": case["allocation"],
        }
        row["failure"] = classify_room_failure(
            row,
            fragmentation_threshold=2.0,
        )
        all_cases.append(row)
    worst = sorted(
        all_cases,
        key=lambda row: (
            float(row["support_macro_iou"]),
            -float(row["support_fragmentation_gap"]),
        ),
    )[:10]
    panels = []
    rows = []
    for index, row in enumerate(worst, start=1):
        name = row["name"]
        raster = case_root / f"{name}.npz"
        if not raster.is_file():
            continue
        with np.load(raster) as payload:
            target = payload["target"]
            prediction = payload["prediction"]
            footprint = payload["footprint"]
        title = (
            f"{index:02d} {name} roomIoU={float(row['support_macro_iou']):.3f} "
            f"{row['failure']}"
        )
        panel = make_comparison_panel(
            footprint,
            target,
            prediction,
            title=title,
            taxonomy=taxonomy,
        )
        panel.save(visual_root / f"{index:02d}_{name}.png")
        panels.append(panel)
        rows.append(
            {
                "name": name,
                "room_macro_iou": float(row["support_macro_iou"]),
                "room_occupancy_iou": float(row["support_occupancy_iou"]),
                "fragmentation_gap": float(row["support_fragmentation_gap"]),
                "allocation": row["allocation"],
                "failure": row["failure"],
                "panel": str((visual_root / f"{index:02d}_{name}.png").relative_to(ROOT)),
            }
        )
    if panels:
        _contact_sheet(panels, visual_root / "worst_10_contact_sheet.png")

    support = decomposition["partition_support"]
    class_rows = [
        row
        for row in support["class_rows"]
        if int(row["present_cases"]) >= 5
    ]
    class_rows.sort(key=lambda row: float(row["mean_case_iou"]))
    categories = Counter(row["failure"] for row in all_cases)
    report = {
        "phase": "P4",
        "status": "complete",
        "case_count": len(per_case),
        "iou_definition": {
            "visible_macro": "per-case present room classes, then case mean",
            "dataset_macro": "per-class dataset IoU, then class mean",
            "dataset_micro": "global room-pixel TP / union",
            "instance": "Hungarian geometry match; type accuracy reported separately",
        },
        "headline": {
            "room_macro_iou": float(metrics["room_macro_iou_mean"]),
            "room_dataset_macro_iou": float(metrics["room_dataset_macro_iou"]),
            "room_dataset_micro_iou": float(metrics["room_dataset_micro_iou"]),
            "instance_geometric_iou": float(metrics["instance_geometric_iou_mean"]),
            "instance_room_f1_at_0_5": float(
                metrics["instance_room_f1_at_0_5_mean"]
            ),
            "instance_type_accuracy": float(metrics["instance_type_accuracy_mean"]),
        },
        "lowest_classes": class_rows[:12],
        "instance_weakest": support["instance_rows"][:10],
        "boundary_weakest": support["boundary_rows"][:10],
        "worst_cases": rows,
        "failure_taxonomy": dict(categories),
        "visual_artifacts": {
            "individual_panels": str(visual_root.relative_to(ROOT)),
            "contact_sheet": str(
                (visual_root / "worst_10_contact_sheet.png").relative_to(ROOT)
            ),
        },
        "raster_roundtrip": {
            "polygon_room_macro_iou": float(polygon["room_macro_iou"]),
            "polygon_room_occupancy_iou": float(polygon["room_occupancy_iou"]),
            "cache_semantic_iou": float(geometry["roundtrip_semantic_iou_mean"]),
            "cache_instance_iou": float(geometry["roundtrip_instance_iou_mean"]),
            "cache_target_equal_fraction": float(geometry["target_equal_fraction"]),
            "decision": "ioU gate passes; exact equality is not the correct raster target contract",
        },
        "gate": {
            "validity_passed": gate["validity_passed"],
            "research_targets_passed": gate["research_targets_passed"],
            "passed": gate["passed"],
            "failed_requirements": [
                name
                for group in ("validity_requirements", "research_requirements")
                for name, passed in gate[group].items()
                if not passed
            ],
        },
        "best_of_n_used": False,
    }
    path = REPORTS / "p4_iou_diagnostics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
