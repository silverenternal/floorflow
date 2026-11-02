#!/usr/bin/env python3
"""Build source-linked failure galleries for every mandatory FEVP metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.typed_edit_graph import (
    build_typed_edit_graph,
    compose_typed_edit,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_rows_v2.json",
    )
    parser.add_argument(
        "--validity-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_rows_v2.json",
    )
    parser.add_argument(
        "--typed-rows",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_failure_gallery_v2.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_metric_failures_v2.pdf",
    )
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _case_paths(seed: int, operation: str, name: str) -> tuple[Path, Path, Path, Path]:
    safe = name.replace("/", "__")
    before = (
        ROOT
        / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1/rasters"
        / f"{safe}.npz"
    )
    raw_directory = (
        f"fp_local_edit_{operation}_test503_v1"
        if seed == 19
        else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
    )
    raw = ROOT / "outputs/eval" / raw_directory / "rasters" / f"{safe}.npz"
    gated = (
        ROOT
        / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
        / operation
        / f"{safe}.npz"
    )
    cache = ROOT / "outputs/cache/fp_edit_detection_v1" / operation / f"{safe}.npz"
    return before, raw, gated, cache


def _record_for_name(seed: int, operation: str, name: str) -> dict:
    manifest = _read(
        ROOT
        / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
        / "test/manifest.json"
    )
    return next(row for row in manifest["rows"] if row["name"] == name)


def _typed_visible(
    seed: int,
    operation: str,
    name: str,
    taxonomy: FloorplanTaxonomy,
    arm: str,
) -> dict[str, np.ndarray]:
    before_path, raw_path, _gated_path, cache_path = _case_paths(
        seed,
        operation,
        name,
    )
    before = _load(before_path)
    raw = _load(raw_path)
    with np.load(cache_path) as payload:
        changed = payload["changed"].astype(bool)
    record = _record_for_name(seed, operation, name)
    region = normalized_region_mask(
        before["prediction"].shape,
        record["edit_region_normalized"],
    ) & before["footprint"].astype(bool)
    graph = build_typed_edit_graph(
        before["prediction"],
        region,
        changed,
        after_labels=raw["prediction"],
        room_ids=taxonomy.room_ids,
        wall_id=taxonomy.class_id("wall"),
        opening_ids=taxonomy.opening_ids,
        furniture_ids=taxonomy.furniture_ids,
    )
    composition = compose_typed_edit(
        before["prediction"],
        raw["prediction"],
        before["room_labels"],
        raw["room_labels"],
        before["topology_prediction"],
        raw["topology_prediction"],
        graph,
        region,
        before["footprint"].astype(bool),
        policy=arm,
    )
    return {
        "before": before["prediction"],
        "after": composition.visible,
        "target": raw["target"],
        "region": region,
    }


def _palette(taxonomy: FloorplanTaxonomy) -> np.ndarray:
    return np.asarray(
        [taxonomy.color(index)[:3] for index in range(taxonomy.num_classes)],
        dtype=np.uint8,
    )


def _draw_page(
    pdf: PdfPages,
    title: str,
    panels: list[dict],
    colors: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    rows = max(1, len(panels))
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(8.0, 2.0 * rows + 0.4),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, panel in enumerate(panels):
        for column, key in enumerate(("before", "after", "highlight")):
            axis = axes[row_index, column]
            if key == "highlight":
                axis.imshow(colors[np.asarray(panel["highlight_base"])])
                axis.imshow(panel["overlay"])
            else:
                axis.imshow(colors[np.asarray(panel[key])])
            labels = ("Before", "After or target", panel.get("highlight_title", "Failure"))
            axis.set_title(labels[column], fontsize=7, loc="left")
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            f"{panel['case_id']}\n{panel['detail']}",
            fontsize=5,
        )
    figure.suptitle(title, x=0.01, ha="left", fontsize=9, fontweight="bold")
    pdf.savefig(figure)
    plt.close(figure)


def _overlay(mask: np.ndarray, color=(0.9, 0.1, 0.1, 0.8)) -> np.ndarray:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[np.asarray(mask, dtype=bool)] = color
    return overlay


def _overlap_mask(plan) -> np.ndarray:
    mask = np.zeros((256, 256), dtype=np.uint8)
    polygons = [
        Polygon(room.get("polygon") or []).buffer(0)
        for room in plan.rooms
        if len(room.get("polygon") or []) >= 3
    ]
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            overlap = polygons[left].intersection(polygons[right])
            if overlap.is_empty or overlap.area <= 0.0:
                continue
            geometries = list(overlap.geoms) if hasattr(overlap, "geoms") else [overlap]
            for geometry in geometries:
                if geometry.geom_type != "Polygon":
                    continue
                points = np.rint(np.asarray(geometry.exterior.coords)).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
    return mask


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    colors = _palette(taxonomy)
    outside = _read(args.outside_rows)["rows"]
    validity = _read(args.validity_rows)["rows"]
    typed = _read(args.typed_rows)["rows"]
    gallery = {"status": "complete", "metrics": {}, "figure": str(args.figure.relative_to(ROOT))}

    outside_cases = sorted(
        [
            row
            for row in outside
            if row["no_edit_gating"]["outside_changed_pixels"] > 0
            and row["strict"]["outside_changed_pixels"] == 0
        ],
        key=lambda row: -row["no_edit_gating"]["outside_changed_pixels"],
    )[:3]
    validity_cases = sorted(
        [row for row in validity if row["validity"]["validity"] < 1.0],
        key=lambda row: (
            row["validity"]["no_room_overlap"],
            row["validity"]["semantic_complete"],
        ),
    )[:3]
    watertight_cases = sorted(
        [
            row
            for row in validity
            if row["target_watertight"] and not row["after_watertight"]
        ],
        key=lambda row: (
            -row["watertight"]["hole_count"],
            -row["watertight"]["dangling_endpoint_rate"],
        ),
    )[:3]
    node_cases = sorted(
        typed,
        key=lambda row: row["arms"]["typed_edit_graph"]["node_f1"],
    )[:3]
    edge_cases = sorted(
        typed,
        key=lambda row: row["arms"]["typed_edit_graph"]["edge_f1"],
    )[:3]

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.figure) as pdf:
        panels = []
        for row in outside_cases:
            before_path, raw_path, gated_path, _cache = _case_paths(
                row["seed"],
                row["operation"],
                row["name"],
            )
            before = _load(before_path)["prediction"]
            gated = _load(gated_path)["prediction"]
            raw = _load(raw_path)["prediction"]
            panels.append(
                {
                    "case_id": row["case_id"],
                    "before": before,
                    "after": gated,
                    "highlight_base": raw,
                    "overlay": _overlay(raw != before),
                    "detail": (
                        f"violation={row['no_edit_gating']['locality_violation_rate']:.4f}"
                    ),
                    "highlight_title": "No-gate leakage",
                    "source_row": row,
                }
            )
        gallery["metrics"]["outside_exact"] = [
            {"case_id": panel["case_id"], "detail": panel["detail"]}
            for panel in panels
        ]
        _draw_page(pdf, "Outside exact failure controls", panels, colors)

        panels = []
        for row in validity_cases:
            _before, _raw, gated_path, _cache = _case_paths(
                row["seed"],
                row["operation"],
                row["name"],
            )
            gated = _load(gated_path)
            plan = raster_labels_to_wall_plan(
                gated["topology_prediction"],
                taxonomy,
                room_labels=gated["room_labels"],
            )
            overlap = _overlap_mask(plan)
            panels.append(
                {
                    "case_id": row["case_id"],
                    "before": gated["room_labels"],
                    "after": gated["topology_prediction"],
                    "highlight_base": gated["topology_prediction"],
                    "overlay": _overlay(overlap),
                    "detail": ",".join(row["validity"]["validity_errors"]),
                    "highlight_title": "Validity overlap",
                    "source_row": row,
                }
            )
        gallery["metrics"]["validity"] = [
            {"case_id": panel["case_id"], "detail": panel["detail"]}
            for panel in panels
        ]
        _draw_page(pdf, "Validity failure controls", panels, colors)

        panels = []
        for row in watertight_cases:
            _before, _raw, gated_path, _cache = _case_paths(
                row["seed"],
                row["operation"],
                row["name"],
            )
            gated = _load(gated_path)
            room = np.isin(gated["topology_prediction"], taxonomy.room_ids)
            hole = (~room) & gated["footprint"].astype(bool)
            panels.append(
                {
                    "case_id": row["case_id"],
                    "before": gated["room_labels"],
                    "after": gated["topology_prediction"],
                    "highlight_base": gated["topology_prediction"],
                    "overlay": _overlay(hole),
                    "detail": (
                        f"holes={row['watertight']['hole_count']},"
                        f"dangling={row['watertight']['dangling_endpoint_rate']:.4f}"
                    ),
                    "highlight_title": "Hole candidates",
                    "source_row": row,
                }
            )
        gallery["metrics"]["conditional_watertightness"] = [
            {"case_id": panel["case_id"], "detail": panel["detail"]}
            for panel in panels
        ]
        _draw_page(pdf, "Watertight failure controls", panels, colors)

        for metric in ("node_f1", "edge_overlap"):
            selected = node_cases if metric == "node_f1" else edge_cases
            arm = "typed_edit_graph"
            panels = []
            for row in selected:
                arrays = _typed_visible(
                    int(row["seed"]),
                    row["operation"],
                    row["name"],
                    taxonomy,
                    arm,
                )
                wrong = arrays["after"] != arrays["target"]
                wrong &= ~arrays["region"]
                value = row["arms"][arm]["node_f1" if metric == "node_f1" else "edge_f1"]
                panels.append(
                    {
                        "case_id": f"seed{row['seed']}/{row['operation']}/{row['name']}",
                        "before": arrays["before"],
                        "after": arrays["after"],
                        "highlight_base": arrays["target"],
                        "overlay": _overlay(wrong),
                        "detail": f"{metric}={value:.6f}",
                        "highlight_title": "Outside target mismatch",
                    }
                )
            gallery["metrics"][metric] = [
                {"case_id": panel["case_id"], "detail": panel["detail"]}
                for panel in panels
            ]
            _draw_page(pdf, f"{metric} failure controls", panels, colors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gallery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "metrics": list(gallery["metrics"])}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
