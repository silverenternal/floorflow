#!/usr/bin/env python3
"""Reproduce CE2EPlan's official RPLAN mode-auto anchor and audit comparability."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
_VISUALIZATION = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        default=ROOT / ".data/external/CE2EPlan",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-ged", action="store_true")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/ce2eplan/official_test12002_v1/fid",
    )
    parser.add_argument(
        "--availability",
        type=Path,
        default=ROOT / "outputs/reports/ce2eplan_availability_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/ce2eplan_official_anchor_v1.json",
    )
    parser.add_argument(
        "--matching-gate",
        type=Path,
        default=ROOT / "outputs/reports/ce2eplan_floorflow_matching_gate_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/ce2eplan_official_anchor_v1.md",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/ce2eplan_audit_v1.md",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _init_visualization(path: str) -> None:
    global _VISUALIZATION
    spec = importlib.util.spec_from_file_location("ce2eplan_visualization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import visualization module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _VISUALIZATION = module


def _render_pair(task: dict) -> str:
    if _VISUALIZATION is None:
        raise RuntimeError("visualization module was not initialized")
    from PIL import Image

    gt = _VISUALIZATION.vis_floorplan(
        task["gt"],
        is_syn=False,
        support_boundary=True,
        canvas_size=512,
    )
    prediction = _VISUALIZATION.vis_floorplan(
        task["prediction"],
        is_syn=True,
        support_boundary=True,
        canvas_size=512,
    )
    name = task["name"]
    Image.fromarray(np.asarray(gt, dtype=np.uint8)).save(task["gt_path"] / f"{name}.png")
    Image.fromarray(np.asarray(prediction, dtype=np.uint8)).save(
        task["prediction_path"] / f"{name}.png"
    )
    return name


def _graph(row: dict, prediction: bool) -> nx.Graph:
    graph = nx.Graph()
    for room in row["rooms"]:
        graph.add_node(int(room["id"]), category=int(room["category"]))
    edge_key = "adjacencies_aligned" if prediction else "adjacencies"
    for edge in row.get(edge_key, []):
        source, target = int(edge[0]), int(edge[1])
        if source in graph and target in graph:
            graph.add_edge(source, target)
    return graph


def _node_substitution_cost(left: dict, right: dict) -> int:
    return 0 if left["category"] == right["category"] else 1


def _ged(task: tuple[dict, dict]) -> int:
    gt, prediction = task
    return int(
        nx.graph_edit_distance(
            _graph(gt, False),
            _graph(prediction, True),
            node_subst_cost=_node_substitution_cost,
        )
    )


def _box(room: dict, prediction: bool) -> list[float]:
    return room["box_aligned"] if prediction else room["box"]


def _case_structure(gt: dict, prediction: dict) -> dict:
    rooms = prediction["rooms"]
    boxes = [_box(room, True) for room in rooms]
    valid_boxes = [
        len(value) == 4
        and value[2] > value[0]
        and value[3] > value[1]
        and min(value) >= 0
        and max(value) <= 256
        for value in boxes
    ]
    areas = [
        max(0.0, float(value[2] - value[0]))
        * max(0.0, float(value[3] - value[1]))
        for value in boxes
    ]
    overlap = 0.0
    polygons = [
        box(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        for value in boxes
        if len(value) == 4
    ]
    for index, left in enumerate(polygons):
        for right in polygons[index + 1 :]:
            overlap += left.intersection(right).area
    boundary = prediction.get("boundary") or prediction.get("boundary_aligned")
    boundary_xy = (
        [[float(point[0]), float(point[1])] for point in boundary]
        if boundary
        else []
    )
    boundary_polygon = (
        Polygon(boundary_xy).buffer(0) if len(boundary_xy) >= 3 else Polygon()
    )
    contained = [
        boundary_polygon.covers(
            box(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        )
        for value in boxes
    ]
    room_polygons = []
    for room in rooms:
        coordinates = room.get("r_boundary_aligned") or []
        try:
            polygon = Polygon(coordinates).buffer(0)
        except (TypeError, ValueError):
            polygon = Polygon()
        room_polygons.append(polygon)
    room_areas = sum(polygon.area for polygon in room_polygons)
    room_overlap = 0.0
    for index, left in enumerate(room_polygons):
        for right in room_polygons[index + 1 :]:
            room_overlap += left.intersection(right).area
    room_union = Polygon()
    for polygon in room_polygons:
        room_union = room_union.union(polygon)
    outside_area = sum(
        polygon.difference(boundary_polygon).area
        for polygon in room_polygons
    )
    has_aligned_boundary = all(
        bool(room.get("r_boundary_aligned")) for room in rooms
    )
    valid_ids = {int(room["id"]) for room in rooms}
    valid_edges = all(
        int(edge[0]) in valid_ids and int(edge[1]) in valid_ids
        for edge in prediction.get("adjacencies_aligned", [])
    )
    return {
        "rooms": len(rooms),
        "all_boxes_valid": bool(valid_boxes) and all(valid_boxes),
        "all_boxes_inside_boundary": bool(contained) and all(contained),
        "all_room_polygons_valid": bool(room_polygons)
        and all(not polygon.is_empty and polygon.is_valid for polygon in room_polygons),
        "all_room_polygons_inside_boundary": bool(room_polygons)
        and all(
            boundary_polygon.covers(polygon)
            for polygon in room_polygons
            if not polygon.is_empty
        ),
        "aligned_boundaries_present": has_aligned_boundary,
        "adjacency_ids_valid": valid_edges,
        "box_overlap_ratio": overlap / max(float(sum(areas)), 1.0),
        "room_polygon_overlap_ratio": room_overlap / max(room_areas, 1.0),
        "room_outside_boundary_area_ratio": outside_area / max(room_areas, 1.0),
        "boundary_coverage_ratio": (
            room_union.intersection(boundary_polygon).area
            / max(boundary_polygon.area, 1.0)
        ),
        "gt_rooms": len(gt["rooms"]),
        "same_case_name": gt.get("name") == prediction.get("name"),
    }


def _stats(gt: dict, prediction: dict) -> dict:
    gt_rooms = gt["rooms"]
    rooms = prediction["rooms"]
    gt_edges = [tuple(map(int, edge)) for edge in gt.get("adjacencies", [])]
    pred_edges = [
        tuple(map(int, edge)) for edge in prediction.get("adjacencies_aligned", [])
    ]

    def living_neighbors(values: dict, edges: list[tuple[int, int]]) -> int:
        living_ids = {
            int(room["id"]) for room in values["rooms"] if int(room["category"]) == 0
        }
        return len(
            {
                target if source in living_ids else source
                for source, target in edges
                if source in living_ids or target in living_ids
            }
            - living_ids
        )

    def area_fraction(values: dict, categories: set[int]) -> float:
        areas = []
        for room in values["rooms"]:
            value = room["box_aligned"] if "box_aligned" in room else room["box"]
            areas.append(
                max(0.0, float(value[2] - value[0]))
                * max(0.0, float(value[3] - value[1]))
            )
        total = max(float(sum(areas)), 1.0)
        return sum(
            area
            for area, room in zip(areas, values["rooms"])
            if int(room["category"]) in categories
        ) / total

    return {
        "nr": len(rooms),
        "cl": living_neighbors(prediction, pred_edges),
        "cr": living_neighbors(prediction, pred_edges)
        / max(len(rooms) - 1, 1),
        "al": area_fraction(prediction, {0}),
        "ab": area_fraction(prediction, {1}),
        "ao": area_fraction(prediction, {2, 3, 4, 5}),
        "gt_nr": len(gt_rooms),
        "gt_cl": living_neighbors(gt, gt_edges),
        "gt_cr": living_neighbors(gt, gt_edges) / max(len(gt_rooms) - 1, 1),
        "gt_al": area_fraction(gt, {0}),
        "gt_ab": area_fraction(gt, {1}),
        "gt_ao": area_fraction(gt, {2, 3, 4, 5}),
    }


def _availability(args: argparse.Namespace, gt: list[dict], predictions: list[dict]) -> dict:
    code_root = args.external_root
    dataset_zip = code_root / "dataset/dataset.zip"
    weight_zip = code_root / "scripts/trained_model/trained_model.zip"
    readme = (code_root / "README.md").read_text(encoding="utf-8")
    return {
        "status": "downloaded_run_and_audited",
        "code": {
            "repository": "https://github.com/shidong-wang/CE2EPlan",
            "commit": "774e670941eeda3e6677303ff79e8d517ae0d1c0",
            "license_file_found": False,
        },
        "dataset": {
            "source": "DiffPlanner RPLAN preprocessed release",
            "splits": {"train": 56053, "val": 12018, "test": len(gt)},
            "zip_bytes": dataset_zip.stat().st_size,
            "zip_sha256": sha256(dataset_zip),
        },
        "weights": {
            "source": "https://github.com/shidong-wang/CE2EPlan/releases/download/Trained_model/trained_model.zip",
            "zip_bytes": weight_zip.stat().st_size,
            "zip_sha256": sha256(weight_zip),
            "checkpoint": "scripts/trained_model/b_model300000.pt",
        },
        "runtime": {
            "declared_python": "3.9.18",
            "declared_pytorch": "2.0.0",
            "executed_python": sys.version.split()[0],
            "executed_device": "cuda:0",
            "sample_command": [
                "python sample.py",
                "--dataset rplan",
                "--batch_size 1024",
                "--set_name test",
                "--model_path trained_model/b_model300000.pt",
                "--num_samples 12002",
                "--support_boundary True",
                "--mode auto",
            ],
        },
        "output": {
            "generated_cases": len(predictions),
            "post_processed_cases": sum(
                bool(row.get("adjacencies_aligned")) for row in predictions
            ),
            "boundary_conditioned": "--support_boundary True" in readme,
        },
        "redistribution": (
            "No LICENSE file was found. Keep code, dataset, and weights outside "
            "the redistributable CVPR package."
        ),
    }


def main() -> None:
    args = parse_args()
    gt_path = args.external_root / "dataset/dataset_json/data_test.json"
    prediction_path = (
        args.external_root
        / "scripts/output/fp_ce2e_official_test12002_v1/b_auto.json"
    )
    gt = read_json(gt_path)
    predictions = read_json(prediction_path)
    if not isinstance(gt, list) or not isinstance(predictions, list):
        raise TypeError("CE2EPlan data and predictions must be JSON lists")
    if args.limit is not None:
        gt = gt[: args.limit]
        predictions = predictions[: args.limit]
    gt_by_name = {str(row["name"]): row for row in gt}
    pred_names = [str(row["name"]) for row in predictions]
    if len(set(pred_names)) != len(pred_names):
        raise ValueError("generated case names are not unique")
    pairs = [(gt_by_name[name], prediction) for prediction, name in zip(predictions, pred_names)]
    structures = [_case_structure(left, right) for left, right in pairs]
    stats = [_stats(left, right) for left, right in pairs]

    ged_values = None
    if not args.skip_ged:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            ged_values = list(executor.map(_ged, pairs, chunksize=16))

    gt_images = args.image_root / ("smoke" if args.limit else "gt")
    pred_images = args.image_root / ("smoke" if args.limit else "pred")
    gt_images.mkdir(parents=True, exist_ok=True)
    pred_images.mkdir(parents=True, exist_ok=True)
    visualization_path = args.external_root / "scripts/output/visualization.py"
    tasks = [
        {
            "gt": left,
            "prediction": right,
            "name": name,
            "gt_path": gt_images,
            "prediction_path": pred_images,
        }
        for (left, right), name in zip(pairs, pred_names, strict=True)
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_visualization,
        initargs=(str(visualization_path),),
    ) as executor:
        list(executor.map(_render_pair, tasks, chunksize=32))

    fid_value = None
    if not args.skip_fid:
        import torch
        from pytorch_fid.fid_score import calculate_fid_given_paths

        fid_value = float(
            calculate_fid_given_paths(
                [str(gt_images), str(pred_images)],
                batch_size=128,
                device=torch.device("cuda:0"),
                dims=2048,
                num_workers=max(1, args.workers),
            )
        )

    stat_means = {
        key: float(np.mean([row[key] for row in stats]))
        for key in stats[0]
    }
    metric_ratios = {
        key: stat_means[key] / stat_means[f"gt_{key}"]
        for key in ("nr", "cl", "cr", "al", "ab", "ao")
    }
    validity = {
        "official_case_yield": len(structures) / len(predictions),
        "all_boxes_valid_rate": float(
            np.mean([row["all_boxes_valid"] for row in structures])
        ),
        "all_boxes_inside_boundary_rate": float(
            np.mean([row["all_boxes_inside_boundary"] for row in structures])
        ),
        "all_room_polygons_valid_rate": float(
            np.mean([row["all_room_polygons_valid"] for row in structures])
        ),
        "all_room_polygons_inside_boundary_rate": float(
            np.mean(
                [row["all_room_polygons_inside_boundary"] for row in structures]
            )
        ),
        "aligned_boundaries_present_rate": float(
            np.mean([row["aligned_boundaries_present"] for row in structures])
        ),
        "adjacency_ids_valid_rate": float(
            np.mean([row["adjacency_ids_valid"] for row in structures])
        ),
        "same_case_name_rate": float(
            np.mean([row["same_case_name"] for row in structures])
        ),
        "mean_box_overlap_ratio": float(
            np.mean([row["box_overlap_ratio"] for row in structures])
        ),
        "mean_room_polygon_overlap_ratio": float(
            np.mean([row["room_polygon_overlap_ratio"] for row in structures])
        ),
        "mean_room_outside_boundary_area_ratio": float(
            np.mean(
                [row["room_outside_boundary_area_ratio"] for row in structures]
            )
        ),
        "mean_boundary_coverage_ratio": float(
            np.mean([row["boundary_coverage_ratio"] for row in structures])
        ),
    }
    report = {
        "status": "official_rplan_anchor_reproduced_with_independent_diagnostics",
        "scope": "CE2EPlan boundary-conditioned RPLAN mode=auto",
        "cases": len(pairs),
        "published_anchor": {
            "FID": 1.08,
            "GED": 3.48,
            "Nravg": 0.995,
            "Clavg": 0.998,
            "Cravg": 1.003,
            "Alavg": 0.988,
            "Abavg": 1.007,
            "Aoavg": 1.008,
        },
        "reproduced": {
            "FID": fid_value,
            "GED_mean": float(np.mean(ged_values)) if ged_values is not None else None,
            "GED_exact_rate": (
                float(np.mean(np.asarray(ged_values) == 0))
                if ged_values is not None
                else None
            ),
            "stat_means": stat_means,
            "stat_ratio": metric_ratios,
            "validity": validity,
        },
        "method_notes": {
            "FID": "Official CE2EPlan visualization at 512x512 plus standard pytorch-fid.",
            "GED": "Exact NetworkX GED on room-type nodes and predicted/GT adjacencies.",
            "statistics": "Paper-style ratio of generated mean to GT mean.",
        },
        "claims": {
            "allowed": (
                "CE2EPlan's released checkpoint can be run with its official RPLAN "
                "data, and structural diagnostics are reproducible in this environment."
            ),
            "forbidden": [
                "CE2EPlan is a matched editing baseline.",
                "CE2EPlan and FloorFlow share one RPLAN/CubiCasa leaderboard.",
                "Independent diagnostics exactly reproduce undisclosed evaluator code.",
            ],
        },
    }
    availability = _availability(args, gt, predictions)
    matching = {
        "status": "not_matched",
        "matched_generation_comparison_available": False,
        "reason": (
            "CE2EPlan operates on RPLAN boundary-conditioned generation, while "
            "FloorFlow's current evidence is CubiCasa test503 local-delta editing. "
            "No FloorFlow RPLAN checkpoint or matched boundary-conditioning run exists."
        ),
        "required_to_unblock": (
            "Train or adapt FloorFlow to the same RPLAN split, boundary tensors, "
            "room taxonomy, post-processing, and evaluator."
        ),
        "allowed_role": "cross-dataset context anchor only",
    }
    for path, value in (
        (args.availability, availability),
        (args.output, report),
        (args.matching_gate, matching),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    table = [
        "# CE2EPlan Official RPLAN Audit",
        "",
        "| Metric | Published | Reproduced |",
        "|---|---:|---:|",
        f"| FID | 1.08 | {fid_value:.4f} |" if fid_value is not None else "| FID | 1.08 | not run |",
        (
            f"| GED | 3.48 | {np.mean(ged_values):.4f} |"
            if ged_values is not None
            else "| GED | 3.48 | not run |"
        ),
        "",
        "CE2EPlan is a boundary-conditioned RPLAN generator, not an editing baseline.",
        "It remains a cross-dataset context anchor and is not merged with CubiCasa tables.",
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    doc = [
        "# CE2EPlan Audit v1",
        "",
        "Code commit: `774e670941eeda3e6677303ff79e8d517ae0d1c0`.",
        "",
        "No LICENSE file was found in the pinned repository. The local run is",
        "permitted for audit, but code, RPLAN data, and weights are excluded from",
        "the redistributable CVPR package.",
        "",
        "The official `mode=auto` test run generated 12,002 cases on CUDA0 and",
        "completed the released post-processing. Reproduction metrics are kept",
        "separate from published anchors and from all CubiCasa editing results.",
        "",
        "Matching gate: no FloorFlow-on-RPLAN run exists, so this is not a matched",
        "external comparison.",
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(pairs),
                "FID": fid_value,
                "GED": report["reproduced"]["GED_mean"],
                "stat_ratio": metric_ratios,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
