#!/usr/bin/env python3
"""Run a bounded, legality-preserving proxy-guided macro displacement."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.batched import batch_routing_proxy_loss
from floorflow.chip_flow.ppa_external import (
    export_macro_tcl,
    model_batch,
    parse_orfs_graph,
)
from scripts.run_chip_external_placement import export_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--odb", type=Path, required=True)
    parser.add_argument("--output-placement", type=Path, required=True)
    parser.add_argument("--output-tcl", type=Path, required=True)
    parser.add_argument("--budget-um", type=float, default=20.0)
    parser.add_argument("--step-um", type=float, default=5.0)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--orfs-image",
        default="openroad/orfs:chips-cvpr-703649ba",
    )
    return parser.parse_args()


def graph_from_odb(path: Path, image: str):
    output = export_graph(Namespace(odb=path, orfs_image=image))
    return parse_orfs_graph(output)


def load_boxes(graph, placement: dict) -> tuple[np.ndarray, list[str]]:
    rows = {row["name"]: row for row in placement["macros"]}
    names = [macro["name"] for macro in graph.macros]
    boxes = np.asarray(
        [
            [
                rows[name]["x_um"],
                rows[name]["y_um"],
                rows[name]["width_um"],
                rows[name]["height_um"],
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    return boxes, names


def composite_proxy(boxes: np.ndarray, batch: dict) -> float:
    return float(
        batch_routing_proxy_loss(
            torch.as_tensor(boxes[None], dtype=torch.float64),
            batch["areas"].double(),
            batch["b2b"],
            batch["p2b"],
            batch["pins"].double(),
            batch["n_b2b"],
            batch["n_p2b"],
            batch["n_pins"],
        ).item()
    )


def is_legal(boxes: np.ndarray, core: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = core
    if np.any(boxes[:, 0] < x0) or np.any(boxes[:, 1] < y0):
        return False
    if np.any(boxes[:, 0] + boxes[:, 2] > x1):
        return False
    if np.any(boxes[:, 1] + boxes[:, 3] > y1):
        return False
    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    right = np.minimum(
        boxes[:, None, 0] + boxes[:, None, 2],
        boxes[None, :, 0] + boxes[None, :, 2],
    )
    bottom = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    top = np.minimum(
        boxes[:, None, 1] + boxes[:, None, 3],
        boxes[None, :, 1] + boxes[None, :, 3],
    )
    overlap = (right > left + 1.0e-5) & (top > bottom + 1.0e-5)
    return not np.any(np.triu(overlap, k=1))


def main() -> None:
    args = parse_args()
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    graph = graph_from_odb(args.odb, args.orfs_image)
    boxes, names = load_boxes(graph, placement)
    batch = model_batch(graph, torch.device("cpu"))
    before = composite_proxy(boxes, batch)
    original = boxes.copy()
    moves = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    )
    moved = set()
    for _ in range(args.passes):
        changed = False
        current = composite_proxy(boxes, batch)
        for index in range(len(boxes)):
            best_score = current
            best_boxes = None
            for dx, dy in moves:
                candidate = boxes.copy()
                candidate[index, 0] += dx * args.step_um
                candidate[index, 1] += dy * args.step_um
                displacement = candidate[index, :2] - original[index, :2]
                if np.linalg.norm(displacement) > args.budget_um + 1.0e-9:
                    continue
                if not is_legal(candidate, graph.core):
                    continue
                score = composite_proxy(candidate, batch)
                if score < best_score - 1.0e-9:
                    best_score = score
                    best_boxes = candidate
            if best_boxes is not None:
                boxes = best_boxes
                current = best_score
                moved.add(index)
                changed = True
        if not changed:
            break
    after = composite_proxy(boxes, batch)
    displacement = np.linalg.norm(boxes[:, :2] - original[:, :2], axis=1)
    placement_out = json.loads(json.dumps(placement))
    by_name = {row["name"]: row for row in placement_out["macros"]}
    for name, box in zip(names, boxes, strict=True):
        by_name[name]["x_um"] = float(box[0])
        by_name[name]["y_um"] = float(box[1])
    args.output_placement.parent.mkdir(parents=True, exist_ok=True)
    args.output_tcl.parent.mkdir(parents=True, exist_ok=True)
    args.output_placement.write_text(
        json.dumps(placement_out, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_tcl.write_text(
        export_macro_tcl(
            graph,
            boxes,
            preserve_reference_orientation=True,
        ),
        encoding="utf-8",
    )
    report = {
        "status": "pass",
        "case": placement["case"],
        "source": str(args.placement.resolve().relative_to(ROOT)),
        "proxy_before": before,
        "proxy_after": after,
        "proxy_relative": after / before - 1.0 if before else None,
        "budget_um": args.budget_um,
        "step_um": args.step_um,
        "passes": args.passes,
        "moved_macros": len(moved),
        "displacement_um": {
            "mean": float(displacement.mean()),
            "p50": float(np.quantile(displacement, 0.5)),
            "p90": float(np.quantile(displacement, 0.9)),
            "p95": float(np.quantile(displacement, 0.95)),
            "max": float(displacement.max()),
        },
        "legal_after": is_legal(boxes, graph.core),
    }
    report_path = args.output_placement.with_name(
        f"{args.output_placement.stem}_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
