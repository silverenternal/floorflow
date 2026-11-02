"""Summarize deterministic distribution-overlay evaluation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from floorflow.evaluation.cad_metric_suite import build_cad_metric_report


def summarize_metric_rows(rows: dict[str, list[dict]]) -> dict:
    return {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }


def build_overlay_report(
    *,
    args: argparse.Namespace,
    overlay_settings: object,
    stop: int,
    rows: dict[str, list[dict]],
    object_rows: dict[str, list[dict]],
    constraint_rows: list[dict],
    opening_readout_counts: Counter,
    room_inter: np.ndarray,
    room_union: np.ndarray,
    room_support_inter: int,
    room_support_union: int,
) -> dict:
    summary = summarize_metric_rows(rows)
    objects = summarize_metric_rows(object_rows)
    present_classes = room_union > 0
    dataset_class_iou = np.divide(
        room_inter,
        room_union,
        out=np.zeros_like(room_inter, dtype=np.float64),
        where=present_classes,
    )
    conditional_watertight = [
        row["fpeval_watertight"]
        for row in rows["overlay"]
        if row["target_watertight"] > 0.5
    ]
    dataset_metrics = {
        "room_dataset_macro_iou": float(
            dataset_class_iou[present_classes].mean()
        ),
        "room_dataset_micro_iou": float(
            room_inter.sum() / max(1, room_union.sum())
        ),
        "room_support_micro_iou": float(
            room_support_inter / max(1, room_support_union)
        ),
        "conditional_watertight": (
            float(np.mean(conditional_watertight))
            if conditional_watertight
            else 0.0
        ),
    }
    cad_rows = [
        {
            **rows["overlay"][index],
            **objects["overlay"],
            **constraint_rows[index],
            **dataset_metrics,
            **(
                {"style_separation": args.style_separation}
                if args.style_separation is not None
                else {}
            ),
            **(
                {"seed_iou_gap": args.seed_iou_gap}
                if args.seed_iou_gap is not None
                else {}
            ),
        }
        for index in range(len(rows["overlay"]))
    ]
    baseline = json.loads(args.retrieval_baseline.read_text(encoding="utf-8"))
    return {
        "checkpoint": str(args.checkpoint),
        "ensemble_checkpoint": (
            str(args.ensemble_checkpoint)
            if args.ensemble_checkpoint is not None
            else None
        ),
        "ensemble_weight": args.ensemble_weight,
        "cases": stop - args.offset,
        "condition_mode": (
            "contour_plus_style"
            if args.style_id is not None
            else "contour_only"
        ),
        "style_id": args.style_id,
        "best_of_n_used": False,
        "opening_readout_sources": dict(opening_readout_counts),
        "overlay_settings": overlay_settings.__dict__,
        "legalize_objects": bool(args.legalize_objects),
        "legalization": {
            "opening_wall_radius": args.legalization_opening_wall_radius,
            "closure_radius": args.legalization_closure_radius,
            "dynamic_closure": args.legalization_dynamic_closure,
        },
        "room_metrics": summary,
        "object_metrics": objects,
        "cad_metrics": build_cad_metric_report(cad_rows),
        "constraint_metrics": {
            key: float(np.mean([row[key] for row in constraint_rows]))
            for key in constraint_rows[0]
        },
        "retrieval_baseline_room": baseline["metrics"]["room_macro_iou"],
        "retrieval_prior_weight": args.retrieval_prior_weight,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_metric": args.retrieval_metric,
        "program_checkpoint": (
            str(args.program_checkpoint) if args.program_checkpoint else None
        ),
        "room_segmenter_checkpoint": (
            str(args.room_segmenter_checkpoint)
            if args.room_segmenter_checkpoint
            else None
        ),
        "layout_checkpoint": (
            str(args.layout_checkpoint) if args.layout_checkpoint else None
        ),
        "room_prior_weight": args.room_prior_weight,
        "room_partition_weight": args.room_partition_weight,
        "program_prior_weight": args.program_prior_weight,
        "room_prior_confidence_threshold": (
            args.room_prior_confidence_threshold
        ),
        "room_delta_vs_retrieval": (
            summary["overlay"]["room_macro_iou"]
            - baseline["metrics"]["room_macro_iou"]
        ),
        "overlay_object_delta_vs_raw": {
            key: objects["overlay"][key] - objects["raw"][key]
            for key in objects["raw"]
        },
    }


__all__ = ["build_overlay_report", "summarize_metric_rows"]
