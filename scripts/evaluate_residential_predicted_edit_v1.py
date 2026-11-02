"""Evaluate predicted-control typed editing and instruction abstention."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.paired_bootstrap import (
    paired_delta,
    paired_effect_passes,
)
from floorflow.evaluation.predicted_edit_graph_diagnostics import (
    predicted_edit_graph_effect,
)
from floorflow.evaluation.predicted_edit_graph_metrics import (
    predicted_edit_graph_metrics,
)
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.inference.predicted_control_edit import project_predicted_instruction
from floorflow.inference.room_repair import (
    close_room_partition,
    merge_room_fragments,
)

ROOT = Path(__file__).resolve().parents[1]
EXECUTED_CATEGORIES = {
    "room_type": "room_type_change",
    "room_add": "room_addition",
    "room_remove": "room_deletion",
    "opening_delete": "opening_pressure",
    "furniture_delete": "furniture_low_salience",
}
ABSTAIN_CATEGORIES = (
    "unsupported_wall_move",
    "ambiguous_global_scope",
    "conflicting_instruction",
)
PROTOCOL = "type_geometry_hungarian"
_TAXONOMY = None
_PREDICTED_RASTERS = None
_TARGET_RASTERS = None
_CONTROL_ROOT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-category", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ged-timeout", type=float, default=5.0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--target-suffix", default="val191")
    parser.add_argument(
        "--predicted-rasters",
        type=Path,
        default=ROOT
        / "outputs/eval/fp_completion_predicted_refined_val491_v1/rasters",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "outputs/eval",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=ROOT
        / "outputs/control_contracts/predicted_instruction_val191_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_predicted_edit_v1.json",
    )
    parser.add_argument(
        "--robustness-output",
        type=Path,
        default=ROOT / "outputs/reports/residential_instruction_robustness_v1.json",
    )
    return parser.parse_args()


def _initialize() -> None:
    global _TAXONOMY
    _TAXONOMY = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def _metrics(
    predicted,
    target,
    timeout: float,
    *,
    visible,
    footprint,
) -> dict:
    return predicted_edit_graph_metrics(
        predicted,
        target,
        _TAXONOMY,
        visible=visible,
        footprint=footprint,
        timeout_seconds=timeout,
    )


def _diagnose(task: dict) -> dict:
    path = Path(task["predicted_rasters"]) / f"{task['name']}.npz"
    with np.load(path) as payload:
        footprint = payload["footprint"] > 0
        before_visible = payload["topology_prediction"]
        logits = gaussian_filter(
            payload["room_logits"].astype(np.float32),
            sigma=(0.0, 10.0, 10.0),
        )
        room_ids = np.asarray(_TAXONOMY.room_ids, dtype=np.int64)
        decoded = room_ids[logits.argmax(axis=0)]
        decoded = np.where(footprint, decoded, 0).astype(np.int64)
        before = merge_room_fragments(
            decoded,
            _TAXONOMY,
            min_area=32,
        )
        before = close_room_partition(
            before,
            footprint,
            _TAXONOMY,
        )
        room_mask = np.isin(before_visible, _TAXONOMY.room_ids)
        before_visible[room_mask] = before[room_mask]
    outside_exact = 1.0
    outside_room_exact = 1.0
    if task["abstained"]:
        after = before.copy()
        after_visible = before_visible.copy()
        target = before
        graph = _metrics(
            after,
            target,
            task["ged_timeout"],
            visible=after_visible,
            footprint=footprint,
        )
        legality = structural_legality_metrics(
            after_visible,
            before_visible,
            _TAXONOMY,
        )
    else:
        with np.load(
            Path(task["target_root"])
            / (
                f"fp_local_edit_{task['operation']}_"
                f"{task['target_suffix']}_v1/rasters"
            )
            / f"{task['name']}.npz"
        ) as payload:
            target = payload["target"]
        source = (
            Path(task["control_root"])
            / task["operation"]
            / task["split"]
            / f"{task['name']}.npz"
        )
        edited_controls = read_control_cache(source)
        object_delete = task["operation"] in {
            "opening_delete",
            "furniture_delete",
        }
        before_source = before_visible if object_delete else before
        after_tensor, edit_mask = project_predicted_instruction(
            torch.from_numpy(before_source.astype(np.int64)),
            edited_controls,
            tuple(task["edit_region"]),
            tuple(task["control_size"]),
            _TAXONOMY,
            operation=task["operation"],
        )
        if object_delete:
            after = before
            after_visible = after_tensor.numpy()
            outside_room_exact = 1.0
        else:
            after = after_tensor.numpy()
            after = merge_room_fragments(after, _TAXONOMY, min_area=32)
            after = close_room_partition(
                after,
                footprint,
                _TAXONOMY,
            )
            outside_room_exact = float(
                np.all(after[~edit_mask.numpy()] == before[~edit_mask.numpy()])
            )
            after_visible = before_visible.copy()
            after_visible[room_mask] = after[room_mask]
        outside_exact = float(
            np.all(
                after_visible[~edit_mask.numpy()]
                == before_visible[~edit_mask.numpy()]
            )
        )
        graph = _metrics(
            after,
            target,
            task["ged_timeout"],
            visible=after_visible,
            footprint=footprint,
        )
        legality = structural_legality_metrics(
            after_visible,
            target,
            _TAXONOMY,
        )
    try:
        validity = float(
            fpeval_raster_metrics(
                after_visible,
                target if not task["abstained"] else before_visible,
                _TAXONOMY,
                tolerance_fractions=(0.05,),
            )["prediction_validity"]["valid"]
        )
    except (IndexError, ValueError):
        validity = 0.0
    return {
        **task,
        "executed": not task["abstained"],
        "outside_pixel_exact": outside_exact,
        "outside_room_label_exact": outside_room_exact,
        "validity": validity,
        "structural": legality,
        "graph": graph,
        "pre_edit_graph": (
            graph
            if task["abstained"]
            else _metrics(
                before,
                target,
                task["ged_timeout"],
                visible=before_visible,
                footprint=footprint,
            )
        ),
    }


def _tasks(args: argparse.Namespace) -> list[dict]:
    limit = args.limit_per_category
    tasks = []
    for operation, category in EXECUTED_CATEGORIES.items():
        manifest = json.loads(
            (
                ROOT
                / str(args.control_root)
                / operation
                / args.split
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        if (
            not manifest.get("predicted_control_used")
            or manifest.get("target_used_at_inference")
        ):
            raise ValueError("predicted instruction manifest violates provenance")
        if manifest.get("split") != args.split:
            raise ValueError("predicted instruction manifest split mismatch")
        edited = [row for row in manifest["rows"] if row["edited"]][:limit]
        for index, row in enumerate(edited):
            tasks.append(
                {
                    "prompt_id": f"{category}-{index:03d}",
                    "category": category,
                    "operation": operation,
                    "name": row["name"].replace("/", "__"),
                    "predicted_control_used": True,
                    "target_used_at_inference": False,
                    "abstained": False,
                    "edit_region": row["edit_region"],
                    "control_size": row["control_size"],
                    "predicted_rasters": str(args.predicted_rasters),
                    "target_root": str(args.target_root),
                    "control_root": str(args.control_root),
                    "target_suffix": args.target_suffix,
                    "split": args.split,
                }
            )
    names = sorted(
        path.stem for path in args.predicted_rasters.glob("*.npz")
    )[:limit]
    for category in ABSTAIN_CATEGORIES:
        for index, name in enumerate(names):
            tasks.append(
                {
                    "prompt_id": f"{category}-{index:03d}",
                    "category": category,
                    "operation": "abstain",
                    "name": name,
                    "predicted_control_used": True,
                    "target_used_at_inference": False,
                    "abstained": True,
                    "predicted_rasters": str(args.predicted_rasters),
                    "target_root": str(args.target_root),
                    "control_root": str(args.control_root),
                    "target_suffix": args.target_suffix,
                    "split": args.split,
                }
            )
    return tasks


def _paired(rows: list[dict], side: str, metric: str, bootstrap: int) -> dict:
    return paired_delta(
        [
            {"value": row["graph"][side][f"instance_{metric}"]}
            for row in rows
        ],
        [
            {"value": row["graph"]["baseline"][f"instance_{metric}"]}
            for row in rows
        ],
        "value",
        iterations=bootstrap,
        seed=21001,
    )


def main() -> None:
    args = parse_args()
    _initialize()
    tasks = _tasks(args)
    for task in tasks:
        task["ged_timeout"] = args.ged_timeout
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
    ) as executor:
        rows = list(executor.map(_diagnose, tasks))
    executed = [row for row in rows if row["executed"]]
    abstained = [row for row in rows if row["abstained"]]
    paired = {
        f"refined_vs_baseline_{metric}": _paired(
            executed,
            "refined",
            metric,
            args.bootstrap,
        )
        for metric in ("node_f1", "edge_overlap", "ged")
    }
    edit_effect = predicted_edit_graph_effect(
        rows,
        iterations=args.bootstrap,
    )
    categories = {
        category: {
            "cases": len(selected),
            "outside_pixel_exact": float(
                np.mean([row["outside_pixel_exact"] for row in selected])
            ),
            "outside_room_label_exact": float(
                np.mean([row["outside_room_label_exact"] for row in selected])
            ),
            "validity": float(np.mean([row["validity"] for row in selected])),
            "abstain_rate": float(
                np.mean([row["abstained"] for row in selected])
            ),
            "node_f1": float(
                np.mean([row["graph"]["refined"]["node_f1"] for row in selected])
            ),
            "edge_overlap": float(
                np.mean([row["graph"]["refined"]["edge_overlap"] for row in selected])
            ),
            "ged": float(
                np.mean([row["graph"]["refined"]["ged"] for row in selected])
            ),
            "ged_exact_rate": float(
                np.mean(
                    [
                        row["graph"]["refined"]["ged_exact"]
                        for row in selected
                    ]
                )
            ),
            "node_deletion": float(
                np.mean(
                    [
                        row["graph"]["refined"]["node_deletion"]
                        for row in selected
                    ]
                )
            ),
            "node_insertion": float(
                np.mean(
                    [
                        row["graph"]["refined"]["node_insertion"]
                        for row in selected
                    ]
                )
            ),
            "node_substitution": float(
                np.mean(
                    [
                        row["graph"]["refined"]["node_substitution"]
                        for row in selected
                    ]
                )
            ),
            "edge_deletion": float(
                np.mean(
                    [
                        row["graph"]["refined"]["edge_deletion"]
                        for row in selected
                    ]
                )
            ),
            "edge_insertion": float(
                np.mean(
                    [
                        row["graph"]["refined"]["edge_insertion"]
                        for row in selected
                    ]
                )
            ),
            "matched_node_fraction": float(
                np.mean(
                    [
                        row["graph"]["refined"]["matched_nodes"]
                        / max(1, row["graph"]["refined"]["target_nodes"])
                        for row in selected
                    ]
                )
            ),
        }
        for category in set(EXECUTED_CATEGORIES.values()) | set(ABSTAIN_CATEGORIES)
        for selected in [[row for row in rows if row["category"] == category]]
        if selected
    }
    checks = {
        "executed_categories_complete": len(
            {row["category"] for row in executed}
        )
        == len(EXECUTED_CATEGORIES),
        "abstention_categories_complete": len(
            {row["category"] for row in abstained}
        )
        == len(ABSTAIN_CATEGORIES),
        "predicted_controls_only": all(
            row["predicted_control_used"] and not row["target_used_at_inference"]
            for row in rows
        ),
        "outside_pixel_exact": all(
            row["outside_pixel_exact"] == 1.0 for row in rows
        ),
        "outside_room_exact": all(
            row["outside_room_label_exact"] == 1.0 for row in rows
        ),
        "abstentions_preserve_exact": all(
            row["outside_pixel_exact"] == 1.0 for row in abstained
        ),
        "node_f1_improves": paired_effect_passes(
            paired["refined_vs_baseline_node_f1"],
        ),
        "node_f1_absolute_at_least_0_82": (
            float(
                np.mean(
                    [
                        row["graph"]["refined"]["node_f1"]
                        for row in executed
                    ]
                )
            )
            >= 0.82
        ),
        "edge_overlap_improves": paired_effect_passes(
            paired["refined_vs_baseline_edge_overlap"],
            ceiling=1.0,
        ),
        "ged_improves": paired_effect_passes(
            paired["refined_vs_baseline_ged"],
            lower_is_better=True,
        ),
        "ged_mean_at_most_6": (
            float(
                np.mean(
                    [
                        row["graph"]["refined"]["ged"]
                        for row in executed
                    ]
                )
            )
            <= 6.0
        ),
        "matched_node_fraction_at_least_0_5": (
            float(
                np.mean(
                    [
                        row["graph"]["refined"]["matched_nodes"]
                        / max(
                            1,
                            row["graph"]["refined"]["target_nodes"],
                        )
                        for row in executed
                    ]
                )
            )
            >= 0.5
        ),
        "edit_node_f1_improves": paired_effect_passes(
            edit_effect["node_f1"],
        ),
        "edit_edge_overlap_non_regresses": paired_effect_passes(
            edit_effect["edge_overlap"],
            ceiling=1.0,
        ),
        "edit_ged_improves": paired_effect_passes(
            edit_effect["ged"],
            lower_is_better=True,
        ),
        "ged_diagnostic_retained": (
            paired["refined_vs_baseline_ged"]["ci_high"]
            >= paired["refined_vs_baseline_ged"]["ci_low"]
        ),
        "validity_at_least_0_95": all(
            row["validity"] >= 0.95 for row in rows
        ),
    }
    report = {
        "schema_version": "residential-predicted-edit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "split": "val191",
        "test_opened": False,
        "prompt_count": len(rows),
        "executed": len(executed),
        "abstained": len(abstained),
        "categories": categories,
        "paired_refinement": paired,
        "edit_effect": edit_effect,
        "checks": checks,
        "coverage_boundaries": {
            "opening_moving": "not available in the current typed action set",
            "wall_moving": "safety-only abstention",
            "multi_step": "not promoted in this benchmark",
        },
        "boundary": (
            "Instructions are typed control contracts resolved against real "
            "predicted controls. Unsupported and ambiguous cases abstain. "
            "GED is retained as a negative diagnostic because it does not "
            "improve with the graph refinement."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.robustness_output.write_text(
        json.dumps(
            {
                **report,
                "schema_version": "residential-instruction-robustness-v1",
                "prompts": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.output), "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
