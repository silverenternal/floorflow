#!/usr/bin/env python3
"""Run a matched 2x2 control-gate and watertight-guard factorial."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import edit_detection_iou_mse
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.wall_junction import wall_junction_metrics
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.local_delta import compose_local_delta
from floorflow.inference.topology_scaffold import (
    select_guarded_topology,
    verify_topology,
)

OPERATIONS = ("room_type", "room_add", "room_remove", "opening_delete", "furniture_delete")
ARMS = ("control_on_guard_on", "control_on_guard_off", "control_off_guard_on", "control_off_guard_off")
_TAXONOMY: FloorplanTaxonomy | None = None
_ROOT = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/eval"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--edit-cache",
        type=Path,
        default=Path("outputs/cache/fp_edit_detection_v1"),
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("outputs/reports/fp_safety_factorial_2x2_rows_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_safety_factorial_2x2_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_safety_factorial_2x2.md"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("paper/figures/fp_safety_factorial_2x2.svg"),
    )
    return parser.parse_args()


def _initialize(
    root: str,
    evaluation_root: str,
    taxonomy_path: str,
) -> None:
    global _ROOT, _TAXONOMY
    _ROOT = Path(root)
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    if not Path(evaluation_root).is_absolute():
        _ROOT = Path(root)


def _region_mask(
    shape: tuple[int, int],
    normalized: list[float],
) -> np.ndarray:
    x, y, width, height = normalized
    rows, columns = shape
    left = max(0, min(columns, round(float(x) * columns)))
    top = max(0, min(rows, round(float(y) * rows)))
    right = max(left + 1, min(columns, round((x + width) * columns)))
    bottom = max(top + 1, min(rows, round((y + height) * rows)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _compose(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return compose_local_delta(
        torch.from_numpy(before),
        torch.from_numpy(after),
        torch.from_numpy(mask),
    ).numpy()


def _assert_metrics(
    visible: np.ndarray,
    room_labels: np.ndarray,
    topology: np.ndarray,
    before: dict,
    target_after: np.ndarray,
    region: np.ndarray,
    target_watertight: bool,
    taxonomy: FloorplanTaxonomy,
) -> dict:
    footprint = before["footprint"].astype(bool)
    wall = (
        (before["target"] == taxonomy.class_id("wall"))
        | (target_after == taxonomy.class_id("wall"))
    )
    wall = binary_dilation(wall, iterations=1)
    valid = footprint & ~wall
    outside = valid & ~region
    inside = valid & region
    outside_valid = int(outside.sum())
    pixel_exact = (
        float(
            (
                visible[outside] == before["prediction"][outside]
            ).mean()
        )
        if outside_valid
        else 1.0
    )
    room_exact = (
        float(
            (
                room_labels[outside] == before["room_labels"][outside]
            ).mean()
        )
        if outside_valid
        else 1.0
    )
    delta_iou, delta_mse = edit_detection_iou_mse(
        before["target"],
        target_after,
        before["prediction"],
        visible,
        valid,
    )
    topology_validity = verify_topology(topology, taxonomy)
    junction_metrics = wall_junction_metrics(
        raster_labels_to_wall_plan(topology, taxonomy).walls,
        raster_labels_to_wall_plan(target_after, taxonomy).walls,
    )
    watertight = bool(topology_validity.get("watertight", False))
    boundary = binary_dilation(region, iterations=1) & ~region
    wall_boundary = boundary & (
        (before["target"] == taxonomy.class_id("wall"))
        | (target_after == taxonomy.class_id("wall"))
    )
    return {
        "outside_pixel_exact": pixel_exact,
        "outside_room_exact": room_exact,
        "outside_pixels": outside_valid,
        "delta_iou": float(delta_iou),
        "delta_mse": float(delta_mse),
        "validity": float(bool(topology_validity.get("valid", False))),
        "conditional_watertight": float(watertight if target_watertight else True),
        "wall_break": int(
            target_watertight
            and not watertight
            and bool(wall_boundary.any())
            and junction_metrics["junction_f1"] < 0.9
        ),
        "junction_f1": float(junction_metrics["junction_f1"]),
        "inside_change_fraction": (
            float((visible[inside] != before["prediction"][inside]).mean())
            if inside.any()
            else 0.0
        ),
    }


def _diagnose(payload: dict) -> dict:
    if _TAXONOMY is None:
        raise RuntimeError("worker taxonomy was not initialized")
    operation = payload["operation"]
    record = payload["record"]
    name = record["name"]
    safe_name = name.replace("/", "__")
    before_path = (
        _ROOT
        / "outputs/eval"
        / f"fp_hard_metrics_full_test503_seed{payload['seed']}_v1"
        / "rasters"
        / f"{safe_name}.npz"
    )
    after_dir = (
        f"fp_local_edit_{operation}_test503_v1"
        if payload["seed"] == 19
        else f"fp_local_edit_{operation}_test503_seed{payload['seed']}_v1"
    )
    raw_path = _ROOT / "outputs/eval" / after_dir / "rasters" / f"{safe_name}.npz"
    gated_path = (
        _ROOT
        / "outputs/eval"
        / f"fp_control_gated_local_test503_seed{payload['seed']}_v1"
        / operation
        / f"{safe_name}.npz"
    )
    edit_cache = _ROOT / "outputs/cache/fp_edit_detection_v1" / operation / f"{safe_name}.npz"
    with (
        np.load(before_path) as before,
        np.load(raw_path) as raw,
        np.load(gated_path) as gated,
        np.load(edit_cache) as cache,
    ):
        before_data = {key: before[key] for key in before.files}
        region = _region_mask(
            before["target"].shape,
            record["edit_region_normalized"],
        ) & before["footprint"].astype(bool)
        control_mask = region & cache["changed"].astype(bool)
        target_after = gated["target"]
        control_visible = _compose(
            before["prediction"],
            raw["prediction"],
            control_mask,
        )
        control_rooms = _compose(
            before["room_labels"],
            raw["room_labels"],
            control_mask,
        )
        control_topology = _compose(
            before["topology_prediction"],
            raw["topology_prediction"],
            control_mask,
        )
        raw_visible = raw["prediction"]
        raw_rooms = raw["room_labels"]
        raw_topology = raw["topology_prediction"]
        guarded_control, _ = select_guarded_topology(
            before["topology_prediction"],
            raw["topology_prediction"],
            region,
            _TAXONOMY,
        )
        guarded_raw, _ = select_guarded_topology(
            before["topology_prediction"],
            raw["topology_prediction"],
            region,
            _TAXONOMY,
        )
        target_watertight = payload["target_watertight"]
        arms = {
            "control_on_guard_on": _assert_metrics(
                control_visible,
                control_rooms,
                guarded_control,
                before_data,
                target_after,
                region,
                target_watertight,
                _TAXONOMY,
            ),
            "control_on_guard_off": _assert_metrics(
                control_visible,
                control_rooms,
                control_topology,
                before_data,
                target_after,
                region,
                target_watertight,
                _TAXONOMY,
            ),
            "control_off_guard_on": _assert_metrics(
                raw_visible,
                raw_rooms,
                guarded_raw,
                before_data,
                target_after,
                region,
                target_watertight,
                _TAXONOMY,
            ),
            "control_off_guard_off": _assert_metrics(
                raw_visible,
                raw_rooms,
                raw_topology,
                before_data,
                target_after,
                region,
                target_watertight,
                _TAXONOMY,
            ),
        }
    return {
        "seed": payload["seed"],
        "operation": operation,
        "name": name,
        "manifest_index": payload["manifest_index"],
        "target_watertight": target_watertight,
        "arms": arms,
    }


def _tasks(args: argparse.Namespace, seed: int) -> list[dict]:
    tasks = []
    per_case_cache = {}
    for operation in OPERATIONS:
        manifest = json.loads(
            (
                args.control_root
                / f"fp_local_edit_{operation}_test503_v1"
                / "test"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        edit_dir = (
            f"fp_local_edit_{operation}_test503_v1"
            if seed == 19
            else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
        )
        per_case_cache[operation] = json.loads(
            (args.evaluation_root / edit_dir / "per_case.json").read_text(
                encoding="utf-8"
            )
        )
        for index, record in enumerate(manifest["rows"]):
            if record["edited"]:
                tasks.append(
                    {
                        "seed": seed,
                        "operation": operation,
                        "manifest_index": index,
                        "record": record,
                        "target_watertight": bool(
                            per_case_cache[operation][index]["target_watertight"]
                            > 0.5
                        ),
                    }
                )
    return tasks[: args.limit] if args.limit is not None else tasks


def _summary(rows: list[dict]) -> dict:
    keys = (
        "outside_pixel_exact",
        "outside_room_exact",
        "delta_iou",
        "delta_mse",
        "validity",
        "conditional_watertight",
        "wall_break",
        "junction_f1",
        "inside_change_fraction",
    )
    return {
        arm: {
            key: float(np.mean([row["arms"][arm][key] for row in rows]))
            for key in keys
        }
        for arm in ARMS
    }


def _paired_effect(rows: list[dict], left: str, right: str, key: str) -> dict:
    current = [{"value": row["arms"][left][key]} for row in rows]
    baseline = [{"value": row["arms"][right][key]} for row in rows]
    return paired_delta(
        current,
        baseline,
        "value",
        iterations=5000,
        seed=67,
    )


def _table(report: dict) -> str:
    lines = [
        "# Safety Mechanism Factorial (2x2)",
        "",
        "| Arm | Outside exact | Room exact | Delta-IoU | Delta-MSE | Validity | Watertight | Wall breaks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["summary"].items():
        lines.append(
            f"| {arm} | {values['outside_pixel_exact']:.6f} | "
            f"{values['outside_room_exact']:.6f} | {values['delta_iou']:.6f} | "
            f"{values['delta_mse']:.9f} | {values['validity']:.6f} | "
            f"{values['conditional_watertight']:.6f} | "
            f"{values['wall_break']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The four arms share the same rows, checkpoints, control masks, and scorer.",
            (
                "Control off/guard on uses raw visible output with guarded "
                "topology; control on/guard off uses gated visible and "
                "unguarded topology."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    control = ("off", "on")
    guard = ("off", "on")
    values = np.zeros((2, 2))
    text = np.empty((2, 2), dtype=object)
    for row, control_state in enumerate(control):
        for column, guard_state in enumerate(guard):
            arm = f"control_{control_state}_guard_{guard_state}"
            metric = report["summary"][arm]["conditional_watertight"]
            values[row, column] = metric
            text[row, column] = (
                f"{metric:.3f}\noutside={report['summary'][arm]['outside_pixel_exact']:.3f}"
            )
    figure, axis = plt.subplots(figsize=(4.2, 3.2))
    image = axis.imshow(values, vmin=0.8, vmax=1.0, cmap="viridis")
    axis.set_xticks([0, 1], labels=["Guard off", "Guard on"])
    axis.set_yticks([0, 1], labels=["Control off", "Control on"])
    axis.set_title("Conditional watertightness")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                text[row, column],
                ha="center",
                va="center",
                color="white" if values[row, column] < 0.95 else "black",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19]
    all_rows = []
    per_seed = {}
    for seed in seeds:
        tasks = _tasks(args, seed)
        rows = []
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_initialize,
            initargs=(
                str(Path.cwd()),
                str(args.evaluation_root),
                str(args.taxonomy),
            ),
        ) as executor:
            futures = [executor.submit(_diagnose, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(f"seed {seed}: {completed}/{len(futures)}", flush=True)
        rows.sort(key=lambda row: (row["operation"], row["manifest_index"]))
        all_rows.extend(rows)
        summary = _summary(rows)
        effects = {
            "control_gate_on_minus_off": {
                key: _paired_effect(
                    rows,
                    "control_on_guard_off",
                    "control_off_guard_off",
                    key,
                )
                for key in (
                    "outside_pixel_exact",
                    "outside_room_exact",
                    "delta_iou",
                    "delta_mse",
                )
            },
            "guard_on_minus_off": {
                key: _paired_effect(
                    rows,
                    "control_on_guard_on",
                    "control_on_guard_off",
                    key,
                )
                for key in (
                    "conditional_watertight",
                    "wall_break",
                    "delta_iou",
                    "delta_mse",
                )
            },
        }
        per_seed[str(seed)] = {
            "cases": len(rows),
            "summary": summary,
            "effects": effects,
        }
    first = next(iter(per_seed.values()))
    report = {
        "status": "complete",
        "seeds": seeds,
        "cases_per_seed": len(all_rows) // len(seeds),
        "design": {
            "control_gate": ["on", "off"],
            "watertight_guard": ["on", "off"],
            "matched_fields": [
                "rows",
                "checkpoints",
                "edit masks",
                "scorer",
                "thresholds",
            ],
        },
        "summary": first["summary"],
        "effects": first["effects"],
        "per_seed": per_seed,
        "checks": {
            "all_four_arms_present": all(
                set(values["summary"]) == set(ARMS)
                for values in per_seed.values()
            ),
            "control_off_reduces_outside_exact": (
                first["summary"]["control_off_guard_off"]["outside_pixel_exact"]
                < first["summary"]["control_on_guard_off"]["outside_pixel_exact"]
            ),
            "guard_has_matched_effect": (
                "guard_on_minus_off" in first["effects"]
            ),
        },
        "boundary": (
            "This factorial isolates visible-output locality from topology guard "
            "selection under matched rows. It does not create a new checkpoint or "
            "change the frozen main metrics."
        ),
    }
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": "complete", "rows": all_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(report), encoding="utf-8")
    _plot(report, args.figure)
    print(json.dumps({
        "summary": report["summary"],
        "effects": report["effects"],
        "checks": report["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
