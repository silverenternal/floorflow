#!/usr/bin/env python3
"""Stress-test strict locality at walls, openings, and exterior boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
SUITES = ("wall_adjacent", "opening_covering", "exterior_boundary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
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
        "--rows-output",
        type=Path,
        default=Path("outputs/reports/fp_strict_locality_stress_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_strict_locality_stress_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_strict_locality_stress.md"),
    )
    parser.add_argument(
        "--panels",
        type=Path,
        default=Path("outputs/reports/fp_strict_locality_stress_panels_v1"),
    )
    return parser.parse_args()


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


def _outside_exact(
    prediction: np.ndarray,
    before: np.ndarray,
    outside: np.ndarray,
) -> tuple[float, int, int]:
    total = int(outside.sum())
    changed = int((prediction[outside] != before[outside]).sum())
    return (float(1.0 - changed / total) if total else 1.0, changed, total)


def _task_rows(args: argparse.Namespace) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest = json.loads(
            (
                args.control_root
                / f"fp_local_edit_{operation}_test503_v1"
                / "test"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        for index, record in enumerate(manifest["rows"]):
            if record["edited"]:
                tasks.append(
                    {
                        "operation": operation,
                        "manifest_index": index,
                        "record": record,
                    }
                )
    return tasks


def _paths(
    args: argparse.Namespace,
    seed: int,
    task: dict,
) -> tuple[Path, Path, Path]:
    name = task["record"]["name"].replace("/", "__")
    before = (
        args.evaluation_root
        / f"fp_hard_metrics_full_test503_seed{seed}_v1"
        / "rasters"
        / f"{name}.npz"
    )
    edit_dir = (
        f"fp_local_edit_{task['operation']}_test503_v1"
        if seed == 19
        else f"fp_local_edit_{task['operation']}_test503_seed{seed}_v1"
    )
    raw = args.evaluation_root / edit_dir / "rasters" / f"{name}.npz"
    gated = (
        args.evaluation_root
        / f"fp_control_gated_local_test503_seed{seed}_v1"
        / task["operation"]
        / f"{name}.npz"
    )
    return before, raw, gated


def _evaluate_case(
    args: argparse.Namespace,
    seed: int,
    taxonomy: FloorplanTaxonomy,
    task: dict,
    fpeval_row: dict,
    before_case: dict,
) -> list[dict]:
    before_path, raw_path, gated_path = _paths(args, seed, task)
    with (
        np.load(before_path) as before,
        np.load(raw_path) as raw,
        np.load(gated_path) as gated,
    ):
        footprint = before["footprint"].astype(bool)
        region = _region_mask(
            before["target"].shape,
            task["record"]["edit_region_normalized"],
        ) & footprint
        target_after = gated["target"]
        wall = (
            (before["target"] == taxonomy.class_id("wall"))
            | (target_after == taxonomy.class_id("wall"))
        )
        wall = binary_dilation(wall, iterations=1)
        valid = footprint & ~wall
        outside = valid & ~region
        inside = valid & region
        raw_pixel, raw_pixels_changed, outside_pixels = _outside_exact(
            raw["prediction"],
            before["prediction"],
            outside,
        )
        gated_pixel, gated_pixels_changed, _ = _outside_exact(
            gated["prediction"],
            before["prediction"],
            outside,
        )
        raw_room, raw_rooms_changed, _ = _outside_exact(
            raw["room_labels"],
            before["room_labels"],
            outside,
        )
        gated_room, gated_rooms_changed, _ = _outside_exact(
            gated["room_labels"],
            before["room_labels"],
            outside,
        )
        inside_area = max(1, int(inside.sum()))
        raw_inside_change = float(
            (raw["prediction"][inside] != before["prediction"][inside]).mean()
            if inside.any()
            else 0.0
        )
        gated_inside_change = float(
            (gated["prediction"][inside] != before["prediction"][inside]).mean()
            if inside.any()
            else 0.0
        )
        neighborhood = binary_dilation(region, iterations=2) & footprint
        room_values = target_after[
            neighborhood & np.isin(target_after, np.asarray(taxonomy.room_ids))
        ]
        room_types = {int(value) for value in np.unique(room_values)}
        opening = np.isin(target_after, np.asarray(taxonomy.opening_ids))
        opening_contact = bool(
            (
                binary_dilation(region, iterations=1)
                & opening
                & footprint
            ).any()
        )
        near_wall = bool((neighborhood & (target_after == taxonomy.class_id("wall"))).any())
        near_exterior = bool(
            (binary_dilation(region, iterations=5) & ~footprint).any()
        )
        common = {
            "seed": seed,
            "operation": task["operation"],
            "name": task["record"]["name"],
            "manifest_index": task["manifest_index"],
            "before_path": str(before_path),
            "raw_path": str(raw_path),
            "gated_path": str(gated_path),
            "outside_pixels": outside_pixels,
            "inside_pixels": inside_area,
            "raw_outside_exact_ratio": raw_pixel,
            "gated_outside_exact_ratio": gated_pixel,
            "raw_outside_changes": raw_pixels_changed,
            "gated_outside_changes": gated_pixels_changed,
            "raw_outside_room_exact_ratio": raw_room,
            "gated_outside_room_exact_ratio": gated_room,
            "raw_outside_room_changes": raw_rooms_changed,
            "gated_outside_room_changes": gated_rooms_changed,
            "raw_inside_change_fraction": raw_inside_change,
            "gated_inside_change_fraction": gated_inside_change,
            "validity": float(fpeval_row["prediction_validity"]["valid"]),
            "conditional_watertight": float(
                bool(fpeval_row["prediction_validity"]["watertight"])
                if float(before_case["target_watertight"]) > 0.5
                else True
            ),
        }
    rows = []
    if near_wall and len(room_types) >= 2:
        rows.append({**common, "suite": "wall_adjacent"})
    if opening_contact:
        rows.append({**common, "suite": "opening_covering"})
    if near_exterior:
        rows.append({**common, "suite": "exterior_boundary"})
    return rows


def _load_seed_rows(
    args: argparse.Namespace,
    seed: int,
    taxonomy: FloorplanTaxonomy,
    tasks: list[dict],
) -> list[dict]:
    fpeval = {}
    per_case = {}
    for operation in OPERATIONS:
        report = json.loads(
            (
                args.evaluation_root
                / f"fp_control_gated_topology_fpeval_{operation}_seed{seed}_v1.json"
            ).read_text(encoding="utf-8")
        )
        fpeval[operation] = {row["case"]: row for row in report["rows"]}
        edit_dir = (
            f"fp_local_edit_{operation}_test503_v1"
            if seed == 19
            else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
        )
        per_case[operation] = json.loads(
            (args.evaluation_root / edit_dir / "per_case.json").read_text(
                encoding="utf-8"
            )
        )
    rows = []
    for task in tasks:
        operation = task["operation"]
        safe_name = task["record"]["name"].replace("/", "__")
        rows.extend(
            _evaluate_case(
                args,
                seed,
                taxonomy,
                task,
                fpeval[operation][safe_name],
                per_case[operation][task["manifest_index"]],
            )
        )
    return rows


def _metrics(rows: list[dict]) -> dict:
    keys = (
        "raw_outside_exact_ratio",
        "gated_outside_exact_ratio",
        "raw_outside_room_exact_ratio",
        "gated_outside_room_exact_ratio",
        "raw_outside_changes",
        "gated_outside_changes",
        "raw_outside_room_changes",
        "gated_outside_room_changes",
        "raw_inside_change_fraction",
        "gated_inside_change_fraction",
        "validity",
        "conditional_watertight",
    )
    return {
        key: float(np.mean([row[key] for row in rows])) for key in keys
    }


def _summary(rows: list[dict]) -> dict:
    result = {}
    for suite in SUITES:
        selected = [row for row in rows if row["suite"] == suite]
        result[suite] = {
            "cases": len(selected),
            "mean": _metrics(selected) if selected else {},
            "minimum": {
                key: float(np.min([row[key] for row in selected]))
                if selected
                else None
                for key in (
                    "gated_outside_exact_ratio",
                    "gated_outside_room_exact_ratio",
                    "validity",
                    "conditional_watertight",
                )
            },
        }
    return result


def _save_panels(output: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    seed19 = [row for row in rows if row["seed"] == 19]
    for suite in SUITES:
        selected = sorted(
            [row for row in seed19 if row["suite"] == suite],
            key=lambda row: (
                row["raw_outside_exact_ratio"],
                -row["raw_outside_changes"],
            ),
        )[:3]
        for index, row in enumerate(selected, start=1):
            with (
                np.load(row["before_path"]) as before,
                np.load(row["raw_path"]) as raw,
                np.load(row["gated_path"]) as gated,
            ):
                panels = (
                    ("Before", before["prediction"]),
                    ("Raw after", raw["prediction"]),
                    ("Gated after", gated["prediction"]),
                    ("Target after", gated["target"]),
                    (
                        "Outside change",
                        (gated["prediction"] != before["prediction"]).astype(int),
                    ),
                )
                figure, axes = plt.subplots(1, len(panels), figsize=(18, 3.8))
                for axis, (title, image) in zip(axes, panels, strict=True):
                    axis.imshow(image, cmap="tab20", interpolation="nearest")
                    axis.set_title(title)
                    axis.axis("off")
                figure.suptitle(
                    f"{suite}: {row['operation']} | {row['name']} | "
                    f"raw exact={row['raw_outside_exact_ratio']:.4f}"
                )
                figure.tight_layout()
                filename = (
                    f"{suite}_{index}_"
                    f"{row['name'].replace('/', '__')}_{row['operation']}.png"
                )
                figure.savefig(output / filename, dpi=140)
                plt.close(figure)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(per_seed: dict) -> str:
    lines = [
        "# Strict-Locality Stress Tests",
        "",
        "| Seed | Suite | N | Raw outside | Gated outside | Gated room | Validity | Watertight | Inside change |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, summary in per_seed.items():
        for suite, values in summary.items():
            mean = values["mean"]
            lines.append(
                f"| {seed} | {suite} | {values['cases']} | "
                f"{mean['raw_outside_exact_ratio']:.6f} | "
                f"{mean['gated_outside_exact_ratio']:.6f} | "
                f"{mean['gated_outside_room_exact_ratio']:.6f} | "
                f"{mean['validity']:.6f} | "
                f"{mean['conditional_watertight']:.6f} | "
                f"{mean['gated_inside_change_fraction']:.6f} |"
            )
    lines.extend(
        [
            "",
            (
                "Wall-adjacent means the edit neighborhood contains a wall and "
                "at least two target room types. Opening-covering means a "
                "one-pixel edit dilation intersects a door/window. "
                "Exterior-boundary means a five-pixel dilation crosses the "
                "footprint."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19, 20, 21, 22, 23]
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    tasks = _task_rows(args)
    rows = []
    per_seed = {}
    for seed in seeds:
        seed_rows = _load_seed_rows(args, seed, taxonomy, tasks)
        rows.extend(seed_rows)
        per_seed[str(seed)] = _summary(seed_rows)
        print(f"seed {seed}: {len(seed_rows)} suite rows", flush=True)
    aggregate = _summary(rows)
    checks = {
        "all_suites_at_least_50_per_seed": all(
            summary[suite]["cases"] >= 50
            for summary in per_seed.values()
            for suite in SUITES
        ),
        "gated_outside_exact_one": all(
            values["minimum"]["gated_outside_exact_ratio"] == 1.0
            for values in aggregate.values()
        ),
        "gated_outside_room_exact_one": all(
            values["minimum"]["gated_outside_room_exact_ratio"] == 1.0
            for values in aggregate.values()
        ),
        "validity_one": all(
            values["minimum"]["validity"] == 1.0
            for values in aggregate.values()
        ),
        "watertight_at_least_0_95": all(
            values["mean"]["conditional_watertight"] >= 0.95
            for values in aggregate.values()
        ),
        "inside_response_positive": all(
            values["mean"]["gated_inside_change_fraction"] > 0.0
            for values in aggregate.values()
        ),
    }
    report = {
        "status": "complete",
        "seeds": seeds,
        "cases": len(rows),
        "suite_definitions": {
            "wall_adjacent": (
                "two-pixel edit neighborhood intersects a wall and at least "
                "two target room types"
            ),
            "opening_covering": (
                "one-pixel edit dilation intersects a door or window"
            ),
            "exterior_boundary": (
                "five-pixel edit dilation crosses the footprint"
            ),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "checks": checks,
        "boundary": (
            "Oracle-control stress rows. A suite case may belong to more than "
            "one stress definition."
        ),
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "seeds": seeds,
        "rows": rows,
    })
    _write_json(args.output, report)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(per_seed), encoding="utf-8")
    _save_panels(args.panels, rows)
    print(json.dumps({"cases": len(rows), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
