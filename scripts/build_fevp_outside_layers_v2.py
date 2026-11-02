#!/usr/bin/env python3
"""Build three-layer outside exactness and continuous locality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    load_metric_contract,
    outside_layer_metrics,
    tolerances_from_contract,
)
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
_TAXONOMY: FloorplanTaxonomy | None = None
_TOLERANCES: FEVPTolerances | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_v2.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_rows_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_outside_layers_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_outside_leakage_v2.pdf",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str, contract_path: str) -> None:
    global _TAXONOMY, _TOLERANCES
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))
    _TOLERANCES = tolerances_from_contract(load_metric_contract(contract_path))


def _load(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / "rasters" / f"{name.replace('/', '__')}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _load_gated(root: Path, operation: str, name: str) -> dict[str, np.ndarray]:
    with np.load(root / operation / f"{name.replace('/', '__')}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _load_cache(operation: str, name: str) -> dict[str, np.ndarray]:
    path = (
        ROOT
        / "outputs/cache/fp_edit_detection_v1"
        / operation
        / f"{name.replace('/', '__')}.npz"
    )
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _diagnose(task: dict) -> dict:
    if _TAXONOMY is None or _TOLERANCES is None:
        raise RuntimeError("worker globals were not initialized")
    seed = int(task["seed"])
    operation = task["operation"]
    record = task["record"]
    name = str(record["name"])
    safe_name = name.replace("/", "__")
    before_root = (
        ROOT / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1"
    )
    raw_directory = (
        f"fp_local_edit_{operation}_test503_v1"
        if seed == 19
        else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
    )
    raw_root = ROOT / "outputs/eval" / raw_directory
    gated_root = (
        ROOT / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
    )
    before = _load(before_root, name)
    raw = _load(raw_root, name)
    gated = _load_gated(gated_root, operation, name)
    cache = _load_cache(operation, name)
    region = normalized_region_mask(
        before["prediction"].shape,
        record["edit_region_normalized"],
    ) & before["footprint"].astype(bool)
    wall = (
        before["target"] == _TAXONOMY.class_id("wall")
    ) | (
        gated["target"] == _TAXONOMY.class_id("wall")
    )
    valid_outside = (
        before["footprint"].astype(bool)
        & ~region
        & ~binary_dilation(wall, iterations=1)
    )
    changed = cache["changed"].astype(bool)
    strict = outside_layer_metrics(
        before["room_labels"],
        gated["room_labels"],
        changed,
        region,
        before["footprint"].astype(bool),
        _TAXONOMY,
        before_pixels=before["prediction"],
        after_pixels=gated["prediction"],
        valid_outside=valid_outside,
        tolerances=_TOLERANCES,
    )
    no_gate = outside_layer_metrics(
        before["room_labels"],
        raw["room_labels"],
        changed,
        region,
        before["footprint"].astype(bool),
        _TAXONOMY,
        before_pixels=before["prediction"],
        after_pixels=raw["prediction"],
        valid_outside=valid_outside,
        tolerances=_TOLERANCES,
    )
    return {
        "seed": seed,
        "operation": operation,
        "case_id": f"seed{seed}/{operation}/{name}",
        "name": name,
        "source_hash": _digest(
            before_root / "rasters" / f"{safe_name}.npz"
        ),
        "strict": strict,
        "no_edit_gating": no_gate,
    }


def _tasks(seeds: list[int]) -> list[dict]:
    tasks = []
    for seed in seeds:
        for operation in OPERATIONS:
            manifest = json.loads(
                (
                    ROOT
                    / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
                    / "test/manifest.json"
                ).read_text(encoding="utf-8")
            )
            for record in manifest["rows"]:
                if record["edited"]:
                    tasks.append(
                        {
                            "seed": seed,
                            "operation": operation,
                            "record": record,
                        }
                    )
    return tasks


def _summary(rows: list[dict], arm: str) -> dict:
    keys = (
        "outside_pixel_exact",
        "outside_room_level_exact",
        "outside_room_label_exact",
        "outside_room_polygon_exact",
        "outside_edge_exact",
        "locality_violation_rate",
        "outside_changed_pixels",
        "affected_room_instances",
        "changed_room_instances",
        "region_overlap_room_instances",
        "unaffected_room_instances",
    )
    return {
        key: float(np.mean([row[arm][key] for row in rows]))
        for key in keys
    }


def _group(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: {
            "cases": len(selected),
            "strict": _summary(selected, "strict"),
            "no_edit_gating": _summary(selected, "no_edit_gating"),
        }
        for value, selected in sorted(grouped.items())
    }


def _render_leakage(rows: list[dict], taxonomy: FloorplanTaxonomy, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    selected = sorted(
        [
            row
            for row in rows
            if row["no_edit_gating"]["outside_changed_pixels"] > 0
            and row["strict"]["outside_changed_pixels"] == 0
        ],
        key=lambda row: -row["no_edit_gating"]["outside_changed_pixels"],
    )[:4]
    colors = np.asarray(
        [taxonomy.color(index)[:3] for index in range(taxonomy.num_classes)],
        dtype=np.uint8,
    )
    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(9.0, 2.3 * max(1, len(selected))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, row in enumerate(selected):
        seed = row["seed"]
        operation = row["operation"]
        name = row["name"]
        before = _load(
            ROOT / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1",
            name,
        )
        raw_directory = (
            f"fp_local_edit_{operation}_test503_v1"
            if seed == 19
            else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
        )
        raw = _load(ROOT / "outputs/eval" / raw_directory, name)
        gated = _load_gated(
            ROOT / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1",
            operation,
            name,
        )
        panels = (
            (before["prediction"], "Before"),
            (gated["prediction"], "Strict"),
            (raw["prediction"], "No gate"),
        )
        for column, (labels, title) in enumerate(panels):
            axes[row_index, column].imshow(colors[np.asarray(labels)])
            axes[row_index, column].set_title(
                f"{title} | {operation}",
                fontsize=8,
                loc="left",
            )
            axes[row_index, column].axis("off")
        changed = raw["prediction"] != before["prediction"]
        axes[row_index, 3].imshow(np.zeros((*changed.shape, 3), dtype=np.uint8))
        overlay = np.zeros((*changed.shape, 4), dtype=np.float32)
        overlay[changed] = [0.9, 0.1, 0.1, 0.85]
        axes[row_index, 3].imshow(overlay)
        axes[row_index, 3].set_title("Leakage", fontsize=8, loc="left")
        axes[row_index, 3].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)
    _ = mpl


def main() -> None:
    args = parse_args()
    tasks = _tasks(args.seeds)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
        initargs=(str(args.taxonomy), str(args.contract)),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 500 == 0 or completed == len(futures):
                print(f"outside audit {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["operation"], row["name"]))
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    single_room = [
        row
        for row in rows
        if row["strict"]["changed_room_instances"] == 1
        and row["strict"]["unaffected_room_instances"] >= 1
    ]
    report = {
        "status": "complete",
        "contract_sha256": _digest(args.contract),
        "seeds": args.seeds,
        "cases": len(rows),
        "summary": {
            "all": {
                "strict": _summary(rows, "strict"),
                "no_edit_gating": _summary(rows, "no_edit_gating"),
            },
            "single_room_negative_control": {
                "cases": len(single_room),
                "strict": _summary(single_room, "strict"),
                "no_edit_gating": _summary(single_room, "no_edit_gating"),
            },
            "by_seed": _group(rows, "seed"),
            "by_operation": _group(rows, "operation"),
        },
        "checks": {
            "strict_pixel_exact_one": all(
                row["strict"]["outside_pixel_exact"] == 1.0 for row in rows
            ),
            "strict_room_level_exact_one": all(
                row["strict"]["outside_room_level_exact"] == 1.0 for row in rows
            ),
            "strict_edge_exact_one": all(
                row["strict"]["outside_edge_exact"] == 1.0 for row in rows
            ),
            "strict_violation_rate_zero": all(
                row["strict"]["locality_violation_rate"] == 0.0 for row in rows
            ),
            "no_gate_exposes_leakage": any(
                row["no_edit_gating"]["outside_changed_pixels"] > 0
                for row in rows
            ),
            "single_room_control_nonempty": len(single_room) >= 100,
        },
        "layers": {
            "pixel": "Zero-tolerance uint8 equality on the valid outside domain.",
            "room": "One-to-one instance match, mask, label, 0.5-pixel polygon, and edge preservation.",
            "edge": "Untouched-room adjacency set preserved after one-to-one mapping.",
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": report["status"], "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    strict = report["summary"]["all"]["strict"]
    leaking = report["summary"]["all"]["no_edit_gating"]
    table = [
        "# FEVP Three-Layer Outside Evidence",
        "",
        "| Arm | Pixel exact | Room exact | Label exact | Polygon exact | Edge exact | Violation rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Strict | {strict['outside_pixel_exact']:.6f} | "
            f"{strict['outside_room_level_exact']:.6f} | "
            f"{strict['outside_room_label_exact']:.6f} | "
            f"{strict['outside_room_polygon_exact']:.6f} | "
            f"{strict['outside_edge_exact']:.6f} | "
            f"{strict['locality_violation_rate']:.9f} |"
        ),
        (
            f"| No edit gating | {leaking['outside_pixel_exact']:.6f} | "
            f"{leaking['outside_room_level_exact']:.6f} | "
            f"{leaking['outside_room_label_exact']:.6f} | "
            f"{leaking['outside_room_polygon_exact']:.6f} | "
            f"{leaking['outside_edge_exact']:.6f} | "
            f"{leaking['locality_violation_rate']:.9f} |"
        ),
        "",
        (
            f"Single-room negative controls: "
            f"{len(single_room)} cases. Strict room-level exactness is "
            f"{report['summary']['single_room_negative_control']['strict']['outside_room_level_exact']:.6f}."
        ),
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    _render_leakage(rows, taxonomy, args.figure)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(rows),
                "single_room_controls": len(single_room),
                "checks": report["checks"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
