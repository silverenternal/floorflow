"""Compare corrected FloorFlow and Raster2Seq on identical validation cases."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.external_claim_matrix import build_claim_matrix
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite

ROOT = Path(__file__).resolve().parents[1]
FLOORFLOW = ROOT / "outputs/eval/fp_completion_predicted_refined_val491_v1/rasters"
RASTER2SEQ = (
    ROOT
    / "outputs/external_baselines/raster2seq/predicted_refined_val491_v1/predictions"
)
REPORT = ROOT / "outputs/reports/residential_external_comparison_v1.json"
TABLE = ROOT / "paper/tables/residential_external_claims_v1.md"
METRICS = (
    "room_macro_iou",
    "room_support_iou",
    "node_f1",
    "edge_overlap",
    "ged",
    "wall_f1",
    "opening_f1",
    "furniture_f1",
    "pixel_accuracy",
)
_TAXONOMY = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--floorflow-root", type=Path, default=FLOORFLOW)
    parser.add_argument("--raster2seq-root", type=Path, default=RASTER2SEQ)
    parser.add_argument("--split", default="val491")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT
        / "outputs/reports/residential_external_comparison_rows_v1.json",
    )
    return parser.parse_args()


def _initialize() -> None:
    global _TAXONOMY
    _TAXONOMY = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def _binary_f1(prediction, target) -> float:
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return (
        1.0
        if denominator == 0
        else float(2.0 * intersection / denominator)
    )


def _metrics(prediction, target, footprint) -> dict:
    suite = room_metric_suite(
        prediction,
        target,
        footprint,
        _TAXONOMY,
        predicted_room_support=np.isin(prediction, _TAXONOMY.room_ids),
        target_room_support=np.isin(target, _TAXONOMY.room_ids),
    )
    structural = housemind_style_metrics(
        prediction,
        target,
        footprint,
        _TAXONOMY,
        ged_timeout_seconds=2.0,
    )
    return {
        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
        "room_support_iou": suite["occlusion_aware_support"]["room_support_iou"],
        "node_f1": structural["node_f1"],
        "edge_overlap": structural["edge_overlap"],
        "ged": structural["ged"],
        "wall_f1": _binary_f1(
            prediction == _TAXONOMY.class_id("wall"),
            target == _TAXONOMY.class_id("wall"),
        ),
        "opening_f1": _binary_f1(
            np.isin(prediction, _TAXONOMY.opening_ids),
            np.isin(target, _TAXONOMY.opening_ids),
        ),
        "furniture_f1": _binary_f1(
            np.isin(prediction, _TAXONOMY.furniture_ids),
            np.isin(target, _TAXONOMY.furniture_ids),
        ),
        "pixel_accuracy": float(
            (prediction[footprint] == target[footprint]).mean()
        ),
    }


def _diagnose(task: dict) -> dict:
    with np.load(task["floorflow"]) as payload:
        target = payload["target"].astype(np.uint8)
        footprint = payload["footprint"].astype(bool)
        floorflow = payload["prediction"].astype(np.uint8)
    with np.load(task["raster2seq"]) as payload:
        raster2seq = payload["prediction"].astype(np.uint8)
        if not np.array_equal(target, payload["target"]):
            raise ValueError("target mismatch")
        if not np.array_equal(footprint, payload["footprint"].astype(bool)):
            raise ValueError("footprint mismatch")
    return {
        "name": task["name"],
        "floorflow": _metrics(floorflow, target, footprint),
        "raster2seq": _metrics(raster2seq, target, footprint),
    }


def _summary(rows: list[dict], method: str) -> dict:
    return {
        metric: float(np.mean([row[method][metric] for row in rows]))
        for metric in METRICS
    }


def _paired(rows: list[dict], metric: str, bootstrap: int) -> dict:
    return paired_delta(
        [{"value": row["floorflow"][metric]} for row in rows],
        [{"value": row["raster2seq"][metric]} for row in rows],
        "value",
        iterations=bootstrap,
        seed=26001,
    )


def _table(report: dict, path: Path) -> None:
    lines = [
        "# Matched Residential Reconstruction Evidence",
        "",
        "| Metric | FloorFlow predicted completion | Raster2Seq matched | Delta | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        values = report["methods"]
        effect = report["paired"][metric]
        lines.append(
            f"| {metric} | {values['FloorFlow'][metric]:.6f} | "
            f"{values['Raster2Seq'][metric]:.6f} | "
            f"{effect['mean_delta']:+.6f} | "
            f"[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "| Claim branch | Enabled |",
            "|---|---:|",
        ]
    )
    for name, branch in report["claim_matrix"].items():
        lines.append(f"| {name} | {branch['enabled']} |")
    lines.extend(
        [
            "",
            (
                "Raster2Seq is reconstruction-only. HouseMind-E, CE2EPlan, "
                "RPLAN, TLC-Plan, and FMLM remain context-only or unavailable."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _initialize()
    floorflow_cases = {
        path.name: path
        for path in sorted(args.floorflow_root.glob("*.npz"))
    }
    raster2seq_cases = {
        path.name: path
        for path in sorted(args.raster2seq_root.glob("*.npz"))
    }
    if set(floorflow_cases) != set(raster2seq_cases):
        raise ValueError("matched reconstruction case sets differ")
    tasks = [
        {
            "name": name,
            "floorflow": floorflow_cases[name],
            "raster2seq": raster2seq_cases[name],
        }
        for name in sorted(floorflow_cases)
    ]
    rows = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize,
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"compared {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["name"])
    methods = {
        "FloorFlow": _summary(rows, "floorflow"),
        "Raster2Seq": _summary(rows, "raster2seq"),
    }
    paired = {
        metric: _paired(rows, metric, args.bootstrap)
        for metric in METRICS
    }
    paired_checks = {
        "room_macro_iou_positive": paired["room_macro_iou"]["ci_low"] > 0.0,
        "node_f1_positive": paired["node_f1"]["ci_low"] > 0.0,
        "edge_overlap_positive": paired["edge_overlap"]["ci_low"] > 0.0,
        "ged_negative": paired["ged"]["ci_high"] < 0.0,
    }
    claim_matrix = build_claim_matrix(
        paired_checks=paired_checks,
        matched_executable=True,
    )
    report = {
        "schema_version": "residential-external-comparison-v1",
        "status": (
            "passed"
            if claim_matrix["matched_cubicasa_reconstruction_head_to_head"]["enabled"]
            else "failed"
        ),
        "scope": "matched CubiCasa validation reconstruction",
        "split": args.split,
        "test_opened": False,
        "cases": len(rows),
        "same_controls": True,
        "same_target": True,
        "same_footprint": True,
        "same_resolution": True,
        "methods": methods,
        "paired": paired,
        "paired_checks": paired_checks,
        "claim_matrix": claim_matrix,
        "external_context": {
            "HouseMind": "weights-only/inference unavailable; no matched row",
            "CE2EPlan": "RPLAN generation-only; no shared split",
            "TLC-Plan": "public code unavailable",
            "FMLM": "public code/weights unavailable",
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.write_text(
        json.dumps({"status": "complete", "rows": rows}) + "\n",
        encoding="utf-8",
    )
    _table(report, args.table)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "paired_checks": paired_checks,
                "status": report["status"],
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
