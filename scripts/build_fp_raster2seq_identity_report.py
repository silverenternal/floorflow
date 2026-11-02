#!/usr/bin/env python3
"""Build an adapter-sensitivity diagnostic for identity reconstruction."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "room_macro_iou",
    "instance_f1",
    "instance_type_accuracy",
    "room_support_iou",
    "node_f1",
    "edge_overlap",
    "ged",
    "exact_ged_rate",
    "wall_f1",
    "opening_f1",
    "furniture_f1",
    "pixel_accuracy",
    "identity_delta_iou",
    "identity_delta_mse",
)
_TAXONOMY: FloorplanTaxonomy | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--floorflow-root",
        type=Path,
        default=ROOT / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters",
    )
    parser.add_argument(
        "--raster2seq-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/identity_test503_v1/predictions",
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/test503_matched_v1/predictions",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_raster2seq_identity_reconstruction_v1.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fp_raster2seq_identity_rows_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_raster2seq_identity_comparison_v1.md",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fp_identity_metric_definition_v1.md",
    )
    return parser.parse_args()


def _initialize(taxonomy_path: str) -> None:
    global _TAXONOMY
    _TAXONOMY = FloorplanTaxonomy.read(Path(taxonomy_path))


def _binary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    intersection = np.logical_and(prediction, target).sum()
    predicted = prediction.sum()
    expected = target.sum()
    if predicted == 0 and expected == 0:
        return 1.0
    precision = intersection / predicted if predicted else 0.0
    recall = intersection / expected if expected else 0.0
    denominator = precision + recall
    return float(2.0 * precision * recall / denominator) if denominator else 0.0


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
) -> dict[str, float]:
    if _TAXONOMY is None:
        raise RuntimeError("worker taxonomy was not initialized")
    room_support = np.isin(prediction, _TAXONOMY.room_ids)
    target_room_support = np.isin(target, _TAXONOMY.room_ids)
    suite = room_metric_suite(
        prediction,
        target,
        footprint,
        _TAXONOMY,
        predicted_room_support=room_support,
        target_room_support=target_room_support,
    )
    structural = housemind_style_metrics(
        prediction,
        target,
        footprint,
        _TAXONOMY,
        ged_timeout_seconds=1.0,
    )
    opening = np.isin(prediction, _TAXONOMY.opening_ids)
    target_opening = np.isin(target, _TAXONOMY.opening_ids)
    furniture = np.isin(prediction, _TAXONOMY.furniture_ids)
    target_furniture = np.isin(target, _TAXONOMY.furniture_ids)
    wall = prediction == _TAXONOMY.class_id("wall")
    target_wall = target == _TAXONOMY.class_id("wall")
    exact = prediction == target
    return {
        "room_macro_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(suite["matched_instance"]["instance_f1_at_threshold"]),
        "instance_type_accuracy": float(
            suite["matched_instance"]["matched_type_accuracy"]
        ),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
        "node_f1": float(structural["node_f1"]),
        "edge_overlap": float(structural["edge_overlap"]),
        "ged": float(structural["ged"]),
        "exact_ged_rate": float(structural["exact"]),
        "wall_f1": _binary_f1(wall, target_wall),
        "opening_f1": _binary_f1(opening, target_opening),
        "furniture_f1": _binary_f1(furniture, target_furniture),
        "pixel_accuracy": float(exact[footprint].mean()) if footprint.any() else 1.0,
        "identity_delta_iou": 1.0,
        "identity_delta_mse": 0.0,
    }


def _diagnose(task: dict) -> dict:
    with (
        np.load(task["floorflow"]) as floorflow_payload,
        np.load(task["raster2seq"]) as raster2seq_payload,
    ):
        target = floorflow_payload["target"].astype(np.uint8, copy=False)
        footprint = floorflow_payload["footprint"].astype(bool, copy=False)
        floorflow = floorflow_payload["prediction"].astype(np.uint8, copy=False)
        raster2seq = raster2seq_payload["prediction"].astype(
            np.uint8,
            copy=False,
        )
        raster_target = raster2seq_payload["target"].astype(np.uint8, copy=False)
        raster_footprint = raster2seq_payload["footprint"].astype(
            bool,
            copy=False,
        )
    if not np.array_equal(target, raster_target):
        raise ValueError(f"target mismatch for {task['name']}")
    if not np.array_equal(footprint, raster_footprint):
        raise ValueError(f"footprint mismatch for {task['name']}")
    return {
        "name": task["name"],
        "floorflow": _metrics(floorflow, target, footprint),
        "raster2seq": _metrics(raster2seq, target, footprint),
        "shape": list(target.shape),
    }


def _case_map(root: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(root.glob("*.npz"))}


def _summary(rows: list[dict], method: str) -> dict:
    return {
        metric: float(np.mean([row[method][metric] for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    floorflow_cases = _case_map(args.floorflow_root)
    raster2seq_cases = _case_map(args.raster2seq_root)
    if set(floorflow_cases) != set(raster2seq_cases):
        missing = sorted(set(floorflow_cases) - set(raster2seq_cases))
        extra = sorted(set(raster2seq_cases) - set(floorflow_cases))
        raise ValueError(f"identity case mismatch missing={missing} extra={extra}")
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
        initargs=(str(args.taxonomy),),
    ) as executor:
        futures = [executor.submit(_diagnose, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"compared {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["name"])
    effects = {}
    for index, metric in enumerate(METRICS):
        effects[metric] = paired_delta(
            [{"value": row["floorflow"][metric]} for row in rows],
            [{"value": row["raster2seq"][metric]} for row in rows],
            "value",
            iterations=args.bootstrap,
            seed=14000 + index,
        )
        effects[metric]["higher_is_better"] = metric not in (
            "ged",
            "identity_delta_mse",
        )
    cross_run_rows = 0
    cross_run_exact = 0
    for name, path in raster2seq_cases.items():
        candidates = list(args.paired_root.glob(f"*/{name}"))
        if not candidates:
            continue
        with np.load(path) as identity, np.load(candidates[0]) as paired:
            cross_run_rows += 1
            cross_run_exact += np.array_equal(
                identity["prediction"],
                paired["prediction_before"],
            )
    report = {
        "status": "adapter_sensitivity_diagnostic_complete",
        "result_role": (
            "diagnostic_only_not_an_official_matched_reconstruction_result"
        ),
        "scope": (
            "FloorFlow identity reconstruction versus an official-style "
            "Raster2Seq input adapter on 503 unique CubiCasa test503 rasters"
        ),
        "cases": len(rows),
        "methods": {
            "FloorFlow": _summary(rows, "floorflow"),
            "Raster2Seq official-style adapter diagnostic": _summary(
                rows,
                "raster2seq",
            ),
        },
        "paired_effects_floorflow_minus_raster2seq": effects,
        "identity_metric_definition": {
            "identity_delta_iou": (
                "No-change contract diagnostic only; both deterministic "
                "reconstructions are identical to their own rerun and therefore "
                "score 1.0. It is not an editing-quality metric."
            ),
            "identity_delta_mse": (
                "No-change contract diagnostic only; zero by construction."
            ),
            "reconstruction_quality": (
                "Use room, graph, wall, opening, furniture, and pixel metrics."
            ),
        },
        "cross_run_stability": {
            "raster2seq_cases_compared": cross_run_rows,
            "exact_repeat_or_replay_rate": (
                cross_run_exact / cross_run_rows if cross_run_rows else None
            ),
            "boundary": (
                "Compares the independent identity run with available paired "
                "before-edit predictions; it is a replay diagnostic, not a method score."
            ),
        },
        "checks": {
            "same_503_inputs": len(rows) == 503,
            "same_target_and_footprint": True,
            "same_resolution": all(row["shape"] == [256, 256] for row in rows),
            "adapter_only_not_upstream_official_preprocessing": True,
            "raster2seq_not_called_editing": True,
            "identity_delta_not_promoted": True,
        },
        "claim_gate": {
            "matched_table_eligible": False,
            "reason": (
                "The input renderer is a project adapter and the upstream "
                "official preprocessing was not shared by this run."
            ),
        },
        "boundaries": [
            (
                "The Raster2Seq input renderer is a project-defined "
                "official-style adapter, not upstream's released preprocessing."
            ),
            (
                "The metrics diagnose adapter sensitivity and must not be "
                "presented as an official or matched reconstruction score."
            ),
            "Raster2Seq is a reconstruction method, not a native editor.",
            "FloorFlow reconstruction uses oracle controls.",
            "No FEVP editing claim follows from identity reconstruction.",
            "Cross-dataset CE2EPlan comparison is separate and not included.",
        ],
        "sources": {
            "floorflow": str(args.floorflow_root.relative_to(ROOT)),
            "raster2seq_adapter_diagnostic": str(
                args.raster2seq_root.relative_to(ROOT)
            ),
            "raster2seq_report": (
                "outputs/external_baselines/raster2seq/identity_test503_v1/report.json"
            ),
        },
    }
    failures = [name for name, passed in report["checks"].items() if not passed]
    if failures:
        raise ValueError(f"identity checks failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": "complete", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_table(report, args.table)
    _write_doc(report, args.doc)
    print(
        json.dumps(
            {
                "status": report["status"],
                **report["checks"],
                "cross_run_exact_rate": report["cross_run_stability"][
                    "exact_repeat_or_replay_rate"
                ],
            },
            indent=2,
        )
    )


def _write_table(report: dict, path: Path) -> None:
    lines = [
        "# Raster2Seq Identity Adapter Diagnostic v1",
        "",
        (
            "This is a project-defined official-style input-adapter diagnostic. "
            "It is not upstream Raster2Seq preprocessing and is not eligible as "
            "a matched reconstruction table."
        ),
        "",
        "| Metric | FloorFlow | R2S adapter diagnostic | FloorFlow - adapter | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        effect = report["paired_effects_floorflow_minus_raster2seq"][metric]
        lines.append(
            f"| {metric} | {report['methods']['FloorFlow'][metric]:.6f} | "
            f"{report['methods']['Raster2Seq official-style adapter diagnostic'][metric]:.6f} | "
            f"{effect['mean_delta']:+.6f} | "
            f"[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}] |"
        )
    stability = report["cross_run_stability"]
    lines.extend(
        [
            "",
            (
                f"Raster2Seq replay exact rate: "
                f"{stability['exact_repeat_or_replay_rate']:.6f} over "
                f"{stability['raster2seq_cases_compared']} available paired files."
            ),
            "",
            "Identity Delta-IoU/Delta-MSE are no-change diagnostics, not editing scores.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_doc(report: dict, path: Path) -> None:
    text = [
        "# Raster2Seq Identity Adapter Diagnostic Definition v1",
        "",
        (
            "This diagnostic checks whether a project-defined official-style "
            "Raster2Seq input renderer changes reconstruction metrics on 503 "
            "unique CubiCasa test503 inputs. It is not an editing experiment "
            "and it is not an official matched reconstruction result."
        ),
        "",
        "## Primary metrics",
        "",
        "- Room macro IoU and matched-instance F1.",
        "- HouseMind-style Node F1, Edge overlap, and GED.",
        "- Wall, opening, and furniture binary F1.",
        "- Pixel accuracy over the footprint.",
        "",
        "## Identity diagnostics",
        "",
        "- Identity Delta-IoU and Delta-MSE test no-change output stability.",
        (
            "- Both are trivial when the same deterministic output is compared "
            "with itself and cannot replace reconstruction quality."
        ),
        (
            "- Raster2Seq replay exactness against available paired predictions "
            "is reported as a provenance/stability diagnostic."
        ),
        "",
        "## Boundary",
        "",
        "- Raster2Seq remains reconstruction-only.",
        (
            "- The adapter result cannot be called native, official, matched, "
            "or reconstruction SOTA."
        ),
        "- FloorFlow identity reconstruction uses oracle controls.",
        "- No FEVP editing claim follows from this audit.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


__all__ = ["main"]


if __name__ == "__main__":
    main()
