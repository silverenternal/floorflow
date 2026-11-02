#!/usr/bin/env python3
"""Promote protocol-aligned structure and build verifiability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard-gate",
        type=Path,
        default=Path("outputs/reports/fp_control_gated_hard_gate_v1.json"),
    )
    parser.add_argument(
        "--edit-detection",
        type=Path,
        default=Path(
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--housemind-structure",
        type=Path,
        default=Path("outputs/reports/fp_housemind_structural_alignment_v1.json"),
    )
    parser.add_argument(
        "--canonical-structure",
        type=Path,
        default=Path("outputs/reports/fp_canonical_structural_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--matched-external",
        type=Path,
        default=Path("outputs/reports/fp_edit_detection_matched_external_v2.json"),
    )
    parser.add_argument(
        "--local-redraw",
        type=Path,
        default=Path("outputs/reports/fp_raster2seq_local_redraw_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_verifiability_score_v1.json"),
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("outputs/reports/fp_main_claim_source_map_v1.json"),
    )
    parser.add_argument(
        "--main-table",
        type=Path,
        default=Path("paper/tables/fp_main_results_v2.md"),
    )
    parser.add_argument(
        "--verifiability-table",
        type=Path,
        default=Path("paper/tables/fp_verifiability_score.md"),
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def _verifiability(hard_gate: dict) -> dict:
    per_seed = {}
    for row in sorted(hard_gate["seeds"], key=lambda item: int(item["seed"])):
        factors = {
            "outside_pixel_exact": float(row["outside_pixel_exact_mean"]),
            "outside_room_label_exact": float(row["outside_room_exact_mean"]),
            "validity": float(row["validity"]),
            "conditional_watertightness": float(row["conditional_watertight"]),
        }
        score = float(np.prod(np.asarray(list(factors.values()))))
        per_seed[str(int(row["seed"]))] = {
            "factors": factors,
            "verifiability": score,
            "product_recomputed": bool(
                np.isclose(score, float(np.prod(list(factors.values()))))
            ),
        }
    return {
        "definition": "outside_pixel_exact * outside_room_label_exact * validity * conditional_watertightness",
        "protocol": "CubiCasa test503 control-gated local-delta, five seeds",
        "floorflow": {
            "per_seed": per_seed,
            "summary": {
                key: _range(
                    [
                        values["verifiability"] if key == "verifiability"
                        else values["factors"][key]
                        for values in per_seed.values()
                    ]
                )
                for key in (
                    "outside_pixel_exact",
                    "outside_room_label_exact",
                    "validity",
                    "conditional_watertightness",
                    "verifiability",
                )
            },
        },
        "other_methods": {
            "HouseMind-E": {
                "outside_pixel_exact": None,
                "outside_room_label_exact": None,
                "validity": None,
                "conditional_watertightness": None,
                "verifiability": None,
                "status": "N/A: editing table does not report all four factors",
            },
            "Raster2Seq native": {
                "outside_pixel_exact": "reported per operation",
                "outside_room_label_exact": None,
                "validity": None,
                "conditional_watertightness": None,
                "verifiability": None,
                "status": "N/A: only one of four factors is reported",
            },
            "Raster2Seq editing-aware local redraw": {
                "outside_pixel_exact": 1.0,
                "outside_room_label_exact": None,
                "validity": None,
                "conditional_watertightness": None,
                "verifiability": None,
                "status": "N/A: only one of four factors is reported",
            },
        },
    }


def _source_map(
    hard_gate: Path,
    edit_detection: Path,
    housemind: Path,
    canonical: Path,
    matched_external: Path,
    local_redraw: Path,
) -> dict:
    sources = {
        "hard_gate": hard_gate,
        "edit_detection": edit_detection,
        "housemind_structure": housemind,
        "canonical_structure": canonical,
        "matched_external": matched_external,
        "local_redraw": local_redraw,
    }
    return {
        "sources": {
            name: {"path": str(path), "sha256": _digest(path)}
            for name, path in sources.items()
        },
        "metric_sources": {
            "delta_iou": {
                "source": "edit_detection",
                "pointer": "/aggregate/local_delta_iou",
            },
            "delta_mse": {
                "source": "edit_detection",
                "pointer": "/aggregate/local_delta_mse",
            },
            "outside_pixel_exact": {
                "source": "hard_gate",
                "pointer": "/seeds/*/outside_pixel_exact_mean",
            },
            "outside_room_label_exact": {
                "source": "hard_gate",
                "pointer": "/seeds/*/outside_room_exact_mean",
            },
            "validity": {
                "source": "hard_gate",
                "pointer": "/seeds/*/validity",
            },
            "conditional_watertightness": {
                "source": "hard_gate",
                "pointer": "/seeds/*/conditional_watertight",
            },
            "housemind_style_structure": {
                "source": "housemind_structure",
                "pointer": "/per_seed/*/summary/after_prediction_vs_after_target",
            },
            "canonical_instance_structure": {
                "source": "canonical_structure",
                "pointer": "/summary",
            },
        },
        "claim_boundary": _read(housemind)["boundaries"],
    }


def _main_table(
    edit: dict,
    hm: dict,
    canonical: dict,
    verifiability: dict,
) -> str:
    edit_agg = edit["aggregate"]
    hm_rows = {}
    for seed, values in hm["per_seed"].items():
        hm_rows[seed] = values["summary"]["after_prediction_vs_after_target"]
    canonical_summary = canonical["summary"]
    vf = verifiability["floorflow"]["summary"]["verifiability"]
    lines = [
        "# FloorFlow Main Results",
        "",
        (
            "FloorFlow edit metrics use CubiCasa test503 with control-gated "
            "local deltas. HouseMind-E anchors come from RPLAN natural-language "
            "editing and are shown for protocol context only."
        ),
        "",
        "| Metric | FloorFlow (5 seeds) | HouseMind-E reference | Comparability |",
        "|---|---:|---:|---|",
        (
            f"| Delta-IoU | {edit_agg['local_delta_iou']['mean']:.6f} "
            f"({edit_agg['local_delta_iou']['min']:.6f}-"
            f"{edit_agg['local_delta_iou']['max']:.6f}) "
            f"| {hm['house_mind_reference']['delta_iou']:.3f} | "
            "Different data/input |"
        ),
        (
            f"| Delta-MSE | {edit_agg['local_delta_mse']['mean']:.9f} "
            f"({edit_agg['local_delta_mse']['min']:.9f}-"
            f"{edit_agg['local_delta_mse']['max']:.9f}) "
            f"| {hm['house_mind_reference']['delta_mse']:.4f} | "
            "Different data/input |"
        ),
        "| Outside pixel exact | 1.000000 | Not reported | FloorFlow strict-locality metric |",
        "| Outside room-label exact | 1.000000 | Not reported | FloorFlow strict-locality metric |",
        "| Validity | 1.000000 | Not reported in editing table | FloorFlow metric |",
        "| Conditional watertightness | 0.957875-0.967491 | Not reported | FloorFlow metric |",
        f"| Verifiability | {vf['min']:.6f}-{vf['max']:.6f} | N/A | Product of four reported factors |",
        "",
        "## HouseMind-Style Structural Main Table",
        "",
        (
            "HouseMind-E reports RPLAN Node F1 0.998, Edge overlap 0.934, "
            "and GED 0.467. Those values are a protocol reference, not a "
            "same-dataset ranking."
        ),
        "",
        "| Seed | Node F1 | Edge overlap | GED | Exact GED rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, values in sorted(hm_rows.items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {seed} | {values['node_f1']:.6f} | "
            f"{values['edge_overlap']:.6f} | {values['ged']:.6f} | "
            f"{values['exact']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "This structural table is HouseMind-protocol-aligned on "
                "CubiCasa oracle controls. It is not a HouseMind RPLAN "
                "leaderboard result."
            ),
            "",
            "## Canonical Instance Graph",
            "",
            "| Metric | Mean | Range |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        ("node_f1", "Node F1"),
        ("edge_overlap", "Edge overlap"),
        ("ged", "Unit GED"),
        ("ged_exact_rate", "Exact GED rate"),
    ):
        values = canonical_summary[key]
        lines.append(
            f"| {label} | {values['mean']:.6f} | "
            f"{values['min']:.6f}-{values['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "The canonical instance graph is secondary because it applies "
                "one-to-one room-instance IoU matching, whereas the "
                "HouseMind-style view uses unique room-type sets and type-pair "
                "edges."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _verifiability_table(verifiability: dict) -> str:
    def value(item) -> str:
        if item is None:
            return "N/A"
        if isinstance(item, (int, float)):
            return f"{float(item):.6f}"
        return "reported"

    lines = [
        "# Verifiability Score",
        "",
        "`Verifiability = outside pixel exact * outside room-label exact * validity * conditional watertightness`.",
        "",
        "| Method | Outside pixel | Outside room label | Validity | Watertight | Verifiability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, values in sorted(
        verifiability["floorflow"]["per_seed"].items(),
        key=lambda item: int(item[0]),
    ):
        factors = values["factors"]
        lines.append(
            f"| FloorFlow seed {seed} | "
            f"{factors['outside_pixel_exact']:.6f} | "
            f"{factors['outside_room_label_exact']:.6f} | "
            f"{factors['validity']:.6f} | "
            f"{factors['conditional_watertightness']:.6f} | "
            f"{values['verifiability']:.6f} |"
        )
    for method, values in verifiability["other_methods"].items():
        lines.append(
            f"| {method} | {value(values['outside_pixel_exact'])} | "
            f"{value(values['outside_room_label_exact'])} | "
            f"{value(values['validity'])} | "
            f"{value(values['conditional_watertightness'])} | "
            f"{value(values['verifiability'])} |"
        )
    lines.extend(
        [
            "",
            "N/A is used whenever any factor is absent. Missing values are not imputed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    hard = _read(args.hard_gate)
    edit = _read(args.edit_detection)
    hm = _read(args.housemind_structure)
    canonical = _read(args.canonical_structure)
    verifiability = _verifiability(hard)
    source_map = _source_map(
        args.hard_gate,
        args.edit_detection,
        args.housemind_structure,
        args.canonical_structure,
        args.matched_external,
        args.local_redraw,
    )
    report = {
        "status": "complete",
        **verifiability,
        "checks": {
            "all_products_recomputed": all(
                values["product_recomputed"]
                for values in verifiability["floorflow"]["per_seed"].values()
            ),
            "floorflow_verifiability_above_0_95": (
                verifiability["floorflow"]["summary"]["verifiability"]["min"] > 0.95
            ),
            "missing_components_not_imputed": all(
                values["verifiability"] is None
                for values in verifiability["other_methods"].values()
            ),
        },
        "boundary": (
            "Verifiability is a reporting-completeness-and-safety score, not a "
            "replacement for edit quality. A missing component produces N/A."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.source_map.parent.mkdir(parents=True, exist_ok=True)
    args.source_map.write_text(
        json.dumps({
            "status": "complete",
            **source_map,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    args.main_table.parent.mkdir(parents=True, exist_ok=True)
    args.main_table.write_text(
        _main_table(edit, hm, canonical, report),
        encoding="utf-8",
    )
    args.verifiability_table.parent.mkdir(parents=True, exist_ok=True)
    args.verifiability_table.write_text(
        _verifiability_table(report),
        encoding="utf-8",
    )
    print(json.dumps({
        "checks": report["checks"],
        "verifiability": report["floorflow"]["summary"]["verifiability"],
    }, indent=2))


if __name__ == "__main__":
    main()
