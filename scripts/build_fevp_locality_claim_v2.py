#!/usr/bin/env python3
"""Build the locality-as-control claim and leakage heatmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_v2.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_outside_layers_rows_v2.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_locality_claim_v2.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_locality_claim_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_locality_leakage_heatmap_v2.pdf",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _paths(seed: int, operation: str, name: str) -> tuple[Path, Path, Path]:
    safe = name.replace("/", "__")
    before = (
        ROOT
        / f"outputs/eval/fp_hard_metrics_full_test503_seed{seed}_v1/rasters"
        / f"{safe}.npz"
    )
    raw_directory = (
        f"fp_local_edit_{operation}_test503_v1"
        if seed == 19
        else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
    )
    raw = ROOT / "outputs/eval" / raw_directory / "rasters" / f"{safe}.npz"
    gated = (
        ROOT
        / f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1"
        / operation
        / f"{safe}.npz"
    )
    return before, raw, gated


def _overlay(mask: np.ndarray, rgb: tuple[float, float, float]) -> np.ndarray:
    result = np.zeros((*mask.shape, 4), dtype=np.float32)
    result[mask] = [*rgb, 0.82]
    return result


def _render(rows: list[dict], taxonomy: FloorplanTaxonomy, path: Path) -> None:
    selected = sorted(
        [
            row
            for row in rows
            if row["strict"]["outside_changed_pixels"] == 0
            and row["no_edit_gating"]["outside_changed_pixels"] > 0
        ],
        key=lambda row: -row["no_edit_gating"]["outside_changed_pixels"],
    )[:4]
    colors = np.asarray(
        [taxonomy.color(index)[:3] for index in range(taxonomy.num_classes)],
        dtype=np.uint8,
    )
    figure, axes = plt.subplots(
        len(selected),
        5,
        figsize=(12.0, 2.1 * max(1, len(selected))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, row in enumerate(selected):
        before_path, raw_path, gated_path = _paths(
            row["seed"],
            row["operation"],
            row["name"],
        )
        before = _load(before_path)
        raw = _load(raw_path)
        gated = _load(gated_path)
        pixel_leak = raw["prediction"] != before["prediction"]
        room_leak = raw["room_labels"] != before["room_labels"]
        images = (
            (before["prediction"], None, "Before", None),
            (gated["prediction"], None, "Strict", None),
            (raw["prediction"], None, "No gate", None),
            (before["prediction"], pixel_leak, "Pixel leakage", (0.9, 0.1, 0.1)),
            (
                before["room_labels"],
                room_leak,
                "Room-label leakage",
                (1.0, 0.55, 0.0),
            ),
        )
        for column, (labels, mask, title, color) in enumerate(images):
            axis = axes[row_index, column]
            axis.imshow(colors[np.asarray(labels)])
            if mask is not None and color is not None:
                axis.imshow(_overlay(mask, color))
            axis.set_title(title, fontsize=7, loc="left")
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            f"{row['operation']}\n{row['name']}",
            fontsize=5,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    strict = report["summary"]["all"]["strict"]
    no_gate = report["summary"]["all"]["no_edit_gating"]
    control = report["summary"]["single_room_negative_control"]
    claim = (
        f"Across {report['cases']:,} five-seed cases and "
        f"{control['cases']:,} single-room negative controls, the typed "
        "strict-locality arm produced no auditable outside leakage: pixel, "
        "room-instance, label, polygon, and edge exactness were 1.0 and "
        "locality violation rate was 0.0. Removing edit gating reduced "
        f"room-level exactness to {no_gate['outside_room_level_exact']:.4f} "
        f"and raised violation rate to {no_gate['locality_violation_rate']:.4f}. "
        "The leaking arm is positive sensitivity evidence that the evaluator is "
        "not a constant-pass metric."
    )
    output = {
        "status": "complete",
        "claim_role": "locality_as_controllable_constraint",
        "claim": claim,
        "cases": report["cases"],
        "single_room_negative_controls": control["cases"],
        "strict": strict,
        "no_edit_gating": no_gate,
        "checks": {
            "strict_five_layer_exact": all(
                strict[key] == 1.0
                for key in (
                    "outside_pixel_exact",
                    "outside_room_level_exact",
                    "outside_room_label_exact",
                    "outside_room_polygon_exact",
                    "outside_edge_exact",
                )
            ),
            "strict_violation_zero": strict["locality_violation_rate"] == 0.0,
            "single_room_controls_complete": (
                control["cases"] >= 7000
                and control["strict"]["outside_room_level_exact"] == 1.0
            ),
            "no_gate_counter_evidence": (
                no_gate["outside_room_level_exact"] < 0.5
                and no_gate["locality_violation_rate"] > 0.0
            ),
        },
        "sources": {
            "report": str(args.report.relative_to(ROOT)),
            "rows": str(args.rows.relative_to(ROOT)),
            "figure": str(args.figure.relative_to(ROOT)),
        },
    }
    failures = [key for key, value in output["checks"].items() if not value]
    if failures:
        output["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    doc = [
        "# Locality as a Controllable Constraint",
        "",
        claim,
        "",
        "## Claim Boundary",
        "",
        "- The no-gate decrease is evaluator-sensitivity evidence, not a baseline ranking.",
        "- Strict outside preservation is measured on the declared FEVP domain.",
        "- Single-room controls do not cover global redesign instructions.",
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    _render(rows, FloorplanTaxonomy.read(args.taxonomy), args.figure)
    print(json.dumps({"status": output["status"], **output["checks"]}, indent=2))
    raise SystemExit(0 if output["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
