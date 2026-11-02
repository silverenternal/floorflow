#!/usr/bin/env python3
"""Build a deterministic oracle-control failure analysis for FloorFlow."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.evaluation.room_graph_matching import INSTANCE_PRIMARY_PROTOCOL

FAILURE_FAMILIES = (
    "edit_boundary_wall_break",
    "complex_home_degradation",
    "room_identity_ambiguity",
    "wall_junction_mismatch",
)
PRIORITY = {
    "edit_boundary_wall_break": 0,
    "complex_home_degradation": 1,
    "room_identity_ambiguity": 2,
    "wall_junction_mismatch": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity-rows",
        type=Path,
        default=Path("outputs/reports/fp_complexity_strata_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--wall-rows",
        type=Path,
        default=Path("outputs/reports/fp_wall_junction_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=Path("outputs/reports/fp_control_robustness_val191_v1.json"),
    )
    parser.add_argument(
        "--predicted-control",
        type=Path,
        default=Path("outputs/reports/fp_predicted_control_v2.json"),
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("outputs/reports/fp_failure_analysis_rows_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_failure_analysis_v2.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_failure_analysis.md"),
    )
    parser.add_argument(
        "--panels",
        type=Path,
        default=Path("outputs/reports/fp_failure_panels_v2"),
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _structural_rows() -> dict[tuple[int, str, str], dict]:
    rows = {}
    for seed in (19, 20, 21, 22, 23):
        path = Path("outputs/reports") / (
            "fp_room_graph_diagnostic_rows_final_v1.json"
            if seed == 19
            else f"fp_room_graph_diagnostic_rows_final_seed{seed}_v1.json"
        )
        for row in _read(path)["rows"]:
            rows[(seed, row["operation"], row["name"])] = row
    return rows


def _flags(row: dict, wall: dict, structural: dict) -> dict[str, bool]:
    junction = wall["sources"]["prediction"]
    ambiguity = structural["ambiguity"]
    node_f1 = structural["matching"][INSTANCE_PRIMARY_PROTOCOL]["node_f1"]
    ged = structural["ged"]["ged"]
    wall_break = (
        row["conditional_watertight"] == 0.0
        and junction["junction_f1"] < 0.9
        and row["interface_complexity"] in {"wall_only", "both"}
    )
    return {
        "edit_boundary_wall_break": wall_break,
        "complex_home_degradation": (
            row["target_node_count"] >= 8
            and (node_f1 < 0.85 or ged > 5.0)
        ),
        "room_identity_ambiguity": (
            node_f1 < 0.8
            and (
                ambiguity["merged_predicted_nodes"] > 0
                or ambiguity["split_target_nodes"] > 0
            )
        ),
        "wall_junction_mismatch": junction["junction_f1"] < 0.9,
    }


def _primary_failure(flags: dict[str, bool]) -> str:
    active = [name for name in FAILURE_FAMILIES if flags[name]]
    if not active:
        return "none"
    return min(active, key=lambda name: PRIORITY[name])


def _gated_path(row: dict) -> Path:
    name = row["name"].replace("/", "__")
    return (
        Path("outputs/eval")
        / f"fp_control_gated_local_test503_seed{row['seed']}_v1"
        / row["operation"]
        / f"{name}.npz"
    )


def _rank_key(family: str, row: dict) -> tuple:
    if family == "edit_boundary_wall_break":
        return (row["conditional_watertight"], row["junction_f1"], -row["delta_mse"])
    if family == "complex_home_degradation":
        return (-row["node_f1"], -row["ged"], -row["target_node_count"])
    if family == "room_identity_ambiguity":
        return (-row["node_f1"], -row["merged_nodes"], -row["split_nodes"])
    return (row["junction_f1"], -row["edge_overlap"], -row["ged"])


def _save_panels(output: Path, failures: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for family in FAILURE_FAMILIES:
        selected = sorted(
            [row for row in failures if row["primary_failure"] == family],
            key=lambda row: _rank_key(family, row),
        )[:2]
        for rank, row in enumerate(selected, start=1):
            with np.load(_gated_path(row)) as payload:
                prediction = payload["prediction"]
                target = payload["target"]
            panels = (
                ("Prediction", prediction),
                ("Target", target),
                ("Label difference", (prediction != target).astype(int)),
                (
                    "Room difference",
                    ((prediction != target) & (target > 1)).astype(int),
                ),
            )
            figure, axes = plt.subplots(1, len(panels), figsize=(15, 3.7))
            for axis, (title, image) in zip(axes, panels, strict=True):
                axis.imshow(image, cmap="tab20", interpolation="nearest")
                axis.set_title(title)
                axis.axis("off")
            figure.suptitle(
                f"{family}: {row['operation']} | {row['name']} | "
                f"NodeF1={row['node_f1']:.3f}, GED={row['ged']:.2f}"
            )
            figure.tight_layout()
            filename = (
                f"{family}_{rank}_"
                f"{row['name'].replace('/', '__')}_{row['operation']}.png"
            )
            figure.savefig(output / filename, dpi=140)
            plt.close(figure)


def _oracle_deployable_gap(
    robustness: dict,
    predicted_control: dict,
) -> dict:
    rows = {float(row["drop_fraction"]): row for row in robustness["rows"]}
    clean = rows[0.0]
    mild = rows[0.1]
    return {
        "oracle_clean": {
            "split": "val191",
            "conditional_watertight": clean["conditional_watertight"],
            "edit_cost": clean["fpeval_edit_cost"],
            "room_adjacency_f1": clean["room_adjacency_f1"],
        },
        "oracle_dropout_0_1": {
            "split": "val191",
            "conditional_watertight": mild["conditional_watertight"],
            "edit_cost": mild["fpeval_edit_cost"],
            "room_adjacency_f1": mild["room_adjacency_f1"],
            "retention_vs_clean": mild["retention_vs_clean"],
        },
        "predicted_control": {
            **predicted_control["result"],
            "room_adjacency_f1": None,
        },
        "gap_checks": predicted_control["checks"],
        "interpretation": (
            "The strict-locality structural headline depends on oracle controls. "
            "Predicted controls remain undeployable, and 10% control dropout "
            "already violates the mild watertightness gate."
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _table(report: dict) -> str:
    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f}"

    lines = [
        "# Failure Analysis",
        "",
        "| Family | Multi-label cases | Fraction | Exclusive top-priority cases | Definition |",
        "|---|---:|---:|---:|---|",
    ]
    for family, values in report["families"].items():
        lines.append(
            f"| {family} | {values['cases']} | {values['fraction']:.6f} | "
            f"{values['exclusive_primary_cases']} | {values['definition']} |"
        )
    lines.extend(
        [
            "",
            "| Oracle-to-deployable stage | Watertight | Edit cost | Adjacency F1 |",
            "|---|---:|---:|---:|",
            (
                "| Oracle clean | "
                f"{report['oracle_deployable_gap']['oracle_clean']['conditional_watertight']:.6f} | "
                f"{report['oracle_deployable_gap']['oracle_clean']['edit_cost']:.6f} | "
                f"{report['oracle_deployable_gap']['oracle_clean']['room_adjacency_f1']:.6f} |"
            ),
            (
                "| Oracle 10% dropout | "
                f"{report['oracle_deployable_gap']['oracle_dropout_0_1']['conditional_watertight']:.6f} | "
                f"{report['oracle_deployable_gap']['oracle_dropout_0_1']['edit_cost']:.6f} | "
                f"{report['oracle_deployable_gap']['oracle_dropout_0_1']['room_adjacency_f1']:.6f} |"
            ),
            (
                "| Predicted control | "
                f"{number(report['oracle_deployable_gap']['predicted_control']['conditional_watertight'])} | "
                f"{number(report['oracle_deployable_gap']['predicted_control']['edit_cost'])} | "
                f"{number(report['oracle_deployable_gap']['predicted_control']['room_adjacency_f1'])} |"
            ),
        ]
    )
    lines.extend(
        [
            "",
            (
                "Failure thresholds are fixed in `build_fp_failure_analysis_v2.py`; "
                "multi-label families overlap, while exclusive counts use the "
                "listed priority order."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    complexity = _read(args.complexity_rows)["rows"]
    wall_rows = {
        (row["seed"], row["operation"], row["name"]): row
        for row in _read(args.wall_rows)["rows"]
    }
    structural_rows = _structural_rows()
    failures = []
    denominator = max(1, len(complexity))
    for row in complexity:
        key = (row["seed"], row["operation"], row["name"])
        wall = wall_rows[key]
        structural = structural_rows[key]
        flags = _flags(row, wall, structural)
        if not any(flags.values()):
            continue
        junction = wall["sources"]["prediction"]
        ambiguity = structural["ambiguity"]
        failures.append(
            {
                **row,
                "junction_f1": junction["junction_f1"],
                "degree_exact_rate": junction["degree_exact_rate"],
                "merged_nodes": ambiguity["merged_predicted_nodes"],
                "split_nodes": ambiguity["split_target_nodes"],
                "target_ambiguous_nodes": ambiguity["target_ambiguous_nodes"],
                "flags": flags,
                "primary_failure": _primary_failure(flags),
                "gated_path": str(_gated_path(row)),
            }
        )
    family_rows = {}
    for family in FAILURE_FAMILIES:
        selected = [row for row in failures if row["flags"][family]]
        sorted_cases = sorted(selected, key=lambda row: _rank_key(family, row))
        family_rows[family] = {
            "definition": {
                "edit_boundary_wall_break": (
                    "conditional watertightness fails, junction F1 < 0.9, and "
                    "the edit boundary intersects a wall"
                ),
                "complex_home_degradation": (
                    ">=8 target rooms and Node F1 < 0.85 or GED > 5"
                ),
                "room_identity_ambiguity": (
                    "Node F1 < 0.8 with a predicted merge or target split"
                ),
                "wall_junction_mismatch": "target-matched junction F1 < 0.9",
            }[family],
            "cases": len(selected),
            "fraction": len(selected) / denominator,
            "exclusive_primary_cases": sum(
                row["primary_failure"] == family for row in failures
            ),
            "operations": dict(Counter(row["operation"] for row in selected)),
            "seeds": dict(Counter(row["seed"] for row in selected)),
            "worst_cases": sorted_cases[:10],
        }
    report = {
        "status": "complete",
        "case_count": denominator,
        "flagged_cases": len(failures),
        "unflagged_cases": denominator - len(failures),
        "families": family_rows,
        "primary_failure_counts": dict(
            Counter(row["primary_failure"] for row in failures)
        ),
        "oracle_deployable_gap": _oracle_deployable_gap(
            _read(args.robustness),
            _read(args.predicted_control),
        ),
        "selection_rule": {
            "priority": list(FAILURE_FAMILIES),
            "ordering": (
                "wall break: watertight, junction F1, negative MSE; complex: "
                "negative Node F1, negative GED, negative room count; identity: "
                "negative Node F1, negative merge/split counts; wall: junction F1, "
                "negative edge overlap, negative GED"
            ),
        },
        "boundary": (
            "Failure families are multi-label diagnostics. Oracle-control and "
            "deployable stages use different splits and are not mixed."
        ),
    }
    _write_json(args.rows_output, {
        "status": "complete",
        "rows": failures,
    })
    _write_json(args.output, report)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(report), encoding="utf-8")
    _save_panels(args.panels, failures)
    print(json.dumps({
        "flagged_cases": len(failures),
        "primary_failure_counts": report["primary_failure_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
