#!/usr/bin/env python3
"""Recover matched architecture-necessity evidence from frozen FP outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stress-rows",
        type=Path,
        default=Path("outputs/reports/fp_strict_locality_stress_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--edit-detection",
        type=Path,
        default=Path("outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--hard-gate",
        type=Path,
        default=Path("outputs/reports/fp_control_gated_hard_gate_v1.json"),
    )
    parser.add_argument(
        "--guarded-topology",
        type=Path,
        default=Path("outputs/reports/fp_guarded_topology_test503_seed19_v1.json"),
    )
    parser.add_argument(
        "--architecture-audit",
        type=Path,
        default=Path("outputs/reports/fp_model_architecture_audit_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_architecture_necessity_ablation_v2.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_architecture_necessity_ablation.md"),
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired(
    rows: list[dict],
    current_key: str,
    baseline_key: str,
) -> dict:
    current = [{"value": float(row[current_key])} for row in rows]
    baseline = [{"value": float(row[baseline_key])} for row in rows]
    return paired_delta(
        current,
        baseline,
        "value",
        iterations=5000,
        seed=53,
    )


def _control_gate(stress_rows: list[dict]) -> dict:
    unique = {}
    for row in stress_rows:
        unique[(row["seed"], row["operation"], row["name"])] = row
    rows = list(unique.values())
    suites = {}
    for suite in ("wall_adjacent", "opening_covering", "exterior_boundary"):
        selected = [row for row in stress_rows if row["suite"] == suite]
        suites[suite] = {
            "cases": len(selected),
            "raw_outside_exact": float(
                np.mean([row["raw_outside_exact_ratio"] for row in selected])
            ),
            "gated_outside_exact": float(
                np.mean([row["gated_outside_exact_ratio"] for row in selected])
            ),
            "raw_outside_room_exact": float(
                np.mean([row["raw_outside_room_exact_ratio"] for row in selected])
            ),
            "gated_outside_room_exact": float(
                np.mean([row["gated_outside_room_exact_ratio"] for row in selected])
            ),
        }
    raw_inside = float(
        np.mean([row["raw_inside_change_fraction"] for row in rows])
    )
    gated_inside = float(
        np.mean([row["gated_inside_change_fraction"] for row in rows])
    )
    return {
        "cases": len(rows),
        "raw_outside_exact_mean": float(
            np.mean([row["raw_outside_exact_ratio"] for row in rows])
        ),
        "gated_outside_exact_mean": float(
            np.mean([row["gated_outside_exact_ratio"] for row in rows])
        ),
        "raw_outside_room_exact_mean": float(
            np.mean([row["raw_outside_room_exact_ratio"] for row in rows])
        ),
        "gated_outside_room_exact_mean": float(
            np.mean([row["gated_outside_room_exact_ratio"] for row in rows])
        ),
        "pixel_exact_gain": _paired(
            rows,
            "gated_outside_exact_ratio",
            "raw_outside_exact_ratio",
        ),
        "room_label_exact_gain": _paired(
            rows,
            "gated_outside_room_exact_ratio",
            "raw_outside_room_exact_ratio",
        ),
        "gated_min_pixel_exact": float(
            min(row["gated_outside_exact_ratio"] for row in rows)
        ),
        "gated_min_room_exact": float(
            min(row["gated_outside_room_exact_ratio"] for row in rows)
        ),
        "inside_change_retention": gated_inside / max(raw_inside, 1.0e-12),
        "suites": suites,
    }


def _watertight_gate(
    hard_gate: dict,
    guarded: dict,
) -> dict:
    seed19 = next(row for row in hard_gate["seeds"] if int(row["seed"]) == 19)
    guard_rows = guarded["rows"]
    conditional_guard = [
        row["selected_watertight"] if row["target_watertight"] else True
        for row in guard_rows
    ]
    return {
        "matched_cases": len(guard_rows),
        "unguarded_conditional_watertight": seed19["conditional_watertight"],
        "guarded_conditional_watertight": float(np.mean(conditional_guard)),
        "guard_minus_unguarded": float(
            np.mean(conditional_guard) - seed19["conditional_watertight"]
        ),
        "guard_source_counts": guarded["source_counts"],
        "guard_visible_min_outside_exact": guarded[
            "visible_min_outside_exact_ratio"
        ],
        "decision": (
            "do_not_retain"
            if np.mean(conditional_guard) < seed19["conditional_watertight"]
            else "candidate_retain"
        ),
    }


def _clusters(rows: list[dict], key: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in rows:
        result.setdefault(row["name"], []).append(float(row[key]))
    return result


def _bootstrap_means(
    clusters: dict[str, list[float]],
    keys: np.ndarray,
) -> np.ndarray:
    names = sorted(clusters)
    sums = np.asarray([sum(clusters[name]) for name in names])
    counts = np.asarray([len(clusters[name]) for name in names])
    return sums[keys].sum(axis=1) / counts[keys].sum(axis=1)


def _cluster_pair_gain(
    rows: list[dict],
    current_key: str,
    baseline_key: str,
    *,
    iterations: int = 5000,
    seed: int = 59,
) -> dict:
    current = _clusters(rows, current_key)
    baseline = _clusters(rows, baseline_key)
    names = sorted(current)
    current = {name: current[name] for name in names}
    baseline = {name: baseline[name] for name in names}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), (iterations, len(names)))
    differences = _bootstrap_means(current, sampled) - _bootstrap_means(
        baseline,
        sampled,
    )
    observed = float(
        np.mean([row[current_key] for row in rows])
        - np.mean([row[baseline_key] for row in rows])
    )
    return {
        "mean_delta": observed,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def _cluster_control_gate(stress_rows: list[dict]) -> dict:
    unique = {}
    for row in stress_rows:
        unique[(row["seed"], row["operation"], row["name"])] = row
    rows = list(unique.values())
    return {
        "cases": len(rows),
        "pixel_exact_gain": _cluster_pair_gain(
            rows,
            "gated_outside_exact_ratio",
            "raw_outside_exact_ratio",
        ),
        "room_exact_gain": _cluster_pair_gain(
            rows,
            "gated_outside_room_exact_ratio",
            "raw_outside_room_exact_ratio",
        ),
    }


def _table(report: dict) -> str:
    control = report["control_gate"]
    pixel = control["pixel_exact_gain"]
    room = control["room_label_exact_gain"]
    water = report["watertight_gate"]
    lines = [
        "# Architecture Necessity Ablations",
        "",
        "| Component | Arm A | Arm B | Matched evidence | Decision |",
        "|---|---|---|---|---|",
        (
            "| Control gate | Raw local output outside exact "
            f"{control['raw_outside_exact_mean']:.6f} | Gated outside exact "
            f"{control['gated_outside_exact_mean']:.6f} | "
            f"pixel gain {pixel['mean_delta']:+.6f} "
            f"[{pixel['ci_low']:+.6f}, {pixel['ci_high']:+.6f}]; "
            f"room gain {room['mean_delta']:+.6f} | retain |"
        ),
        (
            f"| Watertight guard | Off {water['unguarded_conditional_watertight']:.6f} | "
            f"On {water['guarded_conditional_watertight']:.6f} | "
            f"matched seed19, {water['matched_cases']} cases | "
            f"{water['decision']} |"
        ),
        (
            "| Graph-token / chip conditioning | Not part of retained floorplan route | "
            "N/A | architecture audit; separate FloorSet/chip track | not applicable |"
        ),
        "",
        (
            "The control gate preserves inside response at "
            f"{control['inside_change_retention']:.4f} of the raw inside change."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stress = _read(args.stress_rows)["rows"]
    edit = _read(args.edit_detection)
    hard_gate = _read(args.hard_gate)
    guarded = _read(args.guarded_topology)
    architecture = _read(args.architecture_audit)
    control = _control_gate(stress)
    control["cluster_bootstrap"] = _cluster_control_gate(stress)
    water = _watertight_gate(hard_gate, guarded)
    report = {
        "status": "complete",
        "control_gate": control,
        "deltas": {
            "raw_delta_iou": edit["aggregate"]["raw_delta_iou"],
            "gated_delta_iou": edit["aggregate"]["local_delta_iou"],
            "raw_delta_mse": edit["aggregate"]["raw_delta_mse"],
            "gated_delta_mse": edit["aggregate"]["local_delta_mse"],
        },
        "watertight_gate": water,
        "graph_token_chip_conditioning": {
            "status": "not_applicable_to_retained_floorplan_route",
            "evidence": [
                "architecture decisions retain room, structure, and furniture heads",
                "the graph-token/chip route is a separate FloorSet experiment",
                "no matched floorplan checkpoint removes graph-token or chip conditioning",
            ],
            "decision": "Do not claim an ablation effect for an architecture absent from the retained route.",
        },
        "component_audit": {
            "decisions": architecture["decisions"],
            "checks": {
                name: values["passed"]
                for name, values in architecture["component_ablations"][
                    "components"
                ].items()
            },
        },
        "checks": {
            "control_gate_gain_ci_excludes_zero": (
                control["cluster_bootstrap"]["pixel_exact_gain"]["ci_low"] > 0.0
                and control["cluster_bootstrap"]["room_exact_gain"]["ci_low"] > 0.0
            ),
            "control_gate_exact_one": (
                control["gated_min_pixel_exact"] == 1.0
                and control["gated_min_room_exact"] == 1.0
            ),
            "inside_response_retained": control["inside_change_retention"] > 0.8,
            "watertight_guard_not_promoted": water["decision"] == "do_not_retain",
            "component_audit_passed": all(
                architecture["component_ablations"]["components"][name][
                    "passed"
                ]
                for name in ("room", "structure", "furniture")
            ),
            "graph_token_chip_not_claimed": True,
        },
        "boundary": (
            "The control-gate and watertight-guard evidence are matched recovery "
            "analyses on existing rows. Graph-token/chip conditioning is outside "
            "the retained floorplan route and is not claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(report), encoding="utf-8")
    print(json.dumps({
        "checks": report["checks"],
        "control_gate": {
            "pixel_gain": control["pixel_exact_gain"],
            "room_gain": control["room_label_exact_gain"],
            "inside_retention": control["inside_change_retention"],
        },
        "watertight_gate": water,
    }, indent=2))


if __name__ == "__main__":
    main()
