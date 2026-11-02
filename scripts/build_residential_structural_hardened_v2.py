"""Consolidate residential structural evidence without mixing metric protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
SOURCES = {
    "fevp_stability": REPORTS / "fevp_fiveseed_stability_v2.json",
    "house_mind_alignment": REPORTS / "fp_housemind_structural_alignment_v1.json",
    "canonical": REPORTS / "fp_canonical_structural_fiveseed_v1.json",
    "typed_matched": REPORTS / "fevp_graph_metrics_v2.json",
    "legality": REPORTS / "fp_control_gated_structural_fiveseed_v1.json",
    "wall_junction": REPORTS / "fp_wall_junction_continuity_fiveseed_v1.json",
    "strict_locality": REPORTS / "fp_strict_locality_stress_fiveseed_v1.json",
    "fevp_validity": REPORTS / "fevp_validity_watertight_v2.json",
}


def load(name: str) -> dict:
    return json.loads(SOURCES[name].read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def summarize_aligned(report: dict) -> dict:
    rows = [
        seed["summary"]["after_prediction_vs_after_target"]
        for seed in report["per_seed"].values()
    ]
    names = (
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_overlap",
        "ged",
        "exact",
        "node_substitution",
        "node_deletion",
        "node_insertion",
        "edge_deletion",
        "edge_insertion",
    )
    return {
        "definition": report["house_mind_style_definition"],
        "sed": "after_prediction_vs_after_target",
        **{
            name: distribution([float(row[name]) for row in rows])
            for name in names
        },
    }


def summarize_wall_junctions(report: dict) -> dict:
    rows = [seed["summary"] for seed in report["per_seed"].values()]
    output = {"definition": report["definition"]}
    for readout in ("prediction", "topology_prediction", "target"):
        names = tuple(rows[0][readout])
        output[readout] = {
            name: distribution([float(row[readout][name]) for row in rows])
            for name in names
        }
    return output


def summarize_validity() -> tuple[dict, dict]:
    strict = load("strict_locality")
    fevp = load("fevp_validity")
    fpeval_by_suite = {
        suite: float(values["mean"]["validity"])
        for suite, values in strict["aggregate"].items()
    }
    five_check = fevp["summary"]["all"]
    reconciliation = {
        "status": "complete",
        "scope": "CubiCasa test503 five-seed strict-locality editing",
        "metrics": {
            "fp_eval_validity": {
                "value": min(fpeval_by_suite.values()),
                "definition": (
                    "Existing FP-Eval validity used by the editing gate."
                ),
                "source_partition": "oracle-control stress suites",
                "by_suite": fpeval_by_suite,
            },
            "fevp_five_check_validity": {
                "value": five_check["validity"]["validity"],
                "definition": (
                    "Conjunction of outer-boundary closure, room-polygon "
                    "closure, no self-intersection, no room overlap, and "
                    "semantic completeness."
                ),
                "components": five_check["validity"],
            },
            "conditional_watertightness": {
                "before_repair": five_check["before_watertight_rate"],
                "after_repair": five_check["after_watertight_rate"],
                "repair_count": five_check["repair_count"],
                "regression_count": five_check["regression_count"],
                "definition": (
                    "Planar-partition watertightness on cases where the "
                    "after-edit reference is eligible."
                ),
            },
        },
        "checks": {
            "fp_eval_and_fevp_are_separate": True,
            "fp_eval_validity_one": min(fpeval_by_suite.values()) == 1.0,
            "fevp_is_five_check_conjunction": (
                fevp["checks"]["validity_is_five_check_conjunction"]
            ),
            "watertight_eligible_before_and_after_separate": (
                "before_watertight_rate" in five_check
                and "after_watertight_rate" in five_check
            ),
        },
        "boundary": (
            "These are different metrics. Do not relabel the FEVP five-check "
            "rate as FP-Eval validity or merge their denominators."
        ),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in {
                "strict_locality": SOURCES["strict_locality"],
                "fevp_validity": SOURCES["fevp_validity"],
            }.items()
        },
    }
    return five_check, reconciliation


def main() -> None:
    stability = load("fevp_stability")
    alignment = summarize_aligned(load("house_mind_alignment"))
    canonical = load("canonical")
    typed = load("typed_matched")
    legality = load("legality")
    walls = summarize_wall_junctions(load("wall_junction"))
    _five_check, reconciliation = summarize_validity()
    aligned_node = alignment["node_f1"]
    aligned_edge = alignment["edge_overlap"]
    aligned_ged = alignment["ged"]
    canonical_node = canonical["summary"]["node_f1"]
    canonical_edge = canonical["summary"]["edge_overlap"]
    canonical_ged = canonical["summary"]["ged"]
    typed_metrics = typed["summary"]["seed_mean"]["typed_edit_graph"]
    checks = {
        "aligned_node_f1_at_least_0_98": aligned_node["min"] >= 0.98,
        "aligned_edge_overlap_at_least_0_92": aligned_edge["min"] >= 0.92,
        "aligned_ged_below_1_0": aligned_ged["max"] < 1.0,
        "canonical_node_f1_at_least_0_92": canonical_node["min"] >= 0.92,
        "canonical_edge_overlap_at_least_0_85": (
            canonical_edge["min"] >= 0.85
        ),
        "canonical_ged_below_2_1": canonical_ged["max"] < 2.1,
        "typed_matched_node_f1_at_least_0_95": (
            typed_metrics["node_f1"] >= 0.95
        ),
        "wall_continuity_at_least_0_94": (
            legality["summary"]["wall_continuity_rate"]["min"] >= 0.94
        ),
        "room_closure_exact": (
            legality["summary"]["room_closure_rate"]["min"] == 1.0
        ),
        "invalid_rate_zero": (
            legality["summary"]["invalid_rate"]["max"] == 0.0
        ),
        "fp_eval_validity_one": all(
            reconciliation["checks"][name]
            for name in (
                "fp_eval_validity_one",
                "fevp_is_five_check_conjunction",
            )
        ),
        "pairwise_orientation_stable": stability["status"] == "complete",
    }
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "residential_floorplan_only structure and validity",
        "dataset": "CubiCasa5K test503, five seeds where applicable",
        "protocols": {
            "typed_matched": {
                "definition": (
                    "Five-seed typed-edit graph rows on identical test503 "
                    "operations; not a leaderboard protocol."
                ),
                "metrics": typed_metrics,
            },
            "house_mind_style_aligned": alignment,
            "canonical_direct": {
                "definition": canonical["canonical_contract"],
                "metrics": canonical["summary"],
            },
            "wall_junction": walls,
        },
        "legality": legality["summary"],
        "validity_reconciliation": reconciliation["metrics"],
        "checks": checks,
        "claim": (
            "Under five fixed seeds, FloorFlow preserves strict outside-edit "
            "exactness with zero locality violation area, maintains room "
            "closure, and reaches 0.98225 aligned Node F1 and 0.92947 aligned "
            "edge overlap. Canonical direct-tiled and typed-matched graph "
            "readouts are reported separately."
        ),
        "boundary": (
            "The aligned, canonical, and typed-matched graph results use "
            "different node/edge definitions and must not be pooled. "
            "No graph or full-CAD leaderboard SOTA is claimed."
        ),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in SOURCES.items()
        },
    }
    output = REPORTS / "residential_structural_hardened_v2.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    validity_output = REPORTS / "residential_validity_reconciliation_v2.json"
    validity_output.write_text(
        json.dumps(reconciliation, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Residential Structure and Validity v2",
        "",
        "| Protocol | Node F1 | Edge overlap | GED |",
        "|---|---:|---:|---:|",
        (
            f"| HouseMind-style aligned | {aligned_node['mean']:.6f} | "
            f"{aligned_edge['mean']:.6f} | {aligned_ged['mean']:.6f} |"
        ),
        (
            f"| Canonical direct-tiled | {canonical_node['mean']:.6f} | "
            f"{canonical_edge['mean']:.6f} | {canonical_ged['mean']:.6f} |"
        ),
        (
            "| Typed matched routing | "
            f"{typed_metrics['node_f1']:.6f} | "
            f"{typed_metrics['edge_overlap']:.6f} | "
            f"{typed_metrics['ged']:.6f} |"
        ),
        "",
        (
            "Aligned ranges: Node F1 "
            f"{aligned_node['min']:.6f}-{aligned_node['max']:.6f}, "
            f"edge overlap {aligned_edge['min']:.6f}-"
            f"{aligned_edge['max']:.6f}, GED {aligned_ged['min']:.6f}-"
            f"{aligned_ged['max']:.6f}."
        ),
        "",
        (
            "FP-Eval validity and FEVP five-check validity are distinct: "
            f"{reconciliation['metrics']['fp_eval_validity']['value']:.6f} "
            "versus "
            f"{reconciliation['metrics']['fevp_five_check_validity']['value']:.6f}."
        ),
    ]
    table = ROOT / "paper/tables/residential_structural_v2.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
