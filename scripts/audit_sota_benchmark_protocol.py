#!/usr/bin/env python3
"""Audit the canonical SOTA output, graph, and attack contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.evaluation.sota_protocol import (
    SCHEMA_VERSION,
    TRACKS,
    CanonicalFloorplan,
    CanonicalFurniture,
    CanonicalOpening,
    CanonicalRoom,
    CanonicalWall,
    canonical_fingerprint,
    evaluate_canonical_pair,
    validate_floorplan,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sota_benchmark_protocol_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/sota_benchmark_protocol_audit_v2.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Door",
            "window:Window",
            "room:LivingRoom",
            "room:Kitchen",
            "furniture:Chair",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def sample_plan() -> CanonicalFloorplan:
    living = ((2.0, 2.0), (15.0, 2.0), (15.0, 17.0), (2.0, 17.0))
    kitchen = ((15.0, 2.0), (30.0, 2.0), (30.0, 17.0), (15.0, 17.0))
    return CanonicalFloorplan(
        case_id="protocol-smoke",
        track="cubicasa_reconstruction",
        width=32,
        height=32,
        footprint=((1.0, 1.0), (31.0, 1.0), (31.0, 18.0), (1.0, 18.0)),
        rooms=(
            CanonicalRoom("room_1", "room:LivingRoom", living, living),
            CanonicalRoom("room_2", "room:Kitchen", kitchen, kitchen),
        ),
        walls=(
            CanonicalWall("wall_1", (1.0, 1.0), (31.0, 1.0), 1.0),
            CanonicalWall("wall_2", (31.0, 1.0), (31.0, 18.0), 1.0),
            CanonicalWall("wall_3", (31.0, 18.0), (1.0, 18.0), 1.0),
            CanonicalWall("wall_4", (1.0, 18.0), (1.0, 1.0), 1.0),
            CanonicalWall("wall_5", (15.0, 2.0), (15.0, 17.0), 1.0),
        ),
        openings=(
            CanonicalOpening("opening_1", "door:Door", (15.0, 8.0), 3.0, "wall_5"),
        ),
        furniture=(
            CanonicalFurniture(
                "furniture_1",
                "furniture:Chair",
                ((4.0, 4.0), (7.0, 4.0), (7.0, 7.0), (4.0, 7.0)),
                "room_1",
            ),
        ),
        adjacency=(("room_1", "room_2"),),
        source_hash="protocol-smoke-source",
    )


def attack_plans(base: CanonicalFloorplan) -> dict[str, CanonicalFloorplan]:
    kitchen = base.rooms[1]
    overlapping_kitchen = replace(
        kitchen,
        polygon=((10.0, 2.0), (30.0, 2.0), (30.0, 17.0), (10.0, 17.0)),
        support=((10.0, 2.0), (30.0, 2.0), (30.0, 17.0), (10.0, 17.0)),
    )
    return {
        "remove_room": replace(base, rooms=base.rooms[:1], adjacency=()),
        "remove_adjacency": replace(base, adjacency=()),
        "move_opening": replace(
            base,
            openings=(replace(base.openings[0], center=(15.0, 14.0)),),
        ),
        "remove_furniture": replace(base, furniture=()),
        "overlap_rooms": replace(
            base,
            rooms=(base.rooms[0], overlapping_kitchen),
        ),
        "remove_wall": replace(base, walls=base.walls[1:]),
    }


def config_checks(config: dict[str, Any]) -> dict[str, bool]:
    contract = config["output_contract"]
    graph = config["graph_contract"]
    tracks = config["tracks"]
    baselines = read_json(ROOT / config["authoritative_external_contract"])
    return {
        "canonical_schema_matches_code": contract["schema_version"]
        == SCHEMA_VERSION,
        "all_code_tracks_declared": set(TRACKS).issubset(
            {track for track in tracks if track != "oracle_control_diagnostic"}
        ),
        "primary_and_secondary_graphs_separated": graph["primary_protocol"]
        != graph["secondary_protocol"],
        "graph_pooling_forbidden": "Never pool" in graph["pooling_rule"],
        "oracle_diagnostic_not_leaderboard": tracks[
            "oracle_control_diagnostic"
        ]["leaderboard"]
        is False,
        "required_entities_present": len(contract["required_entities"]) >= 6,
        "external_contract_present": baselines["version"]
        == "2026-10-29-sota-official-baselines-v2",
        "external_contract_gate_passed": read_json(
            ROOT / "outputs/reports/official_baseline_execution_v2.json"
        )["gate"]["passed"],
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    task_taxonomy = taxonomy()
    tolerances = FEVPTolerances(
        room_match_min_iou=float(config["tolerances"]["room_match_min_iou"]),
        room_polygon_hausdorff_px=float(
            config["tolerances"]["room_polygon_hausdorff_px"]
        ),
        room_min_area_px=float(config["tolerances"]["room_min_area_px"]),
        room_overlap_area_px2=float(
            config["tolerances"]["room_overlap_area_px2"]
        ),
        edge_contact_px=float(config["tolerances"]["edge_contact_px"]),
        wall_junction_cluster_px=float(
            config["tolerances"]["wall_junction_cluster_px"]
        ),
        hole_min_area_px=float(config["tolerances"]["hole_min_area_px"]),
        changed_pixel_epsilon=float(
            config["tolerances"]["changed_pixel_epsilon"]
        ),
    )
    base = sample_plan()
    base_errors = validate_floorplan(
        base,
        task_taxonomy,
        tolerances=tolerances,
        require_complete_cad=True,
    )
    identity = evaluate_canonical_pair(
        base,
        base,
        task_taxonomy,
        tolerances=tolerances,
    )
    expected_attack_metrics = {
        row["id"]: row["expected_sensitivity"]
        for row in config["adversarial_suite"]
    }
    attack_results = {}
    base_fingerprint = canonical_fingerprint(base)
    for name, attacked in attack_plans(base).items():
        result = evaluate_canonical_pair(
            attacked,
            base,
            task_taxonomy,
            tolerances=tolerances,
        )
        attacked_fingerprint = canonical_fingerprint(attacked)
        metrics = {}
        for metric in expected_attack_metrics[name]:
            metrics[metric] = {
                "clean": identity[metric],
                "attacked": result[metric],
                "changed": result[metric] != identity[metric],
            }
        attack_results[name] = {
            "metrics": metrics,
            "sensitive": all(item["changed"] for item in metrics.values()),
            "input_fingerprint_changed": attacked_fingerprint
            != base_fingerprint,
            "evaluator_preserved_input_fingerprint": (
                canonical_fingerprint(base) == base_fingerprint
            ),
        }
    roundtrip = CanonicalFloorplan.from_dict(
        json.loads(json.dumps(base.to_dict()))
    )
    checks = {
        **config_checks(config),
        "clean_plan_valid": not base_errors,
        "json_roundtrip_exact": canonical_fingerprint(roundtrip)
        == base_fingerprint,
        "identity_graph_metrics_exact": all(
            identity[name] == 1.0
            for name in ("node_f1", "edge_overlap", "instance_node_f1")
        ),
        "identity_ged_zero": identity["ged"] == 0.0
        and identity["instance_ged"] == 0.0,
        "identity_hard_metrics_exact": identity["validity"] == 1.0
        and identity["conditional_watertightness"] == 1.0,
        "all_attacks_sensitive": all(
            result["sensitive"] for result in attack_results.values()
        ),
        "evaluator_does_not_mutate_inputs": all(
            result["evaluator_preserved_input_fingerprint"]
            for result in attack_results.values()
        ),
        "six_attacks_executed": len(attack_results) == 6,
    }
    report = {
        "version": config["schema_version"],
        "status": "passed" if all(checks.values()) else "failed",
        "scope": config["scope"],
        "implementation": {
            "path": config["implementations"]["output_and_pair_evaluator"],
            "sha256": sha256(
                ROOT / config["implementations"]["output_and_pair_evaluator"]
            ),
            "config_sha256": sha256(args.config),
            "code_commit": "no_git_history",
        },
        "canonical_example": base.to_dict(),
        "identity_metrics": identity,
        "attacks": attack_results,
        "baseline_recompute": {
            "status": "contract_ready",
            "source": "outputs/reports/official_baseline_execution_v2.json",
            "note": (
                "Legacy anchors are frozen. Canonical re-scoring starts when "
                "the SOTA-03 adapter emits the canonical object."
            ),
        },
        "checks": checks,
        "decision": (
            "Model training may use this contract. Leaderboard rows remain "
            "disabled until every competitor and FloorFlow output uses it."
        ),
        "next_task": "SOTA-03",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
