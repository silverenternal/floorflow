"""Attack-test the channel-native zero-tolerance edit projector."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon

from floorflow.data.constraint_action_dataset import ActionRecord
from floorflow.data.constraint_candidates import (
    CandidateSpec,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.vector_edit_protocol import VectorPlan, VectorRoom
from floorflow.inference.strict_local_projection import (
    native_control_regions,
    outside_change_counts,
    project_controls,
    project_labels,
    project_vector_plan,
    typed_edge_exact,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/zero_tolerance_locality_audit_v2.json"
CANDIDATES = ROOT / "outputs/control_contracts/model_candidates_v1/val/index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-groups", type=int, default=128)
    parser.add_argument("--skip-real", action="store_true")
    return parser.parse_args()


def _real_candidate_audit(
    taxonomy,
    groups: int,
) -> dict:
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    selected_groups = payload["groups"][: int(groups)]
    before_counts = []
    after_counts = []
    negative_candidates = 0
    for group in selected_groups:
        before = read_control_cache(Path(group["source"]))
        edit = LOCAL_EDIT_OPERATIONS[group["operation"]](
            before,
            taxonomy,
            seed=int(group["action_seed"]),
        )
        if edit is None:
            continue
        for candidate in group["candidates"]:
            record = ActionRecord(
                name=group["name"],
                split="val",
                source=group["source"],
                operation=group["operation"],
                seed=int(group["action_seed"]),
                bbox=tuple(candidate["bbox"]),
                control_size=tuple(
                    before["room_logits"].shape[-2:]
                    if group["operation"] in {"room_add", "room_remove", "room_type"}
                    else edit.control_size
                ),
                changed_pixels=int(candidate["changed_pixels"]),
            )
            if candidate["transform"] == "alternate_legal_target":
                alternative = LOCAL_EDIT_OPERATIONS[group["operation"]](
                    before,
                    taxonomy,
                    seed=int(candidate["candidate_seed"]),
                )
                if alternative is None:
                    continue
                after = alternative.controls
            else:
                spec = CandidateSpec(
                    candidate_id=candidate["candidate_id"],
                    group_id=group["group_id"],
                    role=candidate["role"],
                    violation_family=candidate["violation_family"],
                    severity=int(candidate["severity"]),
                    transform=candidate["transform"],
                    seed=int(candidate["candidate_seed"]),
                    operation=group["operation"],
                )
                after = materialize_candidate(
                    before,
                    edit.controls,
                    record,
                    spec,
                    taxonomy,
                    footprint=None,
                ).controls
            region = np.zeros(record.control_size, dtype=bool)
            x, y, width, height = record.bbox
            region[
                max(y, 0) : min(y + height, region.shape[0]),
                max(x, 0) : min(x + width, region.shape[1]),
            ] = True
            regions = native_control_regions(before, region)
            raw = outside_change_counts(before, after, regions)
            projected = project_controls(before, after, regions)
            repaired = outside_change_counts(before, projected, regions)
            before_counts.append(sum(raw.values()))
            after_counts.append(sum(repaired.values()))
            if candidate["role"] == "near_violation":
                negative_candidates += 1
    return {
        "groups": len(selected_groups),
        "candidates": len(after_counts),
        "negative_candidates": negative_candidates,
        "pre_projection_outside_changes": int(sum(before_counts)),
        "post_projection_outside_changes": int(sum(after_counts)),
        "pre_projection_nonzero_candidates": sum(
            value > 0 for value in before_counts
        ),
        "post_projection_nonzero_candidates": sum(
            value > 0 for value in after_counts
        ),
    }


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _controls(taxonomy):
    size = 16
    rooms = torch.full((2, size, size), -8.0)
    rooms[0, :, :8] = 8.0
    rooms[1, :, 8:] = 8.0
    structure = torch.zeros(
        2 + len(taxonomy.opening_ids) + 3,
        size,
        size,
    )
    structure[2, :, 7:9] = 1.0
    structure[3, 2:4, 7:9] = 1.0
    furniture = torch.zeros(
        2 + len(taxonomy.furniture_ids) + 1,
        size,
        size,
    )
    furniture[2, 2:4, 2:4] = 1.0
    support = torch.ones(2 + 4, size, size)
    return {
        "room_logits": rooms,
        "program": torch.tensor([True, True]),
        "structure": structure,
        "furniture": furniture,
        "support": support,
    }


def _plan(taxonomy, *, attacked: bool) -> VectorPlan:
    left = Polygon([(0, 0), (7, 0), (7, 15), (0, 15)])
    right = Polygon([(8, 0), (15, 0), (15, 15), (8, 15)])
    if attacked:
        left = Polygon([(0, 0), (6, 0), (6, 15), (0, 15)])
    mask_left = np.zeros((16, 16), dtype=bool)
    mask_left[:, :8] = True
    mask_right = np.zeros((16, 16), dtype=bool)
    mask_right[:, 8:] = True
    return VectorPlan(
        rooms=(
            VectorRoom(taxonomy.class_names[taxonomy.room_ids[0]], left, mask_left),
            VectorRoom(taxonomy.class_names[taxonomy.room_ids[1]], right, mask_right),
        ),
        walls=(),
        openings=(),
        wall_mask=np.zeros((16, 16), dtype=bool),
        opening_mask=np.zeros((16, 16), dtype=bool),
        source=None,
    )


def main() -> None:
    args = parse_args()
    taxonomy = _taxonomy()
    before = _controls(taxonomy)
    after = {key: value.clone() for key, value in before.items()}
    after["room_logits"][0, 0, 0] = 9.0
    after["structure"][2, 0, 0] = 1.0
    after["structure"][3, 1, 1] = 1.0
    after["furniture"][2, 0, 0] = 1.0
    after["support"][0, 0, 0] = 0.0
    region = np.zeros((16, 16), dtype=bool)
    region[:, 8:] = True
    regions = native_control_regions(before, region)
    projected_controls = project_controls(before, after, regions)
    control_counts = outside_change_counts(
        before,
        projected_controls,
        regions,
    )

    before_labels = np.zeros((16, 16), dtype=np.int64)
    before_labels[:, :8] = taxonomy.room_ids[0]
    before_labels[:, 8:] = taxonomy.room_ids[1]
    after_labels = before_labels.copy()
    after_labels[0, 0] = taxonomy.room_ids[1]
    after_labels[4, 4] = 0
    projected_labels = project_labels(
        before_labels,
        after_labels,
        before_labels,
        region,
        np.ones((16, 16), dtype=bool),
        taxonomy,
    )
    outside = ~region
    pixel_exact = np.array_equal(
        projected_labels[outside],
        before_labels[outside],
    )
    room_label_exact = pixel_exact
    wall_exact = np.array_equal(
        projected_controls["structure"][2][outside],
        before["structure"][2][outside],
    )
    opening_exact = np.array_equal(
        projected_controls["structure"][3][outside],
        before["structure"][3][outside],
    )

    before_plan = _plan(taxonomy, attacked=False)
    attacked_plan = _plan(taxonomy, attacked=True)
    projected_plan = project_vector_plan(
        before_plan,
        attacked_plan,
        region,
        np.ones((16, 16), dtype=bool),
    )
    polygon_exact = projected_plan.rooms[0].polygon.equals(
        before_plan.rooms[0].polygon
    )

    adjacency_before = torch.eye(4, dtype=torch.bool)
    adjacency_before[0, 1] = adjacency_before[1, 0] = True
    adjacency_after = adjacency_before.clone()
    adjacency_after[0, 1] = adjacency_after[1, 0] = False
    unaffected = torch.tensor([False, True, True, True])
    edge_exact = typed_edge_exact(
        adjacency_before,
        adjacency_after,
        unaffected,
    )
    attacks = {
        "one_pixel_label_leak": not bool(
            np.array_equal(
                after_labels[outside],
                before_labels[outside],
            )
        ),
        "micro_hole": bool(after_labels[4, 4] != before_labels[4, 4]),
        "detached_opening": bool(after["structure"][3, 1, 1] == 1),
        "wall_displacement": bool(after["structure"][2, 0, 0] == 1),
        "edge_rewiring": bool(
            not torch.equal(adjacency_before, adjacency_after)
        ),
        "unaffected_polygon_move": bool(
            not attacked_plan.rooms[0].polygon.equals(
                before_plan.rooms[0].polygon
            )
        ),
    }
    real_candidates = (
        {"skipped": True}
        if args.skip_real
        else _real_candidate_audit(taxonomy, args.real_groups)
    )
    checks = {
        "control_outside_zero": all(value == 0 for value in control_counts.values()),
        "pixel_outside_exact": pixel_exact,
        "room_label_outside_exact": room_label_exact,
        "wall_outside_exact": wall_exact,
        "opening_outside_exact": opening_exact,
        "unaffected_polygon_exact": polygon_exact,
        "unaffected_edge_exact": edge_exact,
        "all_attacks_sensitive": all(attacks.values()),
        "real_validation_post_projection_zero": (
            real_candidates.get("post_projection_outside_changes", 0) == 0
        ),
        "real_validation_negative_controls_present": (
            real_candidates.get("negative_candidates", 0) > 0
        ),
    }
    report = {
        "schema_version": "zero-tolerance-locality-audit-v2",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "tolerance": {
            "control_cells": 0,
            "pixels": 0,
            "polygon_vertices": 0,
            "edge_entries": 0,
        },
        "control_outside_changes": control_counts,
        "attacks": attacks,
        "real_validation_candidates": real_candidates,
        "checks": checks,
        "projection_rule": (
            "Declared edit regions are mapped independently to every native "
            "control resolution and applied after all interpolation."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
