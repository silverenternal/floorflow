"""Run the frozen CAGN selector once on the CubiCasa test503 lockbox."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import build_constraint_embedding_features_v1 as feature_builder
import numpy as np
import torch
import train_constraint_aware_gate as runtime

from floorflow.data.constraint_action_dataset import (
    ActionRecord,
    control_graph_state,
    load_cached_footprint,
)
from floorflow.data.constraint_candidate_dataset import load_candidate_index
from floorflow.data.constraint_candidates import (
    CandidateSpec,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.constraint_aware_gate import ConstraintAwareGate

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_cagn_test503_final_v1.json"
FREEZE = ROOT / "outputs/reports/residential_cagn_validation_freeze_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(row: dict) -> ActionRecord:
    return ActionRecord(
        **{
            **row,
            "bbox": tuple(row["bbox"]),
            "control_size": tuple(row["control_size"]),
        }
    )


def _outside_exact(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    record: ActionRecord,
) -> tuple[bool, bool, bool, dict[str, int]]:
    x, y, box_width, box_height = record.bbox
    record_width, record_height = record.control_size
    controls_exact = True
    controls_tolerant = True
    outside_counts = {}
    for key in ("room_logits", "structure", "furniture", "support"):
        delta = (after[key].float() - before[key].float()).abs()
        while delta.ndim > 2:
            delta = delta.amax(dim=0)
        height, width = delta.shape
        left = round(x * width / max(record_width, 1))
        top = round(y * height / max(record_height, 1))
        right = round((x + box_width) * width / max(record_width, 1))
        bottom = round((y + box_height) * height / max(record_height, 1))
        outside = torch.ones_like(delta, dtype=torch.bool)
        outside[
            max(top, 0) : min(bottom, height),
            max(left, 0) : min(right, width),
        ] = False
        tolerant_outside = torch.ones_like(delta, dtype=torch.bool)
        tolerant_outside[
            max(top - 1, 0) : min(bottom + 1, height),
            max(left - 1, 0) : min(right + 1, width),
        ] = False
        outside_count = int((delta > 0)[outside].sum())
        outside_counts[key] = outside_count
        controls_exact &= outside_count == 0
        controls_tolerant &= not bool(
            (delta > 0)[tolerant_outside].any()
        )
    before_labels = before["room_logits"].argmax(dim=0)
    after_labels = after["room_logits"].argmax(dim=0)
    label_height, label_width = before_labels.shape
    label_left = round(x * label_width / max(record_width, 1))
    label_top = round(y * label_height / max(record_height, 1))
    label_right = round(
        (x + box_width) * label_width / max(record_width, 1)
    )
    label_bottom = round(
        (y + box_height) * label_height / max(record_height, 1)
    )
    outside_labels = torch.ones_like(before_labels, dtype=torch.bool)
    outside_labels[
        max(label_top, 0) : min(label_bottom, label_height),
        max(label_left, 0) : min(label_right, label_width),
    ] = False
    labels_exact = bool(
        (before_labels[outside_labels] == after_labels[outside_labels]).all()
    )
    return controls_exact, controls_tolerant, labels_exact, outside_counts


def _graph_metrics(predicted, positive) -> dict[str, float]:
    predicted_nodes = predicted["program"].bool()
    positive_nodes = positive["program"].bool()
    true_positive = int((predicted_nodes & positive_nodes).sum())
    precision = true_positive / max(int(predicted_nodes.sum()), 1)
    recall = true_positive / max(int(positive_nodes.sum()), 1)
    node_f1 = 2 * precision * recall / max(precision + recall, 1.0e-6)
    predicted_edges = predicted["adjacency"] > 0.05
    positive_edges = positive["adjacency"] > 0.05
    edge_tp = int((predicted_edges & positive_edges).sum())
    edge_precision = edge_tp / max(int(predicted_edges.sum()), 1)
    edge_recall = edge_tp / max(int(positive_edges.sum()), 1)
    edge_f1 = (
        2 * edge_precision * edge_recall
        / max(edge_precision + edge_recall, 1.0e-6)
    )
    node_edits = int((predicted_nodes ^ positive_nodes).sum())
    edge_edits = float(
        torch.triu(
            (predicted["adjacency"] - positive["adjacency"]).abs(),
            diagonal=1,
        ).sum()
    )
    return {
        "node_f1": node_f1,
        "edge_f1": edge_f1,
        "weighted_ged": node_edits + edge_edits,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("test503 has already been evaluated")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    selected = freeze["selected"]
    checkpoint_path = ROOT / selected["checkpoint"]
    if _sha256(checkpoint_path) != selected["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash mismatch")
    device = torch.device(args.device)
    features = runtime._load(
        ROOT
        / "outputs/control_contracts/residential_cagn_features_v1"
        / args.split
        / "features.pt",
        device,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = ConstraintAwareGate(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    groups = runtime._predict(
        model,
        features,
        128,
    )["groups"]
    adjusted = runtime._fallback_rows(
        groups,
        alpha=float(selected["alpha"]),
        legal_threshold=float(selected["legal_fallback_threshold"]),
    )
    aggregate = runtime._metrics(adjusted, learned=True)
    selected_index = {
        int(group["group_index"]): int(
            np.argmax(np.where(group["hard"], group["scores"], -np.inf))
        )
        for group in adjusted
    }
    manifest = load_candidate_index(
        ROOT
        / "outputs/control_contracts/residential_cagn_v1"
        / args.split
        / "index.json",
        allow_test=True,
    )
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    cache_root = ROOT / "outputs/cache/cubicasa5k"
    cached_source = None
    cached_before = None
    cached_footprint = None
    cached_edits = {}
    downstream = defaultdict(list)
    outside_controls = []
    outside_controls_tolerant = []
    outside_labels = []
    outside_counts = defaultdict(int)
    for group_index, group in enumerate(manifest["groups"]):
        if group_index not in selected_index:
            continue
        record = _record(group["record"])
        if record.source != cached_source:
            cached_source = record.source
            cached_before = read_control_cache(Path(record.source))
            cached_footprint = load_cached_footprint(cache_root, record)
            cached_edits = {}
        edit = cached_edits.get(record.operation)
        if edit is None:
            edit = LOCAL_EDIT_OPERATIONS[record.operation](
                cached_before,
                taxonomy,
                seed=record.seed,
            )
            cached_edits[record.operation] = edit
        selected_local = selected_index[group_index]
        specs = tuple(CandidateSpec(**row) for row in group["candidates"])
        selected_specs = feature_builder._selected_specs(
            specs,
            int(feature_builder._NEGATIVE_PER_GROUP),
        )
        selected_spec = selected_specs[selected_local]
        positive_spec = selected_specs[0]
        selected_candidate = materialize_candidate(
            cached_before,
            edit.controls,
            record,
            selected_spec,
            taxonomy,
            cached_footprint,
        )
        positive_candidate = materialize_candidate(
            cached_before,
            edit.controls,
            record,
            positive_spec,
            taxonomy,
            cached_footprint,
        )
        selected_record = ActionRecord(
            name=record.name,
            split=record.split,
            source=record.source,
            operation=record.operation,
            seed=record.seed,
            bbox=selected_candidate.bbox,
            control_size=record.control_size,
            changed_pixels=selected_candidate.changed_pixels,
        )
        control_exact, control_tolerant, room_exact, counts = _outside_exact(
            cached_before,
            selected_candidate.controls,
            selected_record,
        )
        outside_controls.append(control_exact)
        outside_controls_tolerant.append(control_tolerant)
        outside_labels.append(room_exact)
        for key, value in counts.items():
            outside_counts[key] += value
        selected_support = selected_candidate.controls["room_logits"].sigmoid().amax(dim=0) > 0.5
        positive_support = positive_candidate.controls["room_logits"].sigmoid().amax(dim=0) > 0.5
        intersection = (selected_support & positive_support).sum()
        union = (selected_support | positive_support).sum()
        downstream["room_iou_to_positive"].append(
            float(intersection / max(int(union), 1))
        )
        downstream["room_mse_to_positive"].append(
            float(
                (
                    selected_candidate.controls["room_logits"].sigmoid()
                    - positive_candidate.controls["room_logits"].sigmoid()
                ).pow(2).mean()
            )
        )
        selected_graph = control_graph_state(selected_candidate.controls, taxonomy)
        positive_graph = control_graph_state(positive_candidate.controls, taxonomy)
        graph_metrics = _graph_metrics(
            {
                "program": selected_candidate.controls["program"],
                "adjacency": selected_graph.adjacency,
            },
            {
                "program": positive_candidate.controls["program"],
                "adjacency": positive_graph.adjacency,
            },
        )
        for key, value in graph_metrics.items():
            downstream[key].append(value)
    summary = {
        key: float(np.mean(values))
        for key, values in downstream.items()
    }
    report = {
        "schema_version": "residential-cagn-test503-final-v1",
        "status": "passed",
        "scope": f"residential_floorplan_only_{args.split}",
        "test_opened_once": args.split == "test",
        "validation_freeze": str(FREEZE.relative_to(ROOT)),
        "frozen_seed": selected["seed"],
        "frozen_checkpoint_sha256": selected["checkpoint_sha256"],
        "groups": aggregate["groups"],
        "selection": aggregate,
        "downstream": {
            **summary,
            "outside_control_exact": float(np.mean(outside_controls)),
            "outside_control_exact_1px": float(
                np.mean(outside_controls_tolerant)
            ),
            "outside_room_label_exact": float(np.mean(outside_labels)),
            "outside_change_pixels_by_control": dict(outside_counts),
        },
        "checks": {
            "hard_mask_exact": all(
                bool(group["invalid_probability_exact_zero"])
                for group in adjusted
            ),
            "selected_violation_at_most_0_005": (
                aggregate["selected_violation_rate"] <= 0.005
            ),
            "outside_control_exact_1px": all(outside_controls_tolerant),
            "outside_room_label_exact": all(outside_labels),
            "no_best_of_n": True,
        },
        "test_access_policy": (
            "Single frozen evaluation; this report must not be used for tuning."
        ),
    }
    if not all(report["checks"].values()):
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": report["checks"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
