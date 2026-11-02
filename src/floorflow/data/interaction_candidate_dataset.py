"""Two-step model-origin interaction corpus for constraint-aware control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.data.constraint_action_dataset import (
    ActionRecord,
    action_features,
    control_graph_state,
    hard_feasibility,
    hard_feasibility_baseline,
)
from floorflow.data.constraint_candidates import (
    VIOLATION_FAMILIES,
    MaterializedCandidate,
    build_candidate_specs,
    constraint_diagnostics,
    is_hard_legal,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.interaction_context import enrich_sequence_context
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.model_candidate_dataset import (
    control_quality,
    load_target_labels,
)
from floorflow.data.taxonomy import FloorplanTaxonomy

INTERACTION_SCHEMA = "model-origin-interaction-groups-v1"
CONTINUATION_ROLES = ("positive", "legal_alternative", "preserve")


def _candidate_record(
    record: ActionRecord,
    *,
    bbox: tuple[int, int, int, int],
    changed_pixels: int,
) -> ActionRecord:
    return ActionRecord(
        name=record.name,
        split=record.split,
        source=record.source,
        operation=record.operation,
        seed=record.seed,
        bbox=bbox,
        control_size=record.control_size,
        changed_pixels=changed_pixels,
    )


def _legal_alternative(
    before: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    record: ActionRecord,
    spec,
    seed: int,
    footprint,
    baseline,
) -> MaterializedCandidate | None:
    edit_function = LOCAL_EDIT_OPERATIONS[record.operation]
    for attempt in range(8):
        edit = edit_function(before, taxonomy, seed=seed + attempt * 31)
        if edit is None:
            continue
        candidate_record = _candidate_record(
            record,
            bbox=edit.bbox,
            changed_pixels=max(edit.changed_pixels, 1),
        )
        flags = hard_feasibility(
            before,
            edit.controls,
            candidate_record,
            taxonomy,
            footprint=footprint,
            baseline=baseline,
        )
        if not all(flags.values()):
            continue
        return MaterializedCandidate(
            controls=edit.controls,
            bbox=edit.bbox,
            changed_pixels=max(edit.changed_pixels, 1),
            violation_labels={family: False for family in VIOLATION_FAMILIES},
            transform="alternate_legal_target",
        )
    return None


def _candidate_row(
    before: dict[str, torch.Tensor],
    positive_after: dict[str, torch.Tensor],
    oracle_after: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint,
    taxonomy: FloorplanTaxonomy,
    record: ActionRecord,
    spec,
    materialized: MaterializedCandidate,
    baseline,
    *,
    path_utility_sum: float,
    step: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    candidate_record = _candidate_record(
        record,
        bbox=materialized.bbox,
        changed_pixels=materialized.changed_pixels,
    )
    flags = hard_feasibility(
        before,
        materialized.controls,
        candidate_record,
        taxonomy,
        footprint=footprint,
        baseline=baseline,
    )
    quality = control_quality(
        materialized.controls,
        oracle_after,
        target,
        taxonomy,
    )
    state_controls = dict(materialized.controls)
    state_controls["room_logits"] = F.interpolate(
        materialized.controls["room_logits"][None],
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )[0]
    state = control_graph_state(state_controls, taxonomy)
    utility = float(quality.utility)
    row = {
        "candidate_id": spec.candidate_id,
        "role": spec.role,
        "violation_family": spec.violation_family,
        "severity": spec.severity,
        "transform": materialized.transform,
        "candidate_seed": spec.seed,
        "bbox": list(materialized.bbox),
        "changed_pixels": materialized.changed_pixels,
        "hard_mask": flags,
        "hard_feasible": is_hard_legal(flags, materialized.violation_labels),
        "violation_labels": materialized.violation_labels,
        "quality": quality.to_dict(),
        "trajectory_utility": (
            path_utility_sum + utility
        ) / max(step + 1, 1),
        "node_features": state.node_features.tolist(),
        "adjacency": state.adjacency.tolist(),
        "global_features": state.global_features.tolist(),
        "action_features": action_features(
            before,
            materialized.controls,
            candidate_record,
        ).tolist(),
        "constraint_diagnostics": constraint_diagnostics(
            before,
            materialized.controls,
            candidate_record,
            taxonomy,
            footprint,
        ).tolist(),
    }
    return row, materialized.controls


def _build_candidates(
    before: dict[str, torch.Tensor],
    oracle_before: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint,
    taxonomy: FloorplanTaxonomy,
    *,
    name: str,
    split: str,
    source: str,
    operation: str,
    seed: int,
    parent_candidate_id: str | None,
    path_utility_sum: float,
    step: int,
) -> tuple[dict, dict[str, dict[str, torch.Tensor]]] | None:
    edit_function = LOCAL_EDIT_OPERATIONS[operation]
    edit = edit_function(before, taxonomy, seed=seed)
    oracle_edit = edit_function(oracle_before, taxonomy, seed=seed)
    if edit is None or oracle_edit is None:
        return None
    record = ActionRecord(
        name=name,
        split=split,
        source=source,
        operation=operation,
        seed=seed,
        bbox=edit.bbox,
        control_size=edit.control_size,
        changed_pixels=edit.changed_pixels,
    )
    baseline = hard_feasibility_baseline(before, taxonomy, footprint)
    specs = list(build_candidate_specs(record)[:2])
    negatives = build_candidate_specs(record)[2:]
    digest = int.from_bytes(
        hashlib.sha256(
            f"{split}:{name}:{operation}:{step}".encode()
        ).digest()[:8],
        "big",
    )
    offset = digest % len(negatives)
    specs.extend(
        negatives[(offset + index) % len(negatives)]
        for index in range(2)
    )
    rows = []
    continuation: dict[str, dict[str, torch.Tensor]] = {}
    for spec in specs:
        materialized = None
        if spec.role == "legal_alternative":
            materialized = _legal_alternative(
                before,
                taxonomy,
                record,
                spec,
                seed + 7919,
                footprint,
                baseline,
            )
        if materialized is None:
            materialized = materialize_candidate(
                before,
                edit.controls,
                record,
                spec,
                taxonomy,
                footprint,
            )
        row, controls = _candidate_row(
            before,
            edit.controls,
            oracle_edit.controls,
            target,
            footprint,
            taxonomy,
            record,
            spec,
            materialized,
            baseline,
            path_utility_sum=path_utility_sum,
            step=step,
        )
        rows.append(row)
        if spec.role in CONTINUATION_ROLES and row["hard_feasible"]:
            continuation[spec.candidate_id] = controls
    preserve_record = ActionRecord(
        name=name,
        split=split,
        source=source,
        operation=operation,
        seed=seed,
        bbox=(0, 0, 1, 1),
        control_size=record.control_size,
        changed_pixels=1,
    )
    preserve_spec = type(specs[0])(
        candidate_id=f"{name}:{operation}:s{step}:preserve",
        group_id=specs[0].group_id,
        role="preserve",
        violation_family=None,
        severity=0,
        transform="preserve",
        seed=seed,
        operation=operation,
    )
    preserve = MaterializedCandidate(
        controls=before,
        bbox=(0, 0, 1, 1),
        changed_pixels=1,
        violation_labels={family: False for family in VIOLATION_FAMILIES},
        transform="preserve",
    )
    row, controls = _candidate_row(
        before,
        before,
        oracle_edit.controls,
        target,
        footprint,
        taxonomy,
        preserve_record,
        preserve_spec,
        preserve,
        baseline,
        path_utility_sum=path_utility_sum,
        step=step,
    )
    rows.append(row)
    if row["hard_feasible"]:
        continuation[preserve_spec.candidate_id] = controls
    group_id = f"{split}:{name}:{operation}:s{step}"
    for row in rows:
        row["candidate_id"] = row["candidate_id"].replace(
            f"{split}:{name}:{operation}",
            group_id,
        )
    continuation = {
        candidate_id.replace(f"{split}:{name}:{operation}", group_id): controls
        for candidate_id, controls in continuation.items()
    }
    group = {
        "group_id": group_id,
        "sequence_id": f"{split}:{name}",
        "step": step,
        "parent_candidate_id": parent_candidate_id,
        "name": name,
        "split": split,
        "source": source,
        "operation": operation,
        "action_seed": seed,
        "predicted_control_used": True,
        "target_used_at_inference": False,
        "candidates": rows,
    }
    return group, continuation


def _build_branch(
    controls: dict[str, torch.Tensor],
    oracle_controls: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint,
    taxonomy: FloorplanTaxonomy,
    *,
    name: str,
    split: str,
    source: str,
    operations: tuple[str, ...],
    step: int,
    parent_candidate_id: str | None,
    path_utility_sum: float,
    seed_base: int,
) -> list[dict]:
    if step >= len(operations):
        return []
    seed = seed_base + step * 104729
    result = _build_candidates(
        controls,
        oracle_controls,
        target,
        footprint,
        taxonomy,
        name=name,
        split=split,
        source=source,
        operation=operations[step],
        seed=seed,
        parent_candidate_id=parent_candidate_id,
        path_utility_sum=path_utility_sum,
        step=step,
    )
    if result is None:
        return []
    group, continuation = result
    groups = [group]
    oracle_edit = LOCAL_EDIT_OPERATIONS[operations[step]](
        oracle_controls,
        taxonomy,
        seed=seed,
    )
    if oracle_edit is None:
        return groups
    for candidate in group["candidates"]:
        if candidate["candidate_id"] not in continuation:
            continue
        groups.extend(
            _build_branch(
                continuation[candidate["candidate_id"]],
                oracle_edit.controls,
                target,
                footprint,
                taxonomy,
                name=name,
                split=split,
                source=source,
                operations=operations,
                step=step + 1,
                parent_candidate_id=candidate["candidate_id"],
                path_utility_sum=(
                    path_utility_sum + candidate["quality"]["utility"]
                ),
                seed_base=seed_base,
            )
        )
    return groups


def build_interaction_groups(
    predicted_root: str | Path,
    split: str,
    taxonomy: FloorplanTaxonomy,
    *,
    target_cache_root: str | Path,
    oracle_control_root: str | Path,
    case_limit: int | None = None,
    steps: int = 2,
    seed: int = 7300,
) -> list[dict]:
    if split not in {"train", "val"}:
        raise ValueError("interaction corpus is restricted to train/val")
    if steps not in {2, 3}:
        raise ValueError("interaction depth must be two or three")
    predicted_root = Path(predicted_root)
    oracle_root = Path(oracle_control_root)
    paths = sorted((predicted_root / split).glob("*.npz"))
    if case_limit is not None:
        paths = paths[: int(case_limit)]
    groups = []
    operation_names = tuple(LOCAL_EDIT_OPERATIONS)
    for case_index, path in enumerate(paths):
        predicted = read_control_cache(path)
        oracle_path = oracle_root / split / path.name
        if not oracle_path.exists():
            raise FileNotFoundError(f"missing oracle intent controls: {oracle_path}")
        oracle = read_control_cache(oracle_path)
        target, footprint = load_target_labels(
            target_cache_root,
            split,
            path.stem,
        )
        selected_operations = []
        seed_base = seed + case_index * 17
        for offset in range(len(operation_names)):
            operation = operation_names[
                (case_index + offset) % len(operation_names)
            ]
            edit = LOCAL_EDIT_OPERATIONS[operation](
                predicted,
                taxonomy,
                seed=seed_base,
            )
            oracle_edit = LOCAL_EDIT_OPERATIONS[operation](
                oracle,
                taxonomy,
                seed=seed_base,
            )
            if edit is not None and oracle_edit is not None:
                selected_operations.append(operation)
            if len(selected_operations) == steps:
                break
        if len(selected_operations) < steps:
            continue
        groups.extend(
            _build_branch(
                predicted,
                oracle,
                target,
                footprint.numpy(),
                taxonomy,
                name=path.stem,
                split=split,
                source=str(path),
                operations=tuple(selected_operations),
                step=0,
                parent_candidate_id=None,
                path_utility_sum=0.0,
                seed_base=seed_base,
            )
        )
        if (case_index + 1) % 16 == 0:
            print(
                f"interaction candidates: {case_index + 1}/{len(paths)} cases",
                flush=True,
            )
    return groups


def write_interaction_index(
    path: str | Path,
    groups: list[dict],
    *,
    split: str,
    predicted_root: str,
    oracle_control_root: str,
) -> dict:
    sequences = {group["sequence_id"] for group in groups}
    step_counts = {
        str(step): sum(group["step"] == step for group in groups)
        for step in sorted({group["step"] for group in groups})
    }
    sequence_context = enrich_sequence_context(groups)
    role_counts = {
        role: sum(
            candidate["role"] == role
            for group in groups
            for candidate in group["candidates"]
        )
        for role in CONTINUATION_ROLES + ("near_violation",)
    }
    payload = {
        "schema_version": INTERACTION_SCHEMA,
        "status": "complete",
        "scope": "residential_floorplan_only",
        "split": split,
        "candidate_origin": "model_predicted_controls_depth_two_recovery_tree",
        "predicted_control_root": predicted_root,
        "oracle_control_root_for_labels_only": oracle_control_root,
        "target_used_at_inference": False,
        "sequence_context": sequence_context,
        "counts": {
            "sequences": len(sequences),
            "groups": len(groups),
            "candidates": sum(len(group["candidates"]) for group in groups),
            "steps": step_counts,
            "roles": role_counts,
        },
        "groups": groups,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def index_sequences(groups: list[dict]) -> dict[str, dict]:
    sequences = {}
    for group in groups:
        sequence = sequences.setdefault(
            group["sequence_id"],
            {
                "root": None,
                "groups": {},
                "children": {},
            },
        )
        sequence["groups"][group["step"]] = group
        if group["step"] == 0:
            sequence["root"] = group
        if group["parent_candidate_id"] is not None:
            sequence["children"][group["parent_candidate_id"]] = group
    return sequences


__all__ = [
    "CONTINUATION_ROLES",
    "INTERACTION_SCHEMA",
    "build_interaction_groups",
    "index_sequences",
    "write_interaction_index",
]
