"""Candidate-group manifest and lazy dataset for constraint-aware gating."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.data.constraint_action_dataset import (
    ACTION_NAMES,
    ActionRecord,
    action_features,
    control_graph_state,
    hard_feasibility,
    load_cached_footprint,
)
from floorflow.data.constraint_candidates import (
    VIOLATION_FAMILIES,
    CandidateSpec,
    build_candidate_specs,
    is_hard_legal,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy

SCHEMA_VERSION = "residential-constraint-candidate-groups-v1"


def _record_payload(record: ActionRecord) -> dict:
    return {
        **record.__dict__,
        "bbox": list(record.bbox),
        "control_size": list(record.control_size),
    }


def _record_from_payload(row: dict) -> ActionRecord:
    return ActionRecord(
        **{
            **row,
            "bbox": tuple(row["bbox"]),
            "control_size": tuple(row["control_size"]),
        }
    )


def write_candidate_index(
    path: str | Path,
    records: list[ActionRecord],
    *,
    split: str,
    control_root: str,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = []
    role_counts = {"positive": 0, "legal_alternative": 0, "near_violation": 0}
    family_counts = dict.fromkeys(VIOLATION_FAMILIES, 0)
    severity_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
    for record in records:
        specs = build_candidate_specs(record)
        groups.append(
            {
                "group_id": specs[0].group_id,
                "record": _record_payload(record),
                "candidates": [
                    {
                        **spec.__dict__,
                    }
                    for spec in specs
                ],
            }
        )
        for spec in specs:
            role_counts[spec.role] += 1
            severity_counts[str(spec.severity)] += 1
            if spec.violation_family is not None:
                family_counts[spec.violation_family] += 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "residential_floorplan_only",
        "split": split,
        "control_root": control_root,
        "action_vocabulary": list(ACTION_NAMES),
        "violation_families": list(VIOLATION_FAMILIES),
        "candidate_policy": {
            "positive": "the reproducible oracle local edit",
            "legal_alternative": "argmax-preserving room-logit reweight inside the edit box",
            "near_violation": "one deterministic typed perturbation per family",
            "severity": "1 + (base_seed + family_index) mod 3",
        },
        "counts": {
            "groups": len(groups),
            "candidates": sum(len(group["candidates"]) for group in groups),
            "roles": role_counts,
            "families": family_counts,
            "severity": severity_counts,
            "operations": {
                operation: sum(record.operation == operation for record in records)
                for operation in ACTION_NAMES
            },
        },
        "groups": groups,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_candidate_index(
    path: str | Path,
    *,
    allow_test: bool = False,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate index schema mismatch")
    allowed = {"train", "val"} | ({"test"} if allow_test else set())
    if payload.get("split") not in allowed:
        raise ValueError("candidate index test access is not enabled")
    return payload


class CandidateActionDataset(torch.utils.data.Dataset):
    """Lazily materialize frozen positive, legal, and violation candidates."""

    def __init__(
        self,
        index_path: str | Path,
        taxonomy: FloorplanTaxonomy,
        cache_root: str | Path = "outputs/cache/cubicasa5k",
    ) -> None:
        payload = load_candidate_index(index_path)
        self.taxonomy = taxonomy
        self.cache_root = Path(cache_root)
        self.items: list[tuple[ActionRecord, CandidateSpec]] = []
        for group in payload["groups"]:
            record = _record_from_payload(group["record"])
            for row in group["candidates"]:
                self.items.append((record, CandidateSpec(**row)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        record, spec = self.items[index]
        before = read_control_cache(Path(record.source))
        edit = LOCAL_EDIT_OPERATIONS[record.operation](
            before,
            self.taxonomy,
            seed=record.seed,
        )
        if edit is None:
            raise RuntimeError(f"saved edit no longer reproduces: {record}")
        footprint = load_cached_footprint(self.cache_root, record)
        materialized = materialize_candidate(
            before,
            edit.controls,
            record,
            spec,
            self.taxonomy,
            footprint,
        )
        candidate_record = ActionRecord(
            name=record.name,
            split=record.split,
            source=record.source,
            operation=record.operation,
            seed=record.seed,
            bbox=materialized.bbox,
            control_size=record.control_size,
            changed_pixels=materialized.changed_pixels,
        )
        before_state = control_graph_state(before, self.taxonomy)
        after_state = control_graph_state(materialized.controls, self.taxonomy)
        flags = hard_feasibility(
            before,
            materialized.controls,
            candidate_record,
            self.taxonomy,
            footprint=footprint,
        )
        violation = torch.tensor(
            [
                materialized.violation_labels[family]
                for family in VIOLATION_FAMILIES
            ],
            dtype=torch.float32,
        )
        return {
            "candidate_id": spec.candidate_id,
            "group_id": spec.group_id,
            "name": record.name,
            "operation": record.operation,
            "action_index": ACTION_NAMES.index(record.operation),
            "role": spec.role,
            "violation_family": spec.violation_family or "none",
            "severity": spec.severity,
            "before_node_features": before_state.node_features,
            "before_adjacency": before_state.adjacency,
            "before_global_features": before_state.global_features,
            "after_node_features": after_state.node_features,
            "after_adjacency": after_state.adjacency,
            "after_global_features": after_state.global_features,
            "action_features": action_features(
                before,
                materialized.controls,
                candidate_record,
            ),
            "hard_feasible": torch.tensor(
                is_hard_legal(flags, materialized.violation_labels),
                dtype=torch.float32,
            ),
            "hard_mask": {
                key: torch.tensor(value) for key, value in flags.items()
            },
            "violation_labels": violation,
            "changed_pixels": torch.tensor(materialized.changed_pixels),
        }


__all__ = [
    "SCHEMA_VERSION",
    "CandidateActionDataset",
    "load_candidate_index",
    "write_candidate_index",
]
