"""Model-origin candidate records and conservative control-space scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.constraint_action_dataset import (
    ACTION_NAMES,
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
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy

MODEL_CANDIDATE_SCHEMA = "model-origin-candidate-groups-v1"


@dataclass(frozen=True)
class ControlQuality:
    room_intent_iou: float
    wall_intent_iou: float
    opening_intent_iou: float
    furniture_intent_iou: float
    room_mse: float
    target_room_iou: float

    @property
    def utility(self) -> float:
        return float(
            0.55 * self.room_intent_iou
            + 0.15 * self.wall_intent_iou
            + 0.1 * self.opening_intent_iou
            + 0.1 * self.furniture_intent_iou
            + 0.1 * self.target_room_iou
            - 0.05 * min(self.room_mse, 1.0)
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "room_intent_iou": self.room_intent_iou,
            "wall_intent_iou": self.wall_intent_iou,
            "opening_intent_iou": self.opening_intent_iou,
            "furniture_intent_iou": self.furniture_intent_iou,
            "room_mse": self.room_mse,
            "target_room_iou": self.target_room_iou,
            "utility": self.utility,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_cache_path(cache_root: str | Path, split: str, name: str) -> Path:
    parts = name.split("__")
    if len(parts) < 2:
        raise ValueError(f"invalid CubiCasa sample name: {name}")
    return Path(cache_root) / split / parts[0] / f"{parts[1]}.npz"


def load_target_labels(
    cache_root: str | Path,
    split: str,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = target_cache_path(cache_root, split, name)
    with np.load(path) as payload:
        return (
            torch.from_numpy(payload["target"].astype(np.int64)),
            torch.from_numpy(payload["footprint"].astype(bool)),
        )


def _binary_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    union = (left.bool() | right.bool()).sum()
    if int(union) == 0:
        return 1.0
    return float((left.bool() & right.bool()).sum() / union)


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == size:
        return mask
    return F.interpolate(
        mask[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0] > 0.5


def control_quality(
    candidate: dict[str, torch.Tensor],
    intent: dict[str, torch.Tensor],
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> ControlQuality:
    """Score candidate controls against an oracle intent and the GT room mask."""

    room_count = len(taxonomy.room_ids)
    room_size = (64, 64)
    candidate_room_probability = F.interpolate(
        candidate["room_logits"].sigmoid()[None],
        size=room_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    candidate_room = _resize_mask(
        candidate["room_logits"].amax(dim=0) > 0.0,
        room_size,
    )
    intent_room = _resize_mask(
        intent["room_logits"].amax(dim=0) > 0.0,
        room_size,
    )
    target_room = torch.isin(target, torch.tensor(taxonomy.room_ids)).float()[None, None]
    target_room = F.interpolate(
        target_room,
        size=room_size,
        mode="nearest",
    )[0, 0] > 0.5
    structure_size = tuple(candidate["structure"].shape[-2:])
    candidate_wall = candidate["structure"][room_count] > 0.5
    intent_wall = intent["structure"][room_count] > 0.5
    candidate_opening = candidate["structure"][room_count + 1] > 0.5
    intent_opening = intent["structure"][room_count + 1] > 0.5
    furniture_size = tuple(candidate["furniture"].shape[-2:])
    candidate_furniture = candidate["furniture"][room_count:-1].amax(dim=0) > 0.5
    intent_furniture = intent["furniture"][room_count:-1].amax(dim=0) > 0.5
    room_target_logits = F.interpolate(
        (target_room.float() * 2.0 - 1.0)[None, None],
        size=room_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return ControlQuality(
        room_intent_iou=_binary_iou(candidate_room, intent_room),
        wall_intent_iou=_binary_iou(
            _resize_mask(candidate_wall, structure_size),
            _resize_mask(intent_wall, structure_size),
        ),
        opening_intent_iou=_binary_iou(candidate_opening, intent_opening),
        furniture_intent_iou=_binary_iou(
            _resize_mask(candidate_furniture, furniture_size),
            _resize_mask(intent_furniture, furniture_size),
        ),
        room_mse=float(
            (
                candidate_room_probability.amax(dim=0)
                - room_target_logits.sigmoid()
            )
            .pow(2)
            .mean()
        ),
        target_room_iou=_binary_iou(candidate_room, target_room),
    )


def build_model_candidate_groups(
    predicted_root: str | Path,
    split: str,
    taxonomy: FloorplanTaxonomy,
    *,
    target_cache_root: str | Path,
    oracle_control_root: str | Path,
    target_footprint_root: str | Path,
    case_limit: int | None = None,
    negative_per_group: int = 2,
    seed: int = 6100,
) -> list[dict]:
    if split not in {"train", "val"}:
        raise ValueError("model candidate construction is restricted to train/val")
    predicted_root = Path(predicted_root)
    oracle_root = Path(oracle_control_root)
    groups = []
    paths = sorted((predicted_root / split).glob("*.npz"))
    if case_limit is not None:
        paths = paths[: int(case_limit)]
    for index, path in enumerate(paths):
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
        baseline = hard_feasibility_baseline(predicted, taxonomy, footprint.numpy())
        for operation_index, (operation, edit_function) in enumerate(
            LOCAL_EDIT_OPERATIONS.items()
        ):
            action_seed = seed + index * 11 + operation_index
            predicted_edit = edit_function(
                predicted,
                taxonomy,
                seed=action_seed,
            )
            oracle_edit = edit_function(
                oracle,
                taxonomy,
                seed=action_seed,
            )
            if predicted_edit is None or oracle_edit is None:
                continue
            record = ActionRecord(
                name=path.stem,
                split=split,
                source=str(path),
                operation=operation,
                seed=action_seed,
                bbox=predicted_edit.bbox,
                control_size=predicted_edit.control_size,
                changed_pixels=predicted_edit.changed_pixels,
            )
            specs = build_candidate_specs(record)
            selected_specs = list(specs[:2])
            negatives = [
                spec
                for spec in specs
                if spec.violation_family is not None
            ]
            digest = hashlib.sha256(
                f"{split}:{path.stem}:{operation}".encode()
            ).digest()
            offset = int.from_bytes(digest[:4], "big") % len(negatives)
            selected_specs.extend(
                negatives[
                    (offset + index) % len(negatives)
                ]
                for index in range(min(int(negative_per_group), len(negatives)))
            )
            candidates = []
            for spec in selected_specs:
                alternative_edit = None
                if spec.role == "legal_alternative":
                    for attempt in range(8):
                        alternative_edit = edit_function(
                            predicted,
                            taxonomy,
                            seed=spec.seed + attempt * 31,
                        )
                        if alternative_edit is None:
                            continue
                        alternative_record = ActionRecord(
                            name=record.name,
                            split=record.split,
                            source=record.source,
                            operation=operation,
                            seed=action_seed,
                            bbox=alternative_edit.bbox,
                            control_size=alternative_edit.control_size,
                            changed_pixels=alternative_edit.changed_pixels,
                        )
                        alternative_flags = hard_feasibility(
                            predicted,
                            alternative_edit.controls,
                            alternative_record,
                            taxonomy,
                            footprint=footprint.numpy(),
                            baseline=baseline,
                        )
                        if all(alternative_flags.values()):
                            break
                        alternative_edit = None
                if alternative_edit is not None:
                    materialized = MaterializedCandidate(
                        controls=alternative_edit.controls,
                        bbox=alternative_edit.bbox,
                        changed_pixels=max(alternative_edit.changed_pixels, 1),
                        violation_labels={
                            family: False for family in VIOLATION_FAMILIES
                        },
                        transform="alternate_legal_target",
                    )
                    candidate_transform = "alternate_legal_target"
                else:
                    materialized = materialize_candidate(
                        predicted,
                        predicted_edit.controls,
                        record,
                        spec,
                        taxonomy,
                        footprint.numpy(),
                    )
                    candidate_transform = materialized.transform
                candidate_record = ActionRecord(
                    name=record.name,
                    split=record.split,
                    source=record.source,
                    operation=operation,
                    seed=action_seed,
                    bbox=materialized.bbox,
                    control_size=record.control_size,
                    changed_pixels=materialized.changed_pixels,
                )
                flags = hard_feasibility(
                    predicted,
                    materialized.controls,
                    candidate_record,
                    taxonomy,
                    footprint=footprint.numpy(),
                    baseline=baseline,
                )
                legal = is_hard_legal(flags, materialized.violation_labels)
                quality = control_quality(
                    materialized.controls,
                    oracle_edit.controls,
                    target,
                    taxonomy,
                )
                diagnostics = constraint_diagnostics(
                    predicted,
                    materialized.controls,
                    candidate_record,
                    taxonomy,
                    footprint.numpy(),
                )
                state_controls = dict(materialized.controls)
                state_controls["room_logits"] = F.interpolate(
                    materialized.controls["room_logits"][None],
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                state = control_graph_state(state_controls, taxonomy)
                candidates.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "role": spec.role,
                        "violation_family": spec.violation_family,
                        "severity": spec.severity,
                        "transform": candidate_transform,
                        "candidate_seed": spec.seed,
                        "bbox": list(materialized.bbox),
                        "changed_pixels": materialized.changed_pixels,
                        "hard_mask": flags,
                        "hard_feasible": legal,
                        "violation_labels": materialized.violation_labels,
                        "quality": quality.to_dict(),
                        "node_features": state.node_features.tolist(),
                        "adjacency": state.adjacency.tolist(),
                        "global_features": state.global_features.tolist(),
                        "action_features": action_features(
                            predicted,
                            materialized.controls,
                            candidate_record,
                        ).tolist(),
                        "constraint_diagnostics": diagnostics.tolist(),
                    }
                )
            preserve_record = ActionRecord(
                name=record.name,
                split=record.split,
                source=record.source,
                operation=operation,
                seed=action_seed,
                bbox=(0, 0, 1, 1),
                control_size=record.control_size,
                changed_pixels=1,
            )
            preserve_flags = hard_feasibility(
                predicted,
                predicted,
                preserve_record,
                taxonomy,
                footprint=footprint.numpy(),
                baseline=baseline,
            )
            preserve_state = control_graph_state(predicted, taxonomy)
            preserve_quality = control_quality(
                predicted,
                oracle_edit.controls,
                target,
                taxonomy,
            )
            candidates.append(
                {
                    "candidate_id": f"{record.name}:{operation}:preserve",
                    "role": "preserve",
                    "violation_family": None,
                    "severity": 0,
                    "transform": "preserve",
                    "candidate_seed": action_seed,
                    "bbox": [0, 0, 1, 1],
                    "changed_pixels": 1,
                    "hard_mask": preserve_flags,
                    "hard_feasible": is_hard_legal(
                        preserve_flags,
                        {family: False for family in VIOLATION_FAMILIES},
                    ),
                    "violation_labels": {
                        family: False for family in VIOLATION_FAMILIES
                    },
                    "quality": preserve_quality.to_dict(),
                    "node_features": preserve_state.node_features.tolist(),
                    "adjacency": preserve_state.adjacency.tolist(),
                    "global_features": preserve_state.global_features.tolist(),
                    "action_features": action_features(
                        predicted,
                        predicted,
                        preserve_record,
                    ).tolist(),
                    "constraint_diagnostics": constraint_diagnostics(
                        predicted,
                        predicted,
                        preserve_record,
                        taxonomy,
                        footprint.numpy(),
                    ).tolist(),
                }
            )
            groups.append(
                {
                    "group_id": f"{split}:{path.stem}:{operation}",
                    "name": path.stem,
                    "split": split,
                    "source": str(path),
                    "source_sha256": sha256_file(path),
                    "oracle_intent_sha256": sha256_file(oracle_path),
                    "target_cache": str(
                        target_cache_path(
                            target_footprint_root,
                            split,
                            path.stem,
                        )
                    ),
                    "operation": operation,
                    "action_seed": action_seed,
                    "predicted_control_used": True,
                    "target_used_at_inference": False,
                    "candidates": candidates,
                }
            )
        if (index + 1) % 16 == 0:
            print(
                f"model candidates: {index + 1}/{len(paths)} cases",
                flush=True,
            )
    return groups


def write_model_candidate_index(
    path: str | Path,
    groups: list[dict],
    *,
    split: str,
    predicted_root: str,
    oracle_control_root: str,
) -> dict:
    role_counts = {
        role: 0
        for role in ("positive", "legal_alternative", "near_violation", "preserve")
    }
    family_counts = dict.fromkeys(VIOLATION_FAMILIES, 0)
    operation_counts = dict.fromkeys(ACTION_NAMES, 0)
    for group in groups:
        operation_counts[group["operation"]] += 1
        for candidate in group["candidates"]:
            role_counts[candidate["role"]] += 1
            if candidate["violation_family"] is not None:
                family_counts[candidate["violation_family"]] += 1
    payload = {
        "schema_version": MODEL_CANDIDATE_SCHEMA,
        "status": "complete",
        "scope": "residential_floorplan_only",
        "split": split,
        "candidate_origin": "real_model_predicted_controls_plus_deterministic_actions",
        "target_used_at_inference": False,
        "predicted_control_root": predicted_root,
        "oracle_control_root_for_labels_only": oracle_control_root,
        "counts": {
            "groups": len(groups),
            "candidates": sum(len(group["candidates"]) for group in groups),
            "roles": role_counts,
            "families": family_counts,
            "operations": operation_counts,
        },
        "groups": groups,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


class ModelCandidateDataset(torch.utils.data.Dataset):
    """Lazy access to a frozen model-origin candidate manifest."""

    def __init__(
        self,
        index_path: str | Path,
        taxonomy: FloorplanTaxonomy,
        *,
        split: str,
    ) -> None:
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != MODEL_CANDIDATE_SCHEMA:
            raise ValueError("model candidate schema mismatch")
        if split not in {"train", "val"} or payload.get("split") != split:
            raise ValueError("model candidate dataset is train/val only")
        self.taxonomy = taxonomy
        self.rows = [
            (group, candidate)
            for group in payload["groups"]
            for candidate in group["candidates"]
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        group, candidate = self.rows[index]
        return {
            "group_id": group["group_id"],
            "operation": group["operation"],
            "role": candidate["role"],
            "node_features": torch.tensor(candidate["node_features"]),
            "adjacency": torch.tensor(candidate["adjacency"]),
            "global_features": torch.tensor(candidate["global_features"]),
            "action_features": torch.tensor(candidate["action_features"]),
            "constraint_diagnostics": torch.tensor(
                candidate["constraint_diagnostics"]
            ),
            "hard_mask": torch.tensor(
                list(candidate["hard_mask"].values()),
                dtype=torch.bool,
            ),
            "hard_feasible": torch.tensor(candidate["hard_feasible"]),
            "violation_labels": torch.tensor(
                [
                    candidate["violation_labels"][family]
                    for family in VIOLATION_FAMILIES
                ],
                dtype=torch.float32,
            ),
            "utility": torch.tensor(candidate["quality"]["utility"]),
            "quality": candidate["quality"],
        }


__all__ = [
    "MODEL_CANDIDATE_SCHEMA",
    "ControlQuality",
    "ModelCandidateDataset",
    "build_model_candidate_groups",
    "control_quality",
    "load_target_labels",
    "sha256_file",
    "target_cache_path",
    "write_model_candidate_index",
]
