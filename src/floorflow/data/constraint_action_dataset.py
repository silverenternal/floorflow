"""Train-only graph/action dataset for constraint-aware gating."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy

ACTION_NAMES = tuple(LOCAL_EDIT_OPERATIONS)
HARD_MASK_FIELDS = (
    "arrays_finite",
    "program_nonempty",
    "room_support_nonempty",
    "change_inside_edit_region",
    "room_within_footprint",
    "structure_within_footprint",
    "openings_near_wall",
    "furniture_within_footprint",
)
CONTROL_KEYS = ("room_logits", "structure", "furniture", "support")


@dataclass(frozen=True)
class ActionRecord:
    name: str
    split: str
    source: str
    operation: str
    seed: int
    bbox: tuple[int, int, int, int]
    control_size: tuple[int, int]
    changed_pixels: int


@dataclass(frozen=True)
class GraphState:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    global_features: torch.Tensor


@dataclass(frozen=True)
class HardFeasibilityBaseline:
    footprint: torch.Tensor
    room_outside_fraction: float
    wall_outside_fraction: float
    opening_far_fraction: float
    furniture_outside_fraction: float


def _control_diff(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    return {
        key: (after[key].float() - before[key].float()).abs()
        for key in CONTROL_KEYS
    }


def control_graph_state(
    controls: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
) -> GraphState:
    """Encode a control bundle as typed room nodes plus global constraint features."""

    room_logits = controls["room_logits"].float()
    room_count = len(taxonomy.room_ids)
    if room_logits.shape[0] != room_count:
        raise ValueError("room logits must have one channel per room class")
    probability = room_logits.sigmoid()
    occupancy = probability.amax(dim=0) > 0.5
    class_map = probability.argmax(dim=0)
    mass = probability.mean(dim=(1, 2))
    confidence = probability.flatten(1).amax(dim=1)
    entropy = -(
        probability.clamp_min(1.0e-6) * probability.clamp_min(1.0e-6).log()
    ).mean(dim=(1, 2)) / np.log(max(room_count, 2))
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, room_logits.shape[-2]),
        torch.linspace(0, 1, room_logits.shape[-1]),
        indexing="ij",
    )
    denom = probability.sum(dim=(1, 2)).clamp_min(1.0e-6)
    center_x = (probability * xs).sum(dim=(1, 2)) / denom
    center_y = (probability * ys).sum(dim=(1, 2)) / denom
    component_count = torch.tensor(
        [
            ndimage.label(
                (class_map == index).cpu().numpy() & occupancy.cpu().numpy()
            )[1]
            for index in range(room_count)
        ],
        dtype=torch.float32,
    )
    program = controls["program"].float()
    adjacency = _room_adjacency(class_map, room_count)
    degree = (adjacency > 0).float().sum(dim=1)
    node_features = torch.stack(
        [
            program,
            mass,
            confidence,
            entropy,
            center_x,
            center_y,
            component_count / 4.0,
            degree / max(room_count - 1, 1),
        ],
        dim=-1,
    )
    structure = controls["structure"].float()
    furniture = controls["furniture"].float()
    support = controls["support"].float()
    wall = structure[room_count]
    openings = structure[room_count + 1]
    mass_distribution = mass / mass.sum().clamp_min(1.0e-6)
    room_type_entropy = -(
        mass_distribution.clamp_min(1.0e-6)
        * mass_distribution.clamp_min(1.0e-6).log()
    ).sum()
    global_features = torch.stack(
        [
            program.sum() / room_count,
            occupancy.float().mean(),
            wall.mean(),
            openings.mean(),
            furniture.mean(),
            support.mean(),
            adjacency.mean(),
            room_type_entropy,
        ]
    )
    return GraphState(node_features, adjacency, global_features)


def _room_adjacency(class_map: torch.Tensor, room_count: int) -> torch.Tensor:
    adjacency = torch.zeros(room_count, room_count, dtype=torch.float32)
    masks = [class_map == index for index in range(room_count)]
    for left, left_mask in enumerate(masks):
        dilated = F.max_pool2d(
            left_mask.float()[None, None],
            3,
            stride=1,
            padding=1,
        )[0, 0] > 0
        for right, right_mask in enumerate(masks):
            if left == right or not left_mask.any() or not right_mask.any():
                continue
            contact = int((dilated & right_mask).sum())
            adjacency[left, right] = contact / max(
                int(left_mask.sum()),
                1,
            )
    adjacency.fill_diagonal_(1.0)
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


def action_features(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    record: ActionRecord,
) -> torch.Tensor:
    operation_index = ACTION_NAMES.index(record.operation)
    one_hot = F.one_hot(
        torch.tensor(operation_index),
        len(ACTION_NAMES),
    ).float()
    width, height = record.control_size
    x, y, box_width, box_height = record.bbox
    differences = _control_diff(before, after)
    total_elements = sum(value.numel() for value in differences.values())
    delta = sum(value.sum() for value in differences.values()) / max(
        total_elements,
        1,
    )
    changed = [
        float((value > 0).float().mean())
        for value in differences.values()
    ]
    geometry = torch.tensor(
        [
            x / max(width, 1),
            y / max(height, 1),
            box_width / max(width, 1),
            box_height / max(height, 1),
            record.changed_pixels / max(width * height, 1),
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [
            one_hot,
            torch.tensor(changed, dtype=torch.float32),
            geometry,
            delta.reshape(1),
        ]
    )


def hard_feasibility(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    record: ActionRecord,
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None = None,
    baseline: HardFeasibilityBaseline | None = None,
) -> dict[str, bool]:
    finite = all(torch.isfinite(after[key]).all() for key in CONTROL_KEYS)
    program_count = int(after["program"].sum())
    room_support = bool(after["room_logits"].sigmoid().amax(dim=0).gt(0.5).any())
    width, height = record.control_size
    changed_inside = True
    for key in CONTROL_KEYS:
        delta = (after[key].float() - before[key].float()).abs()
        reduced = delta
        while reduced.ndim > 2:
            reduced = reduced.amax(dim=0)
        native_height, native_width = reduced.shape
        x, y, box_width, box_height = record.bbox
        native_x = round(x * native_width / max(width, 1))
        native_y = round(y * native_height / max(height, 1))
        native_box_width = max(
            1,
            round(box_width * native_width / max(width, 1)),
        )
        native_box_height = max(
            1,
            round(box_height * native_height / max(height, 1)),
        )
        left = max(native_x - 1, 0)
        top = max(native_y - 1, 0)
        right = min(native_x + native_box_width + 1, native_width)
        bottom = min(native_y + native_box_height + 1, native_height)
        changed = reduced > 0
        if left < right and top < bottom:
            changed_inside &= not bool(changed[:top].any())
            changed_inside &= not bool(changed[bottom:].any())
            changed_inside &= not bool(changed[:, :left].any())
            changed_inside &= not bool(changed[:, right:].any())
        else:
            changed_inside = False
    flags = {
        "arrays_finite": finite,
        "program_nonempty": 0 < program_count <= len(taxonomy.room_ids),
        "room_support_nonempty": room_support,
        "change_inside_edit_region": changed_inside,
    }
    if footprint is None:
        flags.update(
            {
                "room_within_footprint": True,
                "structure_within_footprint": True,
                "openings_near_wall": True,
                "furniture_within_footprint": True,
            }
        )
        return flags
    if baseline is None:
        baseline = hard_feasibility_baseline(
            before,
            taxonomy,
            footprint,
        )
    footprint_tensor = baseline.footprint
    size = tuple(footprint_tensor.shape)
    room_count = len(taxonomy.room_ids)
    room_mask = F.interpolate(
        after["room_logits"].sigmoid().amax(dim=0)[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    structure = after["structure"].float()
    wall = F.interpolate(
        structure[room_count][None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    opening = F.interpolate(
        structure[room_count + 1][None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    furniture = F.interpolate(
        after["furniture"][room_count:-1].amax(dim=0)[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    outside = ~footprint_tensor
    room_outside_after = float((room_mask & outside).sum()) / max(
        float(room_mask.sum()),
        1.0,
    )
    wall_outside_after = float((wall & outside).sum()) / max(
        float(wall.sum()),
        1.0,
    )
    flags["room_within_footprint"] = (
        room_outside_after <= baseline.room_outside_fraction + 0.01
    )
    flags["structure_within_footprint"] = (
        wall_outside_after <= baseline.wall_outside_fraction + 0.01
    )
    if opening.any():
        near_wall = F.max_pool2d(
            wall.float()[None, None],
            7,
            stride=1,
            padding=3,
        )[0, 0] > 0.5
        opening_far_after = float((opening & ~near_wall).sum()) / max(
            float(opening.sum()),
            1.0,
        )
        flags["openings_near_wall"] = (
            opening_far_after <= baseline.opening_far_fraction + 0.05
        )
    else:
        flags["openings_near_wall"] = True
    furniture_outside_after = float((furniture & outside).sum()) / max(
        float(furniture.sum()),
        1.0,
    )
    flags["furniture_within_footprint"] = (
        furniture_outside_after <= baseline.furniture_outside_fraction + 0.01
    )
    return flags


def hard_feasibility_baseline(
    before: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    footprint: np.ndarray | None,
    *,
    size: tuple[int, int] = (128, 128),
) -> HardFeasibilityBaseline:
    """Cache the before-state footprint fractions used by every candidate."""

    footprint_tensor = _resize_footprint(footprint, size)
    outside = ~footprint_tensor
    room_count = len(taxonomy.room_ids)
    room = F.interpolate(
        before["room_logits"].sigmoid().amax(dim=0)[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    structure = before["structure"].float()
    wall = F.interpolate(
        structure[room_count][None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    opening = F.interpolate(
        structure[room_count + 1][None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5
    near_wall = F.max_pool2d(
        wall.float()[None, None],
        7,
        stride=1,
        padding=3,
    )[0, 0] > 0.5
    furniture = F.interpolate(
        before["furniture"][room_count:-1].amax(dim=0)[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5

    def fraction(mask: torch.Tensor, reference: torch.Tensor) -> float:
        return float((mask & reference).sum()) / max(float(mask.sum()), 1.0)

    return HardFeasibilityBaseline(
        footprint=footprint_tensor,
        room_outside_fraction=fraction(room, outside),
        wall_outside_fraction=fraction(wall, outside),
        opening_far_fraction=fraction(opening, ~near_wall),
        furniture_outside_fraction=fraction(furniture, outside),
    )


def _resize_footprint(
    footprint: np.ndarray | None,
    size: tuple[int, int],
) -> torch.Tensor:
    if footprint is None:
        return torch.ones(size, dtype=torch.bool)
    tensor = torch.from_numpy(np.asarray(footprint, dtype=bool))
    if tuple(tensor.shape) == size:
        return tensor
    return F.interpolate(
        tensor.float()[None, None],
        size=size,
        mode="nearest",
    )[0, 0] > 0.5


def load_cached_footprint(cache_root: str | Path, record: ActionRecord) -> np.ndarray:
    root = Path(cache_root)
    path = root / record.split / f"{record.name}.npz"
    if not path.exists():
        parts = record.name.split("__")
        if len(parts) < 2:
            raise FileNotFoundError(f"cannot resolve footprint cache for {record.name}")
        path = root / record.split / parts[0] / f"{parts[1]}.npz"
    with np.load(path) as payload:
        return payload["footprint"].astype(bool, copy=True)


def build_action_records(
    control_root: str | Path,
    split: str,
    taxonomy: FloorplanTaxonomy,
    *,
    seed: int = 2300,
    case_limit: int | None = None,
    allow_test: bool = False,
) -> list[ActionRecord]:
    allowed = {"train", "val"} | ({"test"} if allow_test else set())
    if split not in allowed:
        raise ValueError("test access requires explicit allow_test=True")
    root = Path(control_root) / split
    records = []
    paths = sorted(root.glob("*.npz"))
    if case_limit is not None:
        paths = paths[: int(case_limit)]
    for index, path in enumerate(paths):
        controls = read_control_cache(path)
        for operation_index, (operation, edit_function) in enumerate(
            LOCAL_EDIT_OPERATIONS.items()
        ):
            edit = edit_function(
                controls,
                taxonomy,
                seed=seed + index * 11 + operation_index,
            )
            if edit is None:
                continue
            records.append(
                ActionRecord(
                    name=path.stem,
                    split=split,
                    source=str(path),
                    operation=operation,
                    seed=seed + index * 11 + operation_index,
                    bbox=edit.bbox,
                    control_size=edit.control_size,
                    changed_pixels=edit.changed_pixels,
                )
            )
    return records


def write_action_index(
    path: str | Path,
    records: list[ActionRecord],
    *,
    split: str,
    control_root: str,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        operation: sum(record.operation == operation for record in records)
        for operation in ACTION_NAMES
    }
    payload = {
        "status": "complete",
        "scope": "residential_floorplan_only",
        "split": split,
        "control_root": control_root,
        "seed_policy": "2300 + case_index * 11 + action_index",
        "actions": list(ACTION_NAMES),
        "hard_mask_fields": list(HARD_MASK_FIELDS),
        "counts": counts,
        "cases_with_any_action": len({record.name for record in records}),
        "rows": [
            {
                **record.__dict__,
                "bbox": list(record.bbox),
                "control_size": list(record.control_size),
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class ConstraintActionDataset(torch.utils.data.Dataset):
    """Lazy candidate-action dataset backed by a frozen manifest."""

    def __init__(
        self,
        index_path: str | Path,
        taxonomy: FloorplanTaxonomy,
        cache_root: str | Path = "outputs/cache/cubicasa5k",
    ) -> None:
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if payload["split"] not in {"train", "val"}:
            raise ValueError("test rows are not allowed in the action dataset")
        self.records = [
            ActionRecord(
                **{
                    **row,
                    "bbox": tuple(row["bbox"]),
                    "control_size": tuple(row["control_size"]),
                }
            )
            for row in payload["rows"]
        ]
        self.taxonomy = taxonomy
        self.cache_root = Path(cache_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        before = read_control_cache(Path(record.source))
        edit = LOCAL_EDIT_OPERATIONS[record.operation](
            before,
            self.taxonomy,
            seed=record.seed,
        )
        if edit is None:
            raise RuntimeError(f"saved edit no longer reproduces: {record}")
        after = edit.controls
        before_state = control_graph_state(before, self.taxonomy)
        after_state = control_graph_state(after, self.taxonomy)
        flags = hard_feasibility(
            before,
            after,
            record,
            self.taxonomy,
            footprint=load_cached_footprint(self.cache_root, record),
        )
        return {
            "name": record.name,
            "operation": record.operation,
            "action_index": ACTION_NAMES.index(record.operation),
            "before_node_features": before_state.node_features,
            "before_adjacency": before_state.adjacency,
            "before_global_features": before_state.global_features,
            "after_node_features": after_state.node_features,
            "after_adjacency": after_state.adjacency,
            "after_global_features": after_state.global_features,
            "action_features": action_features(before, after, record),
            "hard_feasible": torch.tensor(
                all(flags.values()),
                dtype=torch.float32,
            ),
            "hard_mask": {key: torch.tensor(value) for key, value in flags.items()},
            "changed_pixels": torch.tensor(record.changed_pixels),
        }


__all__ = [
    "ACTION_NAMES",
    "ActionRecord",
    "ConstraintActionDataset",
    "GraphState",
    "HardFeasibilityBaseline",
    "build_action_records",
    "control_graph_state",
    "hard_feasibility",
    "hard_feasibility_baseline",
    "load_cached_footprint",
    "write_action_index",
]
