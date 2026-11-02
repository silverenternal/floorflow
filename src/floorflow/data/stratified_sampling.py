"""Fine-grained floorplan strata and train-only curriculum sampling."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, label
from torch.utils.data import Dataset

from floorflow.data.taxonomy import FURNITURE_RARE, ROOM_RARE, FloorplanTaxonomy


def _count_components(mask: np.ndarray, min_area: int = 8) -> int:
    components, count = label(np.asarray(mask, dtype=bool))
    return sum(
        int((components == index).sum()) >= min_area
        for index in range(1, int(count) + 1)
    )


def _bin(value: int, edges: tuple[int, ...]) -> int:
    return sum(value >= edge for edge in edges)


def _entropy(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    probabilities = [count / total for count in counts if count > 0]
    return float(-sum(value * math.log(value) for value in probabilities))


def _present_ids(target: np.ndarray, class_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(class_id)
        for class_id in class_ids
        if np.any(target == int(class_id))
    )


@dataclass(frozen=True)
class FloorplanStratum:
    name: str
    split: str
    corpus: str
    floor_id: int
    room_class_count: int
    room_component_count: int
    rare_room_pixels: int
    opening_pixels: int
    furniture_pixels: int
    footprint_pixels: int
    footprint_fill_fraction: float
    footprint_aspect: float
    boundary_pixels: int
    room_count_bin: int
    object_count_bin: int
    complexity_bin: int
    has_rare: bool
    stratum: str
    present_room_classes: tuple[int, ...] = field(default_factory=tuple)
    present_opening_classes: tuple[int, ...] = field(default_factory=tuple)
    present_furniture_classes: tuple[int, ...] = field(default_factory=tuple)
    room_entropy: float = 0.0
    room_area_imbalance: float = 0.0
    opening_component_count: int = 0
    furniture_component_count: int = 0
    instance_count_bin: int = 0
    fine_stratum: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FloorplanStratum:
        return cls(**payload)


def compute_floorplan_stratum(
    name: str,
    split: str,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> FloorplanStratum:
    target = np.asarray(target)
    footprint = np.asarray(footprint, dtype=bool)
    room_masks = [target == class_id for class_id in taxonomy.room_ids]
    room_present = [mask for mask in room_masks if mask.any()]
    room_components = sum(_count_components(mask) for mask in room_masks)
    rare_room_pixels = int((target == taxonomy.class_id(ROOM_RARE)).sum())
    rare_furniture_pixels = int(
        (target == taxonomy.class_id(FURNITURE_RARE)).sum()
    )
    opening_pixels = int(
        np.isin(target, np.asarray(taxonomy.opening_ids)).sum()
    )
    furniture_pixels = int(
        np.isin(target, np.asarray(taxonomy.furniture_ids)).sum()
    )
    footprint_pixels = int(footprint.sum())
    coordinates = np.argwhere(footprint)
    if len(coordinates):
        extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        aspect = float(max(extent) / max(1, min(extent)))
        fill_fraction = footprint_pixels / max(1, int(np.prod(extent)))
    else:
        aspect = 1.0
        fill_fraction = 0.0
    boundary = footprint & ~binary_erosion(footprint, border_value=0)
    present_room_classes = _present_ids(target, taxonomy.room_ids)
    present_opening_classes = _present_ids(target, taxonomy.opening_ids)
    present_furniture_classes = _present_ids(target, taxonomy.furniture_ids)
    room_areas = [
        int((target == class_id).sum()) for class_id in present_room_classes
    ]
    opening_masks = [
        target == class_id for class_id in taxonomy.opening_ids
    ]
    furniture_masks = [
        target == class_id for class_id in taxonomy.furniture_ids
    ]
    opening_component_count = sum(
        _count_components(mask) for mask in opening_masks
    )
    furniture_component_count = sum(
        _count_components(mask) for mask in furniture_masks
    )
    room_entropy = _entropy(room_areas)
    room_area_imbalance = (
        float(max(room_areas) / max(1, min(room_areas)))
        if len(room_areas) > 1
        else 1.0
    )
    instance_count_bin = _bin(
        room_components + opening_component_count + furniture_component_count,
        (8, 20, 40, 80),
    )
    room_count_bin = _bin(len(room_present), (3, 6, 10))
    object_count_bin = _bin(
        opening_pixels + furniture_pixels,
        (64, 512, 2048, 8192),
    )
    complexity_bin = min(3, round(aspect - 1.0))
    corpus = name.split("/", 1)[0]
    has_rare = bool(rare_room_pixels or rare_furniture_pixels)
    stratum = (
        f"{corpus}|rooms{room_count_bin}|objects{object_count_bin}"
        f"|complex{complexity_bin}|rare{int(has_rare)}"
    )
    room_signature = ",".join(map(str, present_room_classes))
    opening_signature = ",".join(map(str, present_opening_classes))
    furniture_signature = ",".join(map(str, present_furniture_classes))
    fine_stratum = (
        f"{stratum}|instances{instance_count_bin}"
        f"|room[{room_signature}]|open[{opening_signature}]"
        f"|furniture[{furniture_signature}]"
    )
    return FloorplanStratum(
        name=name,
        split=split,
        corpus=corpus,
        floor_id=int(name.rsplit("-", 1)[-1])
        if name.rsplit("-", 1)[-1].isdigit()
        else 1,
        room_class_count=len(room_present),
        room_component_count=room_components,
        rare_room_pixels=rare_room_pixels,
        opening_pixels=opening_pixels,
        furniture_pixels=furniture_pixels,
        footprint_pixels=footprint_pixels,
        footprint_fill_fraction=float(fill_fraction),
        footprint_aspect=aspect,
        boundary_pixels=int(boundary.sum()),
        room_count_bin=room_count_bin,
        object_count_bin=object_count_bin,
        complexity_bin=complexity_bin,
        has_rare=has_rare,
        stratum=stratum,
        present_room_classes=present_room_classes,
        present_opening_classes=present_opening_classes,
        present_furniture_classes=present_furniture_classes,
        room_entropy=room_entropy,
        room_area_imbalance=room_area_imbalance,
        opening_component_count=opening_component_count,
        furniture_component_count=furniture_component_count,
        instance_count_bin=instance_count_bin,
        fine_stratum=fine_stratum,
    )


def curriculum_sample_weights(
    entries: Sequence[FloorplanStratum],
    *,
    progress: float = 0.0,
    rare_boost: float = 2.0,
    curriculum_strength: float = 1.0,
    maximum_weight: float = 10.0,
) -> np.ndarray:
    """Balance strata while gradually shifting toward complex samples."""

    progress = min(1.0, max(0.0, float(progress)))
    curriculum_strength = min(1.0, max(0.0, float(curriculum_strength)))
    counts = Counter(entry.stratum for entry in entries)
    weights = np.asarray(
        [1.0 / math.sqrt(max(1, counts[entry.stratum])) for entry in entries],
        dtype=np.float64,
    )
    for index, entry in enumerate(entries):
        if entry.has_rare:
            weights[index] *= float(rare_boost)
        normalized_rooms = min(1.0, entry.room_class_count / 10.0)
        curriculum_focus = abs(normalized_rooms - progress)
        weights[index] *= math.exp(
            -1.5 * curriculum_strength * curriculum_focus
        )
        weights[index] *= 1.0 + curriculum_strength * progress * min(
            1.0,
            entry.object_count_bin / 4.0,
        )
    weights = np.clip(weights, 1.0 / maximum_weight, maximum_weight)
    return weights / weights.sum()


class CurriculumStratifiedSampler(torch.utils.data.Sampler[int]):
    """Weighted sampler whose mixture follows training progress."""

    def __init__(
        self,
        entries: Sequence[FloorplanStratum],
        *,
        num_samples: int | None = None,
        rare_boost: float = 2.0,
        curriculum_strength: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not entries:
            raise ValueError("stratified sampler requires entries")
        self.entries = tuple(entries)
        self.num_samples = int(num_samples or len(entries))
        self.rare_boost = float(rare_boost)
        self.curriculum_strength = float(curriculum_strength)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.epoch = int(epoch)
        self.total_epochs = max(1, int(total_epochs))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        progress = self.epoch / max(1, self.total_epochs)
        weights = curriculum_sample_weights(
            self.entries,
            progress=progress,
            rare_boost=self.rare_boost,
            curriculum_strength=self.curriculum_strength,
        )
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            torch.from_numpy(weights),
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        yield from indices.tolist()


class EpochRandomSubsetSampler(torch.utils.data.Sampler[int]):
    """Uniform full-pool subset with deterministic per-epoch shuffle."""

    def __init__(
        self,
        length: int,
        num_samples: int,
        *,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0 or num_samples > length:
            raise ValueError("num_samples must be in [1, length]")
        self.length = int(length)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.length, generator=generator)[
            : self.num_samples
        ].tolist()


def stratified_loss_weights(
    entries: Sequence[FloorplanStratum],
    *,
    rare_boost: float = 1.5,
    maximum_weight: float = 2.0,
) -> np.ndarray:
    """Generate per-sample weights without changing data visitation."""

    counts = Counter(entry.stratum for entry in entries)
    weights = np.asarray(
        [1.0 / math.sqrt(max(1, counts[entry.stratum])) for entry in entries],
        dtype=np.float64,
    )
    for index, entry in enumerate(entries):
        if entry.has_rare:
            weights[index] *= float(rare_boost)
        if entry.room_class_count in {0, 3}:
            weights[index] *= 1.25
        weights[index] = min(float(maximum_weight), max(0.25, weights[index]))
    return weights / weights.mean()


class StratifiedWeightDataset(Dataset):
    """Attach train-derived loss weights while preserving sample ordering."""

    def __init__(self, base: Dataset, entries: Sequence[FloorplanStratum]) -> None:
        if len(base) != len(entries):
            raise ValueError("base dataset and strata must have equal length")
        self.base = base
        self.taxonomy = base.taxonomy
        self.weights = torch.from_numpy(
            stratified_loss_weights(entries).astype(np.float32)
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], self.weights[index])


def load_floorplan_strata(
    path: str | Path,
    split: str,
) -> list[FloorplanStratum]:
    entries = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("split") == split:
                entries.append(FloorplanStratum.from_dict(payload))
    if not entries:
        raise ValueError(f"no strata found for split={split!r} in {path}")
    return entries


__all__ = [
    "CurriculumStratifiedSampler",
    "EpochRandomSubsetSampler",
    "FloorplanStratum",
    "StratifiedWeightDataset",
    "compute_floorplan_stratum",
    "curriculum_sample_weights",
    "load_floorplan_strata",
    "stratified_loss_weights",
]
