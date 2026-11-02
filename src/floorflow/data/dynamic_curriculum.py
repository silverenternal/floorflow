"""Training-dynamics data maps adapted from Dataset Cartography."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class DataDynamicsPoint:
    name: str
    visits: int
    confidence: float
    variability: float
    correctness: float
    loss: float
    room_precision: float
    room_recall: float
    room_recall_variability: float
    opening_accuracy: float
    furniture_accuracy: float
    forgetfulness: int
    region: str
    difficulty: float
    room_macro_iou: float = float("nan")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, float("nan")),
    )


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _task_need(
    confidence: np.ndarray,
    room_recall: np.ndarray,
    opening_accuracy: np.ndarray,
    furniture_accuracy: np.ndarray,
    variability: np.ndarray,
    task_focus: float,
) -> np.ndarray:
    """Combine generic difficulty with the CAD families used by the task."""

    room_gap = np.where(np.isfinite(room_recall), 1.0 - room_recall, 0.0)
    opening_gap = np.where(
        np.isfinite(opening_accuracy),
        1.0 - opening_accuracy,
        0.0,
    )
    furniture_gap = np.where(
        np.isfinite(furniture_accuracy),
        1.0 - furniture_accuracy,
        0.0,
    )
    element_count = np.isfinite(opening_accuracy).astype(float) + np.isfinite(
        furniture_accuracy
    ).astype(float)
    element_gap = (opening_gap + furniture_gap) / np.maximum(1.0, element_count)
    structural_need = 0.75 * room_gap + 0.25 * element_gap
    valid_variability = variability[np.isfinite(variability)]
    variability_scale = (
        max(1.0e-8, float(np.quantile(valid_variability, 0.75)))
        if valid_variability.size
        else 1.0
    )
    normalized_variability = variability / variability_scale
    normalized_variability = np.where(
        np.isfinite(normalized_variability),
        normalized_variability,
        0.0,
    )
    return (
        (1.0 - confidence)
        + 0.35 * normalized_variability
        + 0.75 * float(task_focus) * structural_need
    )


class DataDynamicsTracker:
    """Accumulate per-sample learning dynamics without changing visitation."""

    def __init__(self, names: list[str] | tuple[str, ...], taxonomy: FloorplanTaxonomy):
        if not names:
            raise ValueError("data dynamics tracker requires sample names")
        if len(set(names)) != len(names):
            raise ValueError("sample names must be unique")
        self.names = tuple(names)
        self.index = {name: index for index, name in enumerate(self.names)}
        self.taxonomy = taxonomy
        self.size = len(self.names)
        self.visits = np.zeros(self.size, dtype=np.int64)
        self.forgetfulness = np.zeros(self.size, dtype=np.int64)
        self.last_correct = np.full(self.size, -1.0, dtype=np.float64)
        metric_names = (
            "confidence",
            "correctness",
            "loss",
            "room_precision",
            "room_recall",
            "room_macro_iou",
            "opening_accuracy",
            "furniture_accuracy",
        )
        self._sums = {
            name: np.zeros(self.size, dtype=np.float64) for name in metric_names
        }
        self._squares = {
            name: np.zeros(self.size, dtype=np.float64) for name in metric_names
        }
        self._counts = {
            name: np.zeros(self.size, dtype=np.int64) for name in metric_names
        }

    def _update_metric(self, metric: str, index: int, value: float) -> None:
        if not _is_finite(value):
            return
        self._sums[metric][index] += float(value)
        self._squares[metric][index] += float(value) ** 2
        self._counts[metric][index] += 1

    def update(
        self,
        names: tuple[str, ...] | list[str],
        logits: torch.Tensor,
        target: torch.Tensor,
        footprint: torch.Tensor,
    ) -> None:
        """Record one prediction snapshot for each sample in a training batch."""

        if len(names) != logits.shape[0]:
            raise ValueError("one data-map name is required per prediction")
        if footprint.ndim == 4:
            footprint = footprint[:, 0]
        valid = footprint.bool() & torch.isfinite(logits).all(dim=1)
        target = target.long()
        with torch.no_grad():
            probability = logits.detach().float().softmax(dim=1)
            prediction = probability.argmax(dim=1)
            true_probability = probability.gather(
                1,
                target.clamp(0, logits.shape[1] - 1).unsqueeze(1),
            ).squeeze(1)
            correct = (prediction == target) & valid
            loss = torch.nn.functional.cross_entropy(
                logits.detach().float(),
                target,
                reduction="none",
            )
            room_ids = torch.tensor(
                self.taxonomy.room_ids,
                device=logits.device,
                dtype=torch.long,
            )
            opening_ids = torch.tensor(
                self.taxonomy.opening_ids,
                device=logits.device,
                dtype=torch.long,
            )
            furniture_ids = torch.tensor(
                self.taxonomy.furniture_ids,
                device=logits.device,
                dtype=torch.long,
            )
            predicted_room = torch.isin(prediction, room_ids)
            target_room = torch.isin(target, room_ids) & valid
            target_opening = torch.isin(target, opening_ids) & valid
            target_furniture = torch.isin(target, furniture_ids) & valid
            room_lookup = torch.full(
                (logits.shape[1],),
                -1,
                device=logits.device,
                dtype=torch.long,
            )
            room_lookup[room_ids] = torch.arange(
                len(room_ids),
                device=logits.device,
            )
            predicted_room_target = room_lookup[prediction]
            target_room_target = room_lookup[target]
            predicted_one_hot = torch.nn.functional.one_hot(
                predicted_room_target.clamp_min(0),
                len(room_ids),
            ).permute(0, 3, 1, 2).float()
            predicted_one_hot = predicted_one_hot * (
                predicted_room_target >= 0
            )[:, None, :, :]
            target_one_hot = torch.nn.functional.one_hot(
                target_room_target.clamp_min(0),
                len(room_ids),
            ).permute(0, 3, 1, 2).float()
            room_region = target_room[:, None]
            predicted_one_hot = predicted_one_hot * room_region
            target_one_hot = target_one_hot * room_region
            intersection = (predicted_one_hot * target_one_hot).sum(dim=(2, 3))
            predicted_mass = predicted_one_hot.sum(dim=(2, 3))
            target_mass = target_one_hot.sum(dim=(2, 3))
            union = predicted_mass + target_mass - intersection
            present = target_mass > 0
            per_class_iou = intersection / union.clamp_min(1.0)
            room_macro_iou = (
                per_class_iou.masked_fill(~present, 0.0).sum(dim=1)
                / present.sum(dim=1).clamp_min(1)
            )
            room_precision = _safe_div(
                (correct & predicted_room).flatten(1).sum(1).float(),
                (predicted_room & valid).flatten(1).sum(1).float(),
            )
            room_recall = _safe_div(
                (correct & target_room).flatten(1).sum(1).float(),
                target_room.flatten(1).sum(1).float(),
            )
            opening_accuracy = _safe_div(
                (correct & target_opening).flatten(1).sum(1).float(),
                target_opening.flatten(1).sum(1).float(),
            )
            furniture_accuracy = _safe_div(
                (correct & target_furniture).flatten(1).sum(1).float(),
                target_furniture.flatten(1).sum(1).float(),
            )
            confidence = _safe_div(
                (true_probability * valid).flatten(1).sum(1),
                valid.flatten(1).sum(1).float(),
            )
            correctness = _safe_div(
                correct.flatten(1).sum(1).float(),
                valid.flatten(1).sum(1).float(),
            )
            mean_loss = _safe_div(
                (loss * valid).flatten(1).sum(1),
                valid.flatten(1).sum(1).float(),
            )
            values = {
                "confidence": confidence.cpu().tolist(),
                "correctness": correctness.cpu().tolist(),
                "loss": mean_loss.cpu().tolist(),
                "room_precision": room_precision.cpu().tolist(),
                "room_recall": room_recall.cpu().tolist(),
                "room_macro_iou": room_macro_iou.cpu().tolist(),
                "opening_accuracy": opening_accuracy.cpu().tolist(),
                "furniture_accuracy": furniture_accuracy.cpu().tolist(),
            }
        for batch_index, name in enumerate(names):
            sample_index = self.index[name]
            self.visits[sample_index] += 1
            for metric, metric_values in values.items():
                self._update_metric(metric, sample_index, metric_values[batch_index])
            current_correct = float(values["correctness"][batch_index])
            if (
                _is_finite(self.last_correct[sample_index])
                and self.last_correct[sample_index] >= 0.5
                and _is_finite(current_correct)
                and current_correct < 0.5
            ):
                self.forgetfulness[sample_index] += 1
            self.last_correct[sample_index] = current_correct

    def _mean(self, metric: str, index: int) -> float:
        count = self._counts[metric][index]
        return (
            float(self._sums[metric][index] / count) if count else float("nan")
        )

    def _std(self, metric: str, index: int) -> float:
        count = self._counts[metric][index]
        if not count:
            return float("nan")
        mean = self._sums[metric][index] / count
        variance = max(0.0, self._squares[metric][index] / count - mean**2)
        return float(math.sqrt(variance))

    def snapshot(self) -> list[DataDynamicsPoint]:
        confidence = np.asarray(
            [self._mean("confidence", index) for index in range(self.size)]
        )
        variability = np.asarray(
            [self._std("confidence", index) for index in range(self.size)]
        )
        visits = self.visits.copy()
        valid = visits > 0
        regions = np.full(self.size, "unseen", dtype=object)
        if valid.any():
            confidence_q25 = float(np.nanquantile(confidence[valid], 0.25))
            confidence_q75 = float(np.nanquantile(confidence[valid], 0.75))
            variability_q50 = float(np.nanquantile(variability[valid], 0.50))
            variability_q75 = float(np.nanquantile(variability[valid], 0.75))
            for index in range(self.size):
                if visits[index] < 2:
                    regions[index] = "warming"
                elif variability[index] >= variability_q75:
                    regions[index] = "ambiguous"
                elif confidence[index] <= confidence_q25:
                    regions[index] = "hard"
                elif (
                    confidence[index] >= confidence_q75
                    and variability[index] <= variability_q50
                ):
                    regions[index] = "easy"
                else:
                    regions[index] = "typical"
        room_recall = np.asarray(
            [self._mean("room_recall", index) for index in range(self.size)]
        )
        opening_accuracy = np.asarray(
            [self._mean("opening_accuracy", index) for index in range(self.size)]
        )
        furniture_accuracy = np.asarray(
            [
                self._mean("furniture_accuracy", index)
                for index in range(self.size)
            ]
        )
        raw_difficulty = _task_need(
            confidence,
            room_recall,
            opening_accuracy,
            furniture_accuracy,
            variability,
            task_focus=1.0,
        )
        finite_difficulty = raw_difficulty[np.isfinite(raw_difficulty)]
        center = float(finite_difficulty.mean()) if finite_difficulty.size else 0.0
        scale = (
            float(finite_difficulty.std()) if finite_difficulty.size else 1.0
        )
        scale = max(scale, 1.0e-6)
        difficulty = (raw_difficulty - center) / scale
        points = []
        for index, name in enumerate(self.names):
            points.append(
                DataDynamicsPoint(
                    name=name,
                    visits=int(visits[index]),
                    confidence=self._mean("confidence", index),
                    variability=self._std("confidence", index),
                    correctness=self._mean("correctness", index),
                    loss=self._mean("loss", index),
                    room_precision=self._mean("room_precision", index),
                    room_recall=float(room_recall[index]),
                    room_macro_iou=self._mean("room_macro_iou", index),
                    room_recall_variability=self._std("room_recall", index),
                    opening_accuracy=float(opening_accuracy[index]),
                    furniture_accuracy=float(furniture_accuracy[index]),
                    forgetfulness=int(self.forgetfulness[index]),
                    region=str(regions[index]),
                    difficulty=float(difficulty[index]),
                )
            )
        return points


def data_map_weights(
    points: list[DataDynamicsPoint] | tuple[DataDynamicsPoint, ...],
    *,
    strength: float = 0.35,
    floor: float = 0.75,
    ceiling: float = 1.5,
    task_focus: float = 1.0,
    eligible: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Convert difficulty coordinates into bounded loss-only weights."""

    if floor <= 0 or ceiling < floor:
        raise ValueError("require 0 < floor <= ceiling")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if task_focus < 0:
        raise ValueError("task_focus must be non-negative")
    usable = [
        point
        for point in points
        if point.visits >= 2
        and math.isfinite(point.difficulty)
        and (eligible is None or eligible.get(point.name, False))
    ]
    if not usable:
        return {point.name: 1.0 for point in points}
    need = _task_need(
        np.asarray([point.confidence for point in usable]),
        np.asarray([point.room_recall for point in usable]),
        np.asarray([point.opening_accuracy for point in usable]),
        np.asarray([point.furniture_accuracy for point in usable]),
        np.asarray([point.variability for point in usable]),
        task_focus=task_focus,
    )
    centered = need - float(need.mean())
    raw = np.exp(np.clip(float(strength) * centered, -1.5, 1.5))
    raw = np.clip(raw, float(floor), float(ceiling))
    raw /= float(raw.mean())
    weights = {point.name: 1.0 for point in points}
    weights.update(
        {
            point.name: float(weight)
            for point, weight in zip(usable, raw, strict=True)
        }
    )
    return weights


def load_data_map(
    path: str | Path,
    *,
    epoch: int | None = None,
) -> list[DataDynamicsPoint]:
    """Read one epoch of a previously generated per-sample data map."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty data map: {path}")
    selected_epoch = (
        int(epoch)
        if epoch is not None
        else max(int(row["epoch"]) for row in rows)
    )
    fields = set(DataDynamicsPoint.__dataclass_fields__)
    points = [
        DataDynamicsPoint(
            **{key: value for key, value in row.items() if key in fields}
        )
        for row in rows
        if int(row["epoch"]) == selected_epoch
    ]
    if not points:
        raise ValueError(f"no rows for epoch={selected_epoch} in {path}")
    return points


def write_data_map(
    points: list[DataDynamicsPoint],
    path: str | Path,
    *,
    epoch: int,
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(
                json.dumps({"epoch": int(epoch), **point.to_dict()}) + "\n"
            )
    regions: dict[str, int] = {}
    for point in points:
        regions[point.region] = regions.get(point.region, 0) + 1
    return {
        "epoch": int(epoch),
        "path": str(path),
        "samples": len(points),
        "regions": regions,
        "mean_confidence": float(
            np.nanmean([point.confidence for point in points])
        ),
        "mean_variability": float(
            np.nanmean([point.variability for point in points])
        ),
        "mean_correctness": float(
            np.nanmean([point.correctness for point in points])
        ),
        "mean_room_recall": float(
            np.nanmean([point.room_recall for point in points])
        ),
        "mean_room_macro_iou": float(
            np.nanmean([point.room_macro_iou for point in points])
        ),
        "total_forgetfulness": int(
            sum(point.forgetfulness for point in points)
        ),
    }


__all__ = [
    "DataDynamicsPoint",
    "DataDynamicsTracker",
    "data_map_weights",
    "load_data_map",
    "write_data_map",
]
