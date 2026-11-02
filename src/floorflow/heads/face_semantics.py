"""Room-type reasoning over faces of a planar wall graph."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import RasterWallPlan


@dataclass(frozen=True)
class FaceTargets:
    masks: list[np.ndarray]
    adjacency: np.ndarray
    classes: np.ndarray
    purity: np.ndarray


def build_face_targets(
    plan: RasterWallPlan,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> FaceTargets:
    height, width = room_labels.shape
    masks = []
    class_ids = []
    purity_values = []
    room_wall_ids = []
    room_ids = np.asarray(taxonomy.room_ids)
    for room in plan.rooms:
        if len(room.get("polygon", [])) < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(np.asarray(room["polygon"])).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        values = room_labels[mask.astype(bool)]
        room_values = values[np.isin(values, room_ids)]
        if room_values.size == 0:
            continue
        counts = np.bincount(room_values, minlength=taxonomy.num_classes)
        best_global_class = int(counts.argmax())
        best_class = int(np.flatnonzero(room_ids == best_global_class)[0])
        masks.append(mask.astype(bool))
        class_ids.append(best_class)
        purity_values.append(
            float(counts[best_global_class] / room_values.size)
        )
        room_wall_ids.append(set(room.get("walls", [])))
    adjacency = np.eye(len(masks), dtype=np.float32)
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            shared = room_wall_ids[left] & room_wall_ids[right]
            if shared:
                adjacency[left, right] = adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    return FaceTargets(
        masks=masks,
        adjacency=adjacency / degree,
        classes=np.asarray(class_ids, dtype=np.int64),
        purity=np.asarray(purity_values, dtype=np.float32),
    )


def _pad_faces(
    values: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    fill: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not values:
        return (
            torch.zeros(0, 0, device=device, dtype=dtype),
            torch.zeros(0, dtype=torch.bool, device=device),
        )
    count = max(len(item) for item in values)
    rows, valid = [], []
    for item in values:
        padding = count - len(item)
        rows.append(F.pad(item, (0, 0, 0, padding), value=fill))
        valid.append(
            torch.cat(
                [
                    torch.ones(len(item), dtype=torch.bool, device=item.device),
                    torch.zeros(
                        padding,
                        dtype=torch.bool,
                        device=item.device,
                    ),
                ]
            )
        )
    return torch.stack(rows), torch.stack(valid)


class FaceSemanticHead(nn.Module):
    """Pool image features inside wall faces and exchange adjacent messages."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(in_channels, hidden_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_room_classes)

    def forward(
        self,
        features: torch.Tensor,
        masks: list[list[torch.Tensor]],
        adjacency: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, channels, height, width = features.shape
        ys = torch.arange(height, device=features.device, dtype=features.dtype)
        xs = torch.arange(width, device=features.device, dtype=features.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pooled_rows = []
        descriptor_rows = []
        for batch_index in range(batch):
            face_pools = []
            face_descriptors = []
            for mask in masks[batch_index]:
                values = torch.as_tensor(
                    mask,
                    device=features.device,
                    dtype=features.dtype,
                )
                area = values.sum().clamp_min(1.0)
                face_pools.append(
                    torch.einsum("chw,hw->c", features[batch_index], values)
                    / area
                )
                centroid_x = (grid_x * values).sum() / area / max(1, width - 1)
                centroid_y = (grid_y * values).sum() / area / max(1, height - 1)
                face_descriptors.append(
                    torch.stack(
                        [
                            area / float(height * width),
                            centroid_x,
                            centroid_y,
                            values.amax(dim=0).sum() / float(width),
                            values.amax(dim=1).sum() / float(height),
                            values.sum().sqrt() / float(height),
                        ]
                    )
                )
            pooled_rows.append(torch.stack(face_pools) if face_pools else torch.zeros(0, channels))
            descriptor_rows.append(
                torch.stack(face_descriptors)
                if face_descriptors
                else torch.zeros(0, 6)
            )
        pooled, valid = _pad_faces(
            pooled_rows,
            device=features.device,
            dtype=features.dtype,
        )
        descriptors, _ = _pad_faces(
            descriptor_rows,
            device=features.device,
            dtype=features.dtype,
        )
        max_faces = pooled.shape[1]
        adjacency_tensor = torch.zeros(
            batch,
            max_faces,
            max_faces,
            device=features.device,
            dtype=features.dtype,
        )
        for batch_index, matrix in enumerate(adjacency):
            count = len(matrix)
            if count:
                adjacency_tensor[batch_index, :count, :count] = torch.as_tensor(
                    matrix,
                    device=features.device,
                    dtype=features.dtype,
                )
        nodes = (
            self.feature_projection(pooled)
            + self.geometry_projection(descriptors)
        )
        messages = torch.einsum("bij,bjh->bih", adjacency_tensor, nodes)
        nodes = self.norm(nodes + self.message(torch.cat([nodes, messages], dim=-1)))
        return {
            "class_logits": self.classifier(nodes),
            "face_mask": valid,
            "nodes": nodes,
        }


class PooledFaceSemanticHead(nn.Module):
    """Adjacency message passing over precomputed per-face features."""

    def __init__(
        self,
        input_dim: int,
        num_room_classes: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_room_classes)

    def forward(
        self,
        nodes: list[torch.Tensor],
        adjacency: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        padded, valid = _pad_faces(
            [self.input_projection(item) for item in nodes],
            device=nodes[0].device if nodes else torch.device("cpu"),
            dtype=nodes[0].dtype if nodes else torch.float32,
        )
        max_faces = padded.shape[1]
        adjacency_tensor = torch.zeros(
            len(nodes),
            max_faces,
            max_faces,
            device=padded.device,
            dtype=padded.dtype,
        )
        for batch_index, matrix in enumerate(adjacency):
            count = len(matrix)
            if count:
                adjacency_tensor[batch_index, :count, :count] = torch.as_tensor(
                    matrix,
                    device=padded.device,
                    dtype=padded.dtype,
                )
        messages = torch.einsum("bij,bjh->bih", adjacency_tensor, padded)
        hidden = self.norm(
            padded + self.message(torch.cat([padded, messages], dim=-1))
        )
        return {
            "class_logits": self.classifier(hidden),
            "face_mask": valid,
            "nodes": hidden,
        }


def pool_face_evidence(
    features: torch.Tensor,
    masks: list[np.ndarray],
) -> torch.Tensor:
    """Mean-pool an evidence map inside each GT/predicted wall face."""

    rows = []
    for mask in masks:
        values = torch.as_tensor(
            mask,
            device=features.device,
            dtype=features.dtype,
        ).unsqueeze(0)
        area = values.sum().clamp_min(1.0)
        rows.append(
            torch.einsum("chw,nhw->nc", features, values) / area
        )
    return torch.cat(rows, dim=0) if rows else features.new_zeros(0, features.shape[0])


def face_semantic_loss(
    predictions: dict[str, torch.Tensor],
    targets: list[FaceTargets],
    *,
    min_purity: float = 0.3,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = predictions["class_logits"]
    selected_logits = []
    selected_targets = []
    for batch_index, target in enumerate(targets):
        count = min(len(target.classes), logits.shape[1])
        for face_index in range(count):
            if target.purity[face_index] >= float(min_purity):
                selected_logits.append(logits[batch_index, face_index])
                selected_targets.append(int(target.classes[face_index]))
    if not selected_logits:
        zero = logits.sum() * 0.0
        return zero, {"face_semantic_ce": zero.detach()}
    loss = F.cross_entropy(
        torch.stack(selected_logits),
        torch.tensor(selected_targets, device=logits.device),
        weight=class_weights,
    )
    return loss, {"face_semantic_ce": loss.detach()}


@torch.no_grad()
def face_semantic_metrics(
    predictions: dict[str, torch.Tensor],
    targets: list[FaceTargets],
    *,
    min_purity: float = 0.3,
) -> dict[str, float]:
    correct = 0
    total = 0
    for batch_index, target in enumerate(targets):
        count = min(len(target.classes), predictions["class_logits"].shape[1])
        for face_index in range(count):
            if target.purity[face_index] < min_purity:
                continue
            predicted = int(
                predictions["class_logits"][batch_index, face_index].argmax()
            )
            correct += int(predicted == int(target.classes[face_index]))
            total += 1
    return {"face_semantic_accuracy": correct / max(1, total)}
