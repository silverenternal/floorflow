"""Query-based room instance prediction and Hungarian matching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.model.backbone import FloorplanFlowModel
from floorflow.representation.core import extract_room_instances
from floorflow.data.taxonomy import FloorplanTaxonomy


def labels_to_room_targets(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    max_rooms: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_classes = []
    batch_boxes = []
    batch_masks = []
    height, width = labels.shape[-2:]
    for item in labels.cpu().numpy():
        instances = sorted(extract_room_instances(item, taxonomy), key=lambda value: value.area, reverse=True)[:max_rooms]
        classes = torch.zeros(max_rooms, dtype=torch.long)
        boxes = torch.zeros(max_rooms, 4)
        mask = torch.zeros(max_rooms, dtype=torch.bool)
        for index, instance in enumerate(instances):
            x0, y0, x1, y1 = instance.bbox
            classes[index] = instance.class_id
            boxes[index] = torch.tensor(
                [
                    (x0 + x1) / (2.0 * width),
                    (y0 + y1) / (2.0 * height),
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ]
            )
            mask[index] = True
        batch_classes.append(classes)
        batch_boxes.append(boxes)
        batch_masks.append(mask)
    return (
        torch.stack(batch_classes).to(labels.device),
        torch.stack(batch_boxes).to(labels.device),
        torch.stack(batch_masks).to(labels.device),
    )


class RoomSetHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_queries: int = 32) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.class_head = nn.Linear(hidden_dim, max(1, num_classes - 1))
        self.box_head = nn.Linear(hidden_dim, 4)
        self.valid_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = features.flatten(2).transpose(1, 2)
        memory = self.memory_norm(memory)
        query = self.queries.to(features.dtype).expand(features.shape[0], -1, -1)
        attended, _ = self.attention(self.query_norm(query), memory, memory, need_weights=False)
        hidden = self.shared(attended)
        return {
            "class_logits": self.class_head(hidden),
            "boxes": torch.sigmoid(self.box_head(hidden)),
            "valid_logits": self.valid_head(hidden).squeeze(-1),
        }


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyxy = torch.stack(
        [
            left[:, 0] - left[:, 2] / 2,
            left[:, 1] - left[:, 3] / 2,
            left[:, 0] + left[:, 2] / 2,
            left[:, 1] + left[:, 3] / 2,
        ],
        dim=1,
    )
    right_xyxy = torch.stack(
        [
            right[:, 0] - right[:, 2] / 2,
            right[:, 1] - right[:, 3] / 2,
            right[:, 0] + right[:, 2] / 2,
            right[:, 1] + right[:, 3] / 2,
        ],
        dim=1,
    )
    top_left = torch.maximum(left_xyxy[:, None, :2], right_xyxy[None, :, :2])
    bottom_right = torch.minimum(left_xyxy[:, None, 2:], right_xyxy[None, :, 2:])
    size = (bottom_right - top_left).clamp_min(0.0)
    intersection = size[..., 0] * size[..., 1]
    area_left = (left[:, 2] * left[:, 3])[:, None]
    area_right = (right[:, 2] * right[:, 3])[None, :]
    return intersection / (area_left + area_right - intersection).clamp_min(1.0e-6)


def room_set_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    class_losses = []
    box_losses = []
    valid_losses = []
    class_prob = predictions["class_logits"].softmax(dim=-1)
    for index in range(target_classes.shape[0]):
        valid_targets = target_mask[index]
        count = int(valid_targets.sum())
        if count == 0:
            valid_losses.append(F.binary_cross_entropy_with_logits(predictions["valid_logits"][index], torch.zeros_like(predictions["valid_logits"][index])))
            continue
        target_cls = target_classes[index, :count] - 1
        target_box = target_boxes[index, :count]
        cls_cost = -class_prob[index][:, target_cls]
        box_cost = torch.cdist(predictions["boxes"][index], target_box, p=1)
        cost = cls_cost + box_cost
        pred_idx, target_idx = linear_sum_assignment(cost.detach().cpu().numpy())
        class_losses.append(F.cross_entropy(predictions["class_logits"][index, pred_idx], target_cls[target_idx]))
        box_losses.append(F.l1_loss(predictions["boxes"][index, pred_idx], target_box[target_idx]))
        valid_target = torch.zeros_like(predictions["valid_logits"][index])
        valid_target[pred_idx] = 1.0
        valid_losses.append(F.binary_cross_entropy_with_logits(predictions["valid_logits"][index], valid_target))
    return (
        sum(class_losses) / max(1, len(class_losses))
        + 2.0 * sum(box_losses) / max(1, len(box_losses))
        + sum(valid_losses) / max(1, len(valid_losses))
    )


@torch.no_grad()
def room_set_metrics(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float]:
    class_correct = 0
    matched_iou = 0.0
    matched = 0
    predicted_count = 0
    for index in range(target_classes.shape[0]):
        valid_targets = target_mask[index]
        count = int(valid_targets.sum())
        selected = predictions["valid_logits"][index] > 0.0
        predicted_count += int(selected.sum())
        if count == 0:
            continue
        pred_boxes = predictions["boxes"][index][selected]
        pred_classes = predictions["class_logits"][index][selected].argmax(dim=-1)
        target_cls = target_classes[index, :count] - 1
        if pred_boxes.numel() == 0:
            continue
        cost = torch.cdist(pred_boxes, target_boxes[index, :count], p=1)
        pred_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
        class_correct += int((pred_classes[pred_idx] == target_cls[target_idx]).sum())
        matched_iou += float(_box_iou(pred_boxes[pred_idx], target_boxes[index, :count][target_idx]).sum())
        matched += len(pred_idx)
    return {
        "matched_class_accuracy": class_correct / max(1, matched),
        "matched_box_iou": matched_iou / max(1, matched),
        "predicted_instances_per_sample": predicted_count / max(1, target_classes.shape[0]),
    }


@torch.no_grad()
def render_room_set(
    predictions: dict[str, torch.Tensor],
    height: int,
    width: int,
) -> torch.Tensor:
    labels = torch.zeros(predictions["boxes"].shape[0], height, width, dtype=torch.long, device=predictions["boxes"].device)
    for batch_index in range(labels.shape[0]):
        selected = torch.nonzero(predictions["valid_logits"][batch_index] > 0.0, as_tuple=False).flatten()
        rectangles = []
        for query_index in selected.tolist():
            box = predictions["boxes"][batch_index, query_index]
            class_id = int(predictions["class_logits"][batch_index, query_index].argmax()) + 1
            x0 = int((box[0] - box[2] / 2).clamp(0, 1) * width)
            y0 = int((box[1] - box[3] / 2).clamp(0, 1) * height)
            x1 = int((box[0] + box[2] / 2).clamp(0, 1) * width)
            y1 = int((box[1] + box[3] / 2).clamp(0, 1) * height)
            rectangles.append((max(1, (x1 - x0) * (y1 - y0)), x0, y0, x1, y1, class_id))
        for _, x0, y0, x1, y1, class_id in sorted(rectangles, reverse=True):
            labels[batch_index, y0:y1, x0:x1] = class_id
    return labels


def compose_room_set_with_semantics(
    semantic_labels: torch.Tensor,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=semantic_labels.device)
    replaceable = torch.isin(semantic_labels, room_ids) | (semantic_labels == 0)
    replaceable &= footprint > 0.5
    return torch.where(replaceable, room_labels, semantic_labels)
