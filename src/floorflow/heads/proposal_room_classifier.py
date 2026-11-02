"""Classify target-free Raster2Seq proposals with FloorFlow room evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.training.proposal_distillation import TeacherTargetBatch


class ProposalRoomClassifier(nn.Module):
    """Predict room identity for fixed proposal masks."""

    def __init__(
        self,
        feature_channels: int,
        num_room_classes: int,
        *,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_channels + num_room_classes * 2 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.room_class = nn.Linear(hidden_dim, num_room_classes)
        self.exists = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.room_class.weight)
        nn.init.zeros_(self.room_class.bias)

    def forward(
        self,
        features: torch.Tensor,
        room_logits: torch.Tensor,
        teacher: TeacherTargetBatch,
        taxonomy: FloorplanTaxonomy,
    ) -> dict[str, torch.Tensor]:
        masks = F.interpolate(
            teacher.masks.float(),
            size=features.shape[-2:],
            mode="nearest",
        )
        mass = masks.flatten(2).sum(dim=-1).clamp_min(1.0)
        pooled_features = torch.einsum(
            "bnhw,bchw->bnc",
            masks,
            features,
        ) / mass[..., None]
        probabilities = F.interpolate(
            room_logits.softmax(dim=1),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pooled_rooms = torch.einsum(
            "bnhw,brhw->bnr",
            masks,
            probabilities,
        ) / mass[..., None]
        room_ids = torch.tensor(
            taxonomy.room_ids,
            device=features.device,
        )
        teacher_classes = teacher.class_ids
        if teacher_classes is None:
            teacher_classes = torch.zeros_like(teacher.valid, dtype=torch.long)
        local_teacher = torch.full_like(
            teacher_classes,
            -1,
            dtype=torch.long,
        )
        for index, room_id in enumerate(room_ids):
            local_teacher[teacher_classes == int(room_id)] = index
        teacher_one_hot = F.one_hot(
            local_teacher.clamp_min(0),
            num_classes=len(room_ids),
        ).to(features.dtype)
        teacher_one_hot = teacher_one_hot * (local_teacher >= 0)[..., None]
        area = masks.mean(dim=(2, 3))
        confidence = teacher.confidence.to(features.dtype)
        hidden = self.input(
            torch.cat(
                [
                    pooled_features,
                    pooled_rooms,
                    teacher_one_hot,
                    area[..., None],
                    confidence[..., None],
                    teacher.valid.float()[..., None],
                ],
                dim=-1,
            )
        )
        teacher_logits = teacher_one_hot * 2.0 - 1.0
        return {
            "room_class_logits": (
                teacher_logits + self.room_class(hidden)
            ),
            "proposal_exists_logits": self.exists(hidden)[..., 0],
            "proposal_features": hidden,
        }


def proposal_classification_loss(
    predictions: dict[str, torch.Tensor],
    teacher: TeacherTargetBatch,
    target_classes: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    class_loss = (
        F.cross_entropy(
            predictions["room_class_logits"][valid],
            target_classes[valid],
        )
        if bool(valid.any())
        else predictions["room_class_logits"].sum() * 0.0
    )
    exists_loss = F.binary_cross_entropy_with_logits(
        predictions["proposal_exists_logits"],
        teacher.valid.float(),
    )
    total = class_loss + 0.25 * exists_loss
    accuracy = (
        (
            predictions["room_class_logits"][valid].argmax(dim=-1)
            == target_classes[valid]
        ).float().mean()
        if bool(valid.any())
        else class_loss.detach()
    )
    return total, {
        "proposal_room_class_loss": class_loss.detach(),
        "proposal_room_exists_loss": exists_loss.detach(),
        "proposal_room_accuracy": accuracy.detach(),
    }


@torch.no_grad()
def compose_proposal_labels(
    predictions: dict[str, torch.Tensor],
    teacher: TeacherTargetBatch,
    taxonomy: FloorplanTaxonomy,
    footprint: torch.Tensor,
    *,
    dense_fill_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=footprint.device,
    )
    class_probability = predictions["room_class_logits"].softmax(dim=-1)
    class_score, class_index = class_probability.max(dim=-1)
    score = (
        class_score
        * predictions["proposal_exists_logits"].sigmoid()
        * teacher.confidence
    )
    output = torch.zeros_like(footprint, dtype=torch.long)
    proposal_masks = F.interpolate(
        teacher.masks.float(),
        size=footprint.shape[-2:],
        mode="nearest",
    ) > 0.5
    teacher_classes = teacher.class_ids
    if teacher_classes is None:
        room_proposals = teacher.valid.bool()
    else:
        room_proposals = teacher.valid.bool() & torch.isin(
            teacher_classes,
            room_ids,
        )
    for batch_index in range(footprint.shape[0]):
        occupied = torch.zeros_like(footprint[batch_index], dtype=torch.bool)
        order = score[batch_index].argsort(descending=True)
        for query in order:
            if not bool(room_proposals[batch_index, query]):
                continue
            mask = proposal_masks[batch_index, query] & footprint[batch_index].bool()
            paint = mask & ~occupied
            output[batch_index][paint] = room_ids[
                int(class_index[batch_index, query])
            ]
            occupied |= paint
        if dense_fill_logits is not None:
            dense = dense_fill_logits[batch_index].argmax(dim=0)
            fill = footprint[batch_index].bool() & ~occupied
            output[batch_index][fill] = room_ids[dense[fill]]
    return output


__all__ = [
    "ProposalRoomClassifier",
    "compose_proposal_labels",
    "proposal_classification_loss",
]
