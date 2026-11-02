"""Runtime bridge from raster-room predictions to cached Raster2Seq teachers."""

from __future__ import annotations

import torch

from floorflow.training.proposal_distillation import (
    ProposalTeacherIndex,
    proposal_distillation_loss,
)


def raster2seq_teacher_loss(
    teacher: ProposalTeacherIndex,
    predictions: dict[str, torch.Tensor],
    sample_keys: list[str] | tuple[str, ...],
    *,
    max_proposals: int,
    device: torch.device,
    class_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = teacher.load(
        sample_keys,
        image_size=predictions["mask_logits"].shape[-1],
        max_proposals=max_proposals,
        device=device,
        compact_proposals=True,
    )
    return proposal_distillation_loss(
        predictions["mask_logits"],
        targets,
        class_logits=(
            predictions["class_logits"] if class_weight > 0.0 else None
        ),
        class_weight=class_weight,
    )


__all__ = ["raster2seq_teacher_loss"]
