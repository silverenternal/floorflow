"""Typed, class-agnostic distillation from Raster2Seq room proposals."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.data.raster2seq_proposal_cache import (
    TeacherProposalCase,
    read_manifest,
    read_teacher_case,
)
from floorflow.objectives.gpu_assignment import (
    padded_exact_assignment_batch,
)


@dataclass(frozen=True)
class TeacherTargetBatch:
    masks: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    sample_keys: tuple[str, ...]
    class_ids: torch.Tensor | None = None


class ProposalTeacherIndex:
    """Lazy access to one frozen teacher manifest by FloorFlow sample key."""

    def __init__(
        self,
        manifest_path: str | Path,
        taxonomy=None,
        *,
        case_cache_size: int = 128,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.taxonomy = taxonomy
        self.case_cache_size = max(0, int(case_cache_size))
        self._case_cache: OrderedDict[str, TeacherProposalCase] = OrderedDict()
        manifest = read_manifest(self.manifest_path)
        self.rows = {
            str(row["sample_key"]): row
            for row in manifest["rows"]
        }

    def _read_case(self, path: Path) -> TeacherProposalCase:
        key = str(path)
        cached = self._case_cache.get(key)
        if cached is not None:
            self._case_cache.move_to_end(key)
            return cached
        case = read_teacher_case(path)
        if self.case_cache_size:
            self._case_cache[key] = case
            while len(self._case_cache) > self.case_cache_size:
                self._case_cache.popitem(last=False)
        return case

    def load(
        self,
        sample_keys: list[str] | tuple[str, ...],
        *,
        image_size: int = 256,
        max_proposals: int = 64,
        device: torch.device | str = "cpu",
        compact_proposals: bool = False,
    ) -> TeacherTargetBatch:
        prepared = []
        for batch_index, sample_key in enumerate(sample_keys):
            lookup_key = str(sample_key)
            try:
                row = self.rows[lookup_key]
            except KeyError as error:
                lookup_key = lookup_key.replace("__", "/", 1)
                try:
                    row = self.rows[lookup_key]
                except KeyError:
                    raise KeyError(
                        f"sample is missing from teacher cache: {sample_key}"
                    ) from error
            case = self._read_case(
                self.root / str(row["cache_file"]),
            )
            if case.sample_key != lookup_key:
                raise ValueError("teacher cache sample key mismatch")
            count = min(len(case.masks), int(max_proposals))
            if count:
                teacher_masks = torch.from_numpy(
                    case.masks[:count].copy()
                )
                if case.image_size != image_size:
                    teacher_masks = F.interpolate(
                        teacher_masks[:, None].float(),
                        size=(image_size, image_size),
                        mode="nearest",
                    )[:, 0] > 0.5
            prepared.append(
                (
                    teacher_masks if count else None,
                    torch.from_numpy(case.existence_scores[:count].copy()),
                    torch.from_numpy(case.class_ids[:count].copy()),
                )
            )
        proposal_slots = (
            max(
                (
                    0 if teacher_masks is None else teacher_masks.shape[0]
                    for teacher_masks, _confidence, _classes in prepared
                ),
                default=0,
            )
            if compact_proposals
            else int(max_proposals)
        )
        masks = torch.zeros(
            (len(sample_keys), proposal_slots, image_size, image_size),
            dtype=torch.bool,
        )
        valid = torch.zeros(
            (len(sample_keys), proposal_slots),
            dtype=torch.bool,
        )
        confidence = torch.zeros(
            (len(sample_keys), proposal_slots),
            dtype=torch.float32,
        )
        class_ids = torch.zeros(
            (len(sample_keys), proposal_slots),
            dtype=torch.long,
        )
        for batch_index, (
            teacher_masks,
            teacher_confidence,
            teacher_classes,
        ) in enumerate(prepared):
            count = int(teacher_confidence.shape[0])
            if count:
                masks[batch_index, :count] = teacher_masks
                valid[batch_index, :count] = True
                confidence[batch_index, :count] = teacher_confidence
                class_ids[batch_index, :count] = teacher_classes
        if self.taxonomy is not None and class_ids.numel():
            local = torch.full(
                (int(self.taxonomy.num_classes),),
                -1,
                dtype=torch.long,
            )
            local[torch.tensor(self.taxonomy.room_ids)] = torch.arange(
                len(self.taxonomy.room_ids)
            )
            class_ids = local[class_ids]
        return TeacherTargetBatch(
            masks=masks.to(device),
            valid=valid.to(device),
            confidence=confidence.to(device),
            sample_keys=tuple(str(key) for key in sample_keys),
            class_ids=class_ids.to(device),
        )


def _mask_matching_cost(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    flat_prediction = predicted.flatten(1)
    flat_teacher = teacher.flatten(1).to(predicted.dtype)
    intersection = flat_prediction @ flat_teacher.transpose(0, 1)
    predicted_area = flat_prediction.sum(dim=1)[:, None]
    teacher_area = flat_teacher.sum(dim=1)[None, :]
    union = predicted_area + teacher_area - intersection
    return 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)


def proposal_distillation_loss(
    mask_logits: torch.Tensor,
    teacher: TeacherTargetBatch,
    *,
    class_logits: torch.Tensor | None = None,
    class_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match student queries to class-agnostic teacher masks."""

    if mask_logits.ndim != 4:
        raise ValueError("mask_logits must be [B,Q,H,W]")
    if teacher.masks.shape[0] != mask_logits.shape[0]:
        raise ValueError("student and teacher batch sizes differ")
    if teacher.masks.shape[-2:] != mask_logits.shape[-2:]:
        raise ValueError("student and teacher mask resolutions differ")
    probabilities = mask_logits.sigmoid()
    matched_losses = []
    background_losses = []
    match_iou = []
    class_losses = []
    class_correct = []
    costs = []
    teacher_batches = []
    for batch_index in range(mask_logits.shape[0]):
        valid_teacher = teacher.valid[batch_index]
        teacher_masks = teacher.masks[batch_index, valid_teacher]
        costs.append(
            _mask_matching_cost(probabilities[batch_index], teacher_masks)
        )
        teacher_batches.append(
            (
                teacher_masks,
                teacher.confidence[batch_index, valid_teacher],
                teacher.class_ids[batch_index, valid_teacher]
                if teacher.class_ids is not None
                else None,
            )
        )
    assignments = padded_exact_assignment_batch(costs)
    for batch_index, ((cost, teacher_batch), assignment) in enumerate(
        zip(zip(costs, teacher_batches), assignments, strict=True)
    ):
        teacher_masks, teacher_confidence, teacher_classes_all = teacher_batch
        if not len(teacher_masks):
            background_losses.append(
                F.binary_cross_entropy_with_logits(
                    mask_logits[batch_index],
                    torch.zeros_like(mask_logits[batch_index]),
                )
            )
            continue
        row_indices, column_indices = assignment
        rows = torch.as_tensor(
            row_indices,
            dtype=torch.long,
            device=mask_logits.device,
        )
        columns = torch.as_tensor(
            column_indices,
            dtype=torch.long,
            device=mask_logits.device,
        )
        matched_targets = teacher_masks[columns].to(mask_logits.dtype)
        weights = (0.5 + 0.5 * teacher_confidence[columns]).clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(
            mask_logits[batch_index, rows],
            matched_targets,
            reduction="none",
        ).flatten(1).mean(dim=1)
        matched_probability = probabilities[batch_index, rows].flatten(1)
        matched_flat = matched_targets.flatten(1)
        intersection = (matched_probability * matched_flat).sum(dim=1)
        denominator = (
            matched_probability + matched_flat
        ).sum(dim=1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        matched_losses.append(((bce + dice) * weights).mean())
        if class_logits is not None and teacher_classes_all is not None:
            teacher_classes = teacher_classes_all[columns]
            class_losses.append(
                F.cross_entropy(
                    class_logits[batch_index, rows],
                    teacher_classes,
                )
            )
            class_correct.append(
                class_logits[batch_index, rows].argmax(dim=-1)
                == teacher_classes
            )
        match_iou.append(
            (
                1.0 - cost[rows, columns]
            ).mean()
        )
        unmatched = torch.ones(
            mask_logits.shape[1],
            dtype=torch.bool,
            device=mask_logits.device,
        )
        unmatched[rows] = False
        if unmatched.any():
            background_losses.append(
                F.binary_cross_entropy_with_logits(
                    mask_logits[batch_index, unmatched],
                    torch.zeros_like(mask_logits[batch_index, unmatched]),
                )
            )
    matched = (
        torch.stack(matched_losses).mean()
        if matched_losses
        else mask_logits.sum() * 0.0
    )
    background = (
        torch.stack(background_losses).mean()
        if background_losses
        else mask_logits.sum() * 0.0
    )
    class_loss = (
        torch.stack(class_losses).mean()
        if class_losses
        else mask_logits.sum() * 0.0
    )
    loss = matched + 0.1 * background + float(class_weight) * class_loss
    return loss, {
        "proposal_distill_loss": loss.detach(),
        "proposal_distill_matched": matched.detach(),
        "proposal_distill_background": background.detach(),
        "proposal_distill_matched_iou": (
            torch.stack(match_iou).mean().detach()
            if match_iou
            else torch.tensor(0.0, device=mask_logits.device)
        ),
        "proposal_distill_class": class_loss.detach(),
        "proposal_distill_class_accuracy": (
            torch.cat(class_correct).float().mean().detach()
            if class_correct
            else mask_logits.sum().detach() * 0.0
        ),
    }


__all__ = [
    "ProposalTeacherIndex",
    "TeacherTargetBatch",
    "proposal_distillation_loss",
]
