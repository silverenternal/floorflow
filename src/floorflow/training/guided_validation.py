"""Validation loop shared by guided CFM training scripts."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.flow.guided_control import (
    guided_source_from_labels,
    sample_guided_floorplans,
)
from floorflow.flow.mask_policies import make_mask_policy
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)
from floorflow.training.masked_conditioning import prepare_masked_condition


@torch.inference_mode()
def validate_guided(
    model,
    loader,
    taxonomy,
    device,
    steps,
    *,
    missing_token=None,
    mask_ratio=None,
    mask_policy=None,
    mask_generator=None,
):
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    values = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        source = guided_source_from_labels(target, footprint, taxonomy)
        prepared = prepare_masked_condition(
            source,
            footprint,
            taxonomy,
            missing_token
            if missing_token is not None
            else torch.zeros(model.target_channels, device=device),
            missing_ratio=0.0 if mask_ratio is None else mask_ratio,
            floor_ids=floor_ids,
            mask_generator=mask_generator,
            mask_policy=mask_policy or make_mask_policy("attribute"),
        )
        source = prepared.control
        control_mask = prepared.missing_mask
        if isinstance(model, MaskedControlFlow):
            sampled = sample_sparse_guided_floorplans(
                model,
                footprint,
                source,
                control_mask,
                steps=steps,
                floor_ids=floor_ids,
            )
        else:
            sampled = sample_guided_floorplans(
                model,
                footprint,
                source,
                steps=steps,
                floor_ids=floor_ids,
            )
        room_scores = sampled[:, room_ids - 1]
        labels = room_ids[room_scores.argmax(dim=1)]
        source_rooms = source[:, room_ids - 1]
        program = source_rooms.amax(dim=(2, 3)) > 0.1
        missing = ~program.any(dim=1)
        if missing.any():
            fallback = room_scores.amax(dim=(2, 3)).argmax(dim=1)
            program = program.clone()
            program[missing, fallback[missing]] = True
        labels = gate_room_labels_to_program(
            labels,
            program,
            taxonomy.room_ids,
        )
        labels = labels * footprint.long()
        for index in range(target.shape[0]):
            support = torch.isin(labels[index], room_ids)
            suite = room_metric_suite(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            values.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in values]))
        for key in values[0]
    }


__all__ = ["validate_guided"]
