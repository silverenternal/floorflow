"""Auxiliary target-side objectives separated from flow orchestration."""

from __future__ import annotations

import torch

from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    encode_target_and_condition,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.relation_gate_balance import (
    relation_gate_regularization,
)
from floorflow.chip_flow.structural_relations import (
    structural_relation_features,
    structural_relation_loss,
)


def training_clean_layout(model, batch: dict) -> tuple:
    """Decode and project the supervised target layout for auxiliary losses."""

    target_global, condition, context = encode_target_and_condition(model, batch)
    latent, _ = decode_for_batch(
        model,
        target_global,
        condition,
        context,
        batch,
    )
    latent = project_conditioned_latent(
        latent.float(),
        batch["target_latent"],
        batch["constraints"],
    )
    return (
        latent,
        decode_boxes_batch(latent, batch["areas"]),
        target_global,
        condition,
        context,
    )


def structural_relation_objective(
    prediction,
    batch: dict,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Compute grouping-aligned relation supervision when enabled."""

    weight = float(getattr(args, "structural_relation_weight", 0.0))
    relation_prediction = getattr(prediction, "structural_relations", None)
    if weight <= 0.0 or relation_prediction is None:
        return torch.zeros((), device=device), weight
    target, mask = structural_relation_features(
        batch["boxes"].float(),
        batch["areas"].float(),
        batch["constraints"],
    )
    return (
        structural_relation_loss(
            relation_prediction.float(),
            target,
            mask,
            beta=args.flow_huber_beta,
        ),
        weight,
    )


def relation_gate_objective(
    model,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Penalize collapsed or inactive relation-family gates."""

    weight = float(getattr(args, "relation_gate_weight", 0.0))
    attention = getattr(getattr(model, "typed", None), "cross_attention", None)
    if weight <= 0.0 or attention is None:
        return torch.zeros((), device=device), weight
    return relation_gate_regularization(attention).to(device), weight


__all__ = [
    "relation_gate_objective",
    "structural_relation_objective",
    "training_clean_layout",
]
