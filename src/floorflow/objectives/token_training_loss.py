"""Grammar-aware dictionary and structural losses for floorplan tokens."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from floorflow.objectives.token_grammar import (
    FloorplanTokenVocabulary,
    grammar_masks_for_ids,
    stage_weights_for_ids,
    token_kind_weights,
    token_sequence_loss,
)
from floorflow.objectives.token_soft_graph import (
    token_soft_graph_consistency_loss,
)
from floorflow.objectives.token_structure_consistency import (
    token_structure_consistency_loss,
)


@dataclass(frozen=True)
class TokenTrainingLossResult:
    total: torch.Tensor
    structure_total: torch.Tensor
    grammar_violation_rate: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)


def token_training_losses(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    vocabulary: FloorplanTokenVocabulary,
    *,
    kind_weights: dict[str, float] | None = None,
    stage_weights: dict[str, float] | None = None,
    structure_weight: float = 0.0,
    room_labels: tuple[str, ...] = (),
) -> TokenTrainingLossResult:
    if "token_logits" not in predictions or "token_ids" not in targets:
        raise ValueError("token loss requires token_logits and token_ids")
    token_valid = targets.get("token_loss_mask")
    if token_valid is None or token_valid.shape != targets["token_ids"].shape:
        token_valid = targets.get("token_mask")
    if token_valid is None:
        token_valid = (
            targets["token_ids"] >= 0
        ) & (
            targets["token_ids"] < predictions["token_logits"].shape[-1]
        )
    grammar_mask = targets.get("token_grammar_mask")
    if grammar_mask is None:
        grammar_mask = grammar_masks_for_ids(
            targets["token_ids"],
            vocabulary,
            token_valid,
        )
    target_allowed = grammar_mask.gather(
        -1,
        targets["token_ids"].clamp(
            0,
            grammar_mask.shape[-1] - 1,
        )[..., None],
    ).squeeze(-1)
    grammar_violation_rate = (
        (~target_allowed & token_valid).sum().float()
        / token_valid.sum().clamp_min(1)
    )
    if "token_kind_ids" in targets and kind_weights:
        weight_vector = torch.tensor(
            [
                float(kind_weights.get(kind, 1.0))
                for kind in vocabulary.kind_names
            ],
            dtype=torch.float32,
            device=targets["token_kind_ids"].device,
        )
        dictionary_weights = weight_vector[
            targets["token_kind_ids"].long().clamp(
                0,
                len(vocabulary.kind_names) - 1,
            )
        ]
    else:
        dictionary_weights = token_kind_weights(
            targets["token_ids"],
            vocabulary,
            kind_weights,
        )
    cad_stage_weights = (
        stage_weights_for_ids(
            targets["token_stage_ids"],
            stage_weights,
        )
        if "token_stage_ids" in targets
        else None
    )
    if dictionary_weights is not None and cad_stage_weights is not None:
        dictionary_weights = dictionary_weights * cad_stage_weights
    elif cad_stage_weights is not None:
        dictionary_weights = cad_stage_weights
    total = token_sequence_loss(
        predictions["token_logits"],
        targets["token_ids"],
        token_valid,
        grammar_mask=grammar_mask,
        token_weights=dictionary_weights,
    )
    structure_total = total * 0.0
    structure_components: dict[str, torch.Tensor] = {}
    if float(structure_weight) > 0.0:
        structure_total, structure_components = token_structure_consistency_loss(
            targets["token_ids"],
            token_valid,
            vocabulary,
            room_exists_logits=predictions["room_exists"],
            wall_exists_logits=predictions["wall_exists"],
            opening_exists_logits=predictions["opening_exists"],
            room_mask_logits=predictions["room_instance_mask_logits"],
            target_valid=targets.get("room_exists"),
            adjacency_logits=predictions["room_adjacency_logits"],
            target_adjacency=targets.get("adjacency"),
            room_class_logits=predictions["room_class"],
            target_room_classes=targets.get("room_classes"),
            wall_exterior_logits=predictions["wall_exterior_logits"],
            target_wall_exterior=targets.get("wall_exterior"),
            wall_valid=targets.get("wall_valid"),
            opening_host_logits=predictions.get("opening_host_logits"),
            target_opening_host=targets.get("opening_host"),
            opening_valid=targets.get("opening_valid"),
            room_point_valid_logits=predictions.get("room_point_valid_logits"),
            target_room_point_valid=targets.get("room_point_valid"),
            room_polygons=predictions.get("room_polygons"),
            target_room_polygons=targets.get("room_polygons"),
            wall_endpoints=predictions["wall_endpoints"],
            junction_points=targets.get("junction_points"),
            target_junction_degree=targets.get("junction_degree"),
            junction_valid=targets.get("junction_valid"),
            predicted_junction_degree_logits=predictions.get(
                "token_junction_degree_logits"
            ),
            predicted_junction_coordinates=predictions.get(
                "token_junction_coordinates"
            ),
            target_junction_points=targets.get("junction_points"),
            codetree_polygons=predictions.get("codetree_polygons"),
            codetree_vq_loss=predictions.get("codetree_vq_loss"),
            target_entity_counts=targets.get("token_entity_counts"),
        )
        if (
            room_labels
            and "opening_host_logits" in predictions
            and "room_exists" in targets
        ):
            soft_graph_total, soft_graph_components = (
                token_soft_graph_consistency_loss(
                    predictions["token_logits"],
                    vocabulary,
                    predictions,
                    token_valid=token_valid,
                    room_valid=targets["room_exists"],
                    wall_valid=targets["wall_valid"],
                    opening_valid=targets["opening_valid"],
                    room_labels=room_labels,
                    catalogue=predictions.get("token_graph_catalogue"),
                )
            )
            structure_total = structure_total + soft_graph_total
            structure_components.update(soft_graph_components)
    return TokenTrainingLossResult(
        total=total,
        structure_total=structure_total,
        grammar_violation_rate=grammar_violation_rate,
        components=structure_components,
    )


__all__ = ["TokenTrainingLossResult", "token_training_losses"]
