"""Loss composition for the unified boundary-conditioned floorplan generator."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.polygon_loop import polygon_loop_loss
from floorflow.heads.room_mask_losses import room_mask_loss
from floorflow.heads.room_program_set import room_program_set_loss
from floorflow.objectives.gpu_assignment import assignment_index_tensors
from floorflow.objectives.gradient_balance import gradient_diagnostics
from floorflow.objectives.room_query_matching import match_room_queries
from floorflow.objectives.room_query_separation import (
    active_query_overlap_loss,
    active_query_presence,
    balanced_room_class_weights,
    supervised_same_class_overlap_loss,
)
from floorflow.objectives.token_grammar import (
    FloorplanTokenVocabulary,
    token_sequence_loss,
)
from floorflow.objectives.token_training_loss import (
    token_training_losses,
)
from floorflow.objectives.typed_state_hierarchy_loss import (
    typed_state_hierarchy_loss,
)
from floorflow.objectives.verifier_policy import verifier_policy_loss
from floorflow.training.opening_loss_matching import match_openings
from floorflow.training.unified_plan_generator_matching import (
    balanced_positive_weight as _balanced_positive_weight,
)
from floorflow.training.unified_plan_generator_matching import (
    match_candidate_queries,
)


def _sample_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    include: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average per-sample valid entries without a device synchronization."""

    weights = valid.to(values.dtype)
    sample_sum = (values * weights).sum(dim=1)
    sample_count = weights.sum(dim=1)
    if include is None:
        include = sample_count.gt(0)
    sample_mean = sample_sum / sample_count.clamp_min(1.0)
    include = include.to(sample_mean.dtype)
    return (sample_mean * include).sum() / include.sum().clamp_min(1.0)


def unified_generator_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    polygon_weight: float = 1.0,
    identity_weight: float = 1.0,
    count_weight: float = 0.5,
    wall_weight: float = 0.5,
    opening_weight: float = 0.25,
    adjacency_weight: float = 0.25,
    query_separation_weight: float = 0.0,
    class_balance_power: float = 0.0,
    joint_mask_match_weight: float = 1.0,
    joint_room_assignment: bool = False,
    class_aware_separation: bool = False,
    supervised_separation: bool = False,
    query_quality_weight: float = 0.0,
    point_validity_weight: float = 0.5,
    room_program_weight: float = 0.0,
    partition_consistency_weight: float = 0.0,
    partition_conflict_only: bool = False,
    partition_label_smoothing: float = 0.0,
    token_weight: float = 0.0,
    token_kind_weights: dict[str, float] | None = None,
    token_stage_weights: dict[str, float] | None = None,
    token_structure_weight: float = 0.0,
    token_vocabulary: FloorplanTokenVocabulary | None = None,
    loss_balancer=None,
    balance_parameters: list[torch.nn.Parameter] | None = None,
    balance_step: int = 0,
    diagnostic_parameters: list[torch.nn.Parameter] | None = None,
    diagnostic_out: dict[str, object] | None = None,
    loss_terms_out: dict[str, torch.Tensor] | None = None,
    hierarchy_weights: dict[str, float] | None = None,
    verifier_policy_weight: float = 0.0,
    verifier_reward_baseline: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match every declared vector and identity head to RPLAN targets."""

    target_valid = targets["room_exists"]
    room_target_counts = targets.get("target_room_counts")
    if room_target_counts is None:
        room_target_counts = [
            int(value) for value in target_valid.sum(dim=1).tolist()
        ]
    room_target_counts = [
        min(int(value), target_valid.shape[1])
        for value in room_target_counts
    ]
    mask_logits = predictions["room_instance_mask_logits"]
    resized_masks = F.interpolate(
        targets["room_masks"].flatten(0, 1)[:, None],
        size=mask_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    target_masks = resized_masks.reshape(
        targets["room_masks"].shape[0],
        targets["room_masks"].shape[1],
        *mask_logits.shape[-2:],
    )
    room_matches = None
    if float(polygon_weight) > 0.0 or float(query_quality_weight) > 0.0:
        room_matches = match_room_queries(
            predictions["room_class"],
            predictions["room_polygons"],
            targets["room_classes"],
            targets["room_polygons"],
            target_valid,
            target_counts=room_target_counts,
            predicted_masks=mask_logits if joint_room_assignment else None,
            target_masks=target_masks if joint_room_assignment else None,
            mask_cost_weight=float(joint_mask_match_weight),
        )
    if float(polygon_weight) > 0.0:
        polygon_total, polygon_components = polygon_loop_loss(
            {
                "class_logits": predictions["room_class"],
                "polygons": predictions["room_polygons"],
                "valid_logits": predictions["room_exists"],
                **(
                    {
                        "room_point_valid_logits": predictions[
                            "room_point_valid_logits"
                        ]
                    }
                    if "room_point_valid_logits" in predictions
                    else {}
                ),
            },
            targets["room_classes"],
            targets["room_polygons"],
            target_valid,
            valid_positive_weight=1.5,
            matches=room_matches,
            target_point_valid=(
                targets.get("room_point_valid")
                if "room_point_valid_logits" in predictions
                else None
            ),
            point_validity_weight=float(point_validity_weight),
            target_counts=room_target_counts,
        )
    else:
        polygon_zero = predictions["room_polygons"].sum() * 0.0
        polygon_total = polygon_zero
        polygon_components = {
            key: polygon_zero.detach()
            for key in (
                "room_polygon_loss",
                "room_polygon_class_loss",
                "room_polygon_corner_loss",
                "room_polygon_chamfer_loss",
                "room_polygon_mask_dice_loss",
                "room_polygon_valid_loss",
                "room_polygon_point_valid_loss",
            )
        }
    quality_loss = mask_logits.sum() * 0.0
    if float(query_quality_weight) > 0.0:
        if "query_quality_logits" not in predictions:
            raise ValueError("query quality loss requires an enabled quality head")
        if room_matches is None:
            raise RuntimeError("query quality requires room matching")
        mask_probability = mask_logits.sigmoid()
        quality_target = torch.zeros_like(predictions["query_quality_logits"])
        pairs = [
            (match.predicted_indices, match.target_indices)
            for match in room_matches
        ]
        queries, targets_index, match_valid = assignment_index_tensors(
            pairs,
            device=mask_logits.device,
        )
        batch_index = torch.arange(
            mask_logits.shape[0],
            device=mask_logits.device,
        )[:, None]
        predicted_mask = mask_probability[batch_index, queries]
        expected_mask = target_masks[batch_index, targets_index]
        intersection = (predicted_mask * expected_mask).sum(dim=(1, 2))
        denominator = (
            predicted_mask.sum(dim=(1, 2))
            + expected_mask.sum(dim=(1, 2))
        )
        dice = 2.0 * intersection / denominator.clamp_min(1.0)
        class_correct = (
            predictions["room_class"][batch_index, queries].argmax(dim=-1)
            == targets["room_classes"][batch_index, targets_index]
        )
        quality_target.scatter_(
            1,
            queries,
            dice * class_correct.to(dice.dtype) * match_valid.to(dice.dtype),
            reduce="add",
        )
        quality_loss = F.binary_cross_entropy_with_logits(
            predictions["query_quality_logits"],
            quality_target,
            pos_weight=predictions["query_quality_logits"].new_tensor(2.0),
        )
    class_weights = (
        balanced_room_class_weights(
            targets["room_classes"],
            target_valid,
            len(taxonomy.room_ids),
            power=float(class_balance_power),
        )
        if float(class_balance_power) > 0.0
        else None
    )
    identity_predictions = {
        "class_logits": predictions["room_instance_class_logits"],
        "mask_logits": mask_logits,
        "aux_mask_logits": predictions["room_instance_aux_mask_logits"],
        "aux_class_logits": predictions["room_instance_aux_class_logits"],
    }
    if "room_instance_partition_logits" in predictions:
        identity_predictions["room_partition_logits"] = predictions[
            "room_instance_partition_logits"
        ]
    identity_total, identity_components = room_mask_loss(
        identity_predictions,
        targets["room_classes"],
        target_masks,
        target_valid,
        class_weights=class_weights,
        program_loss_weight=0.2,
        unmatched_mask_weight=0.25,
        partition_consistency_weight=float(partition_consistency_weight),
        partition_conflict_only=bool(partition_conflict_only),
        partition_label_smoothing=float(partition_label_smoothing),
        target_counts=room_target_counts,
        matches=room_matches if joint_room_assignment else None,
    )
    if float(query_quality_weight) > 0.0:
        identity_total = (
            identity_total
            + float(query_quality_weight) * quality_loss
        )
    program_loss = identity_total * 0.0
    program_components = {}
    if float(room_program_weight) > 0.0:
        if "room_program_class_logits" not in predictions:
            raise ValueError("room program loss requires an enabled program set")
        program_loss, program_components = room_program_set_loss(
            {
                "class_logits": predictions["room_program_class_logits"],
                "presence_logits": predictions[
                    "room_program_presence_logits"
                ],
                "cardinality_logits": predictions[
                    "room_program_cardinality_logits"
                ],
            },
            targets["room_classes"],
            target_valid,
        )
        identity_total = (
            identity_total + float(room_program_weight) * program_loss
        )
    separation_loss = identity_total * 0.0
    if float(query_separation_weight) > 0.0:
        presence = active_query_presence(
            predictions["room_instance_class_logits"]
        )
        separation_loss = (
            supervised_same_class_overlap_loss(
                predictions["room_instance_mask_logits"],
                room_matches,
                targets["room_classes"],
            )
            if supervised_separation
            else active_query_overlap_loss(
                predictions["room_instance_mask_logits"],
                presence,
                class_probability=(
                    predictions["room_instance_class_logits"][
                        ..., :-1
                    ].softmax(dim=-1)
                    if class_aware_separation
                    else None
                ),
            )
        )
        identity_total = (
            identity_total
            + float(query_separation_weight) * separation_loss
        )
    count_total = F.cross_entropy(
        predictions["room_count_logits"],
        targets["room_count"],
    )
    wall_positive_fraction = predictions["wall_exists"].sum() * 0.0
    opening_positive_fraction = predictions["opening_exists"].sum() * 0.0
    if float(wall_weight) > 0.0:
        wall_pairs, positive_rows = match_candidate_queries(
            predictions["wall_endpoints"],
            targets["wall_endpoints"],
            targets["wall_valid"],
            torch.ones_like(predictions["wall_exists"], dtype=torch.bool),
            target_counts=targets.get("wall_target_counts"),
        )
        positive = torch.from_numpy(np.stack(positive_rows)).to(
            predictions["wall_exists"].device,
            dtype=predictions["wall_exists"].dtype,
        )
        wall_positive_fraction = positive.float().mean()
        wall_bce = F.binary_cross_entropy_with_logits(
            predictions["wall_exists"],
            positive,
            pos_weight=_balanced_positive_weight(
                predictions["wall_exists"],
                positive,
            ),
        )
        wall_rows, wall_targets, wall_match_valid = assignment_index_tensors(
            wall_pairs,
            device=predictions["wall_endpoints"].device,
        )
        wall_batch = torch.arange(
            predictions["wall_endpoints"].shape[0],
            device=predictions["wall_endpoints"].device,
        )[:, None]
        predicted_endpoints = predictions["wall_endpoints"][wall_batch, wall_rows]
        expected_endpoints = targets["wall_endpoints"][wall_batch, wall_targets]
        direct = F.smooth_l1_loss(
            predicted_endpoints,
            expected_endpoints,
            reduction="none",
        ).mean(dim=(2, 3))
        reverse = F.smooth_l1_loss(
            predicted_endpoints,
            expected_endpoints[:, :, [1, 0]],
            reduction="none",
        ).mean(dim=(2, 3))
        endpoint_loss = torch.minimum(
            _sample_mean(direct, wall_match_valid),
            _sample_mean(reverse, wall_match_valid),
        )
        type_losses = F.binary_cross_entropy_with_logits(
            predictions["wall_exterior_logits"][wall_batch, wall_rows],
            targets["wall_exterior"][wall_batch, wall_targets],
            reduction="none",
        )
        type_loss = _sample_mean(type_losses, wall_match_valid)
        wall_total = wall_bce
        wall_total = wall_total + 2.0 * endpoint_loss + 0.5 * type_loss
    else:
        wall_total = predictions["wall_exists"].sum() * 0.0
    if float(opening_weight) > 0.0:
        opening_pairs, opening_positive = match_openings(
            predictions["opening_centers"],
            predictions["opening_widths"],
            targets["opening_center"],
            targets["opening_width"],
            targets["opening_valid"],
            target_counts=targets.get("opening_target_counts"),
        )
        opening_total = F.binary_cross_entropy_with_logits(
            predictions["opening_exists"],
            opening_positive,
            pos_weight=_balanced_positive_weight(
                predictions["opening_exists"],
                opening_positive,
            ),
        )
        opening_positive_fraction = opening_positive.float().mean()
        opening_rows, opening_targets, opening_match_valid = (
            assignment_index_tensors(
                opening_pairs,
                device=predictions["opening_centers"].device,
            )
        )
        opening_batch = torch.arange(
            predictions["opening_centers"].shape[0],
            device=predictions["opening_centers"].device,
        )[:, None]
        center_losses = F.smooth_l1_loss(
            predictions["opening_centers"][opening_batch, opening_rows],
            targets["opening_center"][opening_batch, opening_targets],
            reduction="none",
        ).mean(dim=-1)
        width_losses = F.smooth_l1_loss(
            predictions["opening_widths"][opening_batch, opening_rows],
            targets["opening_width"][opening_batch, opening_targets],
            reduction="none",
        )
        host_target = targets["opening_host"][opening_batch, opening_targets]
        host_valid = host_target >= 0
        host_logits = predictions["opening_host_logits"][
            opening_batch,
            opening_rows,
        ]
        host_losses = F.cross_entropy(
            host_logits.reshape(-1, host_logits.shape[-1]),
            host_target.long().clamp_min(0).reshape(-1),
            reduction="none",
        ).reshape_as(host_target)
        opening_total = (
            opening_total
            + 2.0 * _sample_mean(center_losses, opening_match_valid)
            + _sample_mean(width_losses, opening_match_valid)
            + _sample_mean(
                host_losses,
                opening_match_valid & host_valid,
                include=opening_match_valid.any(dim=1),
            )
        )
    else:
        opening_total = predictions["opening_exists"].sum() * 0.0
    if float(adjacency_weight) > 0.0:
        pair_mask = target_valid[:, :, None] & target_valid[:, None, :]
        adjacency_total = F.binary_cross_entropy_with_logits(
            predictions["room_adjacency_logits"][pair_mask],
            targets["adjacency"][pair_mask],
        )
    else:
        adjacency_total = predictions["room_adjacency_logits"].sum() * 0.0
    token_total = count_total * 0.0
    token_structure_total = count_total * 0.0
    token_structure_components: dict[str, torch.Tensor] = {}
    token_grammar_violation_rate = token_total.detach()
    if float(token_weight) > 0.0:
        if "token_logits" not in predictions or "token_ids" not in targets:
            raise ValueError("token loss requires token_logits and token_ids")
        if token_vocabulary is None:
            token_valid = targets.get("token_loss_mask")
            if (
                token_valid is None
                or token_valid.shape != targets["token_ids"].shape
            ):
                token_valid = targets.get("token_mask")
            token_total = token_sequence_loss(
                predictions["token_logits"],
                targets["token_ids"],
                token_valid,
            )
        else:
            token_result = token_training_losses(
                predictions,
                targets,
                token_vocabulary,
                kind_weights=token_kind_weights,
                stage_weights=token_stage_weights,
                structure_weight=float(token_structure_weight),
                room_labels=tuple(
                    taxonomy.class_names[class_id]
                    for class_id in taxonomy.room_ids
                ),
            )
            token_total = token_result.total
            token_structure_total = token_result.structure_total
            token_grammar_violation_rate = token_result.grammar_violation_rate
            token_structure_components = token_result.components
    loss_terms = {
        "polygon": float(polygon_weight) * polygon_total,
        "identity": float(identity_weight) * identity_total,
        "count": float(count_weight) * count_total,
        "wall": float(wall_weight) * wall_total,
        "opening": float(opening_weight) * opening_total,
        "adjacency": float(adjacency_weight) * adjacency_total,
        "token": float(token_weight) * token_total,
        "token_structure": (
            float(token_structure_weight) * token_structure_total
        ),
    }
    if loss_balancer is not None and balance_parameters:
        loss_balancer.maybe_update(
            loss_terms,
            balance_parameters,
            int(balance_step),
        )
    if diagnostic_out is not None and diagnostic_parameters:
        diagnostic_out.update(
            gradient_diagnostics(loss_terms, diagnostic_parameters)
        )
    if loss_terms_out is not None:
        loss_terms_out.update(loss_terms)
    hierarchy_components: dict[str, torch.Tensor] = {}
    if hierarchy_weights:
        hierarchy = typed_state_hierarchy_loss(loss_terms, hierarchy_weights)
        total = hierarchy.total
        hierarchy_components = hierarchy.components
    else:
        total = (
            loss_balancer.weighted_loss(loss_terms)
            if loss_balancer is not None
            else torch.stack(tuple(loss_terms.values())).sum()
        )
    verifier_components: dict[str, torch.Tensor] = {}
    if (
        float(verifier_policy_weight) > 0.0
        and "verifier_reward" in targets
        and "token_logits" in predictions
    ):
        policy_loss, verifier_components = verifier_policy_loss(
            predictions["token_logits"],
            targets["token_ids"],
            targets.get("token_loss_mask", targets["token_mask"]),
            targets["verifier_reward"],
            baseline=float(verifier_reward_baseline),
            reward_mask=targets.get("verifier_reward_mask"),
        )
        total = total + float(verifier_policy_weight) * policy_loss
    components = {
        **polygon_components,
        **{
            key: value
            for key, value in identity_components.items()
            if not key.startswith("_")
        },
        "room_count_ce": count_total.detach(),
        "room_query_separation": separation_loss.detach(),
        "rplan_query_quality_loss": quality_loss.detach(),
        **program_components,
        "room_program_loss": program_loss.detach(),
        "room_class_balance_enabled": count_total.new_tensor(
            float(class_weights is not None)
        ),
        "wall_total_loss": wall_total.detach(),
        "wall_positive_fraction": wall_positive_fraction.detach(),
        "wall_predicted_positive_fraction": (
            predictions["wall_exists"].sigmoid() >= 0.5
        ).float().mean(),
        "opening_total_loss": opening_total.detach(),
        "opening_positive_fraction": opening_positive_fraction.detach(),
        "opening_predicted_positive_fraction": (
            predictions["opening_exists"].sigmoid() >= 0.5
        ).float().mean(),
        "room_adjacency_bce": adjacency_total.detach(),
        "canonical_token_ce": token_total.detach(),
        "canonical_token_structure_loss": token_structure_total.detach(),
        "canonical_token_weight_effective": token_total.new_tensor(
            float(token_weight)
        ),
        "canonical_token_structure_weight_effective": token_total.new_tensor(
            float(token_structure_weight)
        ),
        **token_structure_components,
        "canonical_token_grammar_violation_rate": token_grammar_violation_rate,
        "canonical_token_target_length": (
            targets["token_length"].float().mean()
            if "token_length" in targets
            else token_total.detach()
        ),
        "canonical_token_oov_count": (
            targets["token_oov_count"].float().mean()
            if "token_oov_count" in targets
            else token_total.detach()
        ),
        **hierarchy_components,
        **verifier_components,
        "total_loss": total.detach(),
    }
    return total, components


__all__ = ["match_candidate_queries", "unified_generator_loss"]
