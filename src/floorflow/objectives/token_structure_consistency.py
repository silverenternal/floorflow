"""Differentiable consistency between structural tokens and vector heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.objectives.gpu_assignment import sinkhorn_log_assignment
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.objectives.triton_ops import endpoint_incidence


def _masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    selected = mask.to(losses.dtype)
    return (losses * selected).sum() / selected.sum().clamp_min(1.0)


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = F.cross_entropy(
        flat_logits,
        flat_targets.long().clamp_min(0),
        reduction="none",
    )
    selected = flat_mask.to(losses.dtype)
    return (losses * selected).sum() / selected.sum().clamp_min(1.0)


def token_structure_consistency_loss(
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    *,
    room_exists_logits: torch.Tensor,
    wall_exists_logits: torch.Tensor,
    opening_exists_logits: torch.Tensor,
    room_mask_logits: torch.Tensor,
    target_valid: torch.Tensor | None = None,
    adjacency_logits: torch.Tensor | None = None,
    target_adjacency: torch.Tensor | None = None,
    room_class_logits: torch.Tensor | None = None,
    target_room_classes: torch.Tensor | None = None,
    wall_exterior_logits: torch.Tensor | None = None,
    target_wall_exterior: torch.Tensor | None = None,
    wall_valid: torch.Tensor | None = None,
    opening_host_logits: torch.Tensor | None = None,
    target_opening_host: torch.Tensor | None = None,
    opening_valid: torch.Tensor | None = None,
    room_point_valid_logits: torch.Tensor | None = None,
    target_room_point_valid: torch.Tensor | None = None,
    room_polygons: torch.Tensor | None = None,
    target_room_polygons: torch.Tensor | None = None,
    wall_endpoints: torch.Tensor | None = None,
    junction_points: torch.Tensor | None = None,
    target_junction_degree: torch.Tensor | None = None,
    junction_valid: torch.Tensor | None = None,
    predicted_junction_degree_logits: torch.Tensor | None = None,
    predicted_junction_coordinates: torch.Tensor | None = None,
    target_junction_points: torch.Tensor | None = None,
    codetree_polygons: torch.Tensor | None = None,
    codetree_vq_loss: torch.Tensor | None = None,
    target_entity_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align token structure with the typed vector graph."""

    if token_ids.shape != token_mask.shape:
        raise ValueError("token ids and mask must align")
    targets = {
        name: token_ids.new_zeros(
            token_ids.shape[0],
            dtype=torch.float32,
        )
        for name in ("rooms", "walls", "openings")
    }
    kind_to_target = {
        "ROOM_BEGIN": "rooms",
        "WALL": "walls",
        "OPENING": "openings",
    }
    if target_entity_counts is not None:
        targets["rooms"] = target_entity_counts[:, 0].float()
        targets["walls"] = target_entity_counts[:, 1].float()
        targets["openings"] = target_entity_counts[:, 2].float()
    else:
        entity_class = torch.zeros(
            vocabulary.model_vocab_size,
            dtype=torch.long,
            device=token_ids.device,
        )
        for target_index, target_name in enumerate(targets, start=1):
            kind = next(
                kind_name
                for kind_name, name in kind_to_target.items()
                if name == target_name
            )
            token_kind_ids = [
                index
                for index, value in enumerate(vocabulary.tokens)
                if value.split("|", 1)[0] == kind
            ]
            if token_kind_ids:
                entity_class[torch.tensor(token_kind_ids, device=token_ids.device)] = (
                    target_index
                )
        safe_ids = token_ids.long().clamp(0, entity_class.numel() - 1)
        sequence_class = entity_class[safe_ids] * token_mask.to(torch.long)
        for target_index, target_name in enumerate(targets, start=1):
            targets[target_name] = (
                sequence_class == target_index
            ).sum(dim=1).float()
    predictions = {
        "rooms": room_exists_logits.sigmoid().sum(dim=-1),
        "walls": wall_exists_logits.sigmoid().sum(dim=-1),
        "openings": opening_exists_logits.sigmoid().sum(dim=-1),
    }
    count_losses = {
        name: F.smooth_l1_loss(
            predictions[name] / (targets[name] + 1.0),
            targets[name] / (targets[name] + 1.0),
        )
        for name in targets
    }
    probability = room_mask_logits.sigmoid()
    overlap = (
        probability.sum(dim=1).square()
        - probability.square().sum(dim=1)
    ) * 0.5
    union = (
        max(1, probability.shape[1] - 1) * probability.sum(dim=1)
        - overlap
    )
    partition_excess = (overlap / union.clamp_min(1.0e-6)).mean()
    components = {
        "token_room_count_consistency": count_losses["rooms"].detach(),
        "token_wall_count_consistency": count_losses["walls"].detach(),
        "token_opening_count_consistency": count_losses["openings"].detach(),
        "token_room_partition_excess": partition_excess.detach(),
    }
    total = (
        count_losses["rooms"]
        + count_losses["walls"]
        + count_losses["openings"]
        + 0.2 * partition_excess
    )
    if (
        target_valid is not None
        and adjacency_logits is not None
        and target_adjacency is not None
    ):
        pair_mask = target_valid[:, :, None] & target_valid[:, None, :]
        adjacency_loss = _masked_binary_cross_entropy(
            adjacency_logits,
            target_adjacency,
            pair_mask,
        )
        total = total + 0.25 * adjacency_loss
        components["token_adjacency_consistency"] = adjacency_loss.detach()
    if (
        wall_valid is not None
        and wall_exterior_logits is not None
        and target_wall_exterior is not None
    ):
        wall_type_loss = _masked_binary_cross_entropy(
            wall_exterior_logits,
            target_wall_exterior,
            wall_valid,
        )
        total = total + 0.25 * wall_type_loss
        components["token_wall_type_consistency"] = wall_type_loss.detach()
    if (
        opening_valid is not None
        and opening_host_logits is not None
        and target_opening_host is not None
    ):
        host_valid = opening_valid & (target_opening_host >= 0)
        host_loss = _masked_cross_entropy(
            opening_host_logits,
            target_opening_host,
            host_valid,
        )
        total = total + 0.2 * host_loss
        components["token_opening_host_consistency"] = host_loss.detach()
    if (
        target_valid is not None
        and room_class_logits is not None
        and target_room_classes is not None
    ):
        class_loss = _masked_cross_entropy(
            room_class_logits,
            target_room_classes,
            target_valid,
        )
        total = total + 0.2 * class_loss
        components["token_room_label_consistency"] = class_loss.detach()
    if room_point_valid_logits is not None and target_room_point_valid is not None:
        predicted_count = room_point_valid_logits.sigmoid().sum(dim=-1)
        expected_count = target_room_point_valid.float().sum(dim=-1)
        vertex_loss = F.smooth_l1_loss(
            predicted_count / (expected_count + 1.0),
            expected_count / (expected_count + 1.0),
        )
        total = total + 0.2 * vertex_loss
        components["token_ring_vertex_consistency"] = vertex_loss.detach()
    if (
        room_polygons is not None
        and target_room_polygons is not None
        and target_room_point_valid is not None
        and target_valid is not None
    ):
        counts = target_room_point_valid.long().sum(dim=-1).clamp_min(1)
        last_index = (counts - 1).clamp_max(target_room_polygons.shape[2] - 1)
        batch_index = torch.arange(
            target_room_polygons.shape[0],
            device=target_room_polygons.device,
        )[:, None]
        room_index = torch.arange(
            target_room_polygons.shape[1],
            device=target_room_polygons.device,
        )[None, :]
        expected_last = target_room_polygons[batch_index, room_index, last_index]
        expected_gap = (
            expected_last - target_room_polygons[:, :, 0]
        ).norm(dim=-1)
        predicted_gap = (
            room_polygons[:, :, -1] - room_polygons[:, :, 0]
        ).norm(dim=-1)
        closure_loss = F.smooth_l1_loss(
            predicted_gap[target_valid],
            expected_gap[target_valid],
        )
        total = total + 0.1 * closure_loss
        components["token_ring_closure_consistency"] = closure_loss.detach()
    if (
        wall_endpoints is not None
        and wall_exists_logits is not None
        and junction_points is not None
        and target_junction_degree is not None
        and junction_valid is not None
    ):
        wall_points = wall_endpoints.flatten(1, 2)
        incident = endpoint_incidence(junction_points, wall_points)
        wall_probability = wall_exists_logits.sigmoid().repeat_interleave(
            2,
            dim=-1,
        )
        predicted_degree = (incident * wall_probability[:, None, :]).sum(dim=-1)
        valid = junction_valid.reshape(predicted_degree.shape)
        expected_degree = target_junction_degree.float()
        degree_loss = F.smooth_l1_loss(
            predicted_degree / (expected_degree + 1.0),
            expected_degree / (expected_degree + 1.0),
            reduction="none",
        )
        selected = valid.to(degree_loss.dtype)
        junction_loss = (degree_loss * selected).sum() / selected.sum().clamp_min(
            1.0
        )
        total = total + 0.5 * junction_loss
        components["token_junction_degree_consistency"] = junction_loss.detach()
    if (
        predicted_junction_degree_logits is not None
        and target_junction_degree is not None
        and junction_valid is not None
    ):
        degrees = target_junction_degree.long().clamp_max(
            predicted_junction_degree_logits.shape[-1] - 1
        )
        degree_ce = _masked_cross_entropy(
            predicted_junction_degree_logits,
            degrees,
            junction_valid,
        )
        total = total + 0.5 * degree_ce
        components["token_graph_junction_degree_ce"] = degree_ce.detach()
    if (
        predicted_junction_coordinates is not None
        and target_junction_points is not None
        and junction_valid is not None
    ):
        coordinate_losses = F.smooth_l1_loss(
            predicted_junction_coordinates,
            target_junction_points,
            reduction="none",
        )
        selected = junction_valid.to(coordinate_losses.dtype)
        coordinate_loss = (
            coordinate_losses * selected[..., None]
        ).sum() / (selected.sum() * coordinate_losses.shape[-1]).clamp_min(1.0)
        total = total + 0.5 * coordinate_loss
        components["token_graph_junction_coordinate_loss"] = (
            coordinate_loss.detach()
        )
    if (
        codetree_polygons is not None
        and target_room_polygons is not None
        and target_valid is not None
    ):
        predicted_polygons = codetree_polygons[target_valid]
        expected_polygons = target_room_polygons[target_valid]
        if predicted_polygons.shape[0]:
            predicted_centers = predicted_polygons.mean(dim=1)
            expected_centers = expected_polygons.mean(dim=1)
            assignment_cost = torch.cdist(predicted_centers, expected_centers)
            assignment = sinkhorn_log_assignment(
                assignment_cost[None],
            )
            aligned_polygons = torch.einsum(
                "bst,bspd->btpd",
                assignment,
                predicted_polygons[None],
            )[0]
            codetree_loss = F.smooth_l1_loss(
                aligned_polygons,
                expected_polygons,
            )
        else:
            codetree_loss = codetree_polygons.sum() * 0.0
        total = total + 0.2 * codetree_loss
        components["codetree_polygon_loss"] = codetree_loss.detach()
    if codetree_vq_loss is not None:
        total = total + 0.05 * codetree_vq_loss
        components["codetree_vq_loss"] = codetree_vq_loss.detach()
    return total, components


__all__ = ["token_structure_consistency_loss"]
