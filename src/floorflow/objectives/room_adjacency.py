"""Self-consistent room adjacency supervision from query masks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.geometry.deterministic_pooling import fixed_grid_pool
from floorflow.objectives.per_case_binary import (
    per_case_binary_cross_entropy,
    per_case_soft_dice,
)


def soft_mask_adjacency(
    mask_logits: torch.Tensor,
    *,
    grid: int = 48,
    dilation: int = 1,
) -> torch.Tensor:
    """Infer symmetric adjacency probabilities from query-mask contact."""

    return adjacency_from_probability(
        fixed_grid_pool(mask_logits.sigmoid(), (grid, grid)),
        dilation=dilation,
    )


def adjacency_from_probability(
    probability: torch.Tensor,
    *,
    dilation: int = 1,
) -> torch.Tensor:
    """Infer symmetric adjacency probabilities from soft masks."""

    area = probability.flatten(2).sum(dim=-1).clamp_min(1.0e-6)
    dilated = F.max_pool2d(
        probability,
        kernel_size=2 * int(dilation) + 1,
        stride=1,
        padding=int(dilation),
    )
    contact = torch.einsum("bihw,bjhw->bij", dilated, probability)
    scale = torch.sqrt(area[:, :, None] * area[:, None, :]).clamp_min(
        1.0e-6
    )
    adjacency = (contact / scale).clamp(0.0, 1.0)
    return 0.5 * (adjacency + adjacency.transpose(1, 2))


def room_adjacency_consistency_loss(
    output: dict[str, torch.Tensor],
    max_rooms: int,
    target_masks: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    *,
    presence_floor: float = 0.05,
    assignment_min_iou: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align predicted room adjacency with differentiable mask contact."""

    logits = output["room_adjacency_logits"][:, :max_rooms, :max_rooms]
    masks = (
        output.get("room_instance_mask_logits")
        if "room_instance_mask_logits" in output
        else output["mask_logits"]
    )[:, :max_rooms]
    self_target = soft_mask_adjacency(masks).detach()
    graph_presence = output.get("room_graph_query_presence")
    presence = None
    if graph_presence is not None:
        if graph_presence.shape != masks.shape[:2]:
            raise ValueError(
                "graph presence must match query-mask cardinality"
            )
        presence = graph_presence[:, :max_rooms].float()
    else:
        existence = output.get(
            "room_instance_query_existence_logits",
            output.get("query_existence_logits"),
        )
        if existence is not None:
            presence = existence[:, :max_rooms].sigmoid()
    pair_weight = None
    if presence is None:
        area = masks.sigmoid().flatten(2).mean(dim=-1)
        valid = area > 1.0e-4
    else:
        valid = presence > 0.05
    pair_valid = valid[:, :, None] & valid[:, None, :]
    target = self_target
    if target_masks is not None and target_valid is not None:
        target_masks = target_masks[:, :max_rooms].float()
        target_valid = target_valid[:, :max_rooms].bool()
        predicted_grid = fixed_grid_pool(
            masks.sigmoid(),
            (48, 48),
        )
        target_grid = fixed_grid_pool(target_masks, (48, 48))
        intersection = torch.einsum(
            "bqhw,blhw->bql",
            predicted_grid,
            target_grid,
        )
        denominator = (
            predicted_grid.flatten(2).sum(dim=-1)[:, :, None]
            + target_grid.flatten(2).sum(dim=-1)[:, None, :]
            - intersection
        )
        iou = intersection / denominator.clamp_min(1.0e-6)
        iou = iou.masked_fill(~target_valid[:, None], 0.0)
        best_target, assigned = _one_to_one_room_assignment(
            iou,
            target_valid,
            min_iou=float(assignment_min_iou),
        )
        target_adjacency = adjacency_from_probability(target_grid).detach()
        rows = best_target[:, :, None].expand(
            -1,
            -1,
            target_adjacency.shape[-1],
        )
        columns = best_target[:, None, :].expand(-1, max_rooms, -1)
        target = torch.gather(
            torch.gather(target_adjacency, 1, rows),
            2,
            columns,
        )
        pair_valid = assigned[:, :, None] & assigned[:, None, :]
        if presence is not None:
            pair_presence = presence[:, :, None] * presence[:, None, :]
            floor = float(presence_floor)
            pair_weight = floor + (1.0 - floor) * pair_presence
    diagonal = torch.eye(
        max_rooms,
        device=logits.device,
        dtype=torch.bool,
    )[None]
    pair_valid = pair_valid & ~diagonal
    if bool(pair_valid.any()):
        loss = per_case_binary_cross_entropy(
            logits.sigmoid(),
            target,
            pair_valid,
            weight=pair_weight,
        )
        target_mean = target[pair_valid].mean()
    else:
        loss = logits.sum() * 0.0
        target_mean = target.sum()
    return loss, {
        "room_adjacency_consistency_loss": loss.detach(),
        "room_adjacency_valid_pairs": pair_valid.sum().detach(),
        "room_adjacency_target_mean": target_mean.detach(),
        "room_adjacency_assigned_queries": (
            assigned.sum().detach()
            if target_masks is not None and target_valid is not None
            else pair_valid.sum().detach()
        ),
    }


def room_type_adjacency_consistency_loss(
    output: dict[str, torch.Tensor],
    max_rooms: int,
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    grid: int = 48,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the canonical room-type edge set, not only query contact.

    The evaluated HouseMind-style graph collapses room instances to labels and
    records an edge when any pair of distinct labels touches. Instance-edge
    supervision alone leaves this consumer unconstrained. This loss projects
    query adjacency and class probabilities into exactly that type-edge space.
    """

    if "room_adjacency_logits" not in output:
        raise ValueError("type adjacency requires room_adjacency_logits")
    if "room_instance_class_logits" not in output:
        raise ValueError("type adjacency requires room class logits")
    logits = output["room_adjacency_logits"][:, :max_rooms, :max_rooms]
    full_class_shape = output["room_instance_class_logits"].shape[:2]
    class_logits = output["room_instance_class_logits"][
        :, :max_rooms, :-1
    ]
    if class_logits.shape[:2] != logits.shape[:2]:
        raise ValueError("query adjacency and classes must share query slots")
    class_probability = class_logits.float().softmax(dim=-1)
    graph_presence = output.get("room_graph_query_presence")
    if graph_presence is not None:
        if graph_presence.shape != full_class_shape:
            raise ValueError(
                "graph presence must match query-class cardinality"
            )
        class_probability = (
            class_probability
            * graph_presence[:, :max_rooms, None].float()
        )
    predicted = type_adjacency_probability(
        logits.sigmoid(),
        class_probability,
    )

    rooms = min(max_rooms, target_masks.shape[1])
    target_masks = target_masks[:, :rooms].float()
    target_classes = target_classes[:, :rooms].long()
    target_valid = target_valid[:, :rooms].bool()
    target_contact = adjacency_from_probability(
        fixed_grid_pool(target_masks, (grid, grid))
    ).detach()
    target_one_hot = F.one_hot(
        target_classes.clamp_min(0),
        num_classes=class_probability.shape[-1],
    ).float()
    target_one_hot = target_one_hot * target_valid[..., None].float()
    target = type_adjacency_probability(
        target_contact,
        target_one_hot,
    ).detach()
    present = target_one_hot.amax(dim=1).bool()
    upper = torch.triu(
        torch.ones(
            target.shape[-2:],
            device=target.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )[None]
    relevant = present[:, :, None] | present[:, None, :]
    valid = upper & relevant
    if bool(valid.any()):
        expected = target[valid]
        positive = expected.sum().clamp_min(1.0)
        negative = (1.0 - expected).sum().clamp_min(1.0)
        pos_weight = (negative / positive).detach().clamp(1.0, 20.0)
        loss = per_case_binary_cross_entropy(
            predicted,
            target,
            valid,
            pos_weight=pos_weight,
        )
    else:
        loss = predicted.sum() * 0.0
        expected = target.new_zeros(())
    return loss, {
        "room_type_adjacency_loss": loss.detach(),
        "room_type_adjacency_valid_pairs": valid.sum().detach(),
        "room_type_adjacency_target_mean": expected.mean().detach(),
    }


def room_type_node_consistency_loss(
    output: dict[str, torch.Tensor],
    max_rooms: int,
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the unique-room-type node set used by HouseMind metrics."""

    if "room_instance_class_logits" not in output:
        raise ValueError("type node loss requires room class logits")
    full_class_shape = output["room_instance_class_logits"].shape[:2]
    class_probability = output["room_instance_class_logits"][
        :, :max_rooms, :-1
    ].float().softmax(dim=-1)
    graph_presence = output.get("room_graph_query_presence")
    if graph_presence is None:
        existence = output.get(
            "room_instance_query_existence_logits",
            output.get("query_existence_logits"),
        )
        if existence is None:
            raise ValueError("type node loss requires query existence logits")
        graph_presence = existence.sigmoid()
    elif graph_presence.shape != full_class_shape:
        raise ValueError(
            "graph presence must match query-class cardinality"
        )
    query_presence = graph_presence[:, :max_rooms].float()
    query_type_probability = (
        query_presence[..., None] * class_probability
    ).clamp(1.0e-6, 1.0 - 1.0e-6)
    type_presence = 1.0 - torch.prod(
        1.0 - query_type_probability,
        dim=1,
    ).clamp(1.0e-6, 1.0 - 1.0e-6)
    type_presence = type_presence.clamp(1.0e-6, 1.0 - 1.0e-6)

    rooms = min(max_rooms, target_classes.shape[1])
    target_classes = target_classes[:, :rooms].long()
    target_valid = target_valid[:, :rooms].bool()
    target_one_hot = F.one_hot(
        target_classes.clamp_min(0),
        num_classes=class_probability.shape[-1],
    ).float()
    target = target_one_hot.mul(
        target_valid[..., None].float()
    ).amax(dim=1)
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (negative / positive).detach().clamp(1.0, 20.0)
    loss = per_case_binary_cross_entropy(
        type_presence,
        target,
        pos_weight=pos_weight,
    )
    return loss, {
        "room_type_node_loss": loss.detach(),
        "room_type_node_target_mean": target.mean().detach(),
        "room_type_node_prediction_mean": type_presence.mean().detach(),
    }


def dense_room_type_graph_losses(
    room_probability: torch.Tensor,
    target: torch.Tensor,
    taxonomy,
    *,
    adjacency_weight: float = 0.25,
    node_weight: float = 0.1,
    adjacency_overlap_weight: float = 0.0,
    node_overlap_weight: float = 0.0,
    adjacency_tolerance: int = 2,
    return_loss_terms: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Supervise the final dense semantic partition in type-graph space."""

    if room_probability.ndim != 4:
        raise ValueError("room probability must be [B,C,H,W]")
    if target.shape != room_probability.shape[:1] + room_probability.shape[-2:]:
        room_probability = F.interpolate(
            room_probability.float(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if target.shape != room_probability.shape[:1] + room_probability.shape[-2:]:
        raise ValueError("target must match room probability [B,H,W]")
    probability = room_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=probability.device,
    )
    target_one_hot = F.one_hot(
        target.long(),
        num_classes=taxonomy.num_classes,
    ).permute(0, 3, 1, 2)[:, room_ids].float()
    target_node = target_one_hot.flatten(2).amax(dim=-1).bool()
    predicted_node = probability.flatten(2).amax(dim=-1)
    node_positive = target_node.sum(dim=1, keepdim=True).clamp_min(1.0)
    node_negative = (
        ~target_node
    ).sum(dim=1, keepdim=True).clamp_min(1.0)
    node_bce_loss = per_case_binary_cross_entropy(
        predicted_node,
        target_node.float(),
        pos_weight=(node_negative / node_positive).clamp(1.0, 20.0),
    )
    node_dice_loss = per_case_soft_dice(
        predicted_node,
        target_node.float(),
    )
    node_loss = (
        node_bce_loss
        + float(node_overlap_weight) * node_dice_loss
    )

    def contact(value: torch.Tensor) -> torch.Tensor:
        radius = max(1, int(adjacency_tolerance) + 1)
        neighbor = F.max_pool2d(
            value,
            2 * radius + 1,
            stride=1,
            padding=radius,
        )
        intersection = torch.bmm(
            value.flatten(2),
            neighbor.flatten(2).transpose(1, 2),
        )
        area = value.flatten(2).sum(dim=-1)
        scale = torch.sqrt(area[:, :, None] * area[:, None, :]).clamp_min(
            1.0e-6
        )
        matrix = (intersection / scale).clamp(0.0, 1.0)
        return 0.5 * (matrix + matrix.transpose(1, 2))

    target_contact = contact(target_one_hot)
    edge_target = target_contact > 0.0
    predicted_contact = contact(probability)
    upper = torch.triu(
        torch.ones(
            room_probability.shape[1],
            room_probability.shape[1],
            device=probability.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )[None]
    relevant = target_node[:, :, None] | target_node[:, None, :]
    valid = upper & relevant
    if bool(valid.any()):
        edge_target_float = edge_target.float()
        valid_float = valid.float()
        positive = (
            edge_target_float * valid_float
        ).sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        negative = (
            (1.0 - edge_target_float) * valid_float
        ).sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        edge_bce_loss = per_case_binary_cross_entropy(
            predicted_contact,
            edge_target_float,
            valid,
            pos_weight=(negative / positive).clamp(1.0, 20.0),
        )
        edge_dice_loss = per_case_soft_dice(
            predicted_contact,
            edge_target_float,
            valid,
        )
        edge_loss = (
            edge_bce_loss
            + float(adjacency_overlap_weight) * edge_dice_loss
        )
    else:
        edge_loss = predicted_contact.sum() * 0.0
        edge_bce_loss = edge_loss
        edge_dice_loss = edge_loss
        expected = predicted_contact.new_zeros(())
    total = (
        float(node_weight) * node_loss
        + float(adjacency_weight) * edge_loss
    )
    components = {
        "dense_type_node_loss": node_loss.detach(),
        "dense_type_node_bce_loss": node_bce_loss.detach(),
        "dense_type_node_dice_loss": node_dice_loss.detach(),
        "dense_type_edge_loss": edge_loss.detach(),
        "dense_type_edge_bce_loss": edge_bce_loss.detach(),
        "dense_type_edge_dice_loss": edge_dice_loss.detach(),
        "dense_type_edge_target_mean": edge_target[valid].float().mean().detach()
        if bool(valid.any())
        else expected,
        "dense_type_total": total.detach(),
    }
    if return_loss_terms:
        return total, components, {
            "dense_type_adjacency": edge_loss,
            "dense_type_node": node_loss,
        }
    return total, components


def type_adjacency_probability(
    query_adjacency: torch.Tensor,
    class_probability: torch.Tensor,
) -> torch.Tensor:
    """Project query-to-query contact into room-type edge probabilities."""

    if query_adjacency.ndim != 3:
        raise ValueError("query adjacency must be [B,Q,Q]")
    if class_probability.ndim != 3:
        raise ValueError("class probability must be [B,Q,C]")
    if query_adjacency.shape[:2] != class_probability.shape[:2]:
        raise ValueError("adjacency and classes must share query slots")
    queries = int(query_adjacency.shape[-1])
    eye = torch.eye(
        queries,
        device=query_adjacency.device,
        dtype=torch.bool,
    )[None]
    adjacency = query_adjacency.float().masked_fill(eye, 0.0)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    pair_mass = (
        adjacency[:, :, :, None, None]
        * class_probability[:, :, None, :, None]
        * class_probability[:, None, :, None, :]
    ).sum(dim=(1, 2))
    pair_mass = 0.5 * (
        pair_mass + pair_mass.transpose(1, 2)
    )
    return pair_mass.clamp(0.0, 1.0)


def _one_to_one_room_assignment(
    iou: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    min_iou: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign at most one query to each target room."""

    targets = []
    assigned_rows = []
    values = iou.detach().float().cpu().numpy()
    valid = target_valid.detach().cpu().numpy()
    for batch in range(values.shape[0]):
        best_target = np.zeros(values.shape[1], dtype=np.int64)
        assigned = np.zeros(values.shape[1], dtype=bool)
        count = min(values.shape[1], values.shape[2])
        cost = values[batch].copy()
        cost[:, ~valid[batch]] = -1.0
        rows, columns = linear_sum_assignment(-cost)
        for row, column in zip(rows, columns, strict=True):
            if row >= count or not valid[batch, column]:
                continue
            if values[batch, row, column] < float(min_iou):
                continue
            best_target[row] = int(column)
            assigned[row] = True
        targets.append(best_target)
        assigned_rows.append(assigned)
    return (
        torch.from_numpy(np.stack(targets)).to(iou.device),
        torch.from_numpy(np.stack(assigned_rows)).to(iou.device),
    )


def room_adjacency_proxy_metrics(
    logits: torch.Tensor,
    mask_logits: torch.Tensor,
) -> dict[str, float]:
    """Measure adjacency-head agreement with query-mask contact."""

    rooms = min(logits.shape[-1], mask_logits.shape[1])
    target = soft_mask_adjacency(mask_logits[:, :rooms]).detach()
    pair_mask = ~torch.eye(
        rooms,
        device=logits.device,
        dtype=torch.bool,
    )[None]
    probability = logits[:, :rooms, :rooms].sigmoid()[pair_mask]
    expected = target[pair_mask] >= 0.5
    predicted = probability >= 0.5
    true_positive = int((predicted & expected).sum())
    precision = true_positive / max(1, int(predicted.sum()))
    recall = true_positive / max(1, int(expected.sum()))
    return {
        "room_adjacency_precision": precision,
        "room_adjacency_recall": recall,
        "room_adjacency_f1": 2.0
        * precision
        * recall
        / max(1.0e-8, precision + recall),
        "room_adjacency_mae": float(
            (probability - target[pair_mask]).abs().detach().mean()
        ),
    }


__all__ = [
    "adjacency_from_probability",
    "dense_room_type_graph_losses",
    "room_adjacency_consistency_loss",
    "room_adjacency_proxy_metrics",
    "room_type_adjacency_consistency_loss",
    "room_type_node_consistency_loss",
    "soft_mask_adjacency",
    "type_adjacency_probability",
]
