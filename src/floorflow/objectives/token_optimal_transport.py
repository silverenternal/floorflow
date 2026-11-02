"""Differentiable token-level optimal transport for room instances."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import extract_room_instances


def room_instance_token_targets(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int = 32,
    min_area: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe GT room instances as normalized geometric transport tokens."""

    batch_classes = []
    batch_tokens = []
    batch_valid = []
    height, width = labels.shape[-2:]
    room_lookup = {
        class_id: index
        for index, class_id in enumerate(taxonomy.room_ids)
    }
    for item in labels.detach().cpu().numpy():
        instances = sorted(
            extract_room_instances(item, taxonomy, min_area=min_area),
            key=lambda instance: instance.area,
            reverse=True,
        )[:max_rooms]
        classes = torch.zeros(max_rooms, dtype=torch.long)
        tokens = torch.zeros(max_rooms, 5)
        valid = torch.zeros(max_rooms, dtype=torch.bool)
        for index, instance in enumerate(instances):
            x0, y0, x1, y1 = instance.bbox
            area = float(instance.area) / float(height * width)
            tokens[index] = torch.tensor(
                [
                    area,
                    (x0 + x1) / (2.0 * width),
                    (y0 + y1) / (2.0 * height),
                    area**0.5,
                    1.0,
                ]
            )
            classes[index] = room_lookup.get(instance.class_id, 0)
            valid[index] = True
        batch_classes.append(classes)
        batch_tokens.append(tokens)
        batch_valid.append(valid)
    return (
        torch.stack(batch_classes).to(labels.device),
        torch.stack(batch_tokens).to(labels.device, dtype=torch.float32),
        torch.stack(batch_valid).to(labels.device),
    )


def sinkhorn_plan(
    cost: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 20,
) -> torch.Tensor:
    """Return a doubly stochastic assignment plan without target-informed selection."""

    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, source, target]")
    if not torch.isfinite(cost).all():
        raise ValueError("transport cost must be finite")
    batch, source_count, target_count = cost.shape
    log_kernel = -cost / max(float(epsilon), 1.0e-6)
    log_mu = torch.full(
        (batch, source_count),
        -torch.log(torch.tensor(float(source_count), device=cost.device)),
        device=cost.device,
        dtype=cost.dtype,
    )
    log_nu = torch.full(
        (batch, target_count),
        -torch.log(torch.tensor(float(target_count), device=cost.device)),
        device=cost.device,
        dtype=cost.dtype,
    )
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(max(1, int(iterations))):
        u = log_mu - torch.logsumexp(log_kernel + v[:, None, :], dim=2)
        v = log_nu - torch.logsumexp(log_kernel + u[:, :, None], dim=1)
    return torch.exp(log_kernel + u[:, :, None] + v[:, None, :])


def batch_token_assignment(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Exact minibatch OT on pooled tokens; returns ``[B]`` target indices.

    This is the OT-CFM-style minibatch coupling.  It is applied only while
    sampling training pairs, so it never uses test-time information.
    """

    if source_tokens.shape != target_tokens.shape:
        raise ValueError(
            f"source/target token shapes differ: {source_tokens.shape} "
            f"vs {target_tokens.shape}"
        )
    cost = torch.cdist(
        source_tokens.flatten(1).float(),
        target_tokens.flatten(1).float(),
    ).pow(2)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    order = torch.empty(source_tokens.shape[0], dtype=torch.long)
    order[torch.as_tensor(rows)] = torch.as_tensor(columns)
    return order.to(source_tokens.device)


def paired_token_assignment(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match tokens within each sample and return ``[batch, tokens]`` targets.

    This is the set-valued analogue used when every sample already has a
    conditioning-specific target. It must not be used to reorder endpoint
    tensors whose rows carry different area or constraint features.
    """

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must share shape [B, N, D]")
    orders = []
    for predicted_item, target_item in zip(predicted, target):
        cost = torch.cdist(
            predicted_item.detach().float(),
            target_item.detach().float(),
        ).pow(2)
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        order = torch.empty(target_item.shape[0], dtype=torch.long)
        order[torch.as_tensor(rows)] = torch.as_tensor(columns)
        orders.append(order)
    return torch.stack(orders).to(target.device)


def paired_token_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant set MSE with exact within-sample assignment."""

    order = paired_token_assignment(predicted, target)
    matched = target.gather(
        1,
        order[..., None].expand_as(target),
    )
    return (predicted - matched).pow(2).mean()


def room_transport_cost(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    predicted_classes: torch.Tensor | None = None,
    target_classes: torch.Tensor | None = None,
    class_penalty: float = 1.0,
) -> torch.Tensor:
    """Pair room tokens by geometry with an optional semantic mismatch cost."""

    if predicted_tokens.shape[0] != target_tokens.shape[0]:
        raise ValueError("predicted and target token batches differ")
    cost = torch.cdist(predicted_tokens, target_tokens).pow(2)
    if predicted_classes is not None and target_classes is not None:
        mismatch = (
            predicted_classes[:, :, None] != target_classes[:, None, :]
        ).to(cost.dtype)
        cost = cost + float(class_penalty) * mismatch
    return cost


def room_token_transport_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    predicted_classes: torch.Tensor | None = None,
    target_classes: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    epsilon: float = 0.1,
    iterations: int = 20,
    class_penalty: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sinkhorn-matched token loss; no best-of-N or candidate selection is used."""

    cost = room_transport_cost(
        predicted_tokens,
        target_tokens,
        predicted_classes=predicted_classes,
        target_classes=target_classes,
        class_penalty=class_penalty,
    )
    if target_valid is None:
        valid = torch.ones(
            target_tokens.shape[:2],
            device=target_tokens.device,
            dtype=cost.dtype,
        )
    else:
        valid = target_valid.to(device=cost.device, dtype=cost.dtype)
    unmatch_penalty = cost.amax(dim=(1, 2), keepdim=True).detach() + 1.0
    cost = torch.where(valid[:, None, :].bool(), cost, unmatch_penalty.expand_as(cost))
    plan = sinkhorn_plan(cost, epsilon=epsilon, iterations=iterations)
    valid_count = valid.sum().clamp_min(1.0)
    loss = ((plan * cost) * valid[:, None, :]).sum() / valid_count
    entropy = -(plan * plan.clamp_min(1.0e-8).log()).sum(dim=(1, 2)).mean()
    matched_targets = 0.0
    with torch.no_grad():
        for batch_index in range(cost.shape[0]):
            valid_indices = valid[batch_index].bool().nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            valid_cost = cost[batch_index][:, valid_indices].detach().cpu().numpy()
            rows, _columns = linear_sum_assignment(valid_cost)
            matched_targets += float(len(rows))
    target_recall = torch.tensor(
        matched_targets / valid_count.detach().item(),
        device=cost.device,
        dtype=cost.dtype,
    )
    return loss, {
        "room_transport_loss": loss.detach(),
        "room_transport_entropy": entropy.detach(),
        "room_transport_target_recall": target_recall.detach(),
    }
