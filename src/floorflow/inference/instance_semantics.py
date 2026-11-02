"""Deterministic dense-to-instance semantic consolidation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label
from skimage.draw import line as raster_line


def _footprint_numpy(footprint: torch.Tensor) -> np.ndarray:
    value = footprint.detach().cpu().numpy() > 0
    if value.ndim == 4:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError("footprint must be [B,H,W] or [B,1,H,W]")
    return value


def consolidate_room_instances(
    room_labels: torch.Tensor,
    room_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_class_ids: tuple[int, ...],
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Assign one semantic class per connected predicted room instance."""

    if room_labels.shape[-2:] != room_logits.shape[-2:]:
        raise ValueError("room label and logit spatial shapes differ")
    logit_channels = (
        room_logits.shape[1]
        if room_logits.ndim == 4
        else room_logits.shape[0]
    )
    if logit_channels != len(room_class_ids):
        raise ValueError("room logits must match room_class_ids")
    output = room_labels.clone()
    labels_np = room_labels.detach().cpu().numpy()
    footprint_np = _footprint_numpy(footprint)
    logits_np = room_logits.detach().cpu().numpy()
    ids = np.asarray(room_class_ids, dtype=np.int64)
    for batch_index in range(labels_np.shape[0]):
        updated = labels_np[batch_index].copy()
        for class_index, class_id in enumerate(ids):
            mask = (labels_np[batch_index] == class_id) & footprint_np[batch_index]
            components, count = label(mask)
            for component_id in range(1, int(count) + 1):
                component = components == component_id
                if int(component.sum()) < int(min_area):
                    continue
                scores = logits_np[batch_index][:, component].mean(axis=1)
                new_index = int(scores.argmax())
                updated[component] = ids[new_index]
        output[batch_index] = torch.from_numpy(updated).to(
            device=room_labels.device,
            dtype=room_labels.dtype,
        )
    return output


def consolidate_room_instances_with_prior(
    room_labels: torch.Tensor,
    model_logits: torch.Tensor,
    retrieval_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_class_ids: tuple[int, ...],
    class_weights: torch.Tensor,
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Assign one class per component from model and calibrated retrieval evidence."""

    if room_labels.shape[-2:] != model_logits.shape[-2:]:
        raise ValueError("labels and logits spatial shapes differ")
    if model_logits.shape != retrieval_logits.shape:
        raise ValueError("model and retrieval logits must have equal shape")
    output = room_labels.clone()
    labels_np = room_labels.detach().cpu().numpy()
    footprint_np = _footprint_numpy(footprint)
    ids = np.asarray(room_class_ids, dtype=np.int64)
    for batch_index in range(labels_np.shape[0]):
        updated = labels_np[batch_index].copy()
        for class_id in ids:
            mask = (labels_np[batch_index] == class_id) & footprint_np[batch_index]
            components, count = label(mask)
            for component_id in range(1, int(count) + 1):
                component = components == component_id
                if int(component.sum()) < int(min_area):
                    continue
                spatial = torch.from_numpy(component).to(
                    device=model_logits.device
                )
                model_scores = model_logits[batch_index][:, spatial].mean(dim=1)
                retrieval_scores = retrieval_logits[batch_index][:, spatial].mean(dim=1)
                scores = model_scores + class_weights.to(model_scores) * retrieval_scores
                updated[component] = ids[int(scores.argmax())]
        output[batch_index] = torch.from_numpy(updated).to(
            device=room_labels.device,
            dtype=room_labels.dtype,
        )
    return output


def graph_refine_room_instances(
    room_labels: torch.Tensor,
    model_logits: torch.Tensor,
    retrieval_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_class_ids: tuple[int, ...],
    class_weights: torch.Tensor,
    *,
    neighbor_weight: float = 0.2,
    steps: int = 2,
    min_area: int = 8,
    margin_threshold: float | None = None,
    class_pair_prior: torch.Tensor | None = None,
    class_strength: float = 0.0,
    neighbor_consensus_threshold: float | None = None,
    pair_margin_threshold: float | None = None,
) -> torch.Tensor:
    """Propagate room-type evidence across adjacent predicted instances."""

    if not 0.0 <= float(neighbor_weight) <= 1.0:
        raise ValueError("neighbor_weight must be in [0, 1]")
    output = room_labels.clone()
    labels_np = room_labels.detach().cpu().numpy()
    footprint_np = _footprint_numpy(footprint)
    ids = np.asarray(room_class_ids, dtype=np.int64)
    for batch_index in range(labels_np.shape[0]):
        masks = []
        original_indices = []
        for class_id in ids:
            mask = (labels_np[batch_index] == class_id) & footprint_np[batch_index]
            components, count = label(mask)
            for component_id in range(1, int(count) + 1):
                component = components == component_id
                if int(component.sum()) >= int(min_area):
                    masks.append(component)
                    original_indices.append(
                        int(np.flatnonzero(ids == class_id)[0])
                    )
        if not masks:
            continue
        scores = []
        for component in masks:
            spatial = torch.from_numpy(component).to(model_logits.device)
            model_score = model_logits[batch_index][:, spatial].mean(dim=1)
            retrieval_score = retrieval_logits[batch_index][:, spatial].mean(dim=1)
            scores.append(
                model_score + class_weights.to(model_score) * retrieval_score
            )
        score_tensor = torch.stack(scores)
        update_weight = None
        adjacency = np.eye(len(masks), dtype=np.float32)
        structure = np.ones((3, 3), dtype=bool)
        for left in range(len(masks)):
            dilated = binary_dilation(masks[left], structure=structure)
            for right in range(left + 1, len(masks)):
                if np.logical_and(dilated, masks[right]).any():
                    adjacency[left, right] = adjacency[right, left] = 1.0
        adjacency = torch.from_numpy(adjacency).to(score_tensor)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if class_pair_prior is not None and float(class_strength) > 0:
            prior = class_pair_prior.to(score_tensor)
            if prior.shape != (
                len(room_class_ids),
                len(room_class_ids),
            ):
                raise ValueError("class_pair_prior must be square over room classes")
            neighbor_probability = adjacency @ score_tensor.softmax(dim=-1)
            compatibility = torch.einsum(
                "nd,cd->nc",
                neighbor_probability,
                torch.log(prior.clamp_min(1.0e-6)),
            )
            score_tensor = score_tensor + float(class_strength) * compatibility
        else:
            neighbor_probability = adjacency @ score_tensor.softmax(dim=-1)
        probability = score_tensor.softmax(dim=-1)
        if margin_threshold is not None:
            top_values = probability.topk(2, dim=-1).values
            margin = top_values[:, 0] - top_values[:, 1]
            update_weight = float(neighbor_weight) * (
                margin < float(margin_threshold)
            ).float()[:, None]
        for _ in range(max(0, int(steps))):
            if update_weight is None:
                probability = (
                    (1.0 - float(neighbor_weight)) * probability
                    + float(neighbor_weight) * adjacency @ probability
                )
            else:
                probability = (
                    (1.0 - update_weight) * probability
                    + update_weight * adjacency @ probability
                )
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        predicted_indices = probability.argmax(dim=-1)
        original_tensor = torch.tensor(
            original_indices,
            device=predicted_indices.device,
            dtype=predicted_indices.dtype,
        )
        if (
            class_pair_prior is not None
            and neighbor_consensus_threshold is not None
            and pair_margin_threshold is not None
        ):
            consensus = neighbor_probability.max(dim=-1).values
            top_values = probability.topk(2, dim=-1).values
            pair_margin = top_values[:, 0] - top_values[:, 1]
            selective = (
                (predicted_indices != original_tensor)
                & (consensus >= float(neighbor_consensus_threshold))
                & (pair_margin >= float(pair_margin_threshold))
            )
            predicted_indices = torch.where(
                selective,
                predicted_indices,
                original_tensor,
            )
        assignment = ids[predicted_indices.cpu().numpy()]
        updated = labels_np[batch_index].copy()
        for component, class_id in zip(masks, assignment):
            updated[component] = int(class_id)
        output[batch_index] = torch.from_numpy(updated).to(
            device=room_labels.device,
            dtype=room_labels.dtype,
        )
    return output


def enforce_room_instance_count(
    room_labels: torch.Tensor,
    model_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_class_ids: tuple[int, ...],
    target_count: int,
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Merge the lowest-confidence excess instances into adjacent rooms."""

    output = room_labels.clone()
    labels_np = room_labels.detach().cpu().numpy()
    footprint_np = footprint.detach().cpu().numpy() > 0
    if footprint_np.ndim == 4:
        footprint_np = footprint_np[:, 0]
    ids = np.asarray(room_class_ids, dtype=np.int64)
    structure = np.ones((3, 3), dtype=bool)
    for batch_index in range(labels_np.shape[0]):
        updated = labels_np[batch_index].copy()
        logits = (
            model_logits[batch_index]
            if model_logits.ndim == 4
            else model_logits
        )
        if logits.ndim != 3:
            raise ValueError("model_logits must be [B,C,H,W] or [C,H,W]")
        instances = []
        for class_id in ids:
            room_footprint = (
                footprint_np[batch_index]
                if footprint_np.ndim == 3
                else footprint_np
            )
            mask = (updated == class_id) & room_footprint
            components, count = label(mask)
            for component_id in range(1, int(count) + 1):
                component = components == component_id
                area = int(component.sum())
                if area < int(min_area):
                    continue
                spatial = torch.from_numpy(component).to(model_logits.device)
                confidence = float(
                    logits[:, spatial].max(dim=0).values.mean()
                )
                instances.append((area, confidence, component, int(class_id)))
        excess = len(instances) - max(0, int(target_count))
        if excess <= 0:
            continue
        ranked = sorted(
            instances,
            key=lambda row: (row[1] / max(1.0, float(row[0])), row[0]),
        )
        for _area, _confidence, component, class_id in ranked[:excess]:
            dilated = binary_dilation(component, structure=structure)
            neighbors = updated[dilated]
            neighbors = neighbors[np.isin(neighbors, ids)]
            neighbors = neighbors[neighbors != class_id]
            if neighbors.size == 0:
                continue
            values, counts = np.unique(neighbors, return_counts=True)
            updated[component] = int(values[counts.argmax()])
        output[batch_index] = torch.from_numpy(updated).to(
            device=room_labels.device,
            dtype=room_labels.dtype,
        )
    return output


def merge_room_instances_to_count(
    room_labels: torch.Tensor,
    room_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_class_ids: tuple[int, ...],
    target_count: int,
    *,
    min_area: int = 8,
    confidence_weight: float = 0.25,
    same_class_bonus: float = 1.5,
) -> torch.Tensor:
    """Merge the nearest predicted room instances until a count is reached.

    Unlike adjacency-only repair, this handles fragments separated by a wall or
    a narrow background gap. The target count is predicted outside this
    function; an oracle count is valid only for a diagnostic ceiling.
    """

    if room_labels.shape[-2:] != room_logits.shape[-2:]:
        raise ValueError("room label and logit spatial shapes differ")
    logit_channels = (
        room_logits.shape[1]
        if room_logits.ndim == 4
        else room_logits.shape[0]
    )
    if logit_channels != len(room_class_ids):
        raise ValueError("room logits must match room_class_ids")
    output = room_labels.clone()
    labels_np = room_labels.detach().cpu().numpy()
    footprint_np = footprint.detach().cpu().numpy() > 0
    if footprint_np.ndim == 4:
        footprint_np = footprint_np[:, 0]
    logits_np = room_logits.detach().cpu().numpy()
    if logits_np.ndim == 3:
        logits_np = logits_np[None]
    ids = np.asarray(room_class_ids, dtype=np.int64)
    for batch_index in range(labels_np.shape[0]):
        room_footprint = (
            footprint_np[batch_index]
            if footprint_np.ndim == 3
            else footprint_np
        )
        components = []
        for class_index, class_id in enumerate(ids):
            connected, count = label(
                (labels_np[batch_index] == class_id) & room_footprint
            )
            for component_id in range(1, int(count) + 1):
                mask = connected == component_id
                area = int(mask.sum())
                if area < int(min_area):
                    continue
                ys, xs = np.nonzero(mask)
                row = np.maximum(logits_np[batch_index, class_index], 0.0)
                confidence = float(row[mask].mean())
                components.append(
                    {
                        "mask": mask,
                        "area": area,
                        "class": int(class_id),
                        "centroid": (float(xs.mean()), float(ys.mean())),
                        "bounds": (
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                        ),
                            "scores": np.asarray(
                                [
                                    logits_np[batch_index, channel][mask].mean()
                                    for channel in range(len(ids))
                                ],
                                dtype=np.float64,
                            ),
                        "confidence": confidence,
                    }
                )
        desired = max(1, int(target_count))
        if len(components) <= desired:
            continue
        parent = list(range(len(components)))

        def root(index: int, parent: list[int] = parent) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def edge_cost(
            left: int,
            right: int,
            components: list[dict] = components,
            confidence_weight: float = confidence_weight,
        ) -> float:
            left_row = components[left]
            right_row = components[right]
            lx0, ly0, lx1, ly1 = left_row["bounds"]
            rx0, ry0, rx1, ry1 = right_row["bounds"]
            dx = max(0.0, rx0 - lx1, lx0 - rx1)
            dy = max(0.0, ry0 - ly1, ly0 - ry1)
            centroid = np.hypot(
                left_row["centroid"][0] - right_row["centroid"][0],
                left_row["centroid"][1] - right_row["centroid"][1],
            )
            area_penalty = abs(
                np.log((left_row["area"] + 1.0) / (right_row["area"] + 1.0))
            )
            class_bonus = (
                -float(same_class_bonus)
                if left_row["class"] == right_row["class"]
                else 0.0
            )
            return float(
                np.hypot(dx, dy)
                + 0.02 * centroid
                + 0.2 * area_penalty
                + float(confidence_weight)
                * abs(
                    left_row["confidence"]
                    - right_row["confidence"]
                )
                + class_bonus
            )

        edges = sorted(
            (
                edge_cost(left, right),
                left,
                right,
            )
            for left in range(len(components))
            for right in range(left + 1, len(components))
        )
        remaining = len(components)
        for _cost, left, right in edges:
            left_root = root(left)
            right_root = root(right)
            if left_root == right_root:
                continue
            parent[right_root] = left_root
            remaining -= 1
            if remaining <= desired:
                break
        groups: dict[int, list[int]] = {}
        for index in range(len(components)):
            groups.setdefault(root(index), []).append(index)
        updated = labels_np[batch_index].copy()
        for indices in groups.values():
            total_area = sum(components[index]["area"] for index in indices)
            merged_scores = sum(
                components[index]["scores"] * components[index]["area"]
                for index in indices
            ) / max(1, total_area)
            merged_class = int(ids[int(merged_scores.argmax())])
            merged_centroid = (
                sum(
                    components[index]["centroid"][0]
                    * components[index]["area"]
                    for index in indices
                )
                / max(1, total_area),
                sum(
                    components[index]["centroid"][1]
                    * components[index]["area"]
                    for index in indices
                )
                / max(1, total_area),
            )
            for index in indices:
                updated[components[index]["mask"]] = merged_class
                start_y, start_x = (
                    round(value)
                    for value in reversed(components[index]["centroid"])
                )
                end_y, end_x = (
                    round(value) for value in reversed(merged_centroid)
                )
                rows, columns = raster_line(
                    start_y,
                    start_x,
                    end_y,
                    end_x,
                )
                updated[rows, columns] = merged_class
        output[batch_index] = torch.from_numpy(updated).to(
            device=room_labels.device,
            dtype=room_labels.dtype,
        )
    return output


__all__ = [
    "consolidate_room_instances",
    "consolidate_room_instances_with_prior",
    "enforce_room_instance_count",
    "graph_refine_room_instances",
    "merge_room_instances_to_count",
]
