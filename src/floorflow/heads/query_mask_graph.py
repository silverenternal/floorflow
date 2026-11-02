"""Graph construction and readout for query room masks."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_fragment_graph import (
    FragmentGraph,
    _pair_features,
    _shape_features,
)
from floorflow.representation.core import (
    RoomInstance,
    extract_room_instances,
    instance_iou,
)


def build_query_mask_graph(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    mask_threshold: float = 0.5,
    min_area: int = 8,
    neighbors: int = 8,
    node_iou_threshold: float = 0.2,
    edge_iou_threshold: float = 0.1,
) -> FragmentGraph | None:
    """Build one graph whose nodes are predicted room-mask queries."""

    probabilities = mask_logits.sigmoid().detach().cpu()
    class_probability = class_logits.softmax(dim=-1).detach().cpu()
    room_count = len(taxonomy.room_ids)
    foreground = class_probability[..., :room_count]
    existence = existence_logits.sigmoid().detach().cpu()
    height, width = mask_logits.shape[-2:]
    footprint_mask = np.asarray(footprint, dtype=bool)
    queries = []
    for query in range(mask_logits.shape[0]):
        mask = (
            probabilities[query] >= float(mask_threshold)
        ).numpy() & footprint_mask
        if int(mask.sum()) < int(min_area):
            continue
        ys, xs = np.nonzero(mask)
        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        class_id = int(taxonomy.room_ids[int(foreground[query].argmax())])
        queries.append(
            (
                query,
                RoomInstance(class_id, int(mask.sum()), bbox, mask),
                mask,
            )
        )
    if not queries:
        return None
    query_ids = [row[0] for row in queries]
    fragments = [row[1] for row in queries]
    masks = np.stack([row[2] for row in queries])
    targets = extract_room_instances(target, taxonomy, min_area=min_area)
    target_ious = np.zeros((len(fragments), len(targets)), dtype=np.float32)
    for left, fragment in enumerate(fragments):
        for right, expected in enumerate(targets):
            target_ious[left, right] = instance_iou(fragment, expected)
    if len(targets):
        best_target = target_ious.argmax(axis=1)
        best_iou = target_ious.max(axis=1)
    else:
        best_target = np.full(len(fragments), -1, dtype=np.int64)
        best_iou = np.zeros(len(fragments), dtype=np.float32)
    node_target = torch.from_numpy(best_iou >= node_iou_threshold)
    class_means = np.stack(
        [foreground[query].numpy() for query in query_ids]
    ).astype(np.float32)
    confidence = np.asarray(
        [
            float(class_means[index].max() * existence[query])
            for index, query in enumerate(query_ids)
        ],
        dtype=np.float32,
    )
    sorted_class = np.sort(class_means, axis=1)
    margin = sorted_class[:, -1] - sorted_class[:, -2]
    node_rows = []
    for index, (query, fragment, _mask) in enumerate(queries):
        local = probabilities[query][fragment.mask].numpy()
        node_rows.append(
            np.concatenate(
                [
                    class_means[index],
                    np.asarray(
                        [
                            float(existence[query]),
                            float(local.mean()),
                            float(local.std()),
                            float(confidence[index]),
                            float(margin[index]),
                        ],
                        dtype=np.float32,
                    ),
                    _shape_features(fragment, height, width),
                ]
            )
        )
    node_features = torch.from_numpy(np.stack(node_rows)).float()
    centers = np.asarray(
        [
            [
                (fragment.bbox[0] + fragment.bbox[2]) / (2.0 * width),
                (fragment.bbox[1] + fragment.bbox[3]) / (2.0 * height),
            ]
            for fragment in fragments
        ],
        dtype=np.float32,
    )
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    pairs = set()
    for index in range(len(fragments)):
        for neighbor in np.argsort(distance[index])[
            : min(neighbors, len(fragments) - 1)
        ]:
            pairs.add(tuple(sorted((index, int(neighbor)))))
    pairs = sorted(pairs)
    pair_rows = []
    edge_target = []
    for left, right in pairs:
        pair_rows.append(
            _pair_features(
                fragments[left],
                fragments[right],
                class_means[left],
                class_means[right],
                height,
                width,
            )
        )
        edge_target.append(
            bool(
                node_target[left]
                and node_target[right]
                and best_target[left] == best_target[right]
                and best_iou[left] >= edge_iou_threshold
                and best_iou[right] >= edge_iou_threshold
            )
        )
    edge_index = (
        torch.tensor(pairs, dtype=torch.long).T
        if pairs
        else torch.zeros((2, 0), dtype=torch.long)
    )
    base_labels = render_query_masks(
        probabilities,
        torch.tensor(query_ids, dtype=torch.long),
        torch.tensor(
            [fragment.class_id for fragment in fragments],
            dtype=torch.long,
        ),
        footprint_mask,
        mask_threshold=mask_threshold,
    )
    return FragmentGraph(
        node_features=node_features,
        pair_features=(
            torch.from_numpy(np.stack(pair_rows)).float()
            if pair_rows
            else torch.zeros((0, 9), dtype=torch.float32)
        ),
        edge_index=edge_index,
        masks=torch.from_numpy(masks),
        mask_probabilities=probabilities[query_ids].float(),
        class_probability=torch.from_numpy(class_means),
        node_confidence=torch.from_numpy(confidence),
        node_margin=torch.from_numpy(margin).float(),
        node_existence=torch.tensor(
            [float(existence[query]) for query in query_ids],
            dtype=torch.float32,
        ),
        classes=torch.tensor(
            [fragment.class_id for fragment in fragments],
            dtype=torch.long,
        ),
        node_target=node_target,
        edge_target=torch.tensor(edge_target, dtype=torch.bool),
        target_count=len(targets),
        base_labels=base_labels,
        target_labels=target.copy(),
        footprint=footprint_mask.copy(),
        graph_kind="query",
    )


def render_query_masks(
    probability: torch.Tensor,
    query_ids: torch.Tensor,
    classes: torch.Tensor,
    footprint: np.ndarray,
    *,
    mask_threshold: float = 0.5,
    fill_footprint: bool = False,
) -> np.ndarray:
    """Render overlapping query masks by their highest soft-mask score."""

    height, width = probability.shape[-2:]
    output = np.zeros((height, width), dtype=np.int64)
    best = np.zeros((height, width), dtype=np.float32)
    for index, query in enumerate(query_ids.tolist()):
        score = probability[query].numpy()
        update = (
            footprint
            & (score >= float(mask_threshold))
            & (score > best)
        )
        output[update] = int(classes[index])
        best[update] = score[update]
    support = (output > 0) & footprint
    empty = footprint & (output == 0)
    if fill_footprint and support.any() and empty.any():
        _, indices = distance_transform_edt(output == 0, return_indices=True)
        nearest = output[tuple(indices)]
        assign = empty & (nearest > 0)
        output[assign] = nearest[assign]
    return output


__all__ = ["build_query_mask_graph", "render_query_masks"]
