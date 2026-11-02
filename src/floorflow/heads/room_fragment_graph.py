"""Predicted-room fragment graph arbitration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.draw import line as raster_line
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import (
    RoomInstance,
    extract_room_instances,
    instance_iou,
)


@dataclass
class FragmentGraph:
    """One predicted fragment graph with training-only match labels."""

    node_features: torch.Tensor
    pair_features: torch.Tensor
    edge_index: torch.Tensor
    masks: torch.Tensor
    mask_probabilities: torch.Tensor
    class_probability: torch.Tensor
    node_confidence: torch.Tensor
    node_margin: torch.Tensor
    node_existence: torch.Tensor
    classes: torch.Tensor
    node_target: torch.Tensor
    edge_target: torch.Tensor
    target_count: int
    base_labels: np.ndarray
    target_labels: np.ndarray
    footprint: np.ndarray
    graph_kind: str = "dense"


def _shape_features(
    fragment: RoomInstance,
    height: int,
    width: int,
) -> np.ndarray:
    x0, y0, x1, y1 = fragment.bbox
    area = float(fragment.area)
    box_area = float(max(1, (x1 - x0) * (y1 - y0)))
    return np.asarray(
        [
            np.log1p(area) / np.log1p(height * width),
            (x1 - x0) / width,
            (y1 - y0) / height,
            area / max(1.0, height * width),
            area / box_area,
            (x0 + x1) / (2.0 * width),
            (y0 + y1) / (2.0 * height),
            max(1, x1 - x0) / max(1, y1 - y0),
            np.sqrt(area) / max(1.0, np.sqrt(height * width)),
        ],
        dtype=np.float32,
    )


def _node_features(
    fragments: list[RoomInstance],
    probability: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    rows = []
    for fragment in fragments:
        values = probability[:, fragment.mask]
        mean = values.mean(axis=1)
        top = np.sort(mean)[-2:]
        entropy = -float(
            np.sum(np.clip(mean, 1.0e-8, 1.0) * np.log(np.clip(mean, 1.0e-8, 1.0)))
        )
        rows.append(
            np.concatenate(
                [
                    mean,
                    np.asarray(
                        [
                            top[-1],
                            top[-1] - top[0],
                            entropy / np.log(max(2, len(mean))),
                            np.log1p(fragment.area)
                            / np.log1p(height * width),
                        ],
                        dtype=np.float32,
                    ),
                    _shape_features(fragment, height, width),
                ]
            )
        )
    return np.stack(rows)


def _pair_features(
    left: RoomInstance,
    right: RoomInstance,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    lx0, ly0, lx1, ly1 = left.bbox
    rx0, ry0, rx1, ry1 = right.bbox
    gap_x = max(0.0, rx0 - lx1, lx0 - rx1) / width
    gap_y = max(0.0, ry0 - ly1, ly0 - ry1) / height
    left_center = np.asarray(
        [(lx0 + lx1) / (2.0 * width), (ly0 + ly1) / (2.0 * height)]
    )
    right_center = np.asarray(
        [(rx0 + rx1) / (2.0 * width), (ry0 + ry1) / (2.0 * height)]
    )
    contact = binary_dilation(left.mask) & right.mask
    contact_scale = min(np.sqrt(left.area), np.sqrt(right.area))
    return np.asarray(
        [
            gap_x,
            gap_y,
            float(np.linalg.norm(left_center - right_center)),
            abs(np.log((left.area + 1.0) / (right.area + 1.0))),
            float(np.dot(left_probability, right_probability)),
            float(np.abs(left_probability - right_probability).sum()),
            float(left.class_id == right.class_id),
            float(contact.sum()) / max(1.0, contact_scale),
            min(left.area, right.area) / max(1.0, max(left.area, right.area)),
        ],
        dtype=np.float32,
    )


def build_fragment_graph(
    labels: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 8,
    neighbors: int = 8,
    node_iou_threshold: float = 0.2,
    edge_iou_threshold: float = 0.1,
) -> FragmentGraph | None:
    """Build one graph from predicted fragments and target-only labels."""

    fragments = extract_room_instances(labels, taxonomy, min_area=min_area)
    if not fragments:
        return None
    height, width = labels.shape
    targets = extract_room_instances(target, taxonomy, min_area=min_area)
    node_features = torch.from_numpy(
        _node_features(fragments, probability, height, width)
    ).float()
    probabilities = []
    for fragment in fragments:
        probabilities.append(probability[:, fragment.mask].mean(axis=1))
    probabilities = np.stack(probabilities).astype(np.float32)
    confidence = probabilities.max(axis=1)
    sorted_probability = np.sort(probabilities, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
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
    candidate_pairs = set()
    for index in range(len(fragments)):
        for neighbor in np.argsort(distance[index])[: min(neighbors, len(fragments) - 1)]:
            candidate_pairs.add(tuple(sorted((index, int(neighbor)))))
    pairs = sorted(candidate_pairs)
    pair_rows = []
    edge_target = []
    for left, right in pairs:
        pair_rows.append(
            _pair_features(
                fragments[left],
                fragments[right],
                probabilities[left],
                probabilities[right],
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
    pair_features = (
        torch.from_numpy(np.stack(pair_rows)).float()
        if pair_rows
        else torch.zeros((0, 9), dtype=torch.float32)
    )
    return FragmentGraph(
        node_features=node_features,
        pair_features=pair_features,
        edge_index=edge_index,
        masks=torch.from_numpy(
            np.stack([fragment.mask for fragment in fragments])
        ),
        mask_probabilities=torch.from_numpy(
            np.stack([fragment.mask for fragment in fragments])
        ).float(),
        class_probability=torch.from_numpy(probabilities),
        node_confidence=torch.from_numpy(confidence).float(),
        node_margin=torch.from_numpy(margin).float(),
        node_existence=torch.ones(len(fragments)),
        classes=torch.tensor(
            [fragment.class_id for fragment in fragments],
            dtype=torch.long,
        ),
        node_target=node_target,
        edge_target=torch.tensor(edge_target, dtype=torch.bool),
        target_count=len(targets),
        base_labels=labels.copy(),
        target_labels=target.copy(),
        footprint=footprint.astype(bool, copy=True),
        graph_kind="dense",
    )


class FragmentGraphArbiter(nn.Module):
    """Lightweight node-existence and edge-merge model."""

    def __init__(
        self,
        node_features: int,
        pair_features: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        hidden = int(hidden)
        self.node_encoder = nn.Sequential(
            nn.Linear(int(node_features), hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(int(pair_features) + 2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_head = nn.Linear(hidden, 1)
        self.node_head = nn.Linear(hidden, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        pair_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_hidden = self.node_encoder(node_features)
        if edge_index.shape[1]:
            source, target = edge_index
            pair_hidden = self.pair_encoder(
                torch.cat(
                    [
                        pair_features,
                        node_hidden[source],
                        node_hidden[target],
                    ],
                    dim=-1,
                )
            )
            message = torch.zeros_like(node_hidden)
            message.index_add_(0, source, pair_hidden)
            message.index_add_(0, target, pair_hidden)
            degree = torch.zeros(
                node_hidden.shape[0],
                device=node_hidden.device,
                dtype=node_hidden.dtype,
            )
            degree.index_add_(
                0,
                source,
                torch.ones_like(source, dtype=node_hidden.dtype),
            )
            degree.index_add_(
                0,
                target,
                torch.ones_like(target, dtype=node_hidden.dtype),
            )
            node_hidden = self.node_norm(
                node_hidden + message / degree.clamp_min(1.0)[:, None]
            )
        else:
            pair_hidden = pair_features.new_zeros((0, node_hidden.shape[1]))
        return {
            "node_logits": self.node_head(node_hidden).squeeze(-1),
            "edge_logits": self.edge_head(pair_hidden).squeeze(-1),
        }


def _union_find(count: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root
    return [root(index) for index in range(count)]


@torch.inference_mode()
def render_fragment_graph(
    graph: FragmentGraph,
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
    class_mode: str = "guarded",
    low_confidence_margin: float = 0.2,
    query_score_rendering: bool = False,
    fill_footprint: bool = False,
) -> np.ndarray:
    """Suppress low-existence fragments and merge accepted graph edges."""

    if class_mode not in {"safe", "guarded", "unrestricted"}:
        raise ValueError("class_mode must be safe, guarded, or unrestricted")
    keep = node_logits.sigmoid() >= float(node_threshold)
    accepted_edges = []
    if edge_logits.numel():
        source, target = graph.edge_index
        merge = edge_logits.sigmoid() >= float(edge_threshold)
        for left, right, valid in zip(
            source.tolist(),
            target.tolist(),
            merge.tolist(),
        ):
            same_class = int(graph.classes[left]) == int(graph.classes[right])
            uncertain = bool(
                graph.node_margin[left] < float(low_confidence_margin)
                or graph.node_margin[right] < float(low_confidence_margin)
            )
            compatible = (
                class_mode == "unrestricted"
                or same_class
                or (class_mode == "guarded" and uncertain)
            )
            if valid and keep[left] and keep[right] and compatible:
                accepted_edges.append((left, right))
    if class_mode != "unrestricted":
        neighbors = {
            index: set()
            for index in range(len(keep))
        }
        for left, right in accepted_edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        for index in range(len(keep)):
            if bool(keep[index]):
                continue
            if class_mode == "safe":
                recover = any(
                    bool(keep[neighbor])
                    and int(graph.classes[neighbor]) == int(graph.classes[index])
                    for neighbor in neighbors[index]
                )
            else:
                recover = bool(
                    graph.node_margin[index] < float(low_confidence_margin)
                    and any(bool(keep[neighbor]) for neighbor in neighbors[index])
                )
            if not recover:
                keep[index] = True
    if not bool(keep.any()):
        keep = torch.ones_like(keep)
    roots = _union_find(len(keep), accepted_edges)
    groups: dict[int, list[int]] = {}
    for index in range(len(keep)):
        groups.setdefault(roots[index], []).append(index)
    render_groups = [
        group for group in groups.values() if any(bool(keep[index]) for index in group)
    ]
    height, width = graph.masks.shape[-2:]
    output = np.zeros((height, width), dtype=np.int64)
    output_score = (
        np.zeros((height, width), dtype=np.float32)
        if query_score_rendering
        else None
    )
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    cluster_ids = {}
    for group_index, group in enumerate(render_groups):
        members = [index for index in group if bool(keep[index])]
        weights = graph.masks[members].sum(dim=(1, 2)).float()
        if class_mode in {"safe", "guarded"}:
            anchor = max(
                members,
                key=lambda index: float(
                    graph.node_confidence[index]
                    * torch.log1p(
                        graph.masks[index].sum().float()
                    )
                ),
            )
            class_id = int(graph.classes[anchor])
        else:
            scores = (
                graph.class_probability[members]
                * weights[:, None]
            ).sum(dim=0) / weights.sum().clamp_min(1.0)
            class_id = int(room_ids[int(scores.argmax())])
        cluster_ids[group_index] = class_id
        if query_score_rendering:
            cluster_score = (
                graph.mask_probabilities[members].max(dim=0).values.numpy()
            )
            cluster_support = graph.masks[members].any(dim=0).numpy()
            update = cluster_support & (cluster_score > output_score)
            output[update] = class_id
            output_score[update] = cluster_score[update]
        else:
            for index in members:
                output[graph.masks[index].numpy()] = class_id
    for group_index, group in enumerate(render_groups):
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1 :]:
                points = np.stack(
                    [
                        np.asarray(
                            torch.nonzero(graph.masks[left]).float().mean(dim=0)
                        ),
                        np.asarray(
                            torch.nonzero(graph.masks[right]).float().mean(dim=0)
                        ),
                    ]
                )
                rows, columns = raster_line(
                    round(float(points[0, 0])),
                    round(float(points[0, 1])),
                    round(float(points[1, 0])),
                    round(float(points[1, 1])),
                )
                output[rows, columns] = cluster_ids[group_index]
    suppressed = ~keep.numpy() if class_mode != "safe" else np.zeros(
        len(keep),
        dtype=bool,
    )
    if suppressed.any() and render_groups:
        support = graph.masks[suppressed].numpy().any(axis=0)
        if output.any() and support.any():
            _, indices = distance_transform_edt(output == 0, return_indices=True)
            nearest = output[tuple(indices)]
            assign = support & (nearest > 0)
            output[assign] = nearest[assign]
    if fill_footprint:
        empty = graph.footprint & (output == 0)
        if output.any() and empty.any():
            _, indices = distance_transform_edt(output == 0, return_indices=True)
            nearest = output[tuple(indices)]
            assign = empty & (nearest > 0)
            output[assign] = nearest[assign]
    return output


__all__ = [
    "FragmentGraph",
    "FragmentGraphArbiter",
    "build_fragment_graph",
    "render_fragment_graph",
]
