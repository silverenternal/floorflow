"""Space-syntax inspired integration and public-space advantage metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def space_syntax_metrics(
    adjacency: np.ndarray,
    room_labels: Sequence[str],
    *,
    public_labels: Sequence[str] = ("room:LivingRoom",),
) -> dict[str, float]:
    graph = np.asarray(adjacency)
    if graph.shape != (len(room_labels), len(room_labels)):
        raise ValueError("adjacency and room labels must align")
    graph = graph.astype(bool)
    for index in range(len(room_labels)):
        graph[index, index] = False
    integration = []
    for source in range(len(room_labels)):
        distances = _bfs_distances(graph, source)
        reachable = distances[distances > 0]
        integration.append(
            float(np.mean(1.0 / (reachable + 1.0))) if reachable.size else 0.0
        )
    public = [
        value
        for value, label in zip(integration, room_labels, strict=True)
        if label in public_labels
    ]
    private = [
        value
        for value, label in zip(integration, room_labels, strict=True)
        if label not in public_labels
    ]
    return {
        "integration_mean": float(np.mean(integration)) if integration else 0.0,
        "public_integration": float(np.mean(public)) if public else 0.0,
        "private_integration": float(np.mean(private)) if private else 0.0,
        "public_integration_advantage": (
            float(np.mean(public) - np.mean(private))
            if public and private
            else 0.0
        ),
        "connectivity": float(graph.sum() / max(1, graph.size - len(graph))),
    }


def _bfs_distances(graph: np.ndarray, source: int) -> np.ndarray:
    distances = np.full(graph.shape[0], -1, dtype=np.int32)
    distances[source] = 0
    queue = [source]
    for node in queue:
        for neighbour in np.flatnonzero(graph[node]):
            if distances[neighbour] < 0:
                distances[neighbour] = distances[node] + 1
                queue.append(int(neighbour))
    return distances


def adjacency_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    return (logits.detach().sigmoid().cpu().numpy() >= float(threshold)).astype(bool)


__all__ = ["adjacency_from_logits", "space_syntax_metrics"]
