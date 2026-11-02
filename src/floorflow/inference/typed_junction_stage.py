"""Cluster wall endpoints into typed junction vectors."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def typed_junction_rows(
    endpoints: torch.Tensor,
    valid: torch.Tensor,
    *,
    frame: float = 1.0,
    tolerance: float = 0.04,
    wall_indices: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    """Return explicit junction coordinates, degrees, and incident walls."""

    if endpoints.ndim != 3 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("endpoints must be [N,2,2]")
    points = endpoints.detach().float().cpu().numpy()
    active = valid.detach().bool().cpu().numpy()
    if wall_indices is None:
        wall_indices = torch.arange(endpoints.shape[0])
    wall_ids = wall_indices.detach().cpu().numpy()
    clusters: dict[tuple[int, int], dict[str, object]] = {}
    for wall in range(points.shape[0]):
        if not active[wall]:
            continue
        for endpoint in range(2):
            point = points[wall, endpoint]
            key = tuple(
                np.rint(point / max(float(tolerance), 1.0e-6)).astype(int)
            )
            cluster = clusters.setdefault(
                key,
                {"points": [], "walls": set()},
            )
            cluster["points"].append(point)
            cluster["walls"].add(int(wall_ids[wall]))
    rows = []
    for index, cluster in enumerate(
        sorted(clusters.values(), key=lambda value: len(value["walls"]), reverse=True),
        start=1,
    ):
        center = np.mean(np.stack(cluster["points"]), axis=0) * float(frame)
        incident = sorted(int(value) for value in cluster["walls"])
        rows.append(
            {
                "id": f"junction_{index:04d}",
                "point": (float(center[0]), float(center[1])),
                "degree": len(incident),
                "incident_walls": incident,
            }
        )
    return rows


__all__ = ["typed_junction_rows"]
