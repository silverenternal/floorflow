"""Project predicted wall endpoints onto token-conditioned graph junctions."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def project_wall_endpoints_to_token_junctions(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.04,
) -> dict[str, torch.Tensor]:
    if (
        "token_junction_coordinates" not in output
        or "token_junction_degree_logits" not in output
    ):
        return output
    endpoints = output["wall_endpoints"].detach().cpu().numpy().copy()
    wall_probability = output["wall_exists"].detach().sigmoid().cpu().numpy()
    junction_points = (
        output["token_junction_coordinates"].detach().cpu().numpy()
    )
    degree = (
        output["token_junction_degree_logits"].detach().argmax(-1).cpu().numpy()
    )
    for batch in range(endpoints.shape[0]):
        active_junctions = junction_points[batch][degree[batch] > 0]
        if not len(active_junctions):
            continue
        for wall in range(endpoints.shape[1]):
            if wall_probability[batch, wall] < 0.5:
                continue
            for endpoint in range(2):
                point = endpoints[batch, wall, endpoint]
                distances = np.linalg.norm(active_junctions - point, axis=-1)
                nearest = int(np.argmin(distances))
                if float(distances[nearest]) <= threshold:
                    endpoints[batch, wall, endpoint] = active_junctions[nearest]
    projected = dict(output)
    projected["wall_endpoints"] = torch.from_numpy(endpoints).to(
        output["wall_endpoints"].device,
        dtype=output["wall_endpoints"].dtype,
    )
    projected["token_junction_projection_applied"] = torch.ones(
        endpoints.shape[0],
        device=output["wall_endpoints"].device,
    )
    return projected


__all__ = ["project_wall_endpoints_to_token_junctions"]
