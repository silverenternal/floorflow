"""Project routed opening hosts from candidate order to final wall order."""

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import ReadoutOpening
from floorflow.heads.opening_host_router import (
    routed_opening_rows,
    slice_tensor_predictions,
)


def project_routed_openings_to_final_walls(
    predictions: dict[str, torch.Tensor],
    candidate_endpoints: torch.Tensor,
    selected: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    frame: float,
    opening_presence_threshold: float | None = None,
    query_centers: torch.Tensor | None = None,
    query_widths: torch.Tensor | None = None,
    query_families: torch.Tensor | None = None,
    preserve_query_intervals: bool = False,
) -> list[list[ReadoutOpening]]:
    """Consume only selected walls while preserving the router's host identity."""

    batch = candidate_endpoints.shape[0]
    host_index = predictions["host_index"]
    opening_mask = predictions["opening_mask"]
    if candidate_endpoints.ndim != 4 or candidate_endpoints.shape[-2:] != (2, 2):
        raise ValueError("candidate endpoints must be [B,N,2,2]")
    if selected.shape != candidate_endpoints.shape[:2]:
        raise ValueError("selected wall mask must be [B,N]")
    if host_index.shape != opening_mask.shape or host_index.shape[0] != batch:
        raise ValueError("router host and opening masks must share [B,O]")
    output = []
    for sample in range(batch):
        selected_indices = torch.nonzero(
            selected[sample].bool(),
            as_tuple=False,
        ).flatten()
        if selected_indices.numel() == 0:
            output.append([])
            continue
        local = slice_tensor_predictions(predictions, sample)
        active = local["opening_mask"][0].bool()
        if opening_presence_threshold is not None:
            active = active & (
                local["presence_logits"][0]
                >= float(opening_presence_threshold)
            )
        mapping = {
            int(global_index): local_index
            for local_index, global_index in enumerate(
                selected_indices.cpu().tolist()
            )
        }
        remapped = torch.full_like(local["host_index"][0], -1)
        for opening in torch.nonzero(active, as_tuple=False).flatten().tolist():
            global_host = int(local["host_index"][0, opening].cpu())
            if global_host not in mapping:
                raise ValueError(
                    "active opening host is not in the final wall set"
                )
            remapped[opening] = mapping[global_host]
        local["host_index"] = remapped[None]
        local["host_endpoints"] = candidate_endpoints[
            sample,
            selected_indices,
        ][None]
        output.append(
            routed_opening_rows(
                local,
                0,
                taxonomy,
                frame=frame,
                threshold=opening_presence_threshold,
                query_centers=(
                    None
                    if query_centers is None
                    else query_centers[sample : sample + 1]
                ),
                query_widths=(
                    None
                    if query_widths is None
                    else query_widths[sample : sample + 1]
                ),
                query_families=(
                    None
                    if query_families is None
                    else query_families[sample : sample + 1]
                ),
                preserve_query_intervals=preserve_query_intervals,
            )
        )
    return output


__all__ = ["project_routed_openings_to_final_walls"]
