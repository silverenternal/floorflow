"""Data and matching utilities for CAD wall-candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.cad_wall_candidates import wall_candidate_evidence
from floorflow.geometry.wall_graph_readout import (
    WallSkeletonPolicy,
    extract_wall_segments,
)
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.cad_wall_candidate_runtime import (
    candidate_walls_with_sources,
    canonical_room_rows,
    pad_candidate_sources,
    pad_candidate_tensors,
)
from floorflow.inference.canonical_layer_composition import (
    canonical_union_labels,
)


@dataclass(frozen=True)
class CandidateBatch:
    endpoints: torch.Tensor
    valid: torch.Tensor
    evidence: torch.Tensor
    exterior: torch.Tensor
    rooms: list[list[dict]]
    host_walls: list[list]
    control_endpoints: torch.Tensor | None = None
    control_valid: torch.Tensor | None = None
    control_evidence: torch.Tensor | None = None
    proposal_source: torch.Tensor | None = None
    control_proposal_source: torch.Tensor | None = None


def match_candidates(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    tolerance: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    pairs = []
    positive = []
    for batch in range(predicted.shape[0]):
        count = int(target_valid[batch].sum())
        if count == 0:
            pairs.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
            positive.append(np.zeros(predicted.shape[1], dtype=bool))
            continue
        direct = (
            predicted[batch][:, None] - target[batch, :count][None]
        ).abs().mean(dim=(2, 3))
        reverse = (
            predicted[batch][:, None]
            - target[batch, :count][None, :, [1, 0]]
        ).abs().mean(dim=(2, 3))
        pred_length = (predicted[batch, :, 1] - predicted[batch, :, 0]).norm(dim=-1)
        target_length = (
            target[batch, :count, 1] - target[batch, :count, 0]
        ).norm(dim=-1)
        cost = torch.minimum(direct, reverse) + 0.5 * (
            pred_length[:, None] - target_length[None]
        ).abs()
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        keep = (
            cost[rows, columns].detach().cpu().numpy() <= float(tolerance)
        ) & candidate_valid[batch, rows].cpu().numpy()
        rows = rows[keep].astype(np.int64)
        columns = columns[keep].astype(np.int64)
        mask = np.zeros(predicted.shape[1], dtype=bool)
        mask[rows] = True
        pairs.append((rows, columns))
        positive.append(mask)
    return pairs, positive


def candidate_targets(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    valid: torch.Tensor,
    tolerance: float,
    match_endpoints: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    pairs, positive_rows = match_candidates(
        outputs["endpoints"] if match_endpoints is None else match_endpoints,
        targets["wall_endpoints"],
        targets["wall_exists"],
        valid,
        tolerance,
    )
    expected = (
        torch.from_numpy(np.stack(positive_rows))
        .to(outputs["score_logits"].device)
        .float()
    )
    exterior_target = torch.zeros_like(expected)
    for batch, (rows, columns) in enumerate(pairs):
        if not rows.size:
            continue
        row_ids = torch.from_numpy(rows).to(exterior_target.device)
        col_ids = torch.from_numpy(columns).to(targets["wall_exterior"].device)
        exterior_target[batch, row_ids] = targets["wall_exterior"][
            batch, col_ids
        ].float()
    coordinate_losses = []
    for batch, (rows, columns) in enumerate(pairs):
        if not rows.size:
            continue
        row_ids = torch.from_numpy(rows).to(outputs["endpoints"].device)
        col_ids = torch.from_numpy(columns).to(targets["wall_endpoints"].device)
        direct = torch.nn.functional.smooth_l1_loss(
            outputs["endpoints"][batch, row_ids],
            targets["wall_endpoints"][batch, col_ids],
            reduction="none",
        ).mean()
        reverse = torch.nn.functional.smooth_l1_loss(
            outputs["endpoints"][batch, row_ids],
            targets["wall_endpoints"][batch, col_ids][:, [1, 0]],
            reduction="none",
        ).mean()
        coordinate_losses.append(torch.minimum(direct, reverse))
    coordinate = (
        torch.stack(coordinate_losses).mean()
        if coordinate_losses
        else outputs["endpoints"].sum() * 0.0
    )
    target_budget = targets["wall_exists"].sum(dim=1).float() / float(valid.shape[1])
    exterior_budget = (
        targets["wall_exterior"].sum(dim=1).float() / float(valid.shape[1])
    )
    interior_budget = (
        (targets["wall_exists"] - targets["wall_exterior"]).sum(dim=1).float()
        / float(valid.shape[1])
    )
    return (
        expected,
        coordinate,
        target_budget,
        exterior_budget,
        interior_budget,
        exterior_target,
    )


def build_candidate_batch(
    controls: dict[str, torch.Tensor],
    layer_output: dict,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_room_area: int,
    max_walls: int,
    device: torch.device,
    candidate_mode: str = "boundary",
    visible_wall_boundary_width: int = 7,
) -> CandidateBatch:
    if candidate_mode not in {"boundary", "host", "skeleton_portfolio"}:
        raise ValueError(
            "candidate_mode must be boundary, host, or skeleton_portfolio"
        )
    room_count = len(taxonomy.room_ids)
    target_size = tuple(footprint.shape[-2:])
    layers = layer_output["cad_layers"].resized(target_size)
    full_stage_labels = canonical_union_labels(
        layers.render(footprint, taxonomy),
        taxonomy,
        boundary_width=int(visible_wall_boundary_width),
    )
    layer_wall = layers.wall_logits[:, 0].sigmoid().detach().cpu().numpy()
    runtime_wall = torch.nn.functional.interpolate(
        controls["structure"][:, room_count : room_count + 1].float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    endpoints = []
    valid = []
    evidence = []
    control_endpoints = []
    control_valid = []
    control_evidence = []
    control_sources = []
    proposal_sources = []
    exterior = []
    rooms = []
    host_walls = []
    for index in range(footprint.shape[0]):
        _labels, room_footprint, room_rows = canonical_room_rows(
            layers.room_logits[index],
            footprint[index],
            taxonomy,
            minimum_area=minimum_room_area,
            largest_component_only=True,
        )
        boundary = partition_wall_mask(
            torch.from_numpy(
                layers.room_logits[index].argmax(dim=0).detach().cpu().numpy()
            )[None],
            radius=1,
        )[0].numpy()
        host_mask = (
            (runtime_wall[index].detach().cpu().numpy() >= 0.5) | boundary
        ).astype(np.uint8) * room_footprint.astype(np.uint8)
        runtime_host_walls = extract_wall_segments(host_mask, taxonomy)
        base_walls, base_scores, base_sources = candidate_walls_with_sources(
            room_rows,
            room_footprint,
            layer_wall[index],
        )
        base = pad_candidate_tensors(
            base_walls,
            base_scores,
            frame=float(target_size[-1]),
            max_walls=max_walls,
            device=device,
        )
        if candidate_mode == "host":
            walls = runtime_host_walls
            candidate_rooms = room_rows
            sources = np.full(len(walls), 2, dtype=np.int64)
            scores = (
                wall_candidate_evidence(walls, layer_wall[index], radius=2)
                if walls
                else np.zeros(0, dtype=np.float32)
            )
        elif candidate_mode == "skeleton_portfolio":
            skeleton_sources = {
                2: runtime_host_walls,
                5: extract_wall_segments(
                    full_stage_labels[index].detach().cpu().numpy(),
                    taxonomy,
                    wall_policy=WallSkeletonPolicy(
                        close_kernel=5,
                        simplification_fraction=0.010,
                    ),
                ),
            }
            for source, policy in enumerate(
                (
                    WallSkeletonPolicy(
                        close_kernel=3,
                        simplification_fraction=0.006,
                    ),
                    WallSkeletonPolicy(
                        close_kernel=5,
                        simplification_fraction=0.006,
                    ),
                ),
                start=3,
            ):
                skeleton_sources[source] = extract_wall_segments(
                    host_mask,
                    taxonomy,
                    wall_policy=policy,
                )
            (
                walls,
                scores,
                sources,
            ) = candidate_walls_with_sources(
                room_rows,
                room_footprint,
                layer_wall[index],
                extra_walls_by_source=skeleton_sources,
            )
            candidate_rooms = room_rows
        else:
            walls, scores, sources = (
                base_walls,
                base_scores,
                base_sources,
            )
            candidate_rooms = room_rows
        padded = pad_candidate_tensors(
            walls,
            scores,
            frame=float(target_size[-1]),
            max_walls=max_walls,
            device=device,
        )
        padded_sources = pad_candidate_sources(
            sources,
            max_walls=max_walls,
            device=device,
        )
        control = padded if candidate_mode == "host" else base
        control_source = (
            padded_sources
            if candidate_mode == "host"
            else pad_candidate_sources(
                base_sources,
                max_walls=max_walls,
                device=device,
            )
        )
        control_sources.append(control_source)
        proposal_sources.append(padded_sources)
        control_endpoints.append(control[0])
        control_valid.append(control[1])
        control_evidence.append(control[2])
        endpoints.append(padded[0])
        valid.append(padded[1])
        evidence.append(padded[2])
        boundary_distance = distance_transform_edt(room_footprint)
        phase = np.zeros(max_walls, dtype=bool)
        for wall_index, wall in enumerate(walls[:max_walls]):
            midpoint = 0.5 * (
                np.asarray(wall.start) + np.asarray(wall.end)
            )
            x = int(np.clip(round(midpoint[0]), 0, room_footprint.shape[1] - 1))
            y = int(np.clip(round(midpoint[1]), 0, room_footprint.shape[0] - 1))
            phase[wall_index] = boundary_distance[y, x] <= 2.5
        exterior.append(torch.from_numpy(phase).to(device))
        host_walls.append(runtime_host_walls)
        rooms.append(candidate_rooms)
    return CandidateBatch(
        endpoints=torch.stack(endpoints),
        valid=torch.stack(valid),
        evidence=torch.stack(evidence),
        exterior=torch.stack(exterior),
        rooms=rooms,
        host_walls=host_walls,
        control_endpoints=torch.stack(control_endpoints),
        control_valid=torch.stack(control_valid),
        control_evidence=torch.stack(control_evidence),
        proposal_source=torch.stack(proposal_sources),
        control_proposal_source=torch.stack(control_sources),
    )


__all__ = [
    "CandidateBatch",
    "build_candidate_batch",
    "candidate_targets",
    "match_candidates",
]
