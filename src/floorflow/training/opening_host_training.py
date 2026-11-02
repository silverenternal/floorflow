"""Build and match predicted opening queries to candidate wall hosts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutWall
from floorflow.inference.opening_heatmap_decoder import (
    learned_opening_detections_from_maps,
)
from floorflow.training.cad_wall_candidate_training import (
    CandidateBatch,
    match_candidates,
)


@dataclass(frozen=True)
class OpeningQueries:
    centers: torch.Tensor
    widths: torch.Tensor
    family_logits: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class OpeningHostTargets:
    hosts: torch.Tensor
    center_fraction: torch.Tensor
    width_fraction: torch.Tensor
    types: torch.Tensor
    matched: torch.Tensor


def opening_query_maps(
    controls: dict[str, torch.Tensor],
    *,
    frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent foreground and family maps for query extraction."""

    family_logits = controls.get("opening_family_logits")
    foreground = controls.get("opening_foreground_probability")
    if foreground is None or family_logits is None:
        product = controls["opening_family_probability"].float()
        if foreground is None:
            foreground = product.amax(dim=1)
        if family_logits is None:
            family_logits = product.clamp_min(1.0e-6).log()
    foreground = foreground.float()
    family_logits = family_logits.float()
    if tuple(foreground.shape[-2:]) != (int(frame), int(frame)):
        foreground = F.interpolate(
            foreground[:, None],
            size=(int(frame), int(frame)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    if tuple(family_logits.shape[-2:]) != (int(frame), int(frame)):
        family_logits = F.interpolate(
            family_logits,
            size=(int(frame), int(frame)),
            mode="bilinear",
            align_corners=False,
        )
    if family_logits.shape[1] != 2:
        raise ValueError("opening family logits must have two channels")
    if foreground.shape != family_logits.shape[:1] + family_logits.shape[-2:]:
        raise ValueError("opening foreground and family maps must align")
    return foreground, family_logits


def detection_opening_queries(
    controls: dict[str, torch.Tensor],
    candidates: CandidateBatch,
    taxonomy: FloorplanTaxonomy,
    *,
    frame: float,
    threshold: float = 0.3,
    max_openings: int = 48,
    device: torch.device,
    host_walls: list[list[ReadoutWall]] | None = None,
) -> OpeningQueries:
    foreground, family_logits = opening_query_maps(
        controls,
        frame=int(frame),
    )
    probability = foreground.detach().cpu().numpy()
    family_probability = family_logits.detach().cpu().numpy()
    centers = np.zeros((candidates.valid.shape[0], max_openings, 2), dtype=np.float32)
    widths = np.zeros((candidates.valid.shape[0], max_openings), dtype=np.float32)
    families = np.zeros((candidates.valid.shape[0], max_openings, 2), dtype=np.float32)
    valid = np.zeros((candidates.valid.shape[0], max_openings), dtype=bool)
    for index in range(candidates.valid.shape[0]):
        host_plan = RasterWallPlan(
            frame=float(frame),
            walls=(
                candidates.host_walls[index]
                if host_walls is None
                else host_walls[index]
            ),
            rooms=candidates.rooms[index],
            openings=[],
        )
        detections = learned_opening_detections_from_maps(
            host_plan,
            probability[index],
            family_probability[index],
            taxonomy,
            threshold=float(threshold),
            min_area=7,
        )
        detections.sort(key=lambda item: item.score, reverse=True)
        for item, detection in enumerate(detections[:max_openings]):
            centers[index, item] = np.asarray(detection.center) / float(frame)
            widths[index, item] = detection.width / float(frame)
            if detection.family == "window":
                families[index, item] = (-1.0, 1.0)
            else:
                families[index, item] = (1.0, -1.0)
            valid[index, item] = True
    return OpeningQueries(
        centers=torch.from_numpy(centers).to(device),
        widths=torch.from_numpy(widths).to(device),
        family_logits=torch.from_numpy(families).to(device),
        valid=torch.from_numpy(valid).to(device),
    )


def candidate_host_walls(
    candidates: CandidateBatch,
    selected: torch.Tensor,
    *,
    frame: float,
) -> list[list[ReadoutWall]]:
    """Materialize the exact scored wall subset used by the opening layer."""

    rows = []
    for index in range(selected.shape[0]):
        walls = []
        for candidate_index in torch.nonzero(
            selected[index],
            as_tuple=False,
        ).flatten().tolist():
            endpoints = candidates.endpoints[index, candidate_index].detach().cpu()
            walls.append(
                ReadoutWall(
                    start=tuple(
                        float(value) * float(frame)
                        for value in endpoints[0]
                    ),
                    end=tuple(
                        float(value) * float(frame)
                        for value in endpoints[1]
                    ),
                    thickness=3.0,
                    wall_type=(
                        "exterior"
                        if bool(candidates.exterior[index, candidate_index].cpu())
                        else "interior"
                    ),
                )
            )
        rows.append(walls)
    return rows


def target_opening_queries(
    targets: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    device: torch.device,
) -> OpeningQueries:
    classes = targets["opening_classes"]
    family = torch.zeros(
        *classes.shape,
        2,
        device=classes.device,
    )
    for local_index, class_id in enumerate(taxonomy.opening_ids):
        family[..., 1] += (
            classes == local_index
        ).float() * (
            1.0
            if taxonomy.class_names[class_id].startswith("window:")
            else -1.0
        )
    family[..., 0] = -family[..., 1]
    return OpeningQueries(
        centers=targets["opening_centers"].to(device),
        widths=targets["opening_widths"].to(device),
        family_logits=family.to(device),
        valid=targets["opening_exists"].bool().to(device),
    )


def match_opening_host_targets(
    queries: OpeningQueries,
    candidates: CandidateBatch,
    targets: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    candidate_mask: torch.Tensor | None = None,
    center_tolerance: float = 0.08,
) -> OpeningHostTargets:
    batch, openings = queries.valid.shape
    hosts = torch.zeros(batch, openings, dtype=torch.long)
    center_fraction = torch.zeros(batch, openings)
    width_fraction = torch.zeros(batch, openings)
    types = torch.zeros(batch, openings, dtype=torch.long)
    matched = torch.zeros(batch, openings, dtype=torch.bool)
    wall_pairs, _ = match_candidates(
        candidates.endpoints,
        targets["wall_endpoints"],
        targets["wall_exists"],
        candidates.valid,
        0.08,
    )
    for index in range(batch):
        wall_rows, wall_columns = wall_pairs[index]
        target_to_candidate = {
            int(column): int(row)
            for row, column in zip(wall_rows, wall_columns, strict=True)
        }
        count = int(targets["opening_exists"][index].sum())
        if count == 0 or not queries.valid[index].any():
            continue
        distance = torch.cdist(
            queries.centers[index],
            targets["opening_centers"][index, :count],
        )
        cost = distance.cpu().numpy()
        query_family = queries.family_logits[index].argmax(dim=-1)
        target_family = torch.zeros(
            count,
            dtype=torch.long,
            device=targets["opening_classes"].device,
        )
        for local_index, class_id in enumerate(taxonomy.opening_ids):
            target_family += (
                targets["opening_classes"][index, :count] == local_index
            ).long() * (
                1
                if taxonomy.class_names[class_id].startswith("window:")
                else 0
            )
        compatible = (
            query_family[:, None] == target_family[None, :]
        ).cpu().numpy()
        valid_queries = queries.valid[index].cpu().numpy()
        cost = np.where(
            compatible & valid_queries[:, None],
            cost,
            1.0e6,
        )
        rows, columns = linear_sum_assignment(cost)
        for query_index, target_index in zip(rows, columns, strict=True):
            if not queries.valid[index, query_index]:
                continue
            if not np.isfinite(cost[query_index, target_index]):
                continue
            if cost[query_index, target_index] > float(center_tolerance):
                continue
            target_host = int(targets["opening_hosts"][index, target_index])
            candidate_host = target_to_candidate.get(target_host)
            candidate_allowed = (
                candidates.valid[index, candidate_host]
                if candidate_host is not None
                else False
            )
            if candidate_mask is not None and candidate_host is not None:
                candidate_allowed = (
                    candidate_allowed and candidate_mask[index, candidate_host]
                )
            if candidate_host is None or not bool(candidate_allowed):
                continue
            host_wall = targets["wall_endpoints"][index, target_host]
            center = targets["opening_centers"][index, target_index]
            delta = host_wall[1] - host_wall[0]
            length_sq = float((delta**2).sum().clamp_min(1.0e-9))
            fraction = float(
                ((center - host_wall[0]) * delta).sum().clamp(0.0, 1.0)
                / torch.tensor(length_sq, dtype=delta.dtype)
            )
            host_length = float(delta.norm().clamp_min(1.0e-6))
            hosts[index, query_index] = candidate_host
            center_fraction[index, query_index] = fraction
            width_fraction[index, query_index] = min(
                1.0,
                float(targets["opening_widths"][index, target_index])
                / host_length,
            )
            types[index, query_index] = int(
                targets["opening_classes"][index, target_index]
            )
            matched[index, query_index] = True
    return OpeningHostTargets(
        hosts=hosts,
        center_fraction=center_fraction,
        width_fraction=width_fraction,
        types=types,
        matched=matched,
    )


def target_opening_host_targets(
    candidates: CandidateBatch,
    targets: dict[str, torch.Tensor],
    *,
    candidate_mask: torch.Tensor,
) -> OpeningHostTargets:
    wall_pairs, _ = match_candidates(
        candidates.endpoints,
        targets["wall_endpoints"],
        targets["wall_exists"],
        candidates.valid,
        0.08,
    )
    batch, openings = targets["opening_exists"].shape
    hosts = torch.zeros(batch, openings, dtype=torch.long)
    center_fraction = torch.zeros(batch, openings)
    width_fraction = torch.zeros(batch, openings)
    matched = torch.zeros(batch, openings, dtype=torch.bool)
    for index in range(batch):
        wall_rows, wall_columns = wall_pairs[index]
        target_to_candidate = {
            int(column): int(row)
            for row, column in zip(wall_rows, wall_columns, strict=True)
        }
        for opening in range(openings):
            if not bool(targets["opening_exists"][index, opening]):
                continue
            target_host = int(targets["opening_hosts"][index, opening])
            candidate_host = target_to_candidate.get(target_host)
            if candidate_host is None or not bool(
                candidate_mask[index, candidate_host]
            ):
                continue
            host_wall = targets["wall_endpoints"][index, target_host]
            center = targets["opening_centers"][index, opening]
            delta = host_wall[1] - host_wall[0]
            length_sq = float((delta**2).sum().clamp_min(1.0e-9))
            fraction = float(
                ((center - host_wall[0]) * delta).sum().clamp(0.0, 1.0)
                / torch.tensor(length_sq, dtype=delta.dtype)
            )
            host_length = float(delta.norm().clamp_min(1.0e-6))
            hosts[index, opening] = candidate_host
            center_fraction[index, opening] = fraction
            width_fraction[index, opening] = min(
                1.0,
                float(targets["opening_widths"][index, opening])
                / host_length,
            )
            matched[index, opening] = True
    return OpeningHostTargets(
        hosts=hosts,
        center_fraction=center_fraction,
        width_fraction=width_fraction,
        types=targets["opening_classes"].clone(),
        matched=matched,
    )


__all__ = [
    "OpeningHostTargets",
    "OpeningQueries",
    "candidate_host_walls",
    "detection_opening_queries",
    "match_opening_host_targets",
    "opening_query_maps",
    "target_opening_host_targets",
    "target_opening_queries",
]
