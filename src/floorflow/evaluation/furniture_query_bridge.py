"""Checkpoint and overlay adapter for dynamic furniture queries."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_object_instance import (
    DynamicFurnitureInstanceHead,
    render_object_queries,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan


def load_furniture_query_head(path: str | Path, device: torch.device):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = payload["config"]
    checkpoint_taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    head = DynamicFurnitureInstanceHead(
        int(config["in_channels"]),
        len(checkpoint_taxonomy.furniture_ids),
        hidden_dim=int(config.get("hidden_dim", 96)),
        max_queries=int(config.get("max_queries", 24)),
        min_queries=int(config.get("min_queries", 4)),
    )
    head.load_state_dict(payload["head"])
    return head.to(device).eval(), checkpoint_taxonomy


@torch.inference_mode()
def apply_furniture_queries(
    head,
    distribution: dict,
    layers: LayeredFloorplan,
    footprint: torch.Tensor,
    taxonomy,
    *,
    threshold: float,
) -> tuple[LayeredFloorplan, dict[str, float]]:
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=footprint.device,
    )
    room_probability = torch.isin(
        layers.room_labels,
        room_ids,
    ).float()[:, None]
    prediction = head(
        distribution["features"],
        room_probability,
        footprint,
    )
    rendered = render_object_queries(
        prediction["class_logits"],
        prediction["mask_logits"],
        taxonomy.furniture_ids,
        footprint,
        presence_threshold=float(threshold),
        mask_threshold=float(threshold),
        active_queries=prediction["active_queries"],
    )
    furniture = torch.where(
        rendered > 0,
        rendered,
        layers.furniture,
    )
    updated = LayeredFloorplan(
        room_labels=layers.room_labels,
        wall=layers.wall,
        opening=layers.opening,
        furniture=furniture,
        footprint=footprint[:, 0] if footprint.ndim == 4 else footprint,
        taxonomy=taxonomy,
    )
    return updated, {
        "query_budget_mean": float(prediction["query_budget"].float().mean()),
        "rendered_furniture_fraction": float(
            (rendered > 0).float().mean()
        ),
    }


__all__ = ["apply_furniture_queries", "load_furniture_query_head"]
