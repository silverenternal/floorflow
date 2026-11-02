"""Typed non-uniform priors over predicted semantic channels."""

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy

BACKGROUND = 0
ROOM = 1
WALL = 2
OPENING = 3
FURNITURE = 4
OTHER = 5


def channel_family_ids(taxonomy: FloorplanTaxonomy) -> torch.Tensor:
    """Map every class channel to a coarse structural family."""

    families = []
    room_ids = set(taxonomy.room_ids)
    opening_ids = set(taxonomy.opening_ids)
    furniture_ids = set(taxonomy.furniture_ids)
    wall_id = taxonomy.class_id("wall")
    for class_id, name in enumerate(taxonomy.class_names):
        if class_id == 0:
            family = BACKGROUND
        elif class_id == wall_id:
            family = WALL
        elif class_id in room_ids:
            family = ROOM
        elif class_id in opening_ids:
            family = OPENING
        elif class_id in furniture_ids:
            family = FURNITURE
        elif name.startswith("room:"):
            family = ROOM
        else:
            family = OTHER
        families.append(family)
    return torch.tensor(families, dtype=torch.long)


def typed_channel_prior(taxonomy: FloorplanTaxonomy) -> torch.Tensor:
    """Return a non-uniform structural attention prior over all channels."""

    families = channel_family_ids(taxonomy)
    channels = families.numel()
    prior = torch.full((channels, channels), -2.0, dtype=torch.float32)
    prior.diagonal().fill_(0.8)

    room = families == ROOM
    wall = families == WALL
    opening = families == OPENING
    furniture = families == FURNITURE
    background = families == BACKGROUND

    prior[room[:, None] & room[None, :]] = 0.45
    prior[room[:, None] & wall[None, :]] = 1.2
    prior[wall[:, None] & room[None, :]] = 1.2
    prior[room[:, None] & opening[None, :]] = 0.9
    prior[opening[:, None] & room[None, :]] = 0.9
    prior[wall[:, None] & opening[None, :]] = 1.5
    prior[opening[:, None] & wall[None, :]] = 1.5
    prior[furniture[:, None] & room[None, :]] = 0.7
    prior[room[:, None] & furniture[None, :]] = 0.7
    prior[background[:, None] & background[None, :]] = 0.4
    prior.diagonal().fill_(0.8)
    return prior


__all__ = [
    "BACKGROUND",
    "FURNITURE",
    "OPENING",
    "OTHER",
    "ROOM",
    "WALL",
    "channel_family_ids",
    "typed_channel_prior",
]
