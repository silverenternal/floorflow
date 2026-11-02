from __future__ import annotations

import torch

from floorflow.heads.geometry_heads import (
    focal_heatmap_loss,
    opening_candidate_loss,
    structural_candidate_mask,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def test_structural_candidates_cover_room_transition_and_envelope() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    first, second = taxonomy.room_ids[:2]
    labels = torch.full((1, 8, 8), first)
    labels[:, :, 4:] = second
    footprint = torch.ones(1, 8, 8)
    candidates = structural_candidate_mask(
        labels,
        footprint,
        taxonomy,
        radius=1,
    )
    assert candidates.shape == footprint.shape
    assert candidates[0, :, 3:5].all()
    assert candidates[0, 0, :].all()
    assert candidates[0, :, 0].all()


def test_structural_mask_controls_heatmap_negative_loss() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2, 2] = 1.0
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 1:4, 1:4] = 1.0
    logits = torch.full_like(target, 4.0)
    logits[0, 0, 1:4, 1:4] = 0.0
    unrestricted = focal_heatmap_loss(logits, target)
    restricted = focal_heatmap_loss(logits, target, valid_mask=mask)
    assert torch.isfinite(restricted)
    assert restricted != unrestricted


def test_opening_candidate_loss_is_finite_and_trainable() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    first, second = taxonomy.room_ids[:2]
    labels = torch.full((1, 8, 8), first)
    labels[:, :, 4:] = second
    footprint = torch.ones(1, 8, 8)
    opening_logits = torch.randn(1, len(taxonomy.opening_ids), 8, 8, requires_grad=True)
    background_logits = torch.randn(1, 8, 8, requires_grad=True)
    loss, value = opening_candidate_loss(
        {
            "opening_logits": opening_logits,
            "opening_background_logits": background_logits,
        },
        labels,
        footprint,
        taxonomy,
    )
    assert torch.isfinite(loss) and torch.isfinite(value)
    loss.backward()
    assert opening_logits.grad is not None
    assert background_logits.grad is not None
