from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.pixel_distribution import (
    PixelDistributionHead,
    apply_distribution_prior,
    pixel_distribution_loss,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def test_pixel_distribution_head_outputs_probability_and_room_prior() -> None:
    taxonomy = _taxonomy()
    head = PixelDistributionHead(
        32,
        taxonomy.num_classes,
        hidden_channels=24,
        room_ids=taxonomy.room_ids,
    )
    features = torch.randn(2, 32, 16, 16, requires_grad=True)
    output = head(features)
    assert output["logits"].shape == (2, taxonomy.num_classes, 16, 16)
    assert torch.allclose(
        output["probability"].sum(dim=1),
        torch.ones(2, 16, 16),
        atol=1.0e-5,
    )
    assert output["room_logits"].shape == (2, len(taxonomy.room_ids), 16, 16)
    assert 0.0 <= float(output["reliability"].min().detach()) <= 1.0
    output["logits"].mean().backward()
    assert features.grad is not None


def test_distribution_prior_and_iou_loss_are_differentiable() -> None:
    taxonomy = _taxonomy()
    logits = torch.randn(
        1,
        taxonomy.num_classes,
        16,
        16,
        requires_grad=True,
    )
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = taxonomy.room_ids[0]
    footprint = torch.ones(1, 1, 16, 16)
    loss, components = pixel_distribution_loss(
        logits,
        target,
        footprint,
        taxonomy,
        boundary_weight=0.2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert "room_soft_iou_loss" in components
    assert components["ce_term"].requires_grad
    assert components["room_iou_term"].requires_grad
    assert components["tversky_term"].requires_grad
    assert components["boundary_term"].requires_grad
    prior = apply_distribution_prior(
        logits,
        {
            "reliability": torch.full_like(logits[:, :1], 0.5),
        },
    )
    assert prior.shape == logits.shape
    assert torch.equal(prior, logits)


def test_classwise_prior_changes_room_softmax() -> None:
    taxonomy = _taxonomy()
    logits = torch.zeros(
        (1, len(taxonomy.room_ids), 4, 4),
        requires_grad=True,
    )
    room_prior = torch.full_like(logits, 0.25)
    room_prior[:, 0] = 0.75
    gated = apply_distribution_prior(
        logits,
        {"reliability": torch.ones(1, 1, 4, 4)},
        scale=1.0,
        room_prior=room_prior,
    )
    assert not torch.allclose(gated, logits)
    assert gated.shape == logits.shape
    gated.sum().backward()
    assert logits.grad is not None


def test_pixel_room_loss_uses_full_class_ids_not_foreground_offsets() -> None:
    taxonomy = _taxonomy()
    room_id = taxonomy.room_ids[0]
    logits = torch.full(
        (1, taxonomy.num_classes, 8, 8),
        -10.0,
    )
    logits[:, room_id] = 10.0
    target = torch.full((1, 8, 8), room_id, dtype=torch.long)
    footprint = torch.ones(1, 1, 8, 8)
    _, components = pixel_distribution_loss(
        logits,
        target,
        footprint,
        taxonomy,
    )
    assert float(components["room_iou_term"]) < 1.0e-4


def test_pixel_family_loss_handles_furniture_and_support() -> None:
    taxonomy = _taxonomy()
    logits = torch.randn(
        1,
        taxonomy.num_classes,
        8,
        8,
        requires_grad=True,
    )
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:4, 2:4] = taxonomy.furniture_ids[0]
    target[:, 5:7, 5:7] = taxonomy.class_id("structure")
    footprint = torch.ones(1, 1, 8, 8)
    loss, components = pixel_distribution_loss(
        logits,
        target,
        footprint,
        taxonomy,
        family_iou_weight=0.3,
    )
    assert "pixel_furniture_iou_loss" in components
    assert "pixel_support_iou_loss" in components
    loss.backward()
    assert logits.grad is not None
