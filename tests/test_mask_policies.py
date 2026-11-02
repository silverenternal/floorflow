from __future__ import annotations

import pytest
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.mask_policies import (
    AdversarialMaskScheduler,
    AttributeMaskPolicy,
    CompositeMaskPolicy,
    SalienceMaskPolicy,
    SpatialBlockMaskPolicy,
)


def test_attribute_policy_masks_independently_and_preserves_room() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    source = torch.ones(2, taxonomy.num_target_channels, 16, 16)
    footprint = torch.ones(2, 1, 16, 16)
    mask = AttributeMaskPolicy(min_room_channels=1).sample(
        source,
        footprint,
        taxonomy,
        ratio=0.5,
    )
    room_channels = torch.tensor(taxonomy.room_ids) - 1
    assert mask.value.shape == source.shape
    assert torch.all((~mask.group[:, room_channels]).any(dim=1))


def test_composite_policy_combines_implementations() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    source = torch.ones(1, taxonomy.num_target_channels, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    policy = CompositeMaskPolicy(spatial_probability=1.0)
    mask = policy.sample(
        source,
        footprint,
        taxonomy,
        ratio=0.25,
    )
    assert mask.value.shape == source.shape
    assert mask.spatial.any()
    assert mask.group.any()


def test_salience_policy_prefers_low_activation_tokens() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    source = torch.ones(1, taxonomy.num_target_channels, 16, 16)
    source[:, -4:] = 100.0
    footprint = torch.ones(1, 1, 16, 16)
    mask = SalienceMaskPolicy().sample(
        source,
        footprint,
        taxonomy,
        ratio=0.5,
    )
    assert mask.value[:, -4:].float().mean() < mask.value[:, :-4].float().mean()


def test_adversarial_scheduler_updates_policy_weights() -> None:
    scheduler = AdversarialMaskScheduler(
        (
            SpatialBlockMaskPolicy(),
            AttributeMaskPolicy(),
        )
    )
    scheduler.update("spatial", 0.9)
    assert scheduler.weights["spatial"] > scheduler.weights["attribute"]
    assert abs(sum(scheduler.weights.values()) - 1.0) < 1.0e-5


def test_adversarial_scheduler_round_trips_and_prevents_collapse() -> None:
    scheduler = AdversarialMaskScheduler(
        (
            SpatialBlockMaskPolicy(),
            AttributeMaskPolicy(),
        ),
        minimum_probability=0.1,
    )
    for _ in range(20):
        scheduler.update("spatial", 1.0)
        scheduler.update("attribute", 0.0)
    assert scheduler.weights["attribute"] >= 0.1 - 1.0e-6
    restored = AdversarialMaskScheduler(scheduler.policies)
    restored.load_state_dict(scheduler.state_dict())
    assert set(restored.weights) == set(scheduler.weights)
    for name in scheduler.weights:
        assert restored.weights[name] == pytest.approx(scheduler.weights[name])
    assert restored.reward_means == scheduler.reward_means
