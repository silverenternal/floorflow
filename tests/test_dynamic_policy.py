from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.training.dynamic_policy import RoomDynamicPolicy


def test_room_dynamic_policy_reports_family_budgets_and_round_trips() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    generator = LearnedMaskGenerator(taxonomy.num_target_channels, hidden_channels=8)
    policy = RoomDynamicPolicy(
        taxonomy,
        mask_generator=generator,
    )
    source = torch.ones(1, taxonomy.num_target_channels, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    output = policy.propose(
        source,
        footprint,
        torch.zeros(taxonomy.num_target_channels),
        missing_ratio=0.25,
        floor_ids=torch.zeros(1, dtype=torch.long),
    )
    assert set(output.family_budget) == {"room", "wall", "opening", "furniture"}
    assert output.condition.missing_mask.shape == source.shape
    policy.observe_score(0.5)
    state = policy.state_dict()
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.zero_()
    policy.load_state_dict(state)
    assert any(parameter.abs().sum() > 0 for parameter in generator.parameters())
