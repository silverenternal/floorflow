from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.flow.condition_training import (
    build_condition_flow_sample,
    condition_velocity_loss,
)


def test_condition_flow_sample_shapes_and_loss() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
    )
    footprint = torch.ones(2, 32, 32)
    labels = torch.zeros(2, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    sample = build_condition_flow_sample(
        labels,
        footprint,
        channels=taxonomy.num_target_channels,
        matcher="torchcfm",
    )
    assert sample.location.shape == sample.velocity.shape
    assert sample.time.shape == (2,)
    loss = condition_velocity_loss(
        sample.velocity,
        sample.velocity,
        footprint,
    )
    assert float(loss) == 0.0

    inward = build_condition_flow_sample(
        labels,
        footprint,
        channels=taxonomy.num_target_channels,
        matcher="torchcfm",
        inward_progress=0.25,
    )
    assert inward.weight_map is not None
    inward_loss = condition_velocity_loss(
        inward.velocity,
        inward.velocity,
        footprint,
        weight_map=inward.weight_map,
    )
    assert float(inward_loss) == 0.0
