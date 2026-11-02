from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.cfm_bridge import DistributionCFMBridge
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.model.cfm_structure_adapter import enable_cfm_structure
from floorflow.representation.conditioning import prepare_condition


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 5, "Kitchen": 5}),
            "furniture": Counter({"Sink": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _base(taxonomy):
    return FloorplanFlowModel(
        target_channels=taxonomy.num_classes - 1,
        base_channels=16,
        levels=2,
        semantic_taxonomy=taxonomy,
        semantic_channel_attention=False,
        slot_mixing=False,
        dynamic_output_head=False,
        topology_aux=False,
        semantic_endpoint_head=False,
        advanced_streams=False,
        family_streams=False,
        coarse_to_face_cfm=False,
    )


def test_static_condition_features_are_cacheable_across_time() -> None:
    taxonomy = _taxonomy()
    base = enable_cfm_structure(_base(taxonomy), taxonomy)
    base.eval()
    footprint = torch.zeros(1, 1, 16, 16)
    footprint[:, :, 2:14, 2:14] = 1.0
    first = base.encode_condition(
        footprint,
        torch.tensor([0.1]),
    )
    second = base.encode_condition(
        footprint,
        torch.tensor([0.9]),
    )
    for left, right in zip(first, second, strict=True):
        assert torch.allclose(left, right)


def test_typed_family_streams_use_group_specific_state_evidence() -> None:
    taxonomy = _taxonomy()
    base = enable_cfm_structure(_base(taxonomy), taxonomy)
    first = torch.zeros(1, taxonomy.num_classes - 1, 16, 16)
    second = first.clone()
    room = max(0, taxonomy.room_ids[0] - 1)
    second[:, room] = 1.0
    condition = torch.randn(
        1,
        base.condition_encoder.blocks[-1].conv1.out_channels,
        4,
        4,
    )
    first_streams = base.family_stream_evidence(first, condition)
    second_streams = base.family_stream_evidence(second, condition)
    assert not torch.allclose(first_streams["room"], second_streams["room"])
    assert torch.allclose(first_streams["wall"], second_streams["wall"])


def test_efficient_shape_descriptor_is_vectorized_and_finite() -> None:
    footprint = torch.zeros(2, 1, 32, 32)
    footprint[:, :, 4:20, 6:26] = 1.0
    condition = prepare_condition(footprint)
    assert condition.shape == (2, 10, 32, 32)
    assert torch.isfinite(condition).all()


def test_residual_cfm_bridge_starts_from_pixel_state() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    bridge = DistributionCFMBridge(hierarchy, bridge_mode="residual")
    pixel = {
        "probability": torch.rand(1, taxonomy.num_classes, 8, 8),
    }
    element = torch.rand(1, taxonomy.num_classes - 1, 8, 8)
    output = bridge(
        pixel,
        {},
        torch.ones(1, 1, 8, 8),
        element_probability=element,
    )
    assert torch.allclose(output["state"], output["pixel_state"])
