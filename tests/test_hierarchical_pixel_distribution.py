from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.pixel_head import HierarchicalPixelDistributionHead
from floorflow.heads.pixel_distribution import PixelDistributionHead


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_hierarchical_pixel_head_is_identity_at_initialization() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    assert hierarchy.groups["room"] == taxonomy.room_ids
    head = HierarchicalPixelDistributionHead(
        12,
        hierarchy,
        hidden_channels=12,
        token_grid=4,
    )
    features = torch.randn(2, 12, 16, 16)
    output = head(features)
    base = head.base(features)
    assert torch.allclose(output["logits"], base["logits"])
    assert output["room_logits"].shape == (2, len(taxonomy.room_ids), 16, 16)
    assert output["family_tokens"].shape == (2, 5, 12)
    assert torch.isfinite(output["routing_entropy"])
    weights = output["multi_scale_weights"]
    assert weights.requires_grad
    assert weights.shape == (2, 6, taxonomy.num_classes, 16, 16)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))


def test_hierarchical_gates_and_attention_receive_gradients() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    head = HierarchicalPixelDistributionHead(
        12,
        hierarchy,
        hidden_channels=12,
        token_grid=4,
    )
    output = head(torch.randn(1, 12, 16, 16))
    loss = output["room_logits"].mean() + output["family_tokens"].mean()
    loss.backward()
    assert head.opening_parent_scale.grad is not None
    assert head.room_parent_scale.grad is not None
    assert head.furniture_parent_scale.grad is not None
    assert head.route.q_proj.weight.grad is not None


def test_legacy_pixel_checkpoint_loads_with_out_zero_dynamic_router() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    source = HierarchicalPixelDistributionHead(
        12,
        hierarchy,
        hidden_channels=12,
        token_grid=4,
    )
    legacy = {
        name: value
        for name, value in source.base.state_dict().items()
        if not name.startswith("dynamic_output.")
    }
    target = HierarchicalPixelDistributionHead(
        12,
        hierarchy,
        hidden_channels=12,
        token_grid=4,
    )
    target.base.load_state_dict(legacy)
    assert target.base.dynamic_output.router[-1].weight.abs().sum() == 0


def test_dynamic_head_variant_contracts() -> None:
    taxonomy = _taxonomy()
    features = torch.randn(1, 12, 8, 8)
    for variant in ("fixed", "mixture", "generated", "query"):
        head = PixelDistributionHead(
            12,
            taxonomy.num_classes,
            hidden_channels=12,
            room_ids=taxonomy.room_ids,
            dynamic_head_variant=variant,
        )
        output = head(features)
        assert output["logits"].shape == (1, taxonomy.num_classes, 8, 8)
        if variant in {"fixed", "query"}:
            assert output["multi_scale_weights"] is None
        else:
            assert output["multi_scale_weights"].requires_grad
        if variant == "query":
            assert output["dynamic_query_diagnostics"]["query_budget"].shape == (1,)
