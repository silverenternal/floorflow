from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.sparse_layers import prepare_state_edge_cache
from floorflow.chip_flow.state_conditioned_edge_bias import (
    StateConditionedEdgeBias,
)


def _cache() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    areas = torch.ones(2, 4)
    adjacency = torch.zeros(2, 4, 4)
    cache = prepare_state_edge_cache(
        areas=areas,
        b2b=torch.tensor([[[0.0, 1.0, 1.0], [2.0, 3.0, 0.5]]] * 2),
        p2b=torch.zeros(2, 1, 3),
        pins=torch.zeros(2, 1, 2),
        n_b2b=torch.tensor([2, 2]),
        n_p2b=torch.tensor([0, 0]),
        n_pins=torch.tensor([1, 1]),
    )
    return adjacency, cache


def test_state_conditioned_edge_bias_is_zero_initialized_and_symmetric() -> None:
    torch.manual_seed(31)
    module = StateConditionedEdgeBias()
    adjacency, cache = _cache()
    state = torch.randn(2, 4, 3)
    output = module(state, adjacency, cache)
    torch.testing.assert_close(output, adjacency)

    with torch.no_grad():
        module.network[-1].bias.fill_(0.25)
    output = module(state, adjacency, cache)
    torch.testing.assert_close(output, output.transpose(1, 2))
    assert output.diagonal(dim1=1, dim2=2).abs().max() == 0.0


def test_state_conditioned_edge_bias_has_direct_gradient() -> None:
    module = StateConditionedEdgeBias()
    adjacency, cache = _cache()
    state = torch.randn(2, 4, 3, requires_grad=True)
    output = module(state, adjacency, cache)
    output.square().mean().backward()
    assert module.network[-1].weight.grad is not None
    assert state.grad is not None


def test_state_conditioned_edge_bias_preserves_adjacency_dtype() -> None:
    module = StateConditionedEdgeBias()
    adjacency, cache = _cache()
    output = module(torch.randn(2, 4, 3), adjacency.half(), cache)
    assert output.dtype == torch.float16


def test_state_conditioned_edge_bias_is_degree_normalized() -> None:
    module = StateConditionedEdgeBias()
    adjacency, cache = _cache()
    with torch.no_grad():
        module.network[-1].bias.fill_(10.0)
    output = module(torch.randn(2, 4, 3), adjacency, cache)
    assert output.max() <= 1.0
    assert output.min() >= -1.0


def test_checkpoint_reconstruction_allows_only_new_edge_bias_parameters() -> None:
    torch.manual_seed(37)
    base = HierarchicalTokenFlowModel(hidden=24, heads=3, depth=1)
    candidate = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        state_conditioned_edge_bias=True,
    )
    incompatible = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith("state_conditioned_edge_bias.")
        for name in incompatible.missing_keys
    )
