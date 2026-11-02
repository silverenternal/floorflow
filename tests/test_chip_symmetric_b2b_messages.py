from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.sparse_layers import GraphMessageBlock


def _run_block(symmetric: bool) -> torch.Tensor:
    torch.manual_seed(3)
    block = GraphMessageBlock(16, symmetric_edges=symmetric)
    if symmetric:
        block.symmetric_gate.data.fill_(1.0)
    tokens = torch.zeros(1, 2, 16)
    tokens[0, 1] = torch.randn(16)
    return block(
        tokens,
        b2b=torch.tensor([[[0.0, 1.0, 1.0]]]),
        p2b=None,
        pins=None,
        n_b2b=torch.tensor([1]),
        n_p2b=None,
        n_pins=None,
    )


def test_symmetric_b2b_messages_reach_both_endpoints() -> None:
    directional = _run_block(False)
    symmetric = _run_block(True)
    assert not torch.allclose(directional[0, 0], symmetric[0, 0])
    torch.testing.assert_close(directional[0, 1], symmetric[0, 1])


def test_symmetric_b2b_path_is_identity_at_warm_start() -> None:
    torch.manual_seed(7)
    directional = GraphMessageBlock(16)
    torch.manual_seed(7)
    symmetric = GraphMessageBlock(16, symmetric_edges=True)
    tokens = torch.randn(1, 3, 16)
    kwargs = {
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]]),
        "p2b": None,
        "pins": None,
        "n_b2b": torch.tensor([2]),
        "n_p2b": None,
        "n_pins": None,
    }

    torch.testing.assert_close(directional(tokens, **kwargs), symmetric(tokens, **kwargs))
    symmetric.symmetric_gate.data.fill_(1.0)
    assert not torch.allclose(directional(tokens, **kwargs), symmetric(tokens, **kwargs))


def test_symmetric_b2b_messages_preserve_block_permutation_equivariance() -> None:
    torch.manual_seed(5)
    block = GraphMessageBlock(16, symmetric_edges=True)
    block.symmetric_gate.data.fill_(1.0)
    tokens = torch.randn(1, 4, 16)
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]])
    counts = torch.tensor([2])
    original = block(
        tokens,
        b2b=b2b,
        p2b=None,
        pins=None,
        n_b2b=counts,
        n_p2b=None,
        n_pins=None,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    reordered = block(
        tokens[:, permutation],
        b2b=torch.stack(
            [
                torch.stack(
                    [
                        inverse[b2b[0, :, 0].long()],
                        inverse[b2b[0, :, 1].long()],
                        b2b[0, :, 2],
                    ],
                    dim=-1,
                )
            ]
        ),
        p2b=None,
        pins=None,
        n_b2b=counts,
        n_p2b=None,
        n_pins=None,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_symmetric_b2b_checkpoint_round_trip() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        symmetric_b2b_messages=True,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "symmetric_b2b_messages": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.symmetric_b2b_messages
    assert all(
        layer.symmetric_edges for layer in restored.typed.graph_layers
    )


def test_old_symmetric_checkpoint_adds_zero_reverse_gate() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        symmetric_b2b_messages=True,
    )
    legacy = {
        name: value
        for name, value in model.state_dict().items()
        if not name.endswith(".symmetric_gate")
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "symmetric_b2b_messages": True,
        },
        "model": legacy,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    assert all(
        layer.symmetric_gate.item() == 0.0
        for layer in restored.typed.graph_layers
    )
