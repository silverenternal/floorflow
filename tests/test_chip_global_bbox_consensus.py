from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.global_bbox_consensus import GlobalBBoxConsensus
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel


def test_global_bbox_consensus_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(7)
    module = GlobalBBoxConsensus()
    latent = torch.randn(2, 6, 3)
    global_prediction = torch.tensor(
        [[0.2, -0.1, 2.0, 1.5], [-0.3, 0.4, 1.8, 2.2]]
    )
    areas = torch.ones(2, 6)
    output, residual = module(latent, global_prediction, areas)
    assert torch.equal(output, latent)
    assert residual.shape == (2,)
    output.sum().backward()
    assert module.gate.grad is not None
    assert module.gate.grad.abs().sum() > 0
    module.gate.data.fill_(0.5)
    corrected, _ = module(latent, global_prediction, areas)
    assert torch.equal(corrected[..., 2], latent[..., 2])
    assert not torch.equal(corrected[..., :2], latent[..., :2])


def test_global_bbox_consensus_checkpoint_rebuilds_module() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=48,
        heads=3,
        depth=1,
        head_dim=16,
        typed_layers=1,
        global_bbox_consensus=True,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 48,
            "heads": 3,
            "depth": 1,
            "head_dim": 16,
            "typed_layers": 1,
            "global_bbox_consensus": True,
        },
        "ema": model.state_dict(),
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert isinstance(restored.global_bbox_consensus, GlobalBBoxConsensus)


def test_global_bbox_consensus_keeps_preplaced_centers_exact() -> None:
    module = GlobalBBoxConsensus()
    module.gate.data.fill_(1.0)
    prediction = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    constraints = torch.zeros(1, 3, 5)
    constraints[0, 1, 1] = 1.0
    output, _ = module(
        prediction,
        prediction.new_tensor([[1.0, 0.0, 2.0, 1.0]]),
        torch.ones(1, 3),
        constraints=constraints,
    )

    torch.testing.assert_close(output[0, 1], prediction[0, 1])
    assert not torch.equal(output[0, 0, :2], prediction[0, 0, :2])
