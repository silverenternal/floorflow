from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": torch.zeros(2, count, 5),
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def test_port_output_residual_is_zero_initialized() -> None:
    torch.manual_seed(7)
    base = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    output = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        port_consensus_output=True,
    )
    output.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    output_context = output.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)

    base_prediction = base(target, time, base_context).block_state
    output_prediction = output(target, time, output_context).block_state

    torch.testing.assert_close(output_prediction, base_prediction)
    assert output.port_consensus is not None


def test_port_output_residual_changes_x0_and_receives_gradient() -> None:
    torch.manual_seed(11)
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        port_consensus_output=True,
    )
    assert model.port_consensus is not None
    model.port_consensus.gate.data.fill_(1.0)
    torch.nn.init.normal_(model.port_consensus.reduce[-1].weight, std=0.02)
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.5)

    prediction = model(target, time, context).block_state
    base = model.backbone(
        target.block_state,
        time,
        context.denoiser_cond,
    )

    assert not torch.equal(prediction, base)
    prediction.square().mean().backward()
    gradient = model.port_consensus.reduce[-1].weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0.0
