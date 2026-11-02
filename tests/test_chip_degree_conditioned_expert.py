from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.degree_conditioned_expert import DegreeConditionedExpert
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)


def _batch() -> dict:
    count = 8
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 3] = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    constraints[:, :2, 4] = 1
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(count) * 1.2,
                        torch.zeros(count),
                        torch.ones(count),
                        torch.ones(count),
                    ],
                    dim=-1,
                )
            ]
            * 2
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0], [4.0, 6.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0], [2.0, 6.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.5, 0.5]],
                [[0.1, 0.5], [0.7, 0.9], [0.4, 0.3]],
            ]
        ),
        "n_b2b": torch.tensor([3, 3]),
        "n_p2b": torch.tensor([3, 3]),
        "n_pins": torch.tensor([3, 3]),
    }


def _model(degree_expert: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        degree_expert_conditioning=degree_expert,
    )


def test_degree_expert_is_zero_gated_and_trainable() -> None:
    torch.manual_seed(3)
    module = DegreeConditionedExpert(64)
    tokens = torch.randn(2, 8, 64)
    arguments = {
        "b2b": torch.tensor(
            [[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2
        ),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 3.0, 0.5]]] * 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
    }
    output, diagnostics = module(tokens, **arguments)
    assert torch.equal(output, tokens)
    assert diagnostics["degree_expert_gate"].shape == (2, 8)
    module.scale.data.fill_(1.0)
    active, _ = module(tokens, **arguments)
    assert not torch.allclose(active, tokens)
    active.square().mean().backward()
    assert module.expert[-1].weight.grad is not None
    assert module.expert[-1].weight.grad.abs().sum() > 0


def test_degree_expert_preserves_block_permutation_equivariance() -> None:
    torch.manual_seed(5)
    module = DegreeConditionedExpert(64)
    module.scale.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 3.0, 0.5]]])
    counts = torch.tensor([2])
    original, _ = module(
        tokens,
        b2b=b2b,
        p2b=p2b,
        n_b2b=counts,
        n_p2b=counts,
    )
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    inverse = torch.argsort(permutation)
    reordered, _ = module(
        tokens[:, permutation],
        b2b=torch.cat(
            [inverse[b2b[..., :2].long()], b2b[..., 2:]], dim=-1
        ),
        p2b=torch.cat(
            [
                p2b[..., :1],
                inverse[p2b[..., 1:2].long()],
                p2b[..., 2:],
            ],
            dim=-1,
        ),
        n_b2b=counts,
        n_p2b=counts,
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_degree_expert_zero_gate_preserves_base_prediction() -> None:
    torch.manual_seed(7)
    base = _model(False)
    torch.manual_seed(7)
    expert = _model(True)
    expert.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    expert_context = expert.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    torch.testing.assert_close(
        base(target, time, base_context).block_state,
        expert(target, time, expert_context).block_state,
    )
    assert expert.typed.degree_expert is not None


def test_degree_expert_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "degree_expert_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.typed.degree_expert is not None


def test_degree_expert_parameter_budget_is_under_two_percent() -> None:
    module = DegreeConditionedExpert(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.02
