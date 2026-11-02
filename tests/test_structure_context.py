import torch

from floorflow.heads.structure_context import (
    STRUCTURE_CONTEXT_NAMES,
    build_structure_context,
)


def test_structure_context_has_named_geometry_channels() -> None:
    room_probability = torch.tensor(
        [
            [
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.1, 0.9], [0.1, 0.9]],
            ]
        ]
    ).requires_grad_(True)
    structure = torch.zeros(1, 5, 2, 2)
    structure[:, 2] = 0.8
    structure[:, 4] = 1.0
    context, diagnostics = build_structure_context(
        room_probability,
        structure,
        room_count=2,
        opening_count=0,
    )
    assert context.shape == (1, len(STRUCTURE_CONTEXT_NAMES), 2, 2)
    assert context[:, 0].max() > 0.5
    assert diagnostics["wall_proximity"] > 0.5
    context.square().mean().backward()
    assert room_probability.grad is not None
