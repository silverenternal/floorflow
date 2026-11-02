import torch

from floorflow.heads.owner_semantic_reassignment import (
    OwnerSemanticReassignment,
)


def test_owner_semantic_reassignment_uses_dense_class_evidence() -> None:
    module = OwnerSemanticReassignment(support_threshold=0.5)
    dense = torch.full((1, 2, 8, 8), -8.0)
    dense[:, 0, :, :4] = 8.0
    dense[:, 1, :, 4:] = 8.0
    masks = torch.full((1, 2, 8, 8), -8.0)
    masks[:, 0, :, :4] = 8.0
    masks[:, 1, :, 4:] = 8.0
    footprint = torch.ones(1, 1, 8, 8)
    output, diagnostics = module(dense, masks, footprint)
    assert output.shape == dense.shape
    assert output[:, 0, :, :4].mean() > output[:, 1, :, :4].mean()
    assert output[:, 1, :, 4:].mean() > output[:, 0, :, 4:].mean()
    assert diagnostics["support_fraction"] > 0.99


def test_owner_semantic_reassignment_preserves_dense_outside_owner_support() -> None:
    module = OwnerSemanticReassignment(support_threshold=0.9)
    dense = torch.randn(1, 3, 8, 8, requires_grad=True)
    masks = torch.full((1, 1, 8, 8), -8.0)
    masks[:, :, :2] = 8.0
    footprint = torch.ones(1, 1, 8, 8)
    output, _diagnostics = module(dense, masks, footprint)
    dense_probability = dense.softmax(dim=1)
    output_probability = output.exp()
    assert torch.allclose(
        output_probability[:, :, 4:],
        dense_probability[:, :, 4:],
        atol=1.0e-5,
    )
    output.square().mean().backward()
    assert dense.grad is not None
    assert dense.grad.abs().sum() > 0
