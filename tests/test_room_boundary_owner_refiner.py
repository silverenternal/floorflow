import torch

from floorflow.heads.room_boundary_owner_refiner import (
    RoomBoundaryOwnerRefiner,
)
from floorflow.objectives.room_boundary_loss import (
    room_boundary_loss,
    room_boundary_targets,
)


def test_owner_boundary_refiner_is_identity_at_initialization() -> None:
    refiner = RoomBoundaryOwnerRefiner(8, 3, hidden_channels=16)
    features = torch.randn(1, 8, 16, 16)
    masks = torch.randn(1, 3, 16, 16)
    refined, boundary, diagnostics = refiner(features, masks)
    assert torch.equal(refined, masks)
    assert torch.equal(boundary, torch.zeros_like(boundary))
    assert diagnostics["owner_residual_norm"] == 0


def test_owner_boundary_refiner_has_nonzero_gradients() -> None:
    refiner = RoomBoundaryOwnerRefiner(8, 3, hidden_channels=16)
    with torch.no_grad():
        refiner.owner_residual.bias.normal_()
        refiner.boundary.bias.normal_()
    features = torch.randn(1, 8, 16, 16)
    masks = torch.randn(1, 3, 16, 16)
    refined, boundary, _diagnostics = refiner(features, masks)
    (refined.square().mean() + boundary.square().mean()).backward()
    assert refiner.owner_residual.weight.grad is not None
    assert refiner.owner_residual.weight.grad.abs().sum() > 0
    assert refiner.boundary.weight.grad is not None
    assert refiner.boundary.weight.grad.abs().sum() > 0


def test_room_boundary_loss_localizes_instance_edges() -> None:
    masks = torch.zeros(1, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    boundary, union = room_boundary_targets(masks, valid)
    assert union.sum() == 16
    assert 0 < boundary.sum() < union.sum()
    logits = boundary * 10.0 - (1.0 - boundary) * 10.0
    loss, components = room_boundary_loss(logits, masks, valid)
    assert loss < 1.0e-3
    assert components["room_boundary_positive_rate"] > 0
