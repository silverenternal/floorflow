import torch

from floorflow.chip_flow.model import ChipBlockFlowModel, block_features, decode_boxes, encode_latent
from floorflow.chip_physics import (
    boundary_penalty,
    chip_physics_loss,
    overlap_penalty,
)


def _boxes():
    return torch.tensor(
        [[0.0, 0.0, 4.0, 2.0], [4.0, 0.0, 4.0, 2.0], [0.0, 2.0, 8.0, 2.0]]
    )


def test_overlap_is_zero_for_touching_boxes() -> None:
    boxes = _boxes()
    areas = boxes[:, 2] * boxes[:, 3]
    assert float(overlap_penalty(boxes, areas)) == 0.0
    shifted = boxes.clone()
    shifted[1, 0] -= 1.0
    assert float(overlap_penalty(shifted, areas)) > 0.0


def test_boundary_penalty_is_zero_on_required_edges() -> None:
    boxes = _boxes()
    areas = boxes[:, 2] * boxes[:, 3]
    codes = torch.tensor([1.0, 2.0, 4.0])
    assert float(boundary_penalty(boxes, codes, areas)) < 1.0e-6


def test_latent_round_trip_is_exact() -> None:
    boxes = _boxes()
    areas = boxes[:, 2] * boxes[:, 3]
    latent = encode_latent(boxes, areas)
    recovered = decode_boxes(latent, areas)
    assert torch.allclose(recovered, boxes, atol=1.0e-5)


def test_chip_physics_loss_reports_all_terms() -> None:
    boxes = _boxes()
    areas = boxes[:, 2] * boxes[:, 3]
    constraints = torch.zeros(3, 5)
    constraints[:, 4] = torch.tensor([1.0, 2.0, 8.0])
    edges = torch.zeros(0, 3)
    pins = torch.zeros(0, 2)
    terms = chip_physics_loss(boxes, areas, constraints, edges, edges, pins)
    assert set(terms) == {"overlap", "group", "mib", "boundary", "wl", "area", "total"}
    assert all(torch.isfinite(value) for value in terms.values())


def test_chip_flow_model_forward_shapes() -> None:
    boxes = _boxes()
    areas = boxes[:, 2] * boxes[:, 3]
    constraints = torch.zeros(3, 5)
    features = block_features(areas, constraints, torch.zeros(0, 2), torch.zeros(0, 3))
    model = ChipBlockFlowModel(top_k=2)
    latent = encode_latent(boxes, areas)
    velocity = model(latent[None], torch.tensor([0.5]), features[None])
    assert velocity.shape == latent[None].shape
