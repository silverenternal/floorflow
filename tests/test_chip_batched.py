import torch

from floorflow.chip_flow.batched import batch_physics_terms, flow_loss
from floorflow.chip_flow.model import (
    decode_boxes_batch,
    encode_latent,
    encode_latents,
)
from floorflow.chip_physics import chip_physics_loss


def _batch():
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 2.0], [4.0, 0.0, 4.0, 2.0], [0.0, 2.0, 8.0, 2.0]],
            [[0.0, 0.0, 3.0, 2.0], [3.0, 0.0, 3.0, 2.0], [0.0, 2.0, 6.0, 2.0]],
        ]
    )
    areas = boxes[:, :, 2] * boxes[:, :, 3]
    constraints = torch.zeros(2, 3, 5)
    constraints[:, :, 2] = torch.tensor([0.0, 1.0, 1.0])
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 2.0, 4.0], [0.0, 8.0, 5.0]]
    )
    b2b = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
            [[0.0, 2.0, 0.25], [0.0, 0.0, 0.0]],
        ]
    )
    p2b = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
            [[0.0, 2.0, 0.75], [0.0, 0.0, 0.0]],
        ]
    )
    pins = torch.tensor(
        [
            [[8.0, 1.0], [0.0, 2.0]],
            [[6.0, 1.0], [0.0, 0.0]],
        ]
    )
    return (
        boxes,
        areas,
        constraints,
        b2b,
        p2b,
        pins,
        torch.tensor([2, 1]),
        torch.tensor([2, 1]),
        torch.tensor([2, 1]),
    )


def test_batched_physics_matches_reference_terms() -> None:
    batch = _batch()
    terms = batch_physics_terms(*batch)
    for index in range(2):
        reference = chip_physics_loss(
            batch[0][index],
            batch[1][index],
            batch[2][index],
            batch[3][index, : int(batch[6][index])],
            batch[4][index, : int(batch[7][index])],
            batch[5][index, : int(batch[8][index])],
        )
        for name in ("overlap", "mib", "boundary", "wl", "area", "total"):
            assert torch.allclose(terms[name][index], reference[name])


def test_flow_loss_returns_clean_layout_estimate() -> None:
    source = torch.zeros(2, 3, 3)
    target = torch.ones(2, 3, 3)
    time = torch.full((2,), 0.25)
    interpolant = (1.0 - time[:, None, None]) * source + time[:, None, None] * target
    loss, estimate = flow_loss(target - source, interpolant, source, target, time)
    assert float(loss) == 0.0
    assert torch.allclose(estimate, target)
    assert encode_latent(torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([4.0])).shape == (
        1,
        3,
    )


def test_huber_flow_loss_is_finite_and_less_than_quadratic_at_outlier() -> None:
    source = torch.zeros(1, 2, 3)
    target = torch.ones(1, 2, 3)
    time = torch.full((1,), 0.5)
    interpolant = torch.full((1, 2, 3), 100.0)
    velocity = torch.zeros_like(target)
    quadratic, _ = flow_loss(velocity, interpolant, source, target, time)
    huber, _ = flow_loss(
        velocity,
        interpolant,
        source,
        target,
        time,
        huber_beta=0.1,
    )
    assert torch.isfinite(huber)
    assert float(huber) < float(quadratic)


def test_batched_latent_round_trip() -> None:
    boxes, areas = _batch()[:2]
    latent = encode_latents(boxes, areas)
    assert latent.shape == (2, 3, 3)
    assert torch.allclose(decode_boxes_batch(latent, areas), boxes, atol=1.0e-5)


def test_mib_penalty_requires_shared_width_and_height() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 2.0], [3.0, 0.0, 2.0, 4.0]]]
    )
    areas = boxes[..., 2] * boxes[..., 3]
    constraints = torch.zeros(1, 2, 5)
    constraints[..., 2] = 1.0
    terms = batch_physics_terms(
        boxes,
        areas,
        constraints,
        torch.empty(1, 0, 3),
        torch.empty(1, 0, 3),
        torch.empty(1, 0, 2),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )

    assert terms["mib"][0] > 0


def test_token_assignment_is_permutation_invariant() -> None:
    from floorflow.chip_flow.batched import matched_endpoint_loss

    predicted = torch.tensor([[[0.0], [1.0], [2.0]]])
    target = torch.tensor([[[2.0], [0.0], [1.0]]])
    assert float(matched_endpoint_loss(predicted, target)) == 0.0
