from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.trinity_latent import room_batch_from_labels
from floorflow.training.trinity_flow import (
    clamp_room_latent,
    match_room_tokens,
    make_trinity_flow,
    prepare_room_flow,
    room_flow_objective,
)


def _batch():
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    return room_batch_from_labels(
        labels,
        taxonomy,
        torch.ones(1, 1, 32, 32),
        max_rooms=4,
    )


def test_rectified_flow_and_logit_normal_are_reproducible() -> None:
    batch = _batch()
    framing, time_sampler = make_trinity_flow()
    torch.manual_seed(7)
    first = prepare_room_flow(
        batch.z,
        batch.token_mask,
        framing=framing,
        time_sampler=time_sampler,
    )
    torch.manual_seed(7)
    second = prepare_room_flow(
        batch.z,
        batch.token_mask,
        framing=framing,
        time_sampler=time_sampler,
    )
    assert torch.allclose(first.x_t, second.x_t)
    assert torch.allclose(first.target, second.target)
    assert torch.equal(batch.token_mask[0], torch.tensor([True, True, False, False]))


def test_room_flow_objective_has_finite_gradients() -> None:
    batch = _batch()
    framing, time_sampler = make_trinity_flow()
    prediction = (batch.z + 0.01 * torch.randn_like(batch.z)).requires_grad_(True)
    loss, components = room_flow_objective(
        prediction,
        batch.z,
        batch,
        framing=framing,
        time_sampler=time_sampler,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert components["flow"] >= 0
    assert components["physics"] >= 0


def test_room_latent_clamp_rejects_exploding_values() -> None:
    batch = _batch()
    z = torch.tensor(
        [[[100.0, -100.0, 1.0e20]] * 2 + [[float("nan"), 0.0, 0.0]] * 2]
    )
    clamped = clamp_room_latent(z, batch.token_mask)
    assert torch.isfinite(clamped).all()
    assert clamped[0, 0, 0].item() == 3.0
    assert clamped[0, 0, 1].item() == -2.0
    assert not clamped[0, 2:].any()


def test_room_token_matching_is_permutation_invariant() -> None:
    batch = _batch()
    class_logits = torch.zeros(1, 4, 24)
    class_logits[0, 0, batch.room_classes[0, 0]] = 10.0
    class_logits[0, 1, batch.room_classes[0, 1]] = 10.0
    log_area = torch.log1p(batch.area_targets)
    forward = match_room_tokens(batch.z, class_logits, log_area, batch)
    permuted_z = batch.z[:, [1, 0, 2, 3]]
    permuted_classes = class_logits[:, [1, 0, 2, 3]]
    permuted_area = log_area[:, [1, 0, 2, 3]]
    backward = match_room_tokens(
        permuted_z,
        permuted_classes,
        permuted_area,
        batch,
    )
    assert sorted(forward["target_for_prediction"][0].tolist()) == sorted(
        backward["target_for_prediction"][0].tolist()
    )
