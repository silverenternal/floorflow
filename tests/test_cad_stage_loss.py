from collections import Counter

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import build_taxonomy
from floorflow.objectives.cad_stage_loss import cad_stage_loss


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_cad_stage_loss_has_independent_family_gradients() -> None:
    taxonomy = _taxonomy()
    logits = torch.randn(
        1,
        taxonomy.num_classes,
        16,
        16,
        requires_grad=True,
    )
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    footprint = torch.ones(1, 16, 16)
    wall = taxonomy.class_id("wall")
    target[:, 7:9, :] = wall
    target[:, 2:5, 12:14] = taxonomy.opening_ids[0]
    target[:, 10:14, 3:7] = taxonomy.furniture_ids[0]
    target[:, :7, :] = taxonomy.room_ids[0]
    loss, components = cad_stage_loss(logits, target, footprint, taxonomy)
    assert torch.isfinite(loss)
    assert components["stage_wall_loss"] >= 0
    assert components["stage_opening_loss"] >= 0
    assert components["stage_furniture_loss"] >= 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_cad_stage_loss_exposes_atomic_balancing_terms() -> None:
    taxonomy = _taxonomy()
    logits = torch.randn(
        1,
        taxonomy.num_classes,
        16,
        16,
        requires_grad=True,
    )
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    footprint = torch.ones(1, 16, 16)
    target[:, 7:9, :] = taxonomy.class_id("wall")
    target[:, 2:5, 12:14] = taxonomy.opening_ids[0]
    target[:, 10:14, 3:7] = taxonomy.furniture_ids[0]
    _loss, _, terms = cad_stage_loss(
        logits,
        target,
        footprint,
        taxonomy,
        return_loss_terms=True,
    )
    assert set(terms) == {
        "stage_wall",
        "stage_wall_precision",
        "stage_opening",
        "stage_opening_precision",
        "stage_opening_host",
        "stage_opening_type",
        "stage_furniture",
    }
    torch.stack(list(terms.values())).sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_opening_type_loss_averages_cases_before_batch() -> None:
    taxonomy = _taxonomy()
    logits = torch.zeros(
        2,
        taxonomy.num_classes,
        4,
        4,
    )
    opening = taxonomy.opening_ids[0]
    logits[:, opening, :, 0] = 4.0
    logits[0, opening, 1, :] = 4.0
    logits[1, opening, 1, 0] = -3.0
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    target[0, :, 0] = opening
    target[0, 1, :] = opening
    target[1, :, 0] = opening
    target[1, 1, 0] = opening
    footprint = torch.ones(2, 4, 4)
    _, components = cad_stage_loss(
        logits,
        target,
        footprint,
        taxonomy,
    )
    type_target = torch.zeros_like(target)
    opening_logits = logits[:, taxonomy.opening_ids].permute(0, 2, 3, 1)
    losses = [
        F.cross_entropy(
            opening_logits[index][target[index] == opening],
            type_target[index][target[index] == opening],
        )
        for index in range(2)
    ]
    assert torch.allclose(
        components["stage_opening_type_loss"],
        torch.stack(losses).mean(),
    )
