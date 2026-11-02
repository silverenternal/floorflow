import torch

from floorflow.distribution.query_budget import allocate_element_query_budgets
from floorflow.heads.dynamic_object_instance import render_object_queries_soft
from floorflow.objectives.furniture_vector_refine import (
    furniture_vector_loss,
    soft_object_render,
)


def test_soft_render_is_differentiable_and_normalized() -> None:
    class_logits = torch.randn(2, 3, 5, requires_grad=True)
    mask_logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    footprint = torch.ones(2, 1, 8, 8)
    rendered = soft_object_render(class_logits, mask_logits, footprint)
    assert rendered.shape == (2, 5, 8, 8)
    rendered.sum().backward()
    assert class_logits.grad is not None
    assert mask_logits.grad is not None


def test_furniture_vector_loss_prefers_correct_class() -> None:
    footprint = torch.ones(1, 1, 8, 8)
    mask_logits = torch.full((1, 2, 8, 8), 8.0)
    mask_logits[0, 1] = -8.0
    class_ids = (10, 20)
    correct = torch.full((1, 8, 8), 10, dtype=torch.long)
    wrong = torch.full((1, 8, 8), 20, dtype=torch.long)
    good_logits = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]])
    good, _ = furniture_vector_loss(
        good_logits, mask_logits, correct, class_ids, footprint
    )
    bad, _ = furniture_vector_loss(
        good_logits, mask_logits, wrong, class_ids, footprint
    )
    assert bad > good


def test_render_object_queries_soft_matches_soft_render() -> None:
    class_logits = torch.randn(1, 2, 4)
    mask_logits = torch.randn(1, 2, 6, 6)
    footprint = torch.ones(1, 1, 6, 6)
    direct = soft_object_render(class_logits, mask_logits, footprint)
    wrapped = render_object_queries_soft(class_logits, mask_logits, footprint)
    assert torch.allclose(direct, wrapped)


def test_soft_object_render_respects_active_query_budget() -> None:
    class_logits = torch.full((1, 2, 4), -5.0)
    class_logits[:, :, 0] = 5.0
    mask_logits = torch.full((1, 2, 8, 8), -100.0)
    mask_logits[0, 0, 1:3, 1:3] = 8.0
    mask_logits[0, 1, 5:7, 5:7] = 8.0
    rendered = soft_object_render(
        class_logits,
        mask_logits,
        torch.ones(1, 1, 8, 8),
        active_queries=torch.tensor([[True, False]]),
    )
    assert rendered[0, 1, 5:7, 5:7].eq(0).all()
    assert rendered[0, 1, 1:3, 1:3].gt(0).all()


def test_rare_class_recycling_raises_budget_when_concentrated() -> None:
    reference = torch.zeros(2, 1, 8, 8)
    gates = {
        "room": torch.full((2, 1, 8, 8), 0.5),
        "furniture": torch.full((2, 1, 8, 8), 0.2),
    }
    balanced = allocate_element_query_budgets(
        gates,
        reference,
        {"furniture": 24},
        class_density={
            "furniture": torch.full((2, 4), 0.25),
        },
    )
    concentrated = allocate_element_query_budgets(
        gates,
        reference,
        {"furniture": 24},
        class_density={
            "furniture": torch.tensor(
                [[0.9, 0.05, 0.03, 0.02], [0.9, 0.05, 0.03, 0.02]]
            ),
        },
    )
    assert concentrated["furniture"][0] > balanced["furniture"][0]
