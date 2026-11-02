import torch

from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.style_semantic import (
    distribution_style_vector,
    style_prototype_loss,
)


def test_distribution_style_vector_has_expected_dimensions() -> None:
    hierarchy = ChannelHierarchy(
        num_classes=7,
        wall_ids=(1,),
        opening_ids=(2,),
        room_ids=(3, 4),
        furniture_ids=(5,),
        support_ids=(6,),
    )
    probability = torch.softmax(
        torch.randn(2, 7, 8, 8),
        dim=1,
    ).requires_grad_()
    presence_logits = torch.randn(2, 4, requires_grad=True)
    mask_logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    elements = {
        "room": {
            "presence_logits": presence_logits,
            "mask_logits": mask_logits,
        }
    }
    vectors = distribution_style_vector(
        {"probability": probability},
        elements,
        hierarchy,
        torch.ones(2, 1, 8, 8),
    )
    assert vectors.shape == (2, 10)
    prototypes = {
        "center": torch.zeros(10),
        "scale": torch.ones(10),
        "centroids": torch.randn(3, 10),
    }
    loss = style_prototype_loss(vectors, torch.tensor([0, 2]), prototypes)
    assert torch.isfinite(loss)
    loss.backward()
    assert probability.grad is not None
    assert presence_logits.grad is not None
