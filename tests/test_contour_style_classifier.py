import torch

from floorflow.heads.contour_style_classifier import ContourStyleClassifier


def test_contour_style_classifier_shape() -> None:
    model = ContourStyleClassifier(num_styles=5, hidden_channels=8)
    logits = model(torch.ones(2, 64, 64))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
