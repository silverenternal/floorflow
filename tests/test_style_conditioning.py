import pytest
import torch

from floorflow.distribution.style_conditioning import (
    LowRankStyleAdapter,
    StyleElementResidual,
    StylePixelResidual,
    model_style_contrast_loss,
    style_orthogonality_loss,
    style_regularization,
)


def test_style_adapter_bypasses_when_no_style_is_supplied() -> None:
    adapter = LowRankStyleAdapter(8, num_styles=3, rank=2)
    features = torch.randn(2, 8, 4, 4)
    output = adapter(features, None)
    assert torch.equal(output, features)


def test_style_adapter_is_style_specific_after_training() -> None:
    adapter = LowRankStyleAdapter(4, num_styles=2, rank=1)
    with torch.no_grad():
        adapter.embedding.weight.copy_(torch.tensor([[1.0], [2.0]]))
        adapter.down.weight.fill_(1.0)
        adapter.up.weight.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(4, 1, 1, 1)
        )
    features = torch.ones(2, 4, 2, 2)
    output = adapter(features, torch.tensor([0, 1]))
    assert output[0, 0].mean() != output[1, 0].mean()
    assert not torch.equal(output, features)


def test_style_id_validation_and_regularization() -> None:
    adapter = LowRankStyleAdapter(4, num_styles=2, rank=1)
    with pytest.raises(ValueError, match="outside"):
        adapter(torch.randn(1, 4, 2, 2), torch.tensor([2]))
    penalty = style_regularization(adapter, torch.tensor([0, 1]))
    assert penalty.ndim == 0
    assert penalty >= 0


def test_style_contrast_loss_penalizes_identical_style_outputs() -> None:
    adapter = LowRankStyleAdapter(4, num_styles=2, rank=1)
    model = type(
        "Model",
        (),
        {
            "num_styles": 2,
            "hub": type(
                "Hub",
                (),
                {
                    "pixel": type("Head", (), {"style_adapter": adapter})(),
                    "element": type("Head", (), {"style_adapter": None})(),
                },
            )(),
            "velocity_style_adapter": None,
        },
    )()
    loss = model_style_contrast_loss(
        model,
        torch.ones(2, 4, 2, 2),
        torch.tensor([0, 1]),
        margin=1.0,
    )
    assert loss > 0


def test_pixel_style_residual_bypasses_and_changes_foreground_only() -> None:
    residual = StylePixelResidual(num_styles=2, num_foreground_classes=3)
    logits = torch.randn(2, 4, 5, 5)
    assert torch.equal(residual(logits, None), logits)
    with torch.no_grad():
        residual.class_bias.weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.2, 0.3],
                    [-0.4, 0.5, -0.6],
                ]
            )
        )
    output = residual(logits, torch.tensor([0, 1]))
    assert torch.equal(output[:, :1], logits[:, :1])
    assert not torch.equal(output[:, 1:], logits[:, 1:])
    assert not torch.equal(output[0, 1:], output[1, 1:])


def test_element_style_residual_controls_tokens_presence_and_types() -> None:
    residual = StyleElementResidual(
        num_styles=2,
        hidden_channels=4,
        group_class_counts={"room": 3, "furniture": 2},
        queries_per_group={"room": 2, "furniture": 1},
    )
    tokens = torch.randn(2, 2, 4)
    assert torch.equal(residual.augment_tokens(tokens, None), tokens)
    presence = torch.randn(2, 2)
    types = torch.randn(2, 2, 3)
    same_presence, same_types = residual.adjust_logits(
        "room",
        presence,
        types,
        None,
    )
    assert same_presence is presence
    assert same_types is types
    with torch.no_grad():
        residual.token_bias.weight.fill_(0.1)
        residual.presence_bias.weight.fill_(0.2)
        residual.type_bias.weight.fill_(0.3)
    styled = residual.augment_tokens(tokens, torch.tensor([0, 1]))
    assert not torch.equal(styled, tokens)
    new_presence, new_types = residual.adjust_logits(
        "room",
        presence,
        types,
        torch.tensor([0, 1]),
    )
    assert torch.allclose(new_presence - presence, torch.full_like(presence, 0.2))
    assert torch.allclose(new_types - types, torch.full_like(types, 0.3))


def test_style_orthogonality_penalizes_collapsed_directions() -> None:
    pixel = StylePixelResidual(num_styles=2, num_foreground_classes=3)
    model = torch.nn.Module()
    model.pixel = pixel
    with torch.no_grad():
        pixel.class_bias.weight.fill_(1.0)
    collapsed = style_orthogonality_loss(model)
    with torch.no_grad():
        pixel.class_bias.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    orthogonal = style_orthogonality_loss(model)
    assert collapsed > orthogonal
    assert orthogonal < 1.0e-10


def test_style_orthogonality_has_finite_gradient_at_zero_initialization() -> None:
    pixel = StylePixelResidual(num_styles=3, num_foreground_classes=4)
    model = torch.nn.Module()
    model.pixel = pixel
    loss = style_orthogonality_loss(model)
    loss.backward()
    gradient = pixel.class_bias.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
