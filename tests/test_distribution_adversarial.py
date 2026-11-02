from __future__ import annotations

import torch

from floorflow.distribution.adversarial import (
    DistributionAdversarialTrainer,
    DynamicFeatureMaskDiscriminator,
    hinge_discriminator_loss,
    hinge_generator_loss,
)


def test_dynamic_discriminator_shapes_masks_and_drift() -> None:
    discriminator = DynamicFeatureMaskDiscriminator(
        5,
        hidden_channels=16,
        mask_ratio=0.5,
    )
    real = torch.rand(2, 5, 16, 16)
    footprint = torch.ones(2, 1, 16, 16)
    real_logits, first = discriminator(real, footprint, update_ema=True)
    fake_logits, second = discriminator(real + 0.5, footprint)
    assert real_logits.shape == fake_logits.shape == (2,)
    assert first["mask_fraction"] == second["mask_fraction"] == 0.5
    assert float(second["feature_drift"]) > 0.0
    assert torch.isfinite(real_logits).all()


def test_hinge_losses_have_expected_gradient_direction() -> None:
    real = torch.tensor([1.0, 2.0])
    fake = torch.tensor([-1.0, -2.0])
    loss, metrics = hinge_discriminator_loss(real, fake)
    assert float(loss) == 0.0
    assert float(metrics["real_accuracy"]) == 1.0
    fake_parameter = torch.tensor([0.0], requires_grad=True)
    generator_loss, _ = hinge_generator_loss(fake_parameter)
    generator_loss.backward()
    assert fake_parameter.grad is not None


def test_adversarial_trainer_updates_both_sides() -> None:
    discriminator = DynamicFeatureMaskDiscriminator(
        3,
        hidden_channels=8,
        mask_ratio=0.75,
    )
    trainer = DistributionAdversarialTrainer(
        discriminator,
        lr=1.0e-3,
        generator_weight=0.1,
        r1_weight=0.01,
    )
    generated = torch.softmax(torch.randn(2, 3, 8, 8, requires_grad=True), dim=1)
    real = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)
    footprint = torch.ones(2, 1, 8, 8)
    generator_loss, _ = trainer.generator_loss(generated, footprint)
    assert generator_loss.requires_grad
    metrics = trainer.discriminator_step(real, generated, footprint)
    assert torch.isfinite(metrics["discriminator_loss"])
    assert torch.isfinite(metrics["r1_penalty"])
