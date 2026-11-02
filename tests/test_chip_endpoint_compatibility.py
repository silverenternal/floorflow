from __future__ import annotations

from argparse import Namespace

import torch

from floorflow.chip_flow.endpoint_compatibility import (
    endpoint_compatibility_terms,
    normalized_latent_displacement,
)
from floorflow.chip_flow.training_endpoint import (
    apply_training_endpoint_contracts,
)


def _batch() -> dict:
    count = 4
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": torch.zeros(2, count, 5),
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 3.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4]],
                [[0.3, 0.5]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def test_endpoint_compatibility_is_finite_and_differentiable() -> None:
    torch.manual_seed(29)
    latent = torch.randn(2, 4, 3, requires_grad=True)
    terms = endpoint_compatibility_terms(
        latent,
        _batch(),
        iterations=2,
        displacement_weight=1.0,
        gain_weight=0.25,
    )
    assert torch.isfinite(terms.total)
    assert terms.total >= 0
    assert terms.displacement >= 0
    assert terms.projection_gain >= 0
    assert terms.refiner_energy >= 0
    assert terms.refiner_trajectory_energy >= 0
    assert terms.refiner_trajectory_displacement >= 0
    terms.total.backward()
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0


def test_latent_displacement_is_not_scaled_twice() -> None:
    latent = torch.tensor([[[0.25, 0.5, -0.1]]])
    projected = torch.tensor([[[0.30, 0.45, -0.1]]])

    displacement = normalized_latent_displacement(latent, projected)

    torch.testing.assert_close(
        displacement,
        torch.tensor([[[0.05, -0.05]]]),
    )


def test_zero_projection_iterations_still_returns_compatibility_terms() -> None:
    latent = torch.randn(2, 4, 3)
    terms = endpoint_compatibility_terms(latent, _batch(), iterations=0)
    assert terms.displacement == 0
    assert terms.projection_gain == 0
    assert terms.refiner_energy >= 0
    assert torch.isfinite(terms.total)


def test_endpoint_includes_legalized_area_p2b_and_response_targets() -> None:
    torch.manual_seed(30)
    batch = _batch()
    batch["legalizer_displacement"] = torch.randn(2, 4, 2) * 0.01
    batch["legalizer_mask"] = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    latent = torch.zeros(2, 4, 3, requires_grad=True)
    terms = endpoint_compatibility_terms(
        latent,
        batch,
        iterations=1,
        legalized_area_weight=0.1,
        legalized_p2b_weight=0.1,
        legalizer_target_weight=0.1,
        shape_mib_weight=0.1,
        shape_pin_tail_weight=0.1,
        shape_rho_trust_weight=0.1,
        shape_rho_trust_radius=0.0,
    )
    assert terms.legalized_area >= 0
    assert terms.legalized_p2b >= 0
    assert terms.legalizer_target >= 0
    assert terms.shape_pin >= 0
    assert terms.mib_shared >= 0
    assert terms.pin_tail >= 0
    assert terms.rho_trust >= 0
    assert torch.isfinite(terms.total)
    legalizer_gradient, = torch.autograd.grad(
        terms.legalizer_target,
        latent,
        retain_graph=True,
    )
    assert legalizer_gradient.abs().sum() > 0
    terms.total.backward()
    assert latent.grad is not None


def test_projected_shape_pin_terms_reach_the_raw_endpoint() -> None:
    torch.manual_seed(32)
    batch = _batch()
    batch["constraints"][:, :2, 2] = 1.0
    latent = torch.randn(2, 4, 3, requires_grad=True)
    terms = endpoint_compatibility_terms(
        latent,
        batch,
        iterations=2,
        displacement_weight=0.0,
        gain_weight=0.0,
        refiner_energy_weight=0.0,
        refiner_trajectory_weight=0.0,
        shape_mib_weight=1.0,
        shape_pin_tail_weight=0.0,
        shape_rho_trust_weight=0.0,
    )
    gradient, = torch.autograd.grad(terms.mib_shared, latent)

    assert gradient.abs().sum() > 0


def test_second_order_endpoint_path_runs_in_fp32_under_bf16_autocast() -> None:
    torch.manual_seed(31)
    raw = torch.randn(2, 4, 3, dtype=torch.bfloat16, requires_grad=True)
    args = Namespace(
        training_projector=True,
        training_projector_iterations=1,
        training_projector_strength=1.0,
        training_projector_displacement_weight=0.1,
        projector_sparse_pairs=True,
        projector_strategy="conflict_aware",
        projector_grid_size=16,
        endpoint_compatibility_weight=0.05,
        endpoint_compatibility_delay_steps=0,
        endpoint_compatibility_anneal_steps=0,
        endpoint_compatibility_iterations=1,
        endpoint_compatibility_displacement_weight=1.0,
        endpoint_compatibility_gain_weight=0.25,
        endpoint_refiner_energy_weight=1.0,
        endpoint_refiner_trajectory_steps=1,
        endpoint_refiner_trajectory_lr=1.0e-3,
        endpoint_refiner_trajectory_weight=1.0,
        projector_aware_endpoint=False,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        latent, loss, terms = apply_training_endpoint_contracts(
            raw,
            _batch(),
            args,
            step=0,
            device=torch.device("cpu"),
        )
    assert latent.dtype == torch.float32
    assert torch.isfinite(loss)
    assert "endpoint_refiner_trajectory_energy" in terms
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
