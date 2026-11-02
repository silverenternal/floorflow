import torch

from floorflow.chip_flow.projections import (
    prepare_conditioned_projection,
    project_conditioned_latent,
)


def test_projection_uses_only_conditioned_target_rows() -> None:
    latent = torch.zeros(1, 3, 3)
    target = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [9.0, 9.0, 9.0]]]
    )
    constraints = torch.zeros(1, 3, 5)
    constraints[0, 0, 0] = 1
    constraints[0, 1, 1] = 1
    constraints[0, :2, 2] = 1
    projected = project_conditioned_latent(latent, target, constraints)
    assert torch.equal(projected[0, 0, :2], latent[0, 0, :2])
    assert torch.equal(projected[0, 1, :2], target[0, 1, :2])
    expected_rho = target[0, :2, 2].median()
    assert torch.equal(projected[0, :2, 2], expected_rho.expand(2))
    assert projected[0, 2, 2].item() == 0.0
    assert torch.equal(projected[0, 2, :2], latent[0, 2, :2])


def test_projection_vectorizes_multiple_mib_groups() -> None:
    torch.manual_seed(7)
    latent = torch.randn(2, 6, 3, requires_grad=True)
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 0.7],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.8],
            ],
            [
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.6],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 0.2],
            ],
        ]
    )
    constraints = torch.zeros(2, 6, 5)
    constraints[0, :3, 2] = 1
    constraints[0, 3:, 2] = 2
    constraints[0, 0, 0] = 1
    constraints[1, :2, 2] = 3
    constraints[1, 2:, 2] = 4
    constraints[1, 2, 1] = 1

    projected = project_conditioned_latent(latent, target, constraints)
    expected = latent.detach().clone()
    expected[0, :3, 2] = target[0, 0, 2]
    expected[0, 3:, 2] = torch.median(latent.detach()[0, 3:, 2])
    expected[1, :2, 2] = torch.median(latent.detach()[1, :2, 2])
    expected[1, 2:, 2] = target[1, 2, 2]
    expected[1, 2, :2] = target[1, 2, :2]
    assert torch.equal(projected, expected)

    projected[:, :, 2].sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert torch.equal(latent.grad[..., :2], torch.zeros_like(latent.grad[..., :2]))


def test_projection_without_mib_groups_is_identity_for_free_rows() -> None:
    latent = torch.randn(3, 4, 3, requires_grad=True)
    projected = project_conditioned_latent(
        latent,
        torch.zeros_like(latent),
        torch.zeros(3, 4, 5),
    )
    assert torch.equal(projected, latent)
    projected.sum().backward()
    assert latent.grad is not None
    assert torch.equal(latent.grad, torch.ones_like(latent.grad))


def test_mib_median_gradient_matches_torch_median_for_ties() -> None:
    latent = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(latent)
    constraints = torch.zeros(1, 4, 5)
    constraints[0, :, 2] = 1
    projected = project_conditioned_latent(latent, target, constraints)
    projected[..., 2].sum().backward()

    reference = latent.detach().clone().requires_grad_(True)
    reference[..., 2].median().backward(torch.tensor(4.0))
    assert torch.equal(latent.grad, reference.grad)


def test_reused_projection_plan_matches_direct_projection() -> None:
    latent = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 3)
    constraints = torch.zeros(2, 5, 5)
    constraints[:, 0, 0] = 1
    constraints[:, 1, 1] = 1
    constraints[:, :3, 2] = 1
    constraints[:, 3:, 2] = 2

    plan = prepare_conditioned_projection(target, constraints)

    assert torch.equal(plan.apply(latent), project_conditioned_latent(latent, target, constraints))
