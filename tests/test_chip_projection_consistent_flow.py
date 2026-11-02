from __future__ import annotations

import torch

from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints,
)
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchy_source_state,
    integrate_hierarchy_step,
    sample_hierarchy_flow,
)


def _batch() -> dict:
    count = 7
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0]] * 2
    )
    constraints[:, :2, 2] = 1
    constraints[0, :2, 3] = 1
    constraints[0, 2:5, 3] = 2
    constraints[1, :3, 3] = 1
    constraints[1, 3:6, 3] = 3
    boxes = torch.stack(
        [
            torch.stack(
                [
                    torch.arange(count) * 1.2,
                    torch.zeros(count),
                    torch.ones(count),
                    torch.ones(count),
                ],
                dim=-1,
            )
        ]
        * 2
    )
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.0, 0.0]],
                [[0.1, 0.5], [0.7, 0.9], [0.0, 0.0]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        projector=True,
        projector_iterations=2,
    ).eval()


def test_projection_consistent_step_reinterpolates_from_source() -> None:
    torch.manual_seed(37)
    model = _model()
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 7, 3)
    context = model.prepare_context(batch)
    source = hierarchy_source_state(
        batch["source_latent"],
        context,
        torch.Generator().manual_seed(37),
    )
    with torch.no_grad():
        output, projection = integrate_hierarchy_step(
            model,
            source,
            torch.full((2,), 0.5),
            context,
            batch,
            step_size=0.25,
            projector=True,
            projector_iterations=2,
            projector_boundary_axes=False,
            projector_boundary_lock_corners=True,
            projection_mode="projection_consistent",
            source_state=source,
        )

    assert projection is not None
    expected = 0.25 * source.block_state + 0.75 * projection.latent
    torch.testing.assert_close(output.block_state, expected)
    assert projection.max_area_error < 1.0e-5


def test_projection_consistent_sampling_preserves_default_and_is_finite() -> None:
    torch.manual_seed(41)
    model = _model()
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 7, 3)
    default, _ = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(41),
        projector=True,
        projector_iterations=2,
    )
    explicit, _ = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(41),
        projector=True,
        projector_iterations=2,
        projection_mode="post_step",
    )
    consistent, _ = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(41),
        projector=True,
        projector_iterations=2,
        projection_mode="projection_consistent",
    )

    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    assert torch.isfinite(consistent).all()


def test_final_only_projection_matches_one_shot_postprojection() -> None:
    torch.manual_seed(43)
    model = _model()
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 7, 3)
    unprojected, _ = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(43),
        projector=False,
        projector_iterations=2,
    )
    final_only, _ = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(43),
        projector=True,
        projector_iterations=2,
        projection_mode="final_only",
    )
    expected = project_hard_constraints(
        unprojected,
        batch["target_latent"],
        batch["constraints"],
        batch["areas"],
        iterations=2,
    ).latent
    torch.testing.assert_close(final_only, expected, rtol=0, atol=0)


def test_projection_consistent_mode_requires_projector() -> None:
    model = _model()
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 7, 3)
    try:
        sample_hierarchy_flow(
            model,
            batch,
            sampling_steps=1,
            generator=torch.Generator().manual_seed(47),
            projector=False,
            projector_iterations=2,
            projection_mode="projection_consistent",
        )
    except ValueError as error:
        assert "requires projector=True" in str(error)
    else:
        raise AssertionError("projection-consistent mode must require a projector")
