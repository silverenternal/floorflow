from __future__ import annotations

import torch

from floorflow.chip_flow.hard_constraint_projector import (
    _separate_overlaps,
    corrected_velocity,
    project_hard_constraints,
    project_hard_constraints_latent,
)
from floorflow.chip_flow.model import decode_boxes, encode_latents
from floorflow.chip_flow.projector_geometry import (
    candidate_overlap_pairs,
    joint_conflict_energy,
    project_boundary,
)


def _targets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    areas = torch.ones(2, 5)
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    target = encode_latents(boxes, areas)
    source = target.clone()
    source[..., :2] = 0.5
    source[..., 2] = 0.2
    constraints = torch.zeros(2, 5, 5)
    constraints[:, 0, 4] = 1
    constraints[:, 1, 4] = 2
    constraints[:, :2, 2] = 1
    constraints[0, 0, 0] = 1
    constraints[1, 1, 1] = 1
    return source, target, constraints


def test_projector_reduces_overlap_and_preserves_exact_area() -> None:
    source, target, constraints = _targets()
    constraints.zero_()
    output = project_hard_constraints(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=16,
    )
    assert output.remaining_overlap_pairs == 0
    assert output.max_area_error < 1.0e-5


def test_projector_preserves_fixed_and_preplaced_blocks() -> None:
    source, target, constraints = _targets()
    output = project_hard_constraints(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=16,
    )
    assert output.fixed_shape_error < 1.0e-5
    assert output.preplaced_max_error < 1.0e-5


def test_fixed_shape_block_remains_position_movable() -> None:
    source, target, constraints = _targets()
    constraints.zero_()
    constraints[:, 0, 0] = 1
    source_boxes = decode_boxes(source[0], torch.ones(5))
    output = project_hard_constraints(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=16,
    )
    assert output.fixed_shape_error < 1.0e-5
    assert not torch.allclose(output.boxes[0, 0, :2], source_boxes[0, :2])


def test_boundary_axes_lock_prevents_overlap_move_across_coded_edge() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.8, 0.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    constraints = torch.zeros(1, 2, 5)
    constraints[0, 0, 4] = 1
    movable = torch.ones(1, 2, dtype=torch.bool)
    free = _separate_overlaps(boxes, movable, constraints)
    locked = _separate_overlaps(
        boxes,
        movable,
        constraints,
        boundary_axes_locked=True,
    )
    assert free[0, 0, 0] != boxes[0, 0, 0]
    assert locked[0, 0, 0] == boxes[0, 0, 0]
    assert locked[0, 1, 0] > boxes[0, 1, 0]


def test_corner_block_remains_movable_when_corner_lock_is_disabled() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    constraints = torch.zeros(1, 2, 5)
    constraints[0, 0, 4] = 5
    movable = torch.ones(1, 2, dtype=torch.bool)
    locked = _separate_overlaps(
        boxes,
        movable,
        constraints,
        boundary_axes_locked=True,
    )
    released = _separate_overlaps(
        boxes,
        movable,
        constraints,
        boundary_axes_locked=True,
        lock_boundary_corners=False,
    )
    assert torch.equal(locked[0, 0], boxes[0, 0])
    assert not torch.equal(released[0, 0], boxes[0, 0])


def test_projector_shares_mib_shape_and_hits_boundary_codes() -> None:
    source, target, constraints = _targets()
    constraints[:, :2, 0:2] = 0
    output = project_hard_constraints(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=20,
    )
    assert output.mib_shape_violations == 0
    assert output.boundary_violations == 0


def test_projector_does_not_use_free_target_latent() -> None:
    source, target, constraints = _targets()
    constraints.zero_()
    changed = target.clone()
    changed[..., :2].add_(3.0)
    first = project_hard_constraints(source, target, constraints, torch.ones(2, 5))
    second = project_hard_constraints(source, changed, constraints, torch.ones(2, 5))
    soft = ~constraints[..., :2].ne(0).any(dim=-1)
    assert torch.allclose(
        first.boxes[soft],
        second.boxes[soft],
        atol=1.0e-6,
    )


def test_corrected_velocity_pulls_toward_projection() -> None:
    source, target, constraints = _targets()
    constraints.zero_()
    velocity = torch.ones_like(source)
    corrected, output = corrected_velocity(
        velocity,
        source,
        target,
        constraints,
        torch.ones(2, 5),
        step_size=0.25,
        strength=1.0,
    )
    plain = source + velocity * 0.25
    assert torch.isfinite(corrected).all()
    assert not torch.allclose(plain, output.latent)
    assert output.remaining_overlap_pairs <= 1


def test_projector_is_autograd_safe() -> None:
    source, target, constraints = _targets()
    constraints.zero_()
    source.requires_grad_(True)
    projected = project_hard_constraints_latent(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=3,
        sparse_pairs=True,
    )
    projected.square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert source.grad.abs().sum() > 0


def test_grid_candidates_are_sparse_for_separated_boxes() -> None:
    boxes = torch.zeros(1, 8, 4)
    boxes[..., 2:] = 0.5
    boxes[0, :, 0] = torch.arange(8) * 2.0
    boxes[0, :, 1] = torch.arange(8) * 2.0
    batch_index, first, second = candidate_overlap_pairs(boxes, grid_size=16)
    assert batch_index.numel() == first.numel() == second.numel()
    assert first.numel() < 8 * 7 // 2


def test_sparse_and_dense_overlap_separation_are_close() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.7, 0.2, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    movable = torch.ones(1, 2, dtype=torch.bool)
    constraints = torch.zeros(1, 2, 5)
    sparse = _separate_overlaps(
        boxes,
        movable,
        constraints,
        sparse_pairs=True,
    )
    dense = _separate_overlaps(
        boxes,
        movable,
        constraints,
        sparse_pairs=False,
    )
    torch.testing.assert_close(sparse, dense)


def test_boundary_projection_moves_fixed_shape_but_not_preplaced() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]]]
    )
    source = encode_latents(boxes, torch.ones(1, 3))
    constraints = torch.zeros(1, 3, 5)
    constraints[:, 0, 0] = 1
    constraints[:, 0, 4] = 2
    constraints[:, 1, 1] = 1
    constraints[:, 1, 4] = 2
    before = decode_boxes(source[0], torch.ones(3))
    projected = project_boundary(source, constraints, torch.ones(1, 3))
    after = decode_boxes(projected[0], torch.ones(3))
    assert not torch.allclose(after[0, :2], before[0, :2])
    torch.testing.assert_close(after[1], before[1])


def test_conflict_aware_projection_reports_joint_energy() -> None:
    source, target, constraints = _targets()
    source.requires_grad_(True)
    output = project_hard_constraints(
        source,
        target,
        constraints,
        torch.ones(2, 5),
        iterations=4,
        sparse_pairs=True,
        projection_strategy="conflict_aware",
    )
    assert torch.isfinite(
        joint_conflict_energy(
            output.latent,
            constraints,
            torch.ones(2, 5),
        )
    ).all()
    assert output.candidate_pair_count >= 0
    assert output.joint_conflict_energy >= 0
    output.latent.square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
