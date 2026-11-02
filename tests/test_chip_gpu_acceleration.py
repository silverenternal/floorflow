from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn

from floorflow.chip_flow.batch_prefetch import BufferedBatchStream
from floorflow.chip_flow.cache import (
    PackedCaseStore,
    prepare_training_store,
    write_packed_split,
)
from floorflow.chip_flow.projector_geometry import (
    boundary_violation_count,
    joint_conflict_energy,
    mib_shape_violation_count,
)
from floorflow.chip_flow.training_runtime import (
    clip_optimizer_gradients_fail_closed,
    make_optimizer,
)
from floorflow.chip_physics import group_penalty_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA equivalence requires a CUDA device",
)


def _group_reference(
    boxes: torch.Tensor,
    clusters: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    total = 0.0
    for cluster in torch.unique(clusters):
        if float(cluster) <= 0.0:
            continue
        members = torch.nonzero(clusters == cluster).flatten()
        if members.numel() < 2:
            continue
        local = boxes[members]
        x0, y0 = local[:, 0], local[:, 1]
        x1, y1 = x0 + local[:, 2], y0 + local[:, 3]
        gap_x = torch.maximum(x0[:, None], x0[None, :]) - torch.minimum(
            x1[:, None],
            x1[None, :],
        )
        gap_y = torch.maximum(y0[:, None], y0[None, :]) - torch.minimum(
            y1[:, None],
            y1[None, :],
        )
        connected = ((gap_x <= 0) & (gap_y < 0)) | (
            (gap_y <= 0) & (gap_x < 0)
        )
        gap = torch.where(
            connected,
            torch.zeros_like(gap_x),
            torch.maximum(gap_x, gap_y).clamp_min(1.0e-3),
        )
        total += float(minimum_spanning_tree(gap.numpy()).sum())
    return boxes.sum() * 0.0 + total / areas.sum().sqrt().clamp_min(1.0e-6)


def test_group_penalty_cuda_matches_exact_protocol() -> None:
    generator = torch.Generator().manual_seed(7)
    boxes = torch.rand(3, 9, 4, generator=generator)
    boxes[..., 2:] = boxes[..., 2:].abs() + 0.2
    areas = boxes[..., 2] * boxes[..., 3]
    clusters = torch.tensor(
        [[0, 1, 1, 1, 2, 2, 0, 2, 1]] * 3
    )
    expected = torch.stack(
        [
            _group_reference(boxes[index], clusters[index], areas[index])
            for index in range(3)
        ]
    )
    cpu = group_penalty_batch(boxes, clusters, areas, workers=2)
    cuda = group_penalty_batch(
        boxes.cuda(),
        clusters.cuda(),
        areas.cuda(),
        workers=2,
    )
    torch.testing.assert_close(cpu, expected)
    torch.testing.assert_close(cuda.cpu(), expected)


def test_projector_diagnostics_cuda_matches_cpu() -> None:
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 0.0, 2.0, 1.0],
                [4.0, 0.0, 2.0, 1.0],
                [0.5, 2.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 3.0, 1.0],
                [3.0, 0.0, 3.0, 1.0],
                [0.0, 1.0, 2.0, 2.0],
                [2.0, 1.0, 2.0, 2.0],
                [4.0, 1.0, 2.0, 2.0],
            ],
        ]
    )
    constraints = torch.zeros(2, 5, 5)
    constraints[..., 2] = torch.tensor([1, 1, 2, 2, 0])
    constraints[..., 4] = torch.tensor([1, 2, 4, 8, 0])
    areas = boxes[..., 2] * boxes[..., 3]
    latent = torch.stack(
        (
            boxes[..., 0] + boxes[..., 2] / 2,
            boxes[..., 1] + boxes[..., 3] / 2,
            boxes[..., 2] / boxes[..., 3],
        ),
        dim=-1,
    )
    cuda_boxes = boxes.cuda()
    cuda_constraints = constraints.cuda()
    assert boundary_violation_count(cuda_boxes, cuda_constraints) == (
        boundary_violation_count(boxes, constraints)
    )
    assert mib_shape_violation_count(cuda_boxes, cuda_constraints) == (
        mib_shape_violation_count(boxes, constraints)
    )
    torch.testing.assert_close(
        joint_conflict_energy(
            latent.cuda(),
            cuda_constraints,
            areas.cuda(),
        ).cpu(),
        joint_conflict_energy(latent, constraints, areas),
    )


def _case_columns() -> dict:
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 1.0], [2.0, 0.0, 1.0, 1.0], [0.0, 1.0, 3.0, 1.0]]
    )
    return {
        "areas": boxes[:, 2] * boxes[:, 3],
        "boxes": boxes,
        "constraints": torch.tensor(
            [[1, 0, 0, 1, 1], [1, 0, 1, 0, 2], [0, 0, 0, 0, 4]]
        ).float(),
        "features": torch.ones(3, 10),
        "b2b": torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]),
        "p2b": torch.tensor([[0.0, 0.0, 1.0]]),
        "pins": torch.tensor([[0.0, 0.0]]),
        "source": "synthetic",
        "source_index": 0,
    }


def test_device_cache_and_prefetch_match_cpu_source(tmp_path) -> None:
    cases = [_case_columns(), _case_columns()]
    cases[1]["source_index"] = 1
    write_packed_split(cases, tmp_path)
    source = PackedCaseStore(tmp_path)
    resident, status = prepare_training_store(
        source,
        torch.device("cuda:0"),
        enabled=True,
        max_gib=1.0,
    )
    assert status["resident"]
    indices = source.indices(3)
    expected = source.batch(indices)
    actual = resident.batch(indices)
    torch.cuda.synchronize()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value.cuda(), actual[name])
        else:
            assert value == actual[name]
    cached_before = resident.arrays["boxes"].clone()
    with BufferedBatchStream(
        resident,
        np.random.default_rng(4),
        batch_size=1,
        device=torch.device("cuda:0"),
        batches=2,
        allowed_by_count=resident.by_block_count,
        geometry_augment="dihedral",
    ) as stream:
        batches = list(stream)
    assert len(batches) == 2
    assert all(batch["boxes"].is_cuda for batch in batches)
    torch.testing.assert_close(resident.arrays["boxes"], cached_before)


def test_grouped_clipping_cuda_matches_cpu() -> None:
    torch.manual_seed(5)
    cpu = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    cuda = copy.deepcopy(cpu).cuda()
    for (_, parameter), (_, cuda_parameter) in zip(
        cpu.named_parameters(),
        cuda.named_parameters(),
    ):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient
        cuda_parameter.grad = gradient.cuda()
    cpu_optimizer = make_optimizer(
        cpu,
        lr=1.0e-3,
        weight_decay=0.0,
        fused=False,
    )
    cuda_optimizer = make_optimizer(
        cuda,
        lr=1.0e-3,
        weight_decay=0.0,
        fused=True,
    )
    expected = clip_optimizer_gradients_fail_closed(
        cpu,
        cpu_optimizer,
        max_norm=0.5,
    )
    actual = clip_optimizer_gradients_fail_closed(
        cuda,
        cuda_optimizer,
        max_norm=0.5,
    )
    assert actual["global_pre_clip_norm"] == pytest.approx(
        expected["global_pre_clip_norm"],
        rel=1.0e-6,
    )
    assert actual["global_post_clip_norm"] == pytest.approx(
        expected["global_post_clip_norm"],
        rel=1.0e-6,
    )
    for name in expected["groups"]:
        assert actual["groups"][name]["pre_clip_norm"] == pytest.approx(
            expected["groups"][name]["pre_clip_norm"],
            rel=1.0e-6,
        )
