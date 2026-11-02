from __future__ import annotations

import torch

from floorflow.heads.coarse_structure_refiner import CoarseStructureRefiner
from floorflow.heads.joint_control_repair import (
    JointControlRepair,
    fuse_valid_regions,
)


def test_valid_region_fusion_preserves_supplied_control() -> None:
    supplied = torch.ones(1, 2, 4, 4)
    predicted = torch.zeros(1, 2, 4, 4)
    valid = torch.zeros(1, 4, 4, dtype=torch.bool)
    valid[:, :2] = True
    fused = fuse_valid_regions(supplied, predicted, valid)
    assert torch.all(fused[:, :, :2] == 1.0)
    assert torch.all(fused[:, :, 2:] == 0.0)


def test_joint_repair_shapes_and_masked_gradients() -> None:
    model = JointControlRepair(
        3,
        2,
        room_base_channels=4,
        structure_base_channels=4,
    )
    room_logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    footprint = torch.ones(1, 32, 32)
    structure = torch.zeros(1, 8, 8, 8)
    structure[:, 3] = 1.0
    room_valid = torch.zeros(1, 16, 16, dtype=torch.bool)
    room_valid[:, 4:12, 4:12] = True
    structure_valid = torch.zeros(1, 8, 8, dtype=torch.bool)
    structure_valid[:, 2:6, 2:6] = True
    output = model(
        room_logits,
        structure,
        footprint,
        room_valid,
        structure_valid,
    )
    assert output["room_logits"].shape == room_logits.shape
    assert output["structure_control"].shape == structure.shape
    assert torch.all(output["room_logits"][:, :, 4:12, 4:12] == room_logits[:, :, 4:12, 4:12])
    assert torch.all(output["structure_control"][:, :, 2:6, 2:6] == structure[:, :, 2:6, 2:6])
    output["room_logits_full"].sum().backward()
    assert room_logits.grad is not None


def test_structure_refiner_accepts_masked_input() -> None:
    model = CoarseStructureRefiner(8, 2, base_channels=4, masked_input=True)
    control = torch.zeros(1, 8, 8, 8)
    validity = torch.ones(1, 8, 8)
    output = model(control, validity_mask=validity)
    assert output["wall_logits"].shape == (1, 1, 16, 16)


def test_structure_only_mode_preserves_room_control() -> None:
    model = JointControlRepair(
        3,
        2,
        room_base_channels=4,
        structure_base_channels=4,
    )
    room_logits = torch.randn(1, 3, 16, 16)
    output = model(
        room_logits,
        torch.zeros(1, 8, 8, 8),
        torch.ones(1, 32, 32),
        torch.zeros(1, 16, 16),
        torch.zeros(1, 8, 8),
        repair_room=False,
    )
    assert torch.equal(output["room_logits"], room_logits)
    assert torch.equal(output["room_logits_full"], room_logits)


def test_typed_joint_repair_consumes_room_structure_context() -> None:
    model = JointControlRepair(
        3,
        2,
        room_base_channels=4,
        structure_base_channels=4,
        structure_contract="typed_v2",
    )
    room_logits = torch.randn(1, 3, 16, 16)
    structure = torch.zeros(1, 8, 8, 8)
    structure[:, 3] = 0.5
    structure[:, 7] = 1.0
    output = model(
        room_logits,
        structure,
        torch.ones(1, 32, 32),
        torch.zeros(1, 16, 16),
        torch.zeros(1, 8, 8),
        repair_room=False,
    )
    assert output["structure"]["wall_logits"].shape == (1, 1, 8, 8)
    assert output["structure"]["junction_logits"].shape == (1, 1, 8, 8)
    assert output["structure"]["junction_degree_logits"].shape == (1, 5, 8, 8)
    assert output["structure_context_diagnostics"]["room_boundary"] >= 0


def test_joint_repair_exposes_optional_centerline_without_wall_shift() -> None:
    model = JointControlRepair(
        3,
        2,
        room_base_channels=4,
        structure_base_channels=4,
        structure_contract="typed_v2",
        predict_centerline=True,
    )
    room_logits = torch.randn(1, 3, 16, 16)
    structure = torch.zeros(1, 8, 8, 8)
    structure[:, 3] = 0.5
    structure[:, 7] = 1.0
    output = model(
        room_logits,
        structure,
        torch.ones(1, 32, 32),
        torch.zeros(1, 16, 16),
        torch.zeros(1, 8, 8),
        repair_room=False,
    )
    assert output["structure"]["centerline_logits"].shape == (1, 1, 8, 8)
    assert torch.count_nonzero(output["structure"]["centerline_logits"]) == 0
