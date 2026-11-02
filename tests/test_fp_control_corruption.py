from __future__ import annotations

import torch

from floorflow.data.control_perturbation import (
    control_validity_masks,
    corrupt_controls,
    random_spatial_block_mask,
    spatial_block,
)


def test_spatial_block_is_deterministic_and_sized() -> None:
    first = spatial_block((20, 20), 0.25, seed=7)
    second = spatial_block((20, 20), 0.25, seed=7)
    assert torch.equal(first, second)
    assert int(first.sum()) == 100
    assert int(spatial_block((20, 20), 0.0, seed=7).sum()) == 0


def test_control_corruption_preserves_program() -> None:
    controls = {
        "room_logits": torch.ones(4, 8, 8),
        "program": torch.tensor([True, False, True, False]),
        "structure": torch.ones(6, 8, 8),
        "furniture": torch.ones(6, 8, 8),
        "support": torch.ones(6, 8, 8),
    }
    corrupted = corrupt_controls(controls, 0.25, seed=3)
    masks = control_validity_masks(controls, 0.25, seed=3)
    assert torch.equal(corrupted["program"], controls["program"])
    assert int(torch.count_nonzero(corrupted["room_logits"])) < 4 * 8 * 8
    assert float(corrupted["structure"].min()) == 0.0
    assert set(masks) == {
        "valid_room_logits",
        "valid_structure",
        "valid_furniture",
        "valid_support",
    }


def test_random_spatial_block_masks_are_batched() -> None:
    masks = random_spatial_block_mask(
        3,
        (16, 16),
        0.1,
        generator=torch.Generator().manual_seed(17),
    )
    assert masks.shape == (3, 16, 16)
    assert masks.dtype == torch.bool
    assert int(masks[0].sum()) > 0
