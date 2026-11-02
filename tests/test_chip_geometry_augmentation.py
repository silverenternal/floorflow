"""Tests for label-preserving geometry augmentation in chip training."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.chip_flow.model import encode_latents
from floorflow.chip_flow.train_evaluation import augment_geometry


def _batch() -> dict:
    boxes = torch.tensor(
        [
            [1.0, 2.0, 2.0, 1.0],
            [5.0, 4.0, 1.0, 3.0],
            [2.0, 5.0, 3.0, 2.0],
            [6.0, 6.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    areas = boxes[:, 2] * boxes[:, 3]
    constraints = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0, 4.0],
            [1.0, 0.0, 1.0, 2.0, 8.0],
        ]
    )
    features = torch.zeros((4, 10), dtype=torch.float32)
    features[:, 7:9] = 0.25
    return {
        "boxes": boxes[None],
        "areas": areas[None],
        "constraints": constraints[None],
        "features": features[None],
        "pins": torch.tensor(
            [[[1.0, 2.0], [4.5, 5.5], [8.0, 9.0]]],
            dtype=torch.float32,
        ),
        "n_pins": torch.tensor([2]),
        "target_latent": encode_latents(
            boxes[None],
            areas[None],
        ),
    }


def test_horizontal_flip_reflects_target_and_swaps_boundary_codes() -> None:
    batch = _batch()
    original_boxes = batch["boxes"].clone()
    augmented = augment_geometry(
        batch,
        "hflip",
        np.random.default_rng(19),
    )

    assert torch.equal(augmented["areas"], _batch()["areas"])
    assert torch.equal(
        augmented["constraints"][..., :4],
        _batch()["constraints"][..., :4],
    )
    assert augmented["constraints"][0, :, 4].tolist() == [2.0, 1.0, 4.0, 8.0]
    assert augmented["pins"][0, :2, 0].tolist() == [8.0, 4.5]
    assert not torch.equal(augmented["boxes"], original_boxes)
    assert torch.allclose(
        augmented["target_latent"],
        encode_latents(augmented["boxes"], augmented["areas"]),
    )


def test_dihedral_is_reproducible_and_preserves_block_areas() -> None:
    first = augment_geometry(
        _batch(),
        "dihedral",
        np.random.default_rng(20),
    )
    second = augment_geometry(
        _batch(),
        "dihedral",
        np.random.default_rng(20),
    )

    assert first["geometry_augmentation"].tolist() == second[
        "geometry_augmentation"
    ].tolist()
    assert torch.equal(first["boxes"], second["boxes"])
    assert torch.allclose(first["areas"], _batch()["areas"])
    assert torch.allclose(
        first["boxes"][..., 2] * first["boxes"][..., 3],
        first["areas"],
    )


def test_augmentation_supports_multiple_batch_rows() -> None:
    batch = _batch()
    batched = {
        key: torch.cat([value, value], dim=0)
        for key, value in batch.items()
    }
    augmented = augment_geometry(
        batched,
        "dihedral",
        np.random.default_rng(21),
    )

    assert augmented["boxes"].shape[0] == 2
    assert torch.allclose(
        augmented["boxes"][..., 2] * augmented["boxes"][..., 3],
        augmented["areas"],
    )


def test_augmentation_probability_zero_preserves_batch() -> None:
    batch = _batch()
    original = batch["boxes"].clone()
    augmented = augment_geometry(
        batch,
        "dihedral",
        np.random.default_rng(22),
        probability=0.0,
    )
    torch.testing.assert_close(augmented["boxes"], original)
    assert augmented["geometry_augmentation"].tolist() == ["none"]
