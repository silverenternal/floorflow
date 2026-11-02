"""Tests for explicit raster-instance warm-start precedence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_raster_instance_control import compatible_initial_state


def test_preserved_room_encoder_keeps_explicit_checkpoint() -> None:
    current = {
        "room.encoder.weight": torch.zeros(2, 2),
        "mask_decoder.query.weight": torch.zeros(3, 3),
        "mask_decoder.head.weight": torch.zeros(1, 3),
    }
    candidate = {
        "room.encoder.weight": torch.ones(2, 2),
        "mask_decoder.query.weight": torch.ones(3, 3),
        "mask_decoder.head.weight": torch.ones(2, 2),
    }
    selected = compatible_initial_state(
        candidate,
        current,
        exclude_room_encoder=True,
    )
    assert set(selected) == {"mask_decoder.query.weight"}


def test_regular_warm_start_accepts_all_compatible_tensors() -> None:
    current = {
        "room.encoder.weight": torch.zeros(2, 2),
        "mask_decoder.query.weight": torch.zeros(3, 3),
    }
    candidate = {
        "room.encoder.weight": torch.ones(2, 2),
        "mask_decoder.query.weight": torch.ones(4, 4),
    }
    selected = compatible_initial_state(candidate, current)
    assert set(selected) == {"room.encoder.weight"}
