from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.control_cache import read_control_cache, write_control_cache


def test_control_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_control_cache(
        path,
        room_logits=torch.randn(1, 24, 256, 256),
        program=torch.tensor([[True] + [False] * 23]),
        structure=torch.rand(1, 38, 128, 128),
        furniture=torch.rand(1, 53, 192, 192),
        support=torch.rand(1, 28, 192, 192),
        room_size=16,
    )
    payload = read_control_cache(path)
    assert payload["room_logits"].shape == (24, 16, 16)
    assert payload["program"].shape == (24,)
    assert payload["structure"].shape == (38, 128, 128)
    assert payload["furniture"].shape == (53, 192, 192)
    assert payload["support"].shape == (28, 192, 192)


def test_control_cache_sanitizes_nonfinite_and_extreme_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sanitized.npz"
    structure = torch.zeros(1, 4, 8, 8)
    structure[0, 0, 0, 0] = float("inf")
    structure[0, 1, 0, 0] = -100.0
    write_control_cache(
        path,
        room_logits=torch.full((1, 4, 8, 8), 1.0e9),
        program=torch.ones(1, 4, dtype=torch.bool),
        structure=structure,
        furniture=torch.full((1, 4, 8, 8), -float("inf")),
        support=torch.full((1, 4, 8, 8), float("nan")),
        room_size=8,
    )
    payload = read_control_cache(path)
    assert torch.isfinite(payload["room_logits"]).all()
    assert float(payload["room_logits"].max()) == 20.0
    assert float(payload["structure"].min()) == 0.0
    assert float(payload["structure"].max()) == 1.0
    assert float(payload["furniture"].min()) == 0.0
    assert torch.isfinite(payload["support"]).all()
    assert float(payload["support"].max()) == 0.0
