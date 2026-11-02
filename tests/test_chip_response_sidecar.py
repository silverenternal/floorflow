from __future__ import annotations

import numpy as np
import torch

from floorflow.chip_flow.response_sidecar import (
    ResponseSidecar,
    build_response_sidecar,
    load_response_sidecar,
    write_response_sidecar,
)


def test_response_sidecar_roundtrip_and_attachment(tmp_path) -> None:
    sidecar = ResponseSidecar(
        np.asarray([2, 5]),
        np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32),
        np.asarray([1, 3]),
    )
    path = tmp_path / "response.npz"
    write_response_sidecar(path, sidecar)
    batch = {"_indices": np.asarray([5, 2]), "areas": torch.ones(2, 1)}
    load_response_sidecar(path).attach(batch)
    assert batch["legalizer_rung"].tolist() == [3, 1]
    torch.testing.assert_close(
        batch["legalizer_displacement"],
        torch.tensor([[[0.3, 0.4]], [[0.1, 0.2]]]),
    )


def test_response_sidecar_builds_from_boxes() -> None:
    before = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    after = before.clone()
    after[..., 0] += 0.25
    sidecar = build_response_sidecar(
        np.asarray([7]),
        before,
        after,
        torch.ones(1, 1),
        torch.tensor([2]),
    )
    assert sidecar.case_indices.tolist() == [7]
    assert sidecar.rung.tolist() == [2]


def test_response_sidecar_supports_padded_variable_block_counts(tmp_path) -> None:
    displacement = np.asarray(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.0, 0.0]]],
        dtype=np.float32,
    )
    sidecar = ResponseSidecar(
        np.asarray([1, 2]),
        displacement,
        np.asarray([2, 3]),
        lengths=np.asarray([2, 1]),
    )
    path = tmp_path / "variable.npz"
    write_response_sidecar(path, sidecar)
    batch = {"_indices": np.asarray([2]), "areas": torch.ones(1, 1)}
    load_response_sidecar(path).attach(batch)
    torch.testing.assert_close(
        batch["legalizer_displacement"],
        torch.tensor([[[0.5, 0.6]]]),
    )
