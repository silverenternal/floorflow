from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.trinity_backbone_init import (
    initialize_trinity_backbone,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "third_party/Trinity/checkpoints/d192"


def _model(depth: int = 12) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=192,
        heads=3,
        depth=depth,
        head_dim=64,
        typed_layers=1,
    )


def test_trinity_backbone_initializes_strictly_without_touching_typed() -> None:
    model = _model()
    typed_before = model.typed.block_input[0].weight.detach().clone()

    provenance = initialize_trinity_backbone(model, RELEASE)

    reference = load_file(str(RELEASE / "model.safetensors"))
    first_key = next(iter(reference))
    torch.testing.assert_close(
        model.backbone.state_dict()[first_key],
        reference[first_key],
    )
    torch.testing.assert_close(
        model.typed.block_input[0].weight,
        typed_before,
    )
    assert provenance["strict_load"] is True
    assert provenance["backbone_parameters"] == 5_856_400
    assert len(str(provenance["weights_sha256"])) == 64


def test_trinity_backbone_init_rejects_depth_mismatch() -> None:
    with pytest.raises(RuntimeError, match="size mismatch|Missing|Unexpected"):
        initialize_trinity_backbone(_model(depth=10), RELEASE)
