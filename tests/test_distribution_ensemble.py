import torch

from floorflow.inference.distribution_ensemble import (
    blend_distribution_outputs,
)


def test_distribution_ensemble_blends_nested_logits_only() -> None:
    primary = {
        "pixel": {"logits": torch.zeros(1, 3, 2, 2), "name": "primary"},
        "elements": {
            "room": {
                "mask_logits": torch.zeros(1, 2, 2, 2),
                "type_logits": torch.zeros(1, 2, 3),
            }
        },
    }
    secondary = {
        "pixel": {"logits": torch.full((1, 3, 2, 2), 4.0)},
        "elements": {
            "room": {
                "mask_logits": torch.full((1, 2, 2, 2), 2.0),
                "type_logits": torch.full((1, 2, 3), 8.0),
            }
        },
    }
    output = blend_distribution_outputs(primary, secondary, 0.25)
    assert torch.allclose(output["pixel"]["logits"], torch.ones(1, 3, 2, 2))
    assert torch.allclose(
        output["elements"]["room"]["mask_logits"],
        torch.full((1, 2, 2, 2), 0.5),
    )
    assert torch.allclose(
        output["elements"]["room"]["type_logits"],
        torch.full((1, 2, 3), 2.0),
    )
    assert output["pixel"]["name"] == "primary"
