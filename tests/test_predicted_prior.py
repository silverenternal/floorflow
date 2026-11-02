import torch

from floorflow.inference.predicted_prior import low_confidence_prior_mask


def test_low_confidence_mask_stays_target_free_and_respects_footprint() -> None:
    state = torch.tensor(
        [
            [
                [[0.9, 0.8, 0.1], [0.7, 0.2, 0.3]],
                [[0.1, 0.2, 0.0], [0.3, 0.1, 0.1]],
            ]
        ]
    )
    footprint = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])
    mask = low_confidence_prior_mask(
        state,
        footprint,
        hidden_fraction=0.4,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 2, 3)
    assert not bool(mask[0, 0, 1, 2])
    assert int(mask.sum()) == 2
