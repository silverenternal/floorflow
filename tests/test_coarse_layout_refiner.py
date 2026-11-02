import torch

from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner


def test_residual_layout_refiner_starts_as_identity():
    model = CoarseLayoutRefiner(5, base_channels=8, residual=True)
    coarse = torch.randn(2, 5, 32, 32)
    footprint = torch.ones(2, 1, 32, 32)
    output = model(coarse, footprint)
    torch.testing.assert_close(output, coarse)
